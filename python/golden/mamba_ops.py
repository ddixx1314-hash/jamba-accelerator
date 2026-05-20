"""Golden models for tiny Mamba/Jamba-like operations."""

import numpy as np


def _as_i64(value):
    return np.asarray(value, dtype=np.int64)


def fixed_max_signed(bits):
    return (1 << (bits - 1)) - 1


def fixed_min_signed(bits):
    return -(1 << (bits - 1))


def fixed_saturate(value, bits):
    """Saturate an integer or integer array to signed two's-complement range."""
    value = np.asarray(value, dtype=np.int64)
    return np.clip(value, fixed_min_signed(bits), fixed_max_signed(bits)).astype(np.int64)


def fixed_round_shift_right(value, shift):
    """Arithmetic right shift with away-from-zero rounding before the shift."""
    if shift < 0:
        raise ValueError("shift must be non-negative")

    value = np.asarray(value, dtype=np.int64)
    if shift == 0:
        return value

    half = np.int64(1 << (shift - 1))
    rounded = np.where(value >= 0, value + half, value - half)
    return (rounded >> shift).astype(np.int64)


def fixed_multiply_rescale(a, b, out_bits, shift):
    """Multiply, rounded-shift, and saturate to the output domain."""
    product = np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)
    shifted = fixed_round_shift_right(product, shift)
    return fixed_saturate(shifted, out_bits)


def fixed_saturating_add(a, b, out_bits):
    """Add and saturate to the output domain."""
    summed = np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64)
    return fixed_saturate(summed, out_bits)


def selective_scan(u, delta, A, B, C, D=None):
    """Small float selective scan reference.

    Args:
        u:   input  (L, D)
        delta: time delta (L, D)
        A:    state matrix (D, N)
        B:    input projection (L, N) or (L, D, N)
        C:    output projection (L, N) or (L, D, N)
        D:    skip connection (D,) or None

    Returns:
        y: output (L, D)
    """
    u = np.asarray(u, dtype=np.float32)
    delta = np.asarray(delta, dtype=np.float32)
    A = np.asarray(A, dtype=np.float32)
    B = np.asarray(B, dtype=np.float32)
    C = np.asarray(C, dtype=np.float32)

    skip = D
    L, dim = u.shape
    d_a, N = A.shape
    if d_a != dim:
        raise ValueError(f"A must have shape (D, N); got {A.shape} for D={dim}")

    if B.ndim == 2:
        B = np.broadcast_to(B[:, None, :], (L, dim, N))
    if C.ndim == 2:
        C = np.broadcast_to(C[:, None, :], (L, dim, N))
    if B.shape != (L, dim, N):
        raise ValueError(f"B must have shape (L, N) or (L, D, N); got {B.shape}")
    if C.shape != (L, dim, N):
        raise ValueError(f"C must have shape (L, N) or (L, D, N); got {C.shape}")

    h = np.zeros((dim, N), dtype=np.float32)
    y = np.zeros((L, dim), dtype=np.float32)

    for t in range(L):
        dA = np.exp(delta[t, :, None] * A)
        dB = delta[t, :, None] * B[t]
        h = dA * h + dB * u[t, :, None]
        y[t] = (h * C[t]).sum(axis=-1)

    if skip is not None:
        skip = np.asarray(skip, dtype=np.float32)
        y = y + skip * u

    return y


def rms_norm(x, eps=1e-6):
    """RMS Normalization"""
    x = np.asarray(x, dtype=np.float32)
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return np.divide(x, rms, out=np.zeros_like(x), where=rms != 0)


def tiny_mamba_state_update(state, x, a, b):
    """Integer reference for the Chisel tiny state update."""
    state = np.asarray(state, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return state * a + x * b


def serial_selective_scan_step(state, x, a, b, c):
    """Op-by-op reference matching SerialSelectiveScanMini.

    Three operations per lane, lane-serial:
      op0: recurrent = state * a
      op1: next_state = recurrent + x * b
      op2: y = next_state * c   (uses NEW state, matching FSM timing of parallel version)

    Returns (next_state, y) both as int64 arrays.
    """
    state = np.asarray(state, dtype=np.int64)
    x = np.asarray(x, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    c = np.asarray(c, dtype=np.int64)
    recurrent = state * a
    next_state = recurrent + x * b
    y = next_state * c
    return next_state, y


def quantized_mamba_step(state, x, a, b, c, data_bits=8, acc_bits=32):
    """Serial selective scan with precision boundaries applied at each operator stage.

    Simulates how SerialSelectiveScanMini behaves under reduced-precision
    configurations (data_bits controls input/weight width; acc_bits controls
    the accumulator width).

    Returns (next_state, y) both saturated to acc_bits.
    """
    def sat_data(v):
        return fixed_saturate(v, data_bits)

    def sat_acc(v):
        return fixed_saturate(v, acc_bits)

    state = sat_acc(np.asarray(state, dtype=np.int64))
    x = sat_data(np.asarray(x, dtype=np.int64))
    a = sat_data(np.asarray(a, dtype=np.int64))
    b = sat_data(np.asarray(b, dtype=np.int64))
    c = sat_data(np.asarray(c, dtype=np.int64))

    recurrent = sat_acc(state * a)
    next_state = sat_acc(recurrent + x * b)
    y = sat_acc(next_state * c)
    return next_state, y


def quantized_attention_step(q, keys, values, data_bits=8, acc_bits=32):
    """Attention decode with precision boundaries at score and value accumulation.

    Mirrors tiny_attention_decode but applies saturation after each dot product,
    matching how AttentionMixerMini would behave under INT4/INT6/INT8 configs.

    Returns (scores, y) both saturated to acc_bits.
    """
    def sat_data(v):
        return fixed_saturate(v, data_bits)

    def sat_acc(v):
        return fixed_saturate(v, acc_bits)

    q = sat_data(np.asarray(q, dtype=np.int64))
    keys = sat_data(np.asarray(keys, dtype=np.int64))
    values = sat_data(np.asarray(values, dtype=np.int64))

    scores = sat_acc(keys @ q)
    y = sat_acc(scores @ values)
    return scores, y


def tiny_attention_decode(q, keys, values):
    """Integer reference for AttentionDecodeTiny without softmax."""
    q = np.asarray(q, dtype=np.int64)
    keys = np.asarray(keys, dtype=np.int64)
    values = np.asarray(values, dtype=np.int64)

    scores = keys @ q
    y = scores @ values
    return scores, y


def tiny_rms_norm_approx(x, weight):
    """Integer reference for RmsNormApprox."""
    x = np.asarray(x, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.int64)
    mean_square = int(np.sum(x * x) // x.size)
    denominator = 1 if mean_square == 0 else mean_square
    return (x * weight) // denominator, mean_square


def tiny_linear4(x, weight, bias):
    """Integer reference for Linear4."""
    x = np.asarray(x, dtype=np.int64)
    weight = np.asarray(weight, dtype=np.int64)
    bias = np.asarray(bias, dtype=np.int64)
    return weight @ x + bias


def _narrow_signed(value, bits):
    """Truncate to a signed two's-complement value with the given bit width."""
    value = np.asarray(value, dtype=np.int64)
    modulus = np.int64(1 << bits)
    sign = np.int64(1 << (bits - 1))
    return ((value + sign) % modulus - sign).astype(np.int64)


def jamba2_mini_fixture(hidden_size=4, num_layers=4, attention_layer_period=4, context_length=4):
    """Deterministic integer fixture for Jamba2 mini golden traces."""
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if attention_layer_period <= 0:
        raise ValueError("attention_layer_period must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    eye = np.eye(hidden_size, dtype=np.int64)
    reverse_eye = np.flipud(eye)
    ones = np.ones(hidden_size, dtype=np.int64)
    twos = np.full(hidden_size, 2, dtype=np.int64)
    zeros = np.zeros(hidden_size, dtype=np.int64)

    return {
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "attention_layer_period": attention_layer_period,
        "attention_layer_offset": attention_layer_period - 1,
        "context_length": context_length,
        "norm1_weight": ones,
        "norm2_weight": ones,
        "mamba_in_weight": eye,
        "mamba_in_bias": zeros,
        "mamba_b_weight": np.zeros((hidden_size, hidden_size), dtype=np.int64),
        "mamba_b_bias": twos,
        "mamba_c_weight": np.zeros((hidden_size, hidden_size), dtype=np.int64),
        "mamba_c_bias": ones,
        "mamba_a": ones,
        "conv_kernel": np.ones((4, hidden_size), dtype=np.int64),
        "q_weight": eye,
        "q_bias": zeros,
        "k_weight": eye,
        "k_bias": zeros,
        "v_weight": eye,
        "v_bias": zeros,
        "attn_out_weight": eye,
        "attn_out_bias": zeros,
        "mlp_gate_weight": eye,
        "mlp_gate_bias": ones,
        "mlp_up_weight": reverse_eye,
        "mlp_up_bias": zeros,
        "mlp_down_weight": eye,
        "mlp_down_bias": zeros,
        "attention_norm_shift": 2,
    }


def _attention_layer_for_index(layer_index, fixture):
    period = int(fixture["attention_layer_period"])
    offset = int(fixture["attention_layer_offset"])
    return layer_index % period == offset


def mamba_mixer_step(x, state, fixture, conv_history=None):
    """Integer Jamba2 mini Mamba mixer step with token-serial state update."""
    projected = tiny_linear4(x, fixture["mamba_in_weight"], fixture["mamba_in_bias"])
    b = tiny_linear4(x, fixture["mamba_b_weight"], fixture["mamba_b_bias"])
    c = tiny_linear4(x, fixture["mamba_c_weight"], fixture["mamba_c_bias"])
    kernel = _as_i64(fixture["conv_kernel"])
    if kernel.ndim == 1:
        kernel = np.broadcast_to(kernel[None, :], (1, kernel.shape[0]))

    taps = kernel.shape[0]
    if conv_history is None:
        conv_history = np.zeros((max(taps - 1, 0), projected.shape[0]), dtype=np.int64)
    else:
        conv_history = _as_i64(conv_history)

    conv = projected * kernel[0]
    for tap in range(1, taps):
        conv = conv + conv_history[tap - 1] * kernel[tap]

    next_state = tiny_mamba_state_update(state, conv, fixture["mamba_a"], b)
    y = next_state * c

    if taps > 1:
        next_conv_history = np.zeros_like(conv_history)
        if taps > 2:
            next_conv_history[1:] = conv_history[:-1]
        next_conv_history[0] = projected
    else:
        next_conv_history = conv_history

    return {
        "projected": projected,
        "conv": conv,
        "conv_history": next_conv_history,
        "b": b,
        "c": c,
        "state": next_state,
        "y": y,
    }


def _active_cache(cache, valid_count, write_index):
    if valid_count < cache.shape[0]:
        return cache[:valid_count]

    # Return entries from oldest to newest when the circular cache is full.
    return np.concatenate((cache[write_index:], cache[:write_index]), axis=0)


def attention_mixer_step(x, kv_cache, write_index, valid_count, fixture):
    """Integer Jamba2 mini attention mixer with circular KV cache."""
    context_length = int(fixture["context_length"])
    q = tiny_linear4(x, fixture["q_weight"], fixture["q_bias"])
    k = tiny_linear4(x, fixture["k_weight"], fixture["k_bias"])
    v = tiny_linear4(x, fixture["v_weight"], fixture["v_bias"])

    next_cache = np.array(kv_cache, dtype=np.int64, copy=True)
    next_cache[write_index, 0] = k
    next_cache[write_index, 1] = v
    next_write_index = (write_index + 1) % context_length
    next_valid_count = min(valid_count + 1, context_length)

    active = _active_cache(next_cache, next_valid_count, next_write_index)
    keys = active[:, 0, :]
    values = active[:, 1, :]
    scores = keys @ q
    shift = int(fixture["attention_norm_shift"])
    weights = scores >> shift
    y = weights @ values
    y = tiny_linear4(y, fixture["attn_out_weight"], fixture["attn_out_bias"])
    return {
        "q": q,
        "k": k,
        "v": v,
        "scores": scores,
        "weights": weights,
        "y": y,
        "kv_cache": next_cache,
        "kv_write_index": next_write_index,
        "kv_valid_count": next_valid_count,
    }


def dense_mlp_step(x, fixture):
    """Integer Dense MLP using gate/up/down projections and ReLU gate."""
    gate = tiny_linear4(x, fixture["mlp_gate_weight"], fixture["mlp_gate_bias"])
    up = tiny_linear4(x, fixture["mlp_up_weight"], fixture["mlp_up_bias"])
    activated_gate = np.maximum(gate, 0)
    hidden = activated_gate * up
    y = tiny_linear4(hidden, fixture["mlp_down_weight"], fixture["mlp_down_bias"])
    return {
        "gate": gate,
        "up": up,
        "activated_gate": activated_gate,
        "hidden": hidden,
        "y": y,
    }


def jamba2_mini_layer_step(
    x,
    layer_index,
    state,
    kv_cache,
    write_index,
    valid_count,
    fixture,
    conv_history=None,
):
    """One Jamba2 mini layer step: norm, mixer, residual, norm, MLP, residual."""
    norm1, norm1_mean_square = tiny_rms_norm_approx(x, fixture["norm1_weight"])
    if _attention_layer_for_index(layer_index, fixture):
        mixer_type = "attention"
        mixer = attention_mixer_step(norm1, kv_cache, write_index, valid_count, fixture)
        next_state = state
        next_kv_cache = mixer["kv_cache"]
        next_write_index = mixer["kv_write_index"]
        next_valid_count = mixer["kv_valid_count"]
    else:
        mixer_type = "mamba"
        mixer = mamba_mixer_step(norm1, state, fixture, conv_history)
        next_state = mixer["state"]
        next_kv_cache = kv_cache
        next_write_index = write_index
        next_valid_count = valid_count
        conv_history = mixer["conv_history"]

    first_residual = x + mixer["y"]
    norm2, norm2_mean_square = tiny_rms_norm_approx(first_residual, fixture["norm2_weight"])
    mlp = dense_mlp_step(norm2, fixture)
    y = first_residual + mlp["y"]
    return {
        "layer_index": layer_index,
        "mixer_type": mixer_type,
        "input": x,
        "norm1": norm1,
        "norm1_mean_square": norm1_mean_square,
        "mixer": mixer,
        "first_residual": first_residual,
        "norm2": norm2,
        "norm2_mean_square": norm2_mean_square,
        "mlp": mlp,
        "final_residual": y,
        "state": next_state,
        "kv_cache": next_kv_cache,
        "kv_write_index": next_write_index,
        "kv_valid_count": next_valid_count,
        "conv_history": conv_history,
        "moe_dispatch_valid": False,
        "moe_combine_valid": False,
    }


def jamba2_mini_core_trace(tokens, fixture=None):
    """Generate a deterministic multi-token Jamba2 mini core trace."""
    if fixture is None:
        fixture = jamba2_mini_fixture()

    tokens = _as_i64(tokens)
    hidden_size = int(fixture["hidden_size"])
    if tokens.ndim != 2 or tokens.shape[1] != hidden_size:
        raise ValueError(f"tokens must have shape (N, {hidden_size})")

    num_layers = int(fixture["num_layers"])
    context_length = int(fixture["context_length"])
    states = np.zeros((num_layers, hidden_size), dtype=np.int64)
    kv_caches = np.zeros((num_layers, context_length, 2, hidden_size), dtype=np.int64)
    conv_taps = _as_i64(fixture["conv_kernel"]).shape[0]
    conv_histories = np.zeros((num_layers, max(conv_taps - 1, 0), hidden_size), dtype=np.int64)
    write_indices = np.zeros(num_layers, dtype=np.int64)
    valid_counts = np.zeros(num_layers, dtype=np.int64)
    token_traces = []

    for token_index, token in enumerate(tokens):
        x = token
        layer_traces = []
        for layer_index in range(num_layers):
            layer = jamba2_mini_layer_step(
                x=x,
                layer_index=layer_index,
                state=states[layer_index],
                kv_cache=kv_caches[layer_index],
                write_index=int(write_indices[layer_index]),
                valid_count=int(valid_counts[layer_index]),
                fixture=fixture,
                conv_history=conv_histories[layer_index],
            )
            x = layer["final_residual"]
            states[layer_index] = layer["state"]
            kv_caches[layer_index] = layer["kv_cache"]
            write_indices[layer_index] = layer["kv_write_index"]
            valid_counts[layer_index] = layer["kv_valid_count"]
            conv_histories[layer_index] = layer["conv_history"]
            layer_traces.append(layer)

        token_traces.append({
            "token_index": token_index,
            "input": token,
            "layers": layer_traces,
            "output": x,
            "states": np.array(states, copy=True),
            "conv_histories": np.array(conv_histories, copy=True),
            "kv_write_indices": np.array(write_indices, copy=True),
            "kv_valid_counts": np.array(valid_counts, copy=True),
        })

    return {
        "fixture": fixture,
        "tokens": tokens,
        "trace": token_traces,
        "final_states": states,
        "final_conv_histories": conv_histories,
        "final_kv_caches": kv_caches,
        "final_kv_write_indices": write_indices,
        "final_kv_valid_counts": valid_counts,
    }


def _tile_demo_mamba_step(x, state, fixture, conv_history):
    """Match the current-cycle Chisel Mamba mixer visibility used by Jamba2MiniTile."""
    projected = _narrow_signed(tiny_linear4(x, fixture["mamba_in_weight"], fixture["mamba_in_bias"]), 8)
    b = _narrow_signed(tiny_linear4(x, fixture["mamba_b_weight"], fixture["mamba_b_bias"]), 8)
    c = _narrow_signed(tiny_linear4(x, fixture["mamba_c_weight"], fixture["mamba_c_bias"]), 8)
    kernel = _as_i64(fixture["conv_kernel"])

    conv = projected * kernel[0]
    for tap in range(1, kernel.shape[0]):
        conv = conv + conv_history[tap - 1] * kernel[tap]

    scan_x = _narrow_signed(conv, 8)
    next_state = state * fixture["mamba_a"] + scan_x * b
    visible_y = state * c

    next_conv_history = np.zeros_like(conv_history)
    if kernel.shape[0] > 1:
        if kernel.shape[0] > 2:
            next_conv_history[1:] = conv_history[:-1]
        next_conv_history[0] = projected

    return {
        "projected": projected,
        "conv": conv,
        "b": b,
        "c": c,
        "state": next_state,
        "conv_history": next_conv_history,
        "y": visible_y,
    }


def _tile_demo_attention_step(x, kv_cache, write_index, valid_count, fixture):
    """Match the current-cycle Chisel Attention mixer with write-through KV cache."""
    context_length = int(fixture["context_length"])
    q = fixed_saturate(tiny_linear4(x, fixture["q_weight"], fixture["q_bias"]), 8)
    k = fixed_saturate(tiny_linear4(x, fixture["k_weight"], fixture["k_bias"]), 8)
    v = fixed_saturate(tiny_linear4(x, fixture["v_weight"], fixture["v_bias"]), 8)

    next_cache = np.array(kv_cache, dtype=np.int64, copy=True)
    next_cache[write_index, 0] = k
    next_cache[write_index, 1] = v
    next_write_index = (write_index + 1) % context_length
    next_valid_count = min(valid_count + 1, context_length)

    active = _active_cache(next_cache, next_valid_count, next_write_index)
    keys = active[:, 0, :]
    values = active[:, 1, :]
    scores = keys @ q
    weights = fixed_round_shift_right(scores, int(fixture["attention_norm_shift"]))
    raw_y = weights @ values
    out_x = fixed_saturate(raw_y, 8)
    y = tiny_linear4(out_x, fixture["attn_out_weight"], fixture["attn_out_bias"])

    return {
        "q": q,
        "k": k,
        "v": v,
        "scores": scores,
        "weights": weights,
        "raw_y": raw_y,
        "y": y,
        "kv_cache": next_cache,
        "kv_write_index": next_write_index,
        "kv_valid_count": next_valid_count,
    }


def _tile_demo_dense_mlp_step(x, fixture):
    """Match the current DenseMLPMini narrowing behavior."""
    gate = tiny_linear4(x, fixture["mlp_gate_weight"], fixture["mlp_gate_bias"])
    up = tiny_linear4(x, fixture["mlp_up_weight"], fixture["mlp_up_bias"])
    activated_gate = _narrow_signed(np.maximum(gate, 0), 8)
    hidden = _narrow_signed(activated_gate * _narrow_signed(up, 8), 8)
    y = tiny_linear4(hidden, fixture["mlp_down_weight"], fixture["mlp_down_bias"])
    return {
        "gate": gate,
        "up": up,
        "activated_gate": activated_gate,
        "hidden": hidden,
        "y": y,
    }


def jamba2_mini_tile_demo_trace(tokens, fixture=None):
    """Trace the Stage 13 Jamba2MiniTile demo-weight path with Chisel-visible timing."""
    if fixture is None:
        fixture = jamba2_mini_fixture(num_layers=4, attention_layer_period=4, context_length=8)

    tokens = _as_i64(tokens)
    hidden_size = int(fixture["hidden_size"])
    if tokens.ndim != 2 or tokens.shape[1] != hidden_size:
        raise ValueError(f"tokens must have shape (N, {hidden_size})")

    num_layers = int(fixture["num_layers"])
    context_length = int(fixture["context_length"])
    conv_taps = _as_i64(fixture["conv_kernel"]).shape[0]
    states = np.zeros((num_layers, hidden_size), dtype=np.int64)
    conv_histories = np.zeros((num_layers, max(conv_taps - 1, 0), hidden_size), dtype=np.int64)
    kv_caches = np.zeros((num_layers, context_length, 2, hidden_size), dtype=np.int64)
    write_indices = np.zeros(num_layers, dtype=np.int64)
    valid_counts = np.zeros(num_layers, dtype=np.int64)
    token_traces = []

    for token_index, token in enumerate(tokens):
        x = _narrow_signed(token, 8)
        layer_traces = []
        for layer_index in range(num_layers):
            norm1, norm1_mean_square = tiny_rms_norm_approx(x, fixture["norm1_weight"])
            norm1 = _narrow_signed(norm1, 8)

            if _attention_layer_for_index(layer_index, fixture):
                mixer_type = "attention"
                mixer = _tile_demo_attention_step(
                    norm1,
                    kv_caches[layer_index],
                    int(write_indices[layer_index]),
                    int(valid_counts[layer_index]),
                    fixture,
                )
                next_state = states[layer_index]
                next_conv_history = conv_histories[layer_index]
                next_kv_cache = mixer["kv_cache"]
                next_write_index = mixer["kv_write_index"]
                next_valid_count = mixer["kv_valid_count"]
            else:
                mixer_type = "mamba"
                mixer = _tile_demo_mamba_step(norm1, states[layer_index], fixture, conv_histories[layer_index])
                next_state = mixer["state"]
                next_conv_history = mixer["conv_history"]
                next_kv_cache = kv_caches[layer_index]
                next_write_index = int(write_indices[layer_index])
                next_valid_count = int(valid_counts[layer_index])

            first_residual = _narrow_signed(x + mixer["y"], 8)
            norm2, norm2_mean_square = tiny_rms_norm_approx(first_residual, fixture["norm2_weight"])
            norm2 = _narrow_signed(norm2, 8)
            mlp = _tile_demo_dense_mlp_step(norm2, fixture)
            final_residual = first_residual + mlp["y"]

            states[layer_index] = next_state
            conv_histories[layer_index] = next_conv_history
            kv_caches[layer_index] = next_kv_cache
            write_indices[layer_index] = next_write_index
            valid_counts[layer_index] = next_valid_count

            layer_traces.append({
                "layer_index": layer_index,
                "mixer_type": mixer_type,
                "input": x,
                "norm1": norm1,
                "norm1_mean_square": norm1_mean_square,
                "mixer": mixer,
                "first_residual": first_residual,
                "norm2": norm2,
                "norm2_mean_square": norm2_mean_square,
                "mlp": mlp,
                "final_residual": final_residual,
                "state": np.array(next_state, copy=True),
                "kv_write_index": next_write_index,
                "kv_valid_count": next_valid_count,
            })

            x = _narrow_signed(final_residual, 8)

        token_traces.append({
            "token_index": token_index,
            "input": token,
            "layers": layer_traces,
            "output": layer_traces[-1]["final_residual"],
            "states": np.array(states, copy=True),
            "kv_write_indices": np.array(write_indices, copy=True),
            "kv_valid_counts": np.array(valid_counts, copy=True),
        })

    return {
        "fixture": fixture,
        "tokens": tokens,
        "trace": token_traces,
        "final_states": states,
        "final_kv_write_indices": write_indices,
        "final_kv_valid_counts": valid_counts,
    }


def tiny_jamba_core_step(
    x,
    state,
    rms_weight,
    input_weight,
    input_bias,
    gate_weight,
    gate_bias,
    b_weight,
    b_bias,
    c_weight,
    c_bias,
    out_weight,
    out_bias,
    kernel_current,
    mamba_a,
    attention_keys,
    attention_values,
    use_attention=False,
):
    """One-token integer reference for the simplified Jamba2MiniCore test setup."""
    norm_x, mean_square = tiny_rms_norm_approx(x, rms_weight)
    projected_x = tiny_linear4(norm_x, input_weight, input_bias)
    gate = tiny_linear4(norm_x, gate_weight, gate_bias)
    mamba_b = tiny_linear4(norm_x, b_weight, b_bias)
    mamba_c = tiny_linear4(norm_x, c_weight, c_bias)

    conv_y = np.asarray(projected_x, dtype=np.int64) * np.asarray(kernel_current, dtype=np.int64)
    next_state = tiny_mamba_state_update(state, conv_y, mamba_a, mamba_b)
    block_y = next_state * mamba_c + projected_x * gate

    scores, attention_y = tiny_attention_decode(projected_x, attention_keys, attention_values)
    if use_attention:
        block_y = block_y + attention_y

    y = tiny_linear4(block_y, out_weight, out_bias)
    return {
        "mean_square": mean_square,
        "projected_x": projected_x,
        "state": next_state,
        "attention_scores": scores,
        "block_y": block_y,
        "y": y,
    }


if __name__ == "__main__":
    # Quick smoke test
    L, D, N = 4, 8, 4
    u = np.random.randn(L, D).astype(np.float32)
    delta = np.random.randn(L, D).astype(np.float32)
    A = np.random.randn(D, N).astype(np.float32)
    B = np.random.randn(L, N).astype(np.float32)
    C = np.random.randn(L, N).astype(np.float32)

    y = selective_scan(u, delta, A, B, C)
    print(f"SSM output shape: {y.shape}, mean: {y.mean():.6f}")

    x = np.random.randn(16, 64).astype(np.float32)
    normed = rms_norm(x)
    print(f"RMSNorm output var: {normed.var():.6f}")
