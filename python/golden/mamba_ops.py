"""Golden models for tiny Mamba/Jamba-like operations."""

import numpy as np


def _as_i64(value):
    return np.asarray(value, dtype=np.int64)


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
        "conv_kernel": ones,
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


def mamba_mixer_step(x, state, fixture):
    """Integer Jamba2 mini Mamba mixer step with token-serial state update."""
    projected = tiny_linear4(x, fixture["mamba_in_weight"], fixture["mamba_in_bias"])
    b = tiny_linear4(x, fixture["mamba_b_weight"], fixture["mamba_b_bias"])
    c = tiny_linear4(x, fixture["mamba_c_weight"], fixture["mamba_c_bias"])
    conv = projected * _as_i64(fixture["conv_kernel"])
    next_state = tiny_mamba_state_update(state, conv, fixture["mamba_a"], b)
    y = next_state * c
    return {
        "projected": projected,
        "conv": conv,
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


def jamba2_mini_layer_step(x, layer_index, state, kv_cache, write_index, valid_count, fixture):
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
        mixer = mamba_mixer_step(norm1, state, fixture)
        next_state = mixer["state"]
        next_kv_cache = kv_cache
        next_write_index = write_index
        next_valid_count = valid_count

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

    state = np.zeros(hidden_size, dtype=np.int64)
    kv_cache = np.zeros((int(fixture["context_length"]), 2, hidden_size), dtype=np.int64)
    write_index = 0
    valid_count = 0
    token_traces = []

    for token_index, token in enumerate(tokens):
        x = token
        layer_traces = []
        for layer_index in range(int(fixture["num_layers"])):
            layer = jamba2_mini_layer_step(
                x=x,
                layer_index=layer_index,
                state=state,
                kv_cache=kv_cache,
                write_index=write_index,
                valid_count=valid_count,
                fixture=fixture,
            )
            x = layer["final_residual"]
            state = layer["state"]
            kv_cache = layer["kv_cache"]
            write_index = layer["kv_write_index"]
            valid_count = layer["kv_valid_count"]
            layer_traces.append(layer)

        token_traces.append({
            "token_index": token_index,
            "input": token,
            "layers": layer_traces,
            "output": x,
            "state": state,
            "kv_write_index": write_index,
            "kv_valid_count": valid_count,
        })

    return {
        "fixture": fixture,
        "tokens": tokens,
        "trace": token_traces,
        "final_state": state,
        "final_kv_cache": kv_cache,
        "final_kv_write_index": write_index,
        "final_kv_valid_count": valid_count,
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
