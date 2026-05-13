"""Golden models for tiny Mamba/Jamba-like operations."""

import numpy as np


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
