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
