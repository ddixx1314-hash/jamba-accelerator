"""Golden model for Mamba SSM operations

Reference: Jamba 2.0 technical report (arXiv 2501.06603)
"""

import numpy as np


def selective_scan(u, delta, A, B, C, D=None):
    """Selective scan — core Mamba SSM operation.

    Args:
        u:   input  (L, D)
        delta: time delta (L, D)
        A:    state matrix (D, N)
        B:    input projection (L, N)
        C:    output projection (L, N)
        D:    skip connection (D,) or None

    Returns:
        y: output (L, D)
    """
    L, D = u.shape
    _, N = A.shape

    # Discretize A, B
    dA = np.exp(np.outer(delta, A.reshape(-1)))  # (L, D*N) simplified
    dB = delta[:, :, None] * B                  # (L, D, N)

    h = np.zeros((D, N), dtype=np.float32)
    y = np.zeros((L, D), dtype=np.float32)

    for t in range(L):
        h = dA[t] * h + dB[t] * u[t, :, None]
        y[t] = (h * C[t]).sum(axis=-1)

    if D is not None:
        y = y + D * u

    return y


def rms_norm(x, eps=1e-6):
    """RMS Normalization"""
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return x / rms


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
