import numpy as np

from python.golden.mamba_ops import (
    rms_norm,
    selective_scan,
    tiny_attention_decode,
    tiny_jamba_core_step,
    tiny_mamba_state_update,
)


def test_tiny_mamba_state_update_matches_hand_checked_values():
    state = np.array([0, 0, 0, 0], dtype=np.int64)
    x = np.array([1, -2, 3, -4], dtype=np.int64)
    a = np.array([1, 1, 1, 1], dtype=np.int64)
    b = np.array([2, 2, 2, 2], dtype=np.int64)

    assert tiny_mamba_state_update(state, x, a, b).tolist() == [2, -4, 6, -8]


def test_tiny_attention_decode_matches_hand_checked_values():
    q = np.array([1, 2, 0, -1], dtype=np.int64)
    keys = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.int64,
    )
    values = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 1, 1, 1],
            [2, 0, 0, 1],
        ],
        dtype=np.int64,
    )

    scores, y = tiny_attention_decode(q, keys, values)

    assert scores.tolist() == [1, 2, 3, -1]
    assert y.tolist() == [2, 5, 3, 2]


def test_selective_scan_returns_expected_shape_and_skip_connection():
    u = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    delta = np.ones_like(u)
    a = np.ones((2, 1), dtype=np.float32)
    b = np.ones((2, 1), dtype=np.float32)
    c = np.ones((2, 1), dtype=np.float32)
    d = np.array([1.0, -1.0], dtype=np.float32)

    y = selective_scan(u, delta, a, b, c, d)

    assert y.shape == (2, 2)
    np.testing.assert_allclose(y[0], np.array([2.0, 0.0], dtype=np.float32), rtol=1e-6)


def test_rms_norm_normalizes_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]], dtype=np.float32)

    y = rms_norm(x, eps=0.0)

    np.testing.assert_allclose(y[0], np.array([0.84852815, 1.1313709], dtype=np.float32), rtol=1e-6)
    np.testing.assert_allclose(y[1], np.array([0.0, 0.0], dtype=np.float32), rtol=1e-6, atol=1e-6)


def test_tiny_jamba_core_step_matches_chisel_hand_checked_path():
    identity = np.eye(4, dtype=np.int64)
    zeros = np.zeros((4, 4), dtype=np.int64)
    x = np.array([1, 2, 3, 4], dtype=np.int64)
    state = np.zeros(4, dtype=np.int64)

    result = tiny_jamba_core_step(
        x=x,
        state=state,
        rms_weight=np.array([7, 7, 7, 7], dtype=np.int64),
        input_weight=identity,
        input_bias=np.zeros(4, dtype=np.int64),
        gate_weight=identity,
        gate_bias=np.zeros(4, dtype=np.int64),
        b_weight=zeros,
        b_bias=np.array([2, 2, 2, 2], dtype=np.int64),
        c_weight=zeros,
        c_bias=np.array([3, 3, 3, 3], dtype=np.int64),
        out_weight=identity,
        out_bias=np.zeros(4, dtype=np.int64),
        kernel_current=np.array([1, 1, 1, 1], dtype=np.int64),
        mamba_a=np.array([1, 1, 1, 1], dtype=np.int64),
        attention_keys=np.tile(np.array([[1, 0, 0, 0]], dtype=np.int64), (4, 1)),
        attention_values=np.ones((4, 4), dtype=np.int64),
        use_attention=True,
    )

    assert result["mean_square"] == 7
    assert result["projected_x"].tolist() == [1, 2, 3, 4]
    assert result["state"].tolist() == [2, 4, 6, 8]
    assert result["attention_scores"].tolist() == [1, 1, 1, 1]
    assert result["block_y"].tolist() == [11, 20, 31, 44]
    assert result["y"].tolist() == [11, 20, 31, 44]
