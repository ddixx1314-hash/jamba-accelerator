import numpy as np

from python.golden.mamba_ops import (
    attention_mixer_step,
    dense_mlp_step,
    fixed_multiply_rescale,
    fixed_round_shift_right,
    fixed_saturate,
    fixed_saturating_add,
    jamba2_mini_core_trace,
    jamba2_mini_fixture,
    jamba2_mini_layer_step,
    jamba2_mini_tile_demo_trace,
    mamba_mixer_step,
    rms_norm,
    selective_scan,
    tiny_attention_decode,
    tiny_jamba_core_step,
    tiny_mamba_state_update,
)


def test_fixed_saturate_clamps_to_signed_range():
    values = np.array([200, -200, 42], dtype=np.int64)

    assert fixed_saturate(values, 8).tolist() == [127, -128, 42]


def test_fixed_round_shift_right_rounds_away_from_zero():
    values = np.array([7, -7], dtype=np.int64)

    assert fixed_round_shift_right(values, 2).tolist() == [2, -3]


def test_fixed_multiply_rescale_matches_chisel_policy():
    assert int(fixed_multiply_rescale(7, 3, 8, 2)) == 5
    assert int(fixed_multiply_rescale(100, 8, 8, 2)) == 127
    assert int(fixed_multiply_rescale(-7, 3, 8, 2)) == -6


def test_fixed_saturating_add_matches_chisel_policy():
    assert int(fixed_saturating_add(100, 50, 8)) == 127
    assert int(fixed_saturating_add(-100, -50, 8)) == -128
    assert int(fixed_saturating_add(10, -3, 8)) == 7


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


def test_jamba2_mini_mamba_mixer_step_updates_state():
    fixture = jamba2_mini_fixture()
    x = np.array([1, 0, 0, 0], dtype=np.int64)
    state = np.zeros(4, dtype=np.int64)

    result = mamba_mixer_step(x, state, fixture)

    assert result["projected"].tolist() == [1, 0, 0, 0]
    assert result["state"].tolist() == [2, 0, 0, 0]
    assert result["y"].tolist() == [2, 0, 0, 0]


def test_jamba2_mini_mamba_mixer_step_tracks_conv_history():
    fixture = jamba2_mini_fixture()
    state = np.zeros(4, dtype=np.int64)

    first = mamba_mixer_step(np.array([1, 0, 0, 0], dtype=np.int64), state, fixture)
    second = mamba_mixer_step(
        np.array([0, 1, 0, 0], dtype=np.int64),
        first["state"],
        fixture,
        first["conv_history"],
    )

    assert second["conv"].tolist() == [1, 1, 0, 0]
    assert second["state"].tolist() == [4, 2, 0, 0]
    assert second["y"].tolist() == [4, 2, 0, 0]


def test_jamba2_mini_attention_mixer_uses_circular_kv_cache():
    fixture = jamba2_mini_fixture(context_length=2)
    cache = np.zeros((2, 2, 4), dtype=np.int64)
    write_index = 0
    valid_count = 0

    first = attention_mixer_step(np.array([4, 0, 0, 0], dtype=np.int64), cache, write_index, valid_count, fixture)
    second = attention_mixer_step(
        np.array([0, 4, 0, 0], dtype=np.int64),
        first["kv_cache"],
        first["kv_write_index"],
        first["kv_valid_count"],
        fixture,
    )
    third = attention_mixer_step(
        np.array([4, 4, 0, 0], dtype=np.int64),
        second["kv_cache"],
        second["kv_write_index"],
        second["kv_valid_count"],
        fixture,
    )

    assert first["kv_write_index"] == 1
    assert second["kv_write_index"] == 0
    assert third["kv_write_index"] == 1
    assert third["kv_valid_count"] == 2
    assert third["scores"].tolist() == [16, 32]
    assert third["weights"].tolist() == [4, 8]
    assert third["y"].tolist() == [32, 48, 0, 0]


def test_jamba2_mini_dense_mlp_step_is_deterministic():
    fixture = jamba2_mini_fixture()
    x = np.array([1, 2, 3, 4], dtype=np.int64)

    result = dense_mlp_step(x, fixture)

    assert result["gate"].tolist() == [2, 3, 4, 5]
    assert result["up"].tolist() == [4, 3, 2, 1]
    assert result["y"].tolist() == [8, 9, 8, 5]


def test_jamba2_mini_layer_step_records_mixer_and_mlp_residuals():
    fixture = jamba2_mini_fixture()
    x = np.array([1, 0, 0, 0], dtype=np.int64)
    state = np.zeros(4, dtype=np.int64)
    cache = np.zeros((4, 2, 4), dtype=np.int64)

    result = jamba2_mini_layer_step(
        x=x,
        layer_index=0,
        state=state,
        kv_cache=cache,
        write_index=0,
        valid_count=0,
        fixture=fixture,
    )

    assert result["mixer_type"] == "mamba"
    assert result["first_residual"].tolist() == [3, 0, 0, 0]
    assert result["mlp"]["y"].tolist() == [0, 0, 0, 1]
    assert result["final_residual"].tolist() == [3, 0, 0, 1]
    assert result["moe_dispatch_valid"] is False
    assert result["moe_combine_valid"] is False


def test_jamba2_mini_core_trace_has_sparse_attention_and_cache_state():
    fixture = jamba2_mini_fixture(num_layers=4, attention_layer_period=4, context_length=4)
    tokens = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.int64,
    )

    result = jamba2_mini_core_trace(tokens, fixture)
    first_token_layers = result["trace"][0]["layers"]
    second_token_layers = result["trace"][1]["layers"]

    assert [layer["mixer_type"] for layer in first_token_layers] == ["mamba", "mamba", "mamba", "attention"]
    assert first_token_layers[-1]["kv_write_index"] == 1
    assert second_token_layers[-1]["kv_write_index"] == 2
    assert result["final_kv_valid_counts"].tolist() == [0, 0, 0, 2]
    assert result["final_states"][0].tolist() == [4, 2, 0, 0]
    assert result["trace"][0]["output"].shape == (4,)
    assert result["trace"][1]["output"].shape == (4,)


def test_jamba2_mini_tile_demo_trace_matches_chisel_visible_timing():
    fixture = jamba2_mini_fixture(num_layers=4, attention_layer_period=4, context_length=8)
    tokens = np.array(
        [
            [1, 0, 0, 0],
            [2, 0, 0, 0],
        ],
        dtype=np.int64,
    )

    result = jamba2_mini_tile_demo_trace(tokens, fixture)

    assert [layer["mixer_type"] for layer in result["trace"][0]["layers"]] == ["mamba", "mamba", "mamba", "attention"]
    assert result["trace"][0]["output"].tolist() == [3, 0, 0, 3]
    assert result["trace"][1]["output"].tolist() == [6, 0, 0, 3]
    assert result["trace"][0]["states"][0].tolist() == [2, 0, 0, 0]
    assert result["trace"][1]["states"][0].tolist() == [8, 0, 0, 0]
    assert result["trace"][0]["kv_write_indices"].tolist() == [0, 0, 0, 1]
    assert result["trace"][1]["kv_write_indices"].tolist() == [0, 0, 0, 2]
    assert result["trace"][1]["kv_valid_counts"].tolist() == [0, 0, 0, 2]
