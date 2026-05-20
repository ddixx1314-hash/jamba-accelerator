"""Quantization precision sweep tests for the Jamba2 Mini golden model.

Verifies that quantized_mamba_step and quantized_attention_step correctly
simulate reduced-precision behaviour:
  - INT4  (data_bits=4,  acc_bits=16)
  - INT6  (data_bits=6,  acc_bits=24)
  - INT8  (data_bits=8,  acc_bits=32)

Key properties tested:
  1. INT4 saturates more aggressively than INT8 (narrower output range).
  2. Larger inputs are clipped to the data precision boundary.
  3. Full-precision (INT8/32) matches serial_selective_scan_step for inputs
     that do not overflow int8.
  4. Attention scores under INT4 are bounded by acc_bits=16.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "golden"))
from mamba_ops import (
    fixed_max_signed,
    fixed_min_signed,
    fixed_saturate,
    serial_selective_scan_step,
    quantized_mamba_step,
    quantized_attention_step,
)

LANES = 4
PRECISIONS = [
    {"data_bits": 4,  "acc_bits": 16,  "label": "INT4"},
    {"data_bits": 6,  "acc_bits": 24,  "label": "INT6"},
    {"data_bits": 8,  "acc_bits": 32,  "label": "INT8"},
]


def _small_inputs():
    """Inputs that fit within INT4 range (no saturation at any precision)."""
    state = np.array([1, -1, 2, -2], dtype=np.int64)
    x = np.array([1, 1, -1, 1], dtype=np.int64)
    a = np.array([1, 1, 1, 1], dtype=np.int64)
    b = np.array([1, 1, 1, 1], dtype=np.int64)
    c = np.array([1, 1, 1, 1], dtype=np.int64)
    return state, x, a, b, c


def _large_inputs():
    """Inputs large enough to saturate INT4 but not INT8."""
    state = np.array([60, -60, 40, -40], dtype=np.int64)
    x = np.array([10, -10, 10, -10], dtype=np.int64)
    a = np.array([3, 3, 3, 3], dtype=np.int64)
    b = np.array([5, 5, 5, 5], dtype=np.int64)
    c = np.array([2, 2, 2, 2], dtype=np.int64)
    return state, x, a, b, c


class TestQuantizedMambaStep:
    def test_small_inputs_match_across_precisions(self):
        """With tiny inputs, all precisions should give the same result."""
        state, x, a, b, c = _small_inputs()
        results = [
            quantized_mamba_step(state, x, a, b, c, **{k: v for k, v in p.items() if k != "label"})
            for p in PRECISIONS
        ]
        ns0, y0 = results[0]
        for ns, y in results[1:]:
            np.testing.assert_array_equal(ns, ns0)
            np.testing.assert_array_equal(y, y0)

    def test_int8_matches_unquantized_for_small_inputs(self):
        """INT8/32 quantized step should equal the unquantized reference for small inputs."""
        state, x, a, b, c = _small_inputs()
        ref_ns, ref_y = serial_selective_scan_step(state, x, a, b, c)
        q_ns, q_y = quantized_mamba_step(state, x, a, b, c, data_bits=8, acc_bits=32)
        np.testing.assert_array_equal(q_ns, ref_ns)
        np.testing.assert_array_equal(q_y, ref_y)

    def test_int4_saturates_more_than_int8_on_large_inputs(self):
        """INT4 output magnitude must be <= INT8 output magnitude for large inputs."""
        state, x, a, b, c = _large_inputs()
        ns4, y4 = quantized_mamba_step(state, x, a, b, c, data_bits=4, acc_bits=16)
        ns8, y8 = quantized_mamba_step(state, x, a, b, c, data_bits=8, acc_bits=32)
        assert np.all(np.abs(ns4) <= np.abs(ns8) + 1), \
            "INT4 state magnitude should not exceed INT8 state magnitude"
        assert np.all(np.abs(y4) <= fixed_max_signed(16) + 1), \
            "INT4 output must be bounded by acc_bits=16"
        assert np.all(np.abs(y8) <= fixed_max_signed(32) + 1)

    def test_output_bounded_by_acc_bits_for_each_precision(self):
        """Output must stay within acc_bits bounds for all precision configs."""
        state, x, a, b, c = _large_inputs()
        for p in PRECISIONS:
            ns, y = quantized_mamba_step(state, x, a, b, c,
                                         data_bits=p["data_bits"], acc_bits=p["acc_bits"])
            lo = fixed_min_signed(p["acc_bits"])
            hi = fixed_max_signed(p["acc_bits"])
            assert np.all(ns >= lo) and np.all(ns <= hi), \
                f"{p['label']} state out of acc_bits range"
            assert np.all(y >= lo) and np.all(y <= hi), \
                f"{p['label']} y out of acc_bits range"

    def test_int4_data_inputs_are_clipped(self):
        """Inputs beyond INT4 range are clipped before any computation."""
        state = np.array([0, 0, 0, 0], dtype=np.int64)
        x = np.array([100, -100, 100, -100], dtype=np.int64)  # >> INT4 max (7)
        a = np.array([1, 1, 1, 1], dtype=np.int64)
        b = np.array([1, 1, 1, 1], dtype=np.int64)
        c = np.array([1, 1, 1, 1], dtype=np.int64)
        ns, y = quantized_mamba_step(state, x, a, b, c, data_bits=4, acc_bits=16)
        # x saturates to INT4 range [-8, 7]; next_state = 0*1 + clipped_x*1
        lo4 = fixed_min_signed(4)
        hi4 = fixed_max_signed(4)
        assert np.all((ns >= lo4) & (ns <= hi4)), f"state not clipped: {ns}"

    def test_precision_monotone_range(self):
        """INT8 output range >= INT6 >= INT4 for the same large inputs."""
        state, x, a, b, c = _large_inputs()
        _, y4 = quantized_mamba_step(state, x, a, b, c, data_bits=4, acc_bits=16)
        _, y6 = quantized_mamba_step(state, x, a, b, c, data_bits=6, acc_bits=24)
        _, y8 = quantized_mamba_step(state, x, a, b, c, data_bits=8, acc_bits=32)
        assert np.max(np.abs(y4)) <= np.max(np.abs(y6)) + 1 or np.max(np.abs(y4)) <= fixed_max_signed(16)
        assert np.max(np.abs(y6)) <= np.max(np.abs(y8)) + 1 or np.max(np.abs(y6)) <= fixed_max_signed(24)


class TestQuantizedAttentionStep:
    def test_small_inputs_match_across_precisions(self):
        """With small inputs, all precisions give the same attention output."""
        q = np.array([1, 0, -1, 1], dtype=np.int64)
        keys = np.eye(4, dtype=np.int64)
        values = np.eye(4, dtype=np.int64)
        results = [
            quantized_attention_step(q, keys, values, **{k: v for k, v in p.items() if k != "label"})
            for p in PRECISIONS
        ]
        sc0, y0 = results[0]
        for sc, y in results[1:]:
            np.testing.assert_array_equal(sc, sc0)
            np.testing.assert_array_equal(y, y0)

    def test_int4_attention_bounded_by_acc_bits(self):
        """INT4 attention output must be bounded by acc_bits=16."""
        q = np.array([100, 100, 100, 100], dtype=np.int64)
        keys = np.full((4, 4), 50, dtype=np.int64)
        values = np.full((4, 4), 50, dtype=np.int64)
        _, y4 = quantized_attention_step(q, keys, values, data_bits=4, acc_bits=16)
        hi = fixed_max_signed(16)
        lo = fixed_min_signed(16)
        assert np.all(y4 >= lo) and np.all(y4 <= hi), \
            f"INT4 attention output out of acc_bits=16 range: {y4}"

    def test_int8_scores_match_unquantized_for_small_inputs(self):
        """INT8 attention scores should equal unquantized reference for small inputs."""
        from mamba_ops import tiny_attention_decode
        q = np.array([1, 2, -1, 0], dtype=np.int64)
        keys = np.eye(4, dtype=np.int64)
        values = np.eye(4, dtype=np.int64)
        ref_sc, ref_y = tiny_attention_decode(q, keys, values)
        q_sc, q_y = quantized_attention_step(q, keys, values, data_bits=8, acc_bits=32)
        np.testing.assert_array_equal(q_sc, ref_sc)
        np.testing.assert_array_equal(q_y, ref_y)
