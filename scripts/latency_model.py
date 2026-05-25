#!/usr/bin/env python3
"""
M12-C: Analytical Latency Model for UnifiedJamba2MiniLayer
===========================================================

Derives a closed-form cycle-count formula for one token through the
hardware layer, parameterised by lanes (L), macLanes (M), and conv taps (T).

Formula derivation
------------------
The layer FSM has named phases that each consume a deterministic number of
clock cycles:

  T_scheduler(N, L, M) = N × (3 + L²/M) + 2

  where N = number of enabled projection slots in one scheduler invocation,
        L = lanes, M = macLanes.

  Derivation: each slot costs 1 (findSlot) + 1 (launch) + (L²/M + 1) (waitLinear)
  = 3 + L²/M cycles.  After all N slots, one more findSlot finds no candidate,
  and doneReg takes one cycle to propagate to the layer = +2 total overhead.

Mamba token (unfused, standard SSM):
  T = 1                                  # launchMixerProj
    + T_scheduler(3, L, M)               # waitMixerProj (Q/K/V or inp/B/C)
    + 1 + L*T + 1                        # launchConv + waitConv (L*taps MACs)
    + 1 + 3*L + 1                        # launchScan + waitScan (3 ops × L lanes)
    + 1                                  # computeFirstResidual (unfused)
    + 1 + T_scheduler(2, L, M)           # launchMlpGateUp + waitMlpGateUp
    + 1                                  # computeHidden (unfused)
    + 1 + T_scheduler(1, L, M)           # launchMlpDown + waitMlpDown
    + 1                                  # doneState

Attention token (unfused):
  T = 1                                  # launchMixerProj
    + T_scheduler(3, L, M)               # waitMixerProj (Q/K/V)
    + 1                                  # computeAttention (combinational, 1 FSM cycle)
    + 1 + T_scheduler(1, L, M)           # launchAttentionOut + waitAttentionOut
    + 1                                  # computeFirstResidual (unfused)
    + 1 + T_scheduler(2, L, M)           # launchMlpGateUp + waitMlpGateUp
    + 1                                  # computeHidden (unfused)
    + 1 + T_scheduler(1, L, M)           # launchMlpDown + waitMlpDown
    + 1                                  # doneState

Compact forms:
  T_mamba(L, M, T, fused=F, shiftA=A) =
      34 + 6*(L²/M) + L*T + (2 if A else 3)*L - (2 if F else 0)

  T_attn(L, M, fused=F) =
      37 + 7*(L²/M) - (2 if F else 0)

M12 milestone savings (per-token, L=4, M=1, T=4):
  M12-P useShiftA   : −4 cycles  (scan: 3L → 2L ops)
  M11-F fusedOps    : −2 cycles  (eliminate computeFirstResidual + computeHidden)
  M12-K window attn : structural only (no cycle saving, but correctness-gating)
  M12-A columnSkip  : projection-level; affects T_scheduler slots where x is sparse
"""

# ── formula functions ──────────────────────────────────────────────────────────

def t_linear(L: int, M: int) -> int:
    """Cycles for one ConfigurableSerialLinear4 projection (L×L matrix, M mac lanes)."""
    return (L * L) // M + 1          # L²/M computation + 1 done-latch cycle


def t_scheduler(N: int, L: int, M: int) -> int:
    """Cycles for scheduler to process N enabled projection slots.
    This is the cycle count of the waitXxx phase (does NOT include launchXxx)."""
    return N * (3 + t_linear(L, M) - 1) + 2
    # = N * (2 + t_linear) + 2
    # = N * (3 + L²/M) + 2   [since t_linear = L²/M + 1, so 2+t_linear = 3+L²/M]


def t_conv(L: int, taps: int) -> int:
    """Cycles for SerialCausalConvMini (waitConv phase only)."""
    return L * taps + 1              # L*taps MACs + 1 done-latch


def t_scan(L: int, use_shift_a: bool = False) -> int:
    """Cycles for SerialSelectiveScanMini (waitScan phase only)."""
    ops = 2 if use_shift_a else 3
    return ops * L + 1               # ops*L MACs + 1 done-latch


def t_mamba(L: int, M: int, taps: int,
            fused: bool = False, use_shift_a: bool = False) -> int:
    """Total cycles for one Mamba token (countCyclesToDone metric)."""
    scan = t_scan(L, use_shift_a)
    conv = t_conv(L, taps)
    fuse_saving = 2 if fused else 0
    return (
        1                         # launchMixerProj
        + t_scheduler(3, L, M)   # waitMixerProj  (inp/B/C projections)
        + 1 + conv                # launchConv + waitConv
        + 1 + scan                # launchScan + waitScan
        + (0 if fused else 1)     # computeFirstResidual (skipped when fused)
        + 1 + t_scheduler(2, L, M)  # launchMlpGateUp + waitMlpGateUp
        + (0 if fused else 1)     # computeHidden (skipped when fused)
        + 1 + t_scheduler(1, L, M)  # launchMlpDown + waitMlpDown
        + 1                       # doneState
    )


def t_attention(L: int, M: int, fused: bool = False) -> int:
    """Total cycles for one Attention token (countCyclesToDone metric)."""
    return (
        1                         # launchMixerProj
        + t_scheduler(3, L, M)   # waitMixerProj  (Q/K/V projections)
        + 1                       # computeAttention (1 FSM cycle)
        + 1 + t_scheduler(1, L, M)  # launchAttentionOut + waitAttentionOut
        + (0 if fused else 1)     # computeFirstResidual
        + 1 + t_scheduler(2, L, M)  # launchMlpGateUp + waitMlpGateUp
        + (0 if fused else 1)     # computeHidden
        + 1 + t_scheduler(1, L, M)  # launchMlpDown + waitMlpDown
        + 1                       # doneState
    )


# ── phase-by-phase breakdown ───────────────────────────────────────────────────

def breakdown_mamba(L: int, M: int, taps: int,
                    fused: bool = False, use_shift_a: bool = False) -> dict:
    """Return a dict mapping phase name → cycle count for one Mamba token."""
    return {
        "launchMixerProj":         1,
        "waitMixerProj (3 slots)": t_scheduler(3, L, M),
        "launchConv":              1,
        f"waitConv ({L}×{taps} MACs)": t_conv(L, taps),
        "launchScan":              1,
        f"waitScan ({'2' if use_shift_a else '3'}×{L} ops)": t_scan(L, use_shift_a),
        "computeFirstResidual":    0 if fused else 1,
        "launchMlpGateUp":         1,
        "waitMlpGateUp (2 slots)": t_scheduler(2, L, M),
        "computeHidden":           0 if fused else 1,
        "launchMlpDown":           1,
        "waitMlpDown (1 slot)":    t_scheduler(1, L, M),
        "doneState":               1,
    }


def breakdown_attention(L: int, M: int, fused: bool = False) -> dict:
    return {
        "launchMixerProj":              1,
        "waitMixerProj (3 slots)":      t_scheduler(3, L, M),
        "computeAttention":             1,
        "launchAttentionOut":           1,
        "waitAttentionOut (1 slot)":    t_scheduler(1, L, M),
        "computeFirstResidual":         0 if fused else 1,
        "launchMlpGateUp":              1,
        "waitMlpGateUp (2 slots)":      t_scheduler(2, L, M),
        "computeHidden":                0 if fused else 1,
        "launchMlpDown":                1,
        "waitMlpDown (1 slot)":         t_scheduler(1, L, M),
        "doneState":                    1,
    }


# ── validation against empirical measurements ─────────────────────────────────

EMPIRICAL = [
    # (description, mode, L, M, taps, fused, use_shift_a, measured)
    ("Mamba std unfused",            "mamba",    4, 1, 4, False, False, 158),
    ("Mamba std fused",              "mamba",    4, 1, 4, True,  False, 156),
    ("Mamba useShiftA unfused",      "mamba",    4, 1, 4, False, True,  154),
    ("Attention std unfused",        "attention", 4, 1, 4, False, False, 149),
    ("Attention std fused",          "attention", 4, 1, 4, True,  False, 147),
]


def validate():
    print("=" * 68)
    print("M12-C  Analytical Latency Model — Validation")
    print("=" * 68)
    all_ok = True
    for desc, mode, L, M, taps, fused, usa, measured in EMPIRICAL:
        if mode == "mamba":
            pred = t_mamba(L, M, taps, fused, usa)
        else:
            pred = t_attention(L, M, fused)
        ok = "✓" if pred == measured else "✗"
        if pred != measured:
            all_ok = False
        print(f"  {ok}  {desc:<35s}  predicted={pred:4d}  measured={measured:4d}")
    print()
    if all_ok:
        print("  All predictions match empirical measurements.")
    else:
        print("  WARNING: some predictions do not match!")
    print()


# ── projection table ───────────────────────────────────────────────────────────

def projection_table():
    """Show predicted cycle counts for various parameter configurations."""
    print("=" * 68)
    print("M12-C  Latency Projections  (lanes, macLanes, taps)")
    print("=" * 68)
    fmt = "{:>6s}  {:>8s}  {:>5s}  {:>5s}  {:>7s}  {:>7s}  {:>7s}  {:>7s}"
    print(fmt.format("lanes", "macLanes", "taps",
                     "T_lin",
                     "T_mamba", "T_m_fus", "T_attn", "T_a_fus"))
    print("-" * 68)
    for L in [4, 8, 16]:
        for M in [1, 2, 4]:
            if M > L:
                continue
            if L % M != 0:
                continue
            for T in [4]:
                tl = t_linear(L, M)
                tm   = t_mamba(L, M, T, fused=False)
                tm_f = t_mamba(L, M, T, fused=True)
                ta   = t_attention(L, M, fused=False)
                ta_f = t_attention(L, M, fused=True)
                print(fmt.format(
                    str(L), str(M), str(T),
                    str(tl), str(tm), str(tm_f), str(ta), str(ta_f)))
    print()


# ── milestone savings table ────────────────────────────────────────────────────

def milestone_savings_table():
    """Show per-milestone cycle savings for L=4, M=1, T=4."""
    L, M, T = 4, 1, 4
    print("=" * 68)
    print("M12-C  Per-Milestone Cycle Savings  (L=4, M=1, taps=4)")
    print("=" * 68)

    base_mamba = t_mamba(L, M, T, fused=False, use_shift_a=False)
    base_attn  = t_attention(L, M, fused=False)

    rows = [
        ("Baseline (unfused, no ShiftA)",  base_mamba, base_attn, 0, 0),
        ("M11-F fusedOps",
         t_mamba(L, M, T, fused=True,  use_shift_a=False),
         t_attention(L, M, fused=True),
         base_mamba - t_mamba(L, M, T, fused=True,  use_shift_a=False),
         base_attn  - t_attention(L, M, fused=True)),
        ("M12-P useShiftA (Mamba only)",
         t_mamba(L, M, T, fused=False, use_shift_a=True),
         base_attn,
         base_mamba - t_mamba(L, M, T, fused=False, use_shift_a=True),
         0),
        ("M11-F + M12-P combined",
         t_mamba(L, M, T, fused=True,  use_shift_a=True),
         t_attention(L, M, fused=True),
         base_mamba - t_mamba(L, M, T, fused=True,  use_shift_a=True),
         base_attn  - t_attention(L, M, fused=True)),
    ]

    hdr = "{:<35s}  {:>7s}  {:>7s}  {:>8s}  {:>8s}"
    print(hdr.format("Configuration", "T_mamba", "T_attn",
                     "Δ_mamba", "Δ_attn"))
    print("-" * 68)
    for name, tm, ta, dm, da in rows:
        print(hdr.format(name, str(tm), str(ta),
                         f"-{dm}" if dm else "—",
                         f"-{da}" if da else "—"))
    print()


# ── phase breakdown example ────────────────────────────────────────────────────

def phase_breakdown_example():
    L, M, T = 4, 1, 4
    print("=" * 68)
    print(f"M12-C  Phase Breakdown  (L={L}, M={M}, taps={T}, unfused, std)")
    print("=" * 68)

    print("\n  Mamba token:")
    total = 0
    for phase, cyc in breakdown_mamba(L, M, T).items():
        bar = "█" * cyc if cyc <= 60 else "█" * 30 + f"…{cyc}"
        print(f"    {phase:<40s}  {cyc:3d}  {bar}")
        total += cyc
    print(f"    {'TOTAL':<40s}  {total:3d}")

    print("\n  Attention token:")
    total = 0
    for phase, cyc in breakdown_attention(L, M).items():
        bar = "█" * cyc if cyc <= 60 else "█" * 30 + f"…{cyc}"
        print(f"    {phase:<40s}  {cyc:3d}  {bar}")
        total += cyc
    print(f"    {'TOTAL':<40s}  {total:3d}")
    print()


# ── sparse-path projection (M12-A interaction) ────────────────────────────────

def sparse_projection_note():
    """
    M12-A: columnSkip=true in ConfigurableSerialLinear4.

    When the projection input x has k non-zero columns (k ≤ L), the
    column-skip FSM completes in k×L + 2 cycles (instead of L² + 1 standard),
    saving (L-k)×L - 1 cycles per projection.

    This changes t_linear(L, M) for the affected slot(s):
        t_linear_sparse(L, k) = k*L + 2     (macLanes=1 only)

    And thus T_scheduler for a batch of N slots where each has k_i non-zeros:
        T_scheduler_sparse = sum_i (3 + t_linear_sparse(L, k_i)) + 2
    """
    L = 4
    print("=" * 68)
    print("M12-C  M12-A Column-Skip Projection Savings  (L=4, macLanes=1)")
    print("=" * 68)
    print(f"  Standard t_linear(L=4, M=1) = {t_linear(L, 1)} cycles")
    print()
    hdr = "{:>4s}  {:>12s}  {:>12s}  {:>8s}  {:>7s}"
    print(hdr.format("k", "t_lin_sparse", "t_lin_std", "saved", "speedup"))
    print("-" * 52)
    t_std = t_linear(L, 1)
    for k in range(L + 1):
        t_sp = k * L + 2
        saved = t_std - t_sp
        speedup = f"{t_std/t_sp:.1f}×" if t_sp > 0 else "∞"
        note = "  ← same as standard" if saved < 0 else ""
        print(hdr.format(str(k), str(t_sp), str(t_std), str(saved), speedup) + note)
    print()
    print("  For a Mamba token with all projections having k non-zero cols:")
    for k in [0, 1, 2, 3, 4]:
        t_sp = k * L + 2
        # Replace t_linear in mamba formula: 6 projection calls (3+2+1 slots),
        # each using t_sp instead of t_std.
        # T_scheduler_sparse(N, L) = N*(3 + t_sp - 1) + 2 = N*(2 + t_sp) + 2
        def t_sched_sp(N): return N * (2 + t_sp) + 2
        tm_sp = (1 + t_sched_sp(3) + 1 + t_conv(L,4) + 1 + t_scan(L) + 1 +
                 1 + t_sched_sp(2) + 1 + 1 + t_sched_sp(1) + 1)
        tm_std = t_mamba(L, 1, 4)
        print(f"    k={k}: T_mamba_sparse={tm_sp:4d}  T_mamba_std={tm_std}  saved={tm_std-tm_sp}")
    print()


if __name__ == "__main__":
    validate()
    projection_table()
    milestone_savings_table()
    phase_breakdown_example()
    sparse_projection_note()
