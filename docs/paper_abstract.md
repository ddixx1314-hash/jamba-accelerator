# Abstract

Hybrid sequence models such as Jamba2 interleave selective state space model (Mamba)
layers with sparse self-attention and mixture-of-experts (MoE) feed-forward blocks.
This heterogeneous operator mix presents a hardware design challenge: a naive RTL
implementation allocates dedicated multiplier arrays for each of the three operator
families, replicating compute fabric and wasting area. This work studies a unified
resource-reuse fabric for Jamba2-style layers at the mini-prototype scale
(lanes=4, INT8 data path, 4×4 weight matrices).

We propose an operator taxonomy that identifies the 10 linear projections across the
Mamba, Attention, and MoE-lite paths as structurally identical 4×4 integer
matrix-vector multiplies and the primary candidates for hardware sharing. Building on
this taxonomy, we implement and compare four resource-latency tiers for a single
Jamba2 Mini layer. The most unified tier, `UnifiedSerial`, schedules all 10 projections
through a single slot-table FSM sharing one MAC lane, achieving a 48% reduction in
structural multiply-line proxy relative to the fully parallel baseline (50 vs 96),
at the cost of approximately 143 cycles per token per layer.

At the tile level, `SinglePhysicalLayerTile` replaces an L-instance layer scheduler
with one physical compute layer time-multiplexed across L logical layers, reducing the
instance-weighted resource proxy from O(L) to O(1) (e.g., 4-layer Context8: 368→92).
A per-layer state file saves and restores SSM hidden state, causal-conv history, and
KV cache on each layer transition; multi-token functional correctness is verified by
direct comparison against an L-physical-layer reference design.

We additionally analyze INT4/INT6/INT8 quantization — showing that the structural
multiply count is precision-invariant while register bits scale roughly linearly with
data width — a `projectionMacLanes` sweep (1/2/4 lanes) that exposes a resource-latency
Pareto curve, and a zero-skip MAC variant that provides dynamic power reduction for
sparse activations without changing the structural multiplier count.

The full prototype is implemented in Chisel, generating synthesizable SystemVerilog,
and is validated by 219 unit tests and 28 Python golden-model tests. All resource
figures are pre-synthesis structural proxies; FPGA synthesis results are deferred to
future work.
