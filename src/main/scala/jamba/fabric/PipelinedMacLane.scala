package jamba.fabric

import chisel3._
import jamba.common.SignedMath

/** Pipelined signed multiply-accumulate lane: accOut = a * b + accIn.
  *
  * M15-P: Pipeline-depth analysis variant of MacLane. Adds a 1-cycle register
  * stage between the multiply and the accumulate:
  *
  *   Cycle N:   a * b → productReg (registered)
  *   Cycle N+1: productReg + accIn → accOut
  *
  * Trade-off vs. non-pipelined MacLane:
  *   - Reduces combinational depth of the critical path by ~50%:
  *       Non-pipelined: multiply chain + add chain (sequential, deep logic)
  *       Pipelined:     only multiply chain (stage 1) or only add chain (stage 2)
  *   - Expected effect on max clock frequency: up to 2× Fmax improvement (post-synthesis)
  *   - Cost: 1 additional latency cycle per projection slot in the scheduler.
  *     The L²/M+1 formula becomes L²/M+2 when using PipelinedMacLane in
  *     SerialSharedLinear4 (one extra flush cycle at the end).
  *
  * Note: PipelinedMacLane is NOT wired into SerialSharedLinear4 or the unified
  * scheduler in this prototype — it is a standalone component for pipeline-depth
  * analysis and comparison. The structural proxy (multiply-line count) is identical
  * to MacLane; the improvement is in achievable Fmax, not in resource count.
  *
  * See also: docs/paper/paper_ch4_architecture.md §4.8 for the timing model analysis.
  */
class PipelinedMacLane(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(dataWidth > 0, "PipelinedMacLane dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "PipelinedMacLane accWidth should hold the full product")

  val io = IO(new Bundle {
    val a        = Input(SInt(dataWidth.W))
    val b        = Input(SInt(dataWidth.W))
    val accIn    = Input(SInt(accWidth.W))
    val accOut   = Output(SInt(accWidth.W))
    // Pipeline stage outputs (for analysis / verification)
    val stageAValid = Output(Bool())   // stage A (multiply) output is valid
    val stageBValid = Output(Bool())   // stage B (accumulate) output is valid
  })

  // Stage A: multiply → register the product
  val validA  = RegInit(false.B)
  val productReg = RegInit(0.S(accWidth.W))

  validA     := true.B   // always accepting new inputs
  productReg := SignedMath.resize(io.a * io.b, accWidth)

  // Stage B: accumulate with pipelined product
  io.accOut   := SignedMath.resize(productReg +& io.accIn, accWidth)
  io.stageAValid := validA
  io.stageBValid := RegNext(validA, init = false.B)
}
