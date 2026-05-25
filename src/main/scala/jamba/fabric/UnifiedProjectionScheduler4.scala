package jamba.fabric

import chisel3._
import chisel3.util.{Enum, PriorityEncoder, log2Ceil}

object UnifiedProjectionSlots {
  val MambaInput = 0
  val MambaB = 1
  val MambaC = 2
  val AttentionQ = 3
  val AttentionK = 4
  val AttentionV = 5
  val AttentionOut = 6
  val MlpGate = 7
  val MlpUp = 8
  val MlpDown = 9
  val NumSlots = 10

  val MoERouter = 10
  val MoEExpertGate = 11
  val MoEExpertUp = 12
  val MoEExpertDown = 13
  val NumMoESlots = 14
}

/** Schedules named layer projections through one configurable serial linear unit.
  *
  * Unlike SerialProjectionScheduler4, every projection slot owns a separate
  * input vector. This matches a full Jamba-style layer: Mamba input/B/C and
  * attention Q/K/V consume norm1, attention out consumes decoded attention,
  * MLP gate/up consume norm2, and MLP down consumes the hidden activation.
  *
  * Dynamic vector bypass (M10-D):
  *   When `vectorBypass = true`, the scheduler checks each enabled slot's input
  *   vector before launching the MAC.  If all elements of x[slot] are zero the
  *   MAC is skipped and the slot output is set to the bias vector directly.
  *   This saves `lanes × (lanes / projectionMacLanes)` MAC cycles per bypassed
  *   projection.  The bypass count is exposed on `io.vectorBypassCount` so that
  *   test benches and higher-level wrappers can observe runtime sparsity.
  */
class UnifiedProjectionScheduler4(
    numSlots:            Int = UnifiedProjectionSlots.NumSlots,
    dataWidth:           Int = 8,
    accWidth:            Int = 32,
    lanes:               Int = 4,
    projectionMacLanes:  Int = 1,
    zeroSkip:            Boolean = false,
    vectorBypass:        Boolean = false)
    extends Module {
  require(numSlots > 0, "UnifiedProjectionScheduler4 must schedule at least one projection")
  require(dataWidth > 0, "UnifiedProjectionScheduler4 dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "UnifiedProjectionScheduler4 accWidth should hold four products")
  require(lanes > 0, "UnifiedProjectionScheduler4 lanes must be positive")
  require(projectionMacLanes >= 1, "UnifiedProjectionScheduler4 projectionMacLanes must be positive")
  require(projectionMacLanes <= lanes, "UnifiedProjectionScheduler4 projectionMacLanes must be <= lanes")
  require(lanes % projectionMacLanes == 0, "UnifiedProjectionScheduler4 lanes must be divisible by projectionMacLanes")

  private val slotWidth = math.max(1, log2Ceil(numSlots + 1))
  private val bypassCountWidth = 8 // saturates at 255; sufficient for numSlots <= 14

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val slotEnable = Input(Vec(numSlots, Bool()))
    val x = Input(Vec(numSlots, Vec(lanes, SInt(dataWidth.W))))
    val weight = Input(Vec(numSlots, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val bias = Input(Vec(numSlots, Vec(lanes, SInt(accWidth.W))))

    val ready = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val slotIndex = Output(UInt(slotWidth.W))
    val y = Output(Vec(numSlots, Vec(lanes, SInt(accWidth.W))))
    // M10-D: number of slots bypassed (input all-zero) in the most recent run.
    // Always 0 when vectorBypass = false.
    val vectorBypassCount = Output(UInt(bypassCountWidth.W))
  })

  private def zeroX =
    VecInit(Seq.fill(numSlots)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def zeroWeight =
    VecInit(Seq.fill(numSlots)(VecInit(Seq.fill(lanes)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))))
  private def zeroBias =
    VecInit(Seq.fill(numSlots)(VecInit(Seq.fill(lanes)(0.S(accWidth.W)))))
  private def zeroY =
    VecInit(Seq.fill(numSlots)(VecInit(Seq.fill(lanes)(0.S(accWidth.W)))))

  val idle :: findSlot :: launch :: waitLinear :: Nil = Enum(4)
  val state = RegInit(idle)
  val slot = RegInit(0.U(slotWidth.W))
  val doneReg = RegInit(false.B)
  val bypassCountReg = RegInit(0.U(bypassCountWidth.W))

  val slotEnableReg = RegInit(VecInit(Seq.fill(numSlots)(false.B)))
  val xReg = RegInit(zeroX)
  val weightReg = RegInit(zeroWeight)
  val biasReg = RegInit(zeroBias)
  val yReg = RegInit(zeroY)

  val candidates = VecInit((0 until numSlots).map(i => slotEnableReg(i) && (i.U >= slot)))
  val hasCandidate = candidates.asUInt.orR
  val nextSlot = PriorityEncoder(candidates)
  val activeSlot = Mux(slot < numSlots.U, slot, 0.U)

  // M10-D: combinational all-zero check on the candidate slot's input vector.
  // When vectorBypass = false this wire is tied to false (no overhead).
  val candidateInputAllZero: Bool =
    if (vectorBypass)
      VecInit(xReg(nextSlot).map(_ === 0.S(dataWidth.W))).asUInt.andR
    else
      false.B

  val linear = Module(new ConfigurableSerialLinear4(dataWidth, accWidth, lanes, projectionMacLanes, zeroSkip))
  linear.io.start := state === launch
  linear.io.clear := io.clear
  linear.io.x := xReg(activeSlot)
  linear.io.weight := weightReg(activeSlot)
  linear.io.bias := biasReg(activeSlot)

  when(io.clear) {
    state := idle
    slot := 0.U
    doneReg := false.B
    bypassCountReg := 0.U
    slotEnableReg := VecInit(Seq.fill(numSlots)(false.B))
    yReg := zeroY
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      slotEnableReg := io.slotEnable
      xReg := io.x
      weightReg := io.weight
      biasReg := io.bias
      yReg := zeroY
      slot := 0.U
      bypassCountReg := 0.U
      state := findSlot
    }
  }.elsewhen(state === findSlot) {
    doneReg := false.B
    when(hasCandidate) {
      // M10-D: if vectorBypass enabled and input is all-zero, skip the MAC.
      // Output for this slot becomes the bias vector (W·0 + b = b).
      when(candidateInputAllZero) {
        yReg(nextSlot) := biasReg(nextSlot)
        bypassCountReg := Mux(
          bypassCountReg === ((1 << bypassCountWidth) - 1).U,
          bypassCountReg,           // saturate
          bypassCountReg + 1.U
        )
        slot := nextSlot + 1.U     // advance past the bypassed slot
        // stay in findSlot to look for the next candidate
      }.otherwise {
        slot := nextSlot
        state := launch
      }
    }.otherwise {
      state := idle
      doneReg := true.B
    }
  }.elsewhen(state === launch) {
    doneReg := false.B
    state := waitLinear
  }.elsewhen(state === waitLinear) {
    doneReg := false.B
    when(linear.io.done) {
      yReg(activeSlot) := linear.io.y
      slot := slot + 1.U
      state := findSlot
    }
  }

  io.ready := state === idle
  io.busy := state =/= idle
  io.done := doneReg
  io.slotIndex := slot
  io.y := yReg
  io.vectorBypassCount := bypassCountReg
}
