package jamba.fabric

import chisel3._
import chisel3.util.{Enum, MuxCase, log2Ceil}
import jamba.common.{FixedPointMath, SignedMath}
import jamba.norm.RmsNormApprox

/** Jamba2 mini layer using one unified projection scheduler.
  *
  * The projection-heavy paths are time-multiplexed through one
  * UnifiedProjectionScheduler4:
  *   - Mamba input/B/C projections
  *   - Attention Q/K/V/out projections
  *   - Dense MLP gate/up/down projections
  *
  * Causal convolution, selective scan, and attention score/value accumulation
  * remain specialized units in this first unified layer step.
  */
class UnifiedJamba2MiniLayer(
    lanes:              Int = 4,
    taps:               Int = 4,
    contextLength:      Int = 4,
    dataWidth:          Int = 8,
    stateWidth:         Int = 32,
    accWidth:           Int = 32,
    normShift:          Int = 2,
    projectionMacLanes: Int = 1,
    zeroSkipScan:       Boolean = false,
    vectorBypass:       Boolean = false,
    fusedOperators:     Boolean = false,
    useShiftA:          Boolean = false,   // M12-P: power-of-two A in SSM (shift instead of multiply)
    attentionWindowSize: Int   = 0,        // M12-K: sliding window size (0 = full context, Samba-style)
    fuseInnerLaunch:    Boolean = false)   // M14-F: fuse launchConv/launchScan/launchAttentionOut into
                                           //   preceding wait states, saving 2 Mamba + 1 Attention FSM cycles
    extends Module {
  require(lanes > 0, "UnifiedJamba2MiniLayer lanes must be positive")
  require(attentionWindowSize >= 0, "attentionWindowSize must be non-negative")
  require(attentionWindowSize == 0 || attentionWindowSize <= contextLength,
    "attentionWindowSize must be <= contextLength (or 0 for full context)")
  require(taps > 1, "UnifiedJamba2MiniLayer taps must be > 1 (historyIn/Out size = taps - 1)")
  require(contextLength > 0, "UnifiedJamba2MiniLayer contextLength must be positive")
  require(projectionMacLanes >= 1, "UnifiedJamba2MiniLayer projectionMacLanes must be positive")
  require(projectionMacLanes <= lanes, "UnifiedJamba2MiniLayer projectionMacLanes must be <= lanes")
  require(lanes % projectionMacLanes == 0, "UnifiedJamba2MiniLayer lanes must be divisible by projectionMacLanes")

  private val indexWidth = math.max(1, log2Ceil(contextLength))
  private val countWidth = log2Ceil(contextLength + 1)
  private val slots = UnifiedProjectionSlots.NumSlots

  val io = IO(new Bundle {
    val start        = Input(Bool())
    val clear        = Input(Bool())
    val useAttention = Input(Bool())
    val enableMoE    = Input(Bool())
    val x            = Input(Vec(lanes, SInt(dataWidth.W)))

    val norm1Weight = Input(Vec(lanes, SInt(dataWidth.W)))
    val norm2Weight = Input(Vec(lanes, SInt(dataWidth.W)))

    val mambaInputWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaInputBias   = Input(Vec(lanes, SInt(accWidth.W)))
    val mambaBWeight     = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaBBias       = Input(Vec(lanes, SInt(accWidth.W)))
    val mambaCWeight     = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaCBias       = Input(Vec(lanes, SInt(accWidth.W)))
    val mambaA           = Input(Vec(lanes, SInt(dataWidth.W)))
    val mambaKernel      = Input(Vec(taps, Vec(lanes, SInt(dataWidth.W))))

    val qWeight            = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val qBias              = Input(Vec(lanes, SInt(accWidth.W)))
    val kWeight            = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val kBias              = Input(Vec(lanes, SInt(accWidth.W)))
    val vWeight            = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val vBias              = Input(Vec(lanes, SInt(accWidth.W)))
    val attentionOutWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val attentionOutBias   = Input(Vec(lanes, SInt(accWidth.W)))

    val mlpGateWeight    = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpGateBias      = Input(Vec(lanes, SInt(accWidth.W)))
    val mlpUpWeight      = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpUpBias        = Input(Vec(lanes, SInt(accWidth.W)))
    val mlpDownWeight    = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpDownBias      = Input(Vec(lanes, SInt(accWidth.W)))
    val routerWeight     = Input(Vec(2, Vec(lanes, SInt(dataWidth.W))))
    val routerBias       = Input(Vec(2, SInt(accWidth.W)))
    val expertGateWeight = Input(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertGateBias   = Input(Vec(2, Vec(lanes, SInt(accWidth.W))))
    val expertUpWeight   = Input(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertUpBias     = Input(Vec(2, Vec(lanes, SInt(accWidth.W))))
    val expertDownWeight = Input(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertDownBias   = Input(Vec(2, Vec(lanes, SInt(accWidth.W))))

    // State save/restore for SinglePhysicalLayerTile (M7-B)
    val loadState   = Input(Bool())
    val stateIn     = Input(Vec(lanes, SInt(stateWidth.W)))
    val loadHistory = Input(Bool())
    val historyIn   = Input(Vec(taps - 1, Vec(lanes, SInt(dataWidth.W))))
    val loadKvState     = Input(Bool())
    val keyCacheIn      = Input(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))
    val valueCacheIn    = Input(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))
    val kvWriteIndexIn  = Input(UInt(indexWidth.W))
    val kvValidCountIn  = Input(UInt(countWidth.W))

    val ready          = Output(Bool())
    val busy           = Output(Bool())
    val done           = Output(Bool())
    val y              = Output(Vec(lanes, SInt(accWidth.W)))
    val mixerY         = Output(Vec(lanes, SInt(accWidth.W)))
    val firstResidual  = Output(Vec(lanes, SInt(dataWidth.W)))
    val mlpY           = Output(Vec(lanes, SInt(accWidth.W)))
    val stateOut       = Output(Vec(lanes, SInt(stateWidth.W)))
    val historyOut     = Output(Vec(taps - 1, Vec(lanes, SInt(dataWidth.W))))
    val keyCacheOut    = Output(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))
    val valueCacheOut  = Output(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))
    val kvWriteIndex   = Output(UInt(indexWidth.W))
    val kvValidCount   = Output(UInt(countWidth.W))
    val mixerType      = Output(Bool())
    val dispatchValid  = Output(Bool())
    val combineValid   = Output(Bool())
    val selectedExpert = Output(UInt(1.W))
    val projectionBusy        = Output(Bool())
    val projectionSlot        = Output(UInt(log2Ceil(slots + 1).W))
    // M10-D: number of projection slots bypassed (zero-input) in the last token.
    // Always 0 when vectorBypass = false.
    val projectionBypassCount = Output(UInt(8.W))
  })

  private def zeroData = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroAcc = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  private def zeroState = VecInit(Seq.fill(lanes)(0.S(stateWidth.W)))
  private def zeroCache = VecInit(Seq.fill(contextLength)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def narrowBits(v: SInt): SInt = v(dataWidth - 1, 0).asSInt
  private def saturate(v: SInt): SInt = FixedPointMath.saturate(v, dataWidth)

  val states = Enum(19)
  val idle = states(0)
  val launchMixerProj = states(1)
  val waitMixerProj = states(2)
  val launchConv = states(3)
  val waitConv = states(4)
  val launchScan = states(5)
  val waitScan = states(6)
  val computeAttention = states(7)
  val launchAttentionOut = states(8)
  val waitAttentionOut = states(9)
  val computeFirstResidual = states(10)
  val launchMlpGateUp = states(11)
  val waitMlpGateUp = states(12)
  val computeHidden = states(13)
  val launchMlpDown = states(14)
  val waitMlpDown = states(15)
  val launchMoE = states(16)
  val waitMoE = states(17)
  val doneState = states(18)

  val state = RegInit(idle)
  val doneReg = RegInit(false.B)

  val xReg = RegInit(zeroData)
  val useAttentionReg = RegInit(false.B)
  val enableMoEReg = RegInit(false.B)
  val firstResidualReg = RegInit(zeroData)
  val mixerYReg = RegInit(zeroAcc)
  val mlpYReg = RegInit(zeroAcc)
  val yReg = RegInit(zeroAcc)

  val mambaProjectedReg = RegInit(zeroData)
  val mambaBReg = RegInit(zeroData)
  val mambaCReg = RegInit(zeroData)
  val convReg = RegInit(zeroAcc)

  val qReg = RegInit(zeroData)
  val kReg = RegInit(zeroData)
  val vReg = RegInit(zeroData)
  val scoresReg = RegInit(VecInit(Seq.fill(contextLength)(0.S(accWidth.W))))
  val weightsReg = RegInit(VecInit(Seq.fill(contextLength)(0.S(accWidth.W))))
  val rawYReg = RegInit(zeroAcc)
  // M14-F: declare rawYWire early so it can be referenced in slotX(AttentionOut) before the
  // attention combinational block (which connects rawYWire) is elaborated at line ~370.
  val rawYWire = Wire(Vec(lanes, SInt(accWidth.W)))

  val gateReg = RegInit(zeroAcc)
  val upReg = RegInit(zeroAcc)
  val hiddenReg = RegInit(zeroData)

  val keyCache = RegInit(zeroCache)
  val valueCache = RegInit(zeroCache)
  val writeIndex = RegInit(0.U(indexWidth.W))
  val validCount = RegInit(0.U(countWidth.W))

  val norm1 = Module(new RmsNormApprox(lanes, dataWidth, accWidth))
  norm1.io.x := xReg
  norm1.io.weight := io.norm1Weight

  val norm2 = Module(new RmsNormApprox(lanes, dataWidth, accWidth))
  norm2.io.x := firstResidualReg
  norm2.io.weight := io.norm2Weight

  val scheduler = Module(new UnifiedProjectionScheduler4(
    numSlots = slots,
    dataWidth = dataWidth,
    accWidth = accWidth,
    lanes = lanes,
    projectionMacLanes = projectionMacLanes,
    vectorBypass = vectorBypass
  ))
  scheduler.io.clear := io.clear

  val slotEnable = WireDefault(VecInit(Seq.fill(slots)(false.B)))
  val slotX = Wire(Vec(slots, Vec(lanes, SInt(dataWidth.W))))
  val slotWeight = Wire(Vec(slots, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
  val slotBias = Wire(Vec(slots, Vec(lanes, SInt(accWidth.W))))
  for (slot <- 0 until slots) {
    slotX(slot) := zeroData
    for (row <- 0 until lanes) {
      slotWeight(slot)(row) := zeroData
    }
    slotBias(slot) := zeroAcc
  }

  val s = UnifiedProjectionSlots
  slotX(s.MambaInput) := norm1.io.y
  slotWeight(s.MambaInput) := io.mambaInputWeight
  slotBias(s.MambaInput) := io.mambaInputBias
  slotX(s.MambaB) := norm1.io.y
  slotWeight(s.MambaB) := io.mambaBWeight
  slotBias(s.MambaB) := io.mambaBBias
  slotX(s.MambaC) := norm1.io.y
  slotWeight(s.MambaC) := io.mambaCWeight
  slotBias(s.MambaC) := io.mambaCBias

  slotX(s.AttentionQ) := norm1.io.y
  slotWeight(s.AttentionQ) := io.qWeight
  slotBias(s.AttentionQ) := io.qBias
  slotX(s.AttentionK) := norm1.io.y
  slotWeight(s.AttentionK) := io.kWeight
  slotBias(s.AttentionK) := io.kBias
  slotX(s.AttentionV) := norm1.io.y
  slotWeight(s.AttentionV) := io.vWeight
  slotBias(s.AttentionV) := io.vBias
  // M14-F: when fusing launchAttentionOut into computeAttention, use rawYWire (current-cycle
  // combinational value) so the scheduler latches the correct attention output before rawYReg
  // is updated. When fuseInnerLaunch=false, rawYWire is never referenced here (compile-time if).
  slotX(s.AttentionOut) := {
    val fuseAttnActive: Bool =
      if (fuseInnerLaunch) state === computeAttention else false.B
    Mux(fuseAttnActive, VecInit(rawYWire.map(saturate)), VecInit(rawYReg.map(saturate)))
  }
  slotWeight(s.AttentionOut) := io.attentionOutWeight
  slotBias(s.AttentionOut) := io.attentionOutBias

  slotX(s.MlpGate) := norm2.io.y
  slotWeight(s.MlpGate) := io.mlpGateWeight
  slotBias(s.MlpGate) := io.mlpGateBias
  slotX(s.MlpUp) := norm2.io.y
  slotWeight(s.MlpUp) := io.mlpUpWeight
  slotBias(s.MlpUp) := io.mlpUpBias
  slotX(s.MlpDown) := hiddenReg
  slotWeight(s.MlpDown) := io.mlpDownWeight
  slotBias(s.MlpDown) := io.mlpDownBias

  when(state === launchMixerProj && !useAttentionReg) {
    slotEnable(s.MambaInput) := true.B
    slotEnable(s.MambaB) := true.B
    slotEnable(s.MambaC) := true.B
  }.elsewhen(state === launchMixerProj && useAttentionReg) {
    slotEnable(s.AttentionQ) := true.B
    slotEnable(s.AttentionK) := true.B
    slotEnable(s.AttentionV) := true.B
  // M14-F: fused → enable AttentionOut slot in computeAttention itself (scheduler starts there)
  }.elsewhen(if (fuseInnerLaunch) state === computeAttention else state === launchAttentionOut) {
    slotEnable(s.AttentionOut) := true.B
  }.elsewhen(state === launchMlpGateUp) {
    slotEnable(s.MlpGate) := true.B
    slotEnable(s.MlpUp) := true.B
  }.elsewhen(state === launchMlpDown) {
    slotEnable(s.MlpDown) := true.B
  }

  // M14-F: fused → scheduler starts in computeAttention (not launchAttentionOut)
  val launchAttnCondition: Bool =
    if (fuseInnerLaunch) state === computeAttention else state === launchAttentionOut
  scheduler.io.start := state === launchMixerProj || launchAttnCondition ||
    state === launchMlpGateUp || state === launchMlpDown
  scheduler.io.slotEnable := slotEnable
  scheduler.io.x := slotX
  scheduler.io.weight := slotWeight
  scheduler.io.bias := slotBias

  val conv = Module(new SerialCausalConvMini(lanes, taps, dataWidth, accWidth))
  // M14-F: fused conv launch — assert start in waitMixerProj when scheduler.done (!attention).
  // conv.io.x uses scheduler output wire directly (skipping the mambaProjectedReg register stage).
  // When fuseInnerLaunch=false these wires resolve to false.B / mambaProjectedReg (no overhead).
  val fuseConvWire: Bool =
    if (fuseInnerLaunch)
      state === waitMixerProj && scheduler.io.done && !useAttentionReg
    else false.B
  conv.io.start       := state === launchConv || fuseConvWire
  conv.io.clear       := io.clear
  conv.io.loadHistory := io.loadHistory
  conv.io.x           := Mux(fuseConvWire,
    VecInit((0 until lanes).map(l => narrowBits(scheduler.io.y(s.MambaInput)(l)))),
    mambaProjectedReg)
  conv.io.kernel      := io.mambaKernel
  conv.io.historyIn   := io.historyIn

  val scan = Module(new SerialSelectiveScanMini(lanes, dataWidth, stateWidth, accWidth, zeroSkipScan, useShiftA))
  // M14-F: fused scan launch — assert start in waitConv when conv.done.
  // scan.io.x uses conv output wire directly (skipping convReg register stage).
  val fuseScanWire: Bool =
    if (fuseInnerLaunch)
      state === waitConv && conv.io.done
    else false.B
  scan.io.start     := state === launchScan || fuseScanWire
  scan.io.clear     := io.clear
  scan.io.loadState := io.loadState
  scan.io.stateIn   := io.stateIn
  scan.io.a         := io.mambaA
  for (lane <- 0 until lanes) {
    scan.io.x(lane) := Mux(fuseScanWire, narrowBits(conv.io.y(lane)), narrowBits(convReg(lane)))
    scan.io.b(lane) := mambaBReg(lane)
    scan.io.c(lane) := mambaCReg(lane)
  }

  val moe = Module(new UnifiedMoEPathMini(lanes, 2, dataWidth, accWidth, projectionMacLanes))
  moe.io.start := state === launchMoE
  moe.io.clear := io.clear
  moe.io.x := norm2.io.y
  moe.io.routerWeight := io.routerWeight
  moe.io.routerBias := io.routerBias
  moe.io.expertGateWeight := io.expertGateWeight
  moe.io.expertGateBias := io.expertGateBias
  moe.io.expertUpWeight := io.expertUpWeight
  moe.io.expertUpBias := io.expertUpBias
  moe.io.expertDownWeight := io.expertDownWeight
  moe.io.expertDownBias := io.expertDownBias
  moe.io.dispatchReady := true.B
  moe.io.combineReady := true.B

  val fullCache = validCount === contextLength.U
  val physicalRows = Wire(Vec(contextLength, UInt(indexWidth.W)))
  val rowValid = Wire(Vec(contextLength, Bool()))

  // M12-K Sliding Window Attention (Samba-style):
  //   attentionWindowSize=0  → full context (original behaviour, attends to all valid KV entries)
  //   attentionWindowSize=W  → only the W most recent KV entries are attended to.
  //   Older entries are masked out even if they exist in the circular buffer.
  //   Hardware effect: masked terms collapse to 0 in the weighted-sum — same area for RTL
  //   proxy but meaningful for synthesis (masked multipliers can be pruned) and correct
  //   alignment with Samba's Sliding Window Attention design.
  val windowStart = Wire(UInt(countWidth.W))
  if (attentionWindowSize > 0 && attentionWindowSize < contextLength) {
    val wsU = attentionWindowSize.U(countWidth.W)
    windowStart := Mux(validCount > wsU, validCount - wsU, 0.U)
  } else {
    windowStart := 0.U
  }

  for (row <- 0 until contextLength) {
    val shifted = writeIndex + row.U
    val wrappedRow = Mux(shifted >= contextLength.U, shifted - contextLength.U, shifted)
    physicalRows(row) := Mux(fullCache, wrappedRow, row.U)
    // Row is valid if it is within the recent window AND within the actual valid count.
    if (attentionWindowSize > 0 && attentionWindowSize < contextLength) {
      rowValid(row) := row.U >= windowStart && row.U < validCount
    } else {
      rowValid(row) := row.U < validCount
    }
  }

  val scoresWire = Wire(Vec(contextLength, SInt(accWidth.W)))
  val weightsWire = Wire(Vec(contextLength, SInt(accWidth.W)))
  // rawYWire declared earlier (line ~180) so it can be referenced in slotX(AttentionOut)
  for (row <- 0 until contextLength) {
    val products = Wire(Vec(lanes, SInt(accWidth.W)))
    for (lane <- 0 until lanes) {
      products(lane) := SignedMath.resize(qReg(lane) * keyCache(physicalRows(row))(lane), accWidth)
    }
    scoresWire(row) := Mux(rowValid(row), products.reduce(_ +& _), 0.S)
    weightsWire(row) := FixedPointMath.roundedShiftRight(scoresWire(row), normShift)
  }
  for (lane <- 0 until lanes) {
    val weighted = Wire(Vec(contextLength, SInt(accWidth.W)))
    for (row <- 0 until contextLength) {
      weighted(row) := Mux(
        rowValid(row),
        SignedMath.resize(weightsWire(row) * valueCache(physicalRows(row))(lane), accWidth),
        0.S
      )
    }
    rawYWire(lane) := weighted.reduce(_ +& _)
  }

  when(io.clear) {
    state := idle
    doneReg := false.B
    xReg := zeroData
    useAttentionReg := false.B
    enableMoEReg := false.B
    firstResidualReg := zeroData
    mixerYReg := zeroAcc
    mlpYReg := zeroAcc
    yReg := zeroAcc
    mambaProjectedReg := zeroData
    mambaBReg := zeroData
    mambaCReg := zeroData
    convReg := zeroAcc
    qReg := zeroData
    kReg := zeroData
    vReg := zeroData
    scoresReg := VecInit(Seq.fill(contextLength)(0.S(accWidth.W)))
    weightsReg := VecInit(Seq.fill(contextLength)(0.S(accWidth.W)))
    rawYReg := zeroAcc
    gateReg := zeroAcc
    upReg := zeroAcc
    hiddenReg := zeroData
    keyCache := zeroCache
    valueCache := zeroCache
    writeIndex := 0.U
    validCount := 0.U
  }.elsewhen(io.loadKvState) {
    keyCache   := io.keyCacheIn
    valueCache := io.valueCacheIn
    writeIndex := io.kvWriteIndexIn
    validCount := io.kvValidCountIn
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      xReg := io.x
      useAttentionReg := io.useAttention
      enableMoEReg := io.enableMoE
      state := launchMixerProj
    }
  }.elsewhen(state === launchMixerProj) {
    doneReg := false.B
    state := waitMixerProj
  }.elsewhen(state === waitMixerProj) {
    doneReg := false.B
    when(scheduler.io.done) {
      when(useAttentionReg) {
        for (lane <- 0 until lanes) {
          qReg(lane) := saturate(scheduler.io.y(s.AttentionQ)(lane))
          kReg(lane) := saturate(scheduler.io.y(s.AttentionK)(lane))
          vReg(lane) := saturate(scheduler.io.y(s.AttentionV)(lane))
          keyCache(writeIndex)(lane) := saturate(scheduler.io.y(s.AttentionK)(lane))
          valueCache(writeIndex)(lane) := saturate(scheduler.io.y(s.AttentionV)(lane))
        }
        writeIndex := Mux(writeIndex === (contextLength - 1).U, 0.U, writeIndex + 1.U)
        validCount := Mux(validCount === contextLength.U, validCount, validCount + 1.U)
        state := computeAttention
      }.otherwise {
        for (lane <- 0 until lanes) {
          mambaProjectedReg(lane) := narrowBits(scheduler.io.y(s.MambaInput)(lane))
          mambaBReg(lane) := narrowBits(scheduler.io.y(s.MambaB)(lane))
          mambaCReg(lane) := narrowBits(scheduler.io.y(s.MambaC)(lane))
        }
        // M14-F: skip launchConv — conv is already started via fuseConvWire in this cycle
        if (fuseInnerLaunch) { state := waitConv } else { state := launchConv }
      }
    }
  }.elsewhen(state === launchConv) {
    doneReg := false.B
    state := waitConv
  }.elsewhen(state === waitConv) {
    doneReg := false.B
    when(conv.io.done) {
      convReg := conv.io.y
      // M14-F: skip launchScan — scan is already started via fuseScanWire in this cycle
      if (fuseInnerLaunch) { state := waitScan } else { state := launchScan }
    }
  }.elsewhen(state === launchScan) {
    doneReg := false.B
    state := waitScan
  }.elsewhen(state === waitScan) {
    doneReg := false.B
    when(scan.io.done) {
      mixerYReg := scan.io.y
      if (fusedOperators) {
        // M11-F: fuse computeFirstResidual into waitScan — saves 1 FSM cycle per Mamba token.
        // firstResidualReg is needed by norm2 in the next launchMlpGateUp cycle.
        for (lane <- 0 until lanes) {
          firstResidualReg(lane) := narrowBits(xReg(lane) + scan.io.y(lane))
        }
        state := Mux(enableMoEReg, launchMoE, launchMlpGateUp)
      } else {
        state := computeFirstResidual
      }
    }
  }.elsewhen(state === computeAttention) {
    doneReg := false.B
    scoresReg := scoresWire
    weightsReg := weightsWire
    rawYReg := rawYWire
    // M14-F: skip launchAttentionOut — scheduler is already started via launchAttnCondition
    if (fuseInnerLaunch) { state := waitAttentionOut } else { state := launchAttentionOut }
  }.elsewhen(state === launchAttentionOut) {
    doneReg := false.B
    state := waitAttentionOut
  }.elsewhen(state === waitAttentionOut) {
    doneReg := false.B
    when(scheduler.io.done) {
      mixerYReg := scheduler.io.y(s.AttentionOut)
      if (fusedOperators) {
        // M11-F: fuse computeFirstResidual into waitAttentionOut — saves 1 FSM cycle per Attention token.
        for (lane <- 0 until lanes) {
          firstResidualReg(lane) := narrowBits(xReg(lane) + scheduler.io.y(s.AttentionOut)(lane))
        }
        state := Mux(enableMoEReg, launchMoE, launchMlpGateUp)
      } else {
        state := computeFirstResidual
      }
    }
  }.elsewhen(state === computeFirstResidual) {
    doneReg := false.B
    for (lane <- 0 until lanes) {
      firstResidualReg(lane) := narrowBits(xReg(lane) + mixerYReg(lane))
    }
    state := Mux(enableMoEReg, launchMoE, launchMlpGateUp)
  }.elsewhen(state === launchMoE) {
    doneReg := false.B
    state := waitMoE
  }.elsewhen(state === waitMoE) {
    doneReg := false.B
    when(moe.io.done) {
      mlpYReg := moe.io.y
      for (lane <- 0 until lanes) {
        yReg(lane) := firstResidualReg(lane) + moe.io.y(lane)
      }
      state := doneState
    }
  }.elsewhen(state === launchMlpGateUp) {
    doneReg := false.B
    state := waitMlpGateUp
  }.elsewhen(state === waitMlpGateUp) {
    doneReg := false.B
    when(scheduler.io.done) {
      gateReg := scheduler.io.y(s.MlpGate)
      upReg   := scheduler.io.y(s.MlpUp)
      if (fusedOperators) {
        // M11-F: fuse computeHidden into waitMlpGateUp — saves 1 FSM cycle per MLP token.
        // hiddenReg feeds slotX(s.MlpDown) for the launchMlpDown cycle.
        for (lane <- 0 until lanes) {
          val activatedGate = Mux(scheduler.io.y(s.MlpGate)(lane) < 0.S,
                                  0.S,
                                  narrowBits(scheduler.io.y(s.MlpGate)(lane)))
          hiddenReg(lane) := narrowBits(activatedGate * narrowBits(scheduler.io.y(s.MlpUp)(lane)))
        }
        state := launchMlpDown
      } else {
        state := computeHidden
      }
    }
  }.elsewhen(state === computeHidden) {
    doneReg := false.B
    for (lane <- 0 until lanes) {
      val activatedGate = Mux(gateReg(lane) < 0.S, 0.S, narrowBits(gateReg(lane)))
      hiddenReg(lane) := narrowBits(activatedGate * narrowBits(upReg(lane)))
    }
    state := launchMlpDown
  }.elsewhen(state === launchMlpDown) {
    doneReg := false.B
    state := waitMlpDown
  }.elsewhen(state === waitMlpDown) {
    doneReg := false.B
    when(scheduler.io.done) {
      mlpYReg := scheduler.io.y(s.MlpDown)
      for (lane <- 0 until lanes) {
        yReg(lane) := firstResidualReg(lane) + scheduler.io.y(s.MlpDown)(lane)
      }
      state := doneState
    }
  }.elsewhen(state === doneState) {
    doneReg := true.B
    state := idle
  }

  io.ready := state === idle
  io.busy := state =/= idle
  io.done := doneReg
  io.y := yReg
  io.mixerY := mixerYReg
  io.firstResidual := firstResidualReg
  io.mlpY := mlpYReg
  io.stateOut    := scan.io.stateOut
  io.historyOut  := conv.io.historyOut
  io.keyCacheOut := keyCache
  io.valueCacheOut := valueCache
  io.kvWriteIndex := writeIndex
  io.kvValidCount := validCount
  io.mixerType := useAttentionReg
  io.dispatchValid := enableMoEReg && (moe.io.dispatchValid || doneReg)
  io.combineValid := enableMoEReg && (moe.io.combineValid || doneReg)
  io.selectedExpert := Mux(enableMoEReg, moe.io.selectedExpert, 0.U)
  io.projectionBusy        := scheduler.io.busy || moe.io.projectionBusy
  io.projectionSlot        := scheduler.io.slotIndex
  io.projectionBypassCount := scheduler.io.vectorBypassCount
}
