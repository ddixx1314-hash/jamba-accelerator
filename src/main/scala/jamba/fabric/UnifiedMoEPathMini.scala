package jamba.fabric

import chisel3._
import chisel3.util.Enum

/** Token-serial top-1 MoE-lite path through the unified projection scheduler.
  *
  * Router is represented as a 4-output projection: row 0 is expert 0's router
  * score, row 1 is expert 1's router score, and rows 2/3 are unused. After
  * top-1 selection, only the selected expert's gate/up/down projections are
  * scheduled.
  */
class UnifiedMoEPathMini(
    lanes:              Int = 4,
    numExperts:         Int = 2,
    dataWidth:          Int = 8,
    accWidth:           Int = 32,
    projectionMacLanes: Int = 1)
    extends Module {
  require(lanes > 0, "UnifiedMoEPathMini lanes must be positive")
  require(numExperts == 2, "UnifiedMoEPathMini first implementation supports exactly 2 experts")
  require(projectionMacLanes >= 1, "UnifiedMoEPathMini projectionMacLanes must be positive")
  require(projectionMacLanes <= lanes, "UnifiedMoEPathMini projectionMacLanes must be <= lanes")
  require(lanes % projectionMacLanes == 0, "UnifiedMoEPathMini lanes must be divisible by projectionMacLanes")

  private val slots = UnifiedProjectionSlots.NumMoESlots

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))

    val routerWeight = Input(Vec(numExperts, Vec(lanes, SInt(dataWidth.W))))
    val routerBias = Input(Vec(numExperts, SInt(accWidth.W)))
    val expertGateWeight = Input(Vec(numExperts, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertGateBias = Input(Vec(numExperts, Vec(lanes, SInt(accWidth.W))))
    val expertUpWeight = Input(Vec(numExperts, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertUpBias = Input(Vec(numExperts, Vec(lanes, SInt(accWidth.W))))
    val expertDownWeight = Input(Vec(numExperts, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertDownBias = Input(Vec(numExperts, Vec(lanes, SInt(accWidth.W))))

    val dispatchReady = Input(Bool())
    val combineReady = Input(Bool())
    val ready = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val dispatchValid = Output(Bool())
    val combineValid = Output(Bool())
    val selectedExpert = Output(UInt(1.W))
    val routerScores = Output(Vec(numExperts, SInt(accWidth.W)))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
    val projectionBusy = Output(Bool())
  })

  private def zeroData = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroAcc = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  private def zeroRouterWeight =
    VecInit(Seq.fill(lanes)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def zeroBias = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  private def narrowToData(value: SInt): SInt = value(dataWidth - 1, 0).asSInt

  val states = Enum(9)
  val idle = states(0)
  val launchRouter = states(1)
  val waitRouter = states(2)
  val launchExpertGateUp = states(3)
  val waitExpertGateUp = states(4)
  val computeHidden = states(5)
  val launchExpertDown = states(6)
  val waitExpertDown = states(7)
  val doneState = states(8)
  val state = RegInit(idle)
  val doneReg = RegInit(false.B)

  val xReg = RegInit(zeroData)
  val selectedExpertReg = RegInit(0.U(1.W))
  val routerScoresReg = RegInit(VecInit(Seq.fill(numExperts)(0.S(accWidth.W))))
  val gateReg = RegInit(zeroAcc)
  val upReg = RegInit(zeroAcc)
  val hiddenReg = RegInit(zeroData)
  val yReg = RegInit(zeroAcc)

  val scheduler = Module(new UnifiedProjectionScheduler4(
    numSlots = slots,
    dataWidth = dataWidth,
    accWidth = accWidth,
    lanes = lanes,
    projectionMacLanes = projectionMacLanes
  ))
  scheduler.io.clear := io.clear

  val slotEnable = WireDefault(VecInit(Seq.fill(slots)(false.B)))
  val slotX = Wire(Vec(slots, Vec(lanes, SInt(dataWidth.W))))
  val slotWeight = Wire(Vec(slots, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
  val slotBias = Wire(Vec(slots, Vec(lanes, SInt(accWidth.W))))
  for (slot <- 0 until slots) {
    slotX(slot) := zeroData
    slotWeight(slot) := zeroRouterWeight
    slotBias(slot) := zeroBias
  }

  val s = UnifiedProjectionSlots
  val routerWeightAsProjection = Wire(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
  val routerBiasAsProjection = Wire(Vec(lanes, SInt(accWidth.W)))
  routerWeightAsProjection := zeroRouterWeight
  routerBiasAsProjection := zeroBias
  for (expert <- 0 until numExperts) {
    routerWeightAsProjection(expert) := io.routerWeight(expert)
    routerBiasAsProjection(expert) := io.routerBias(expert)
  }

  val expertGateWeight = Mux(selectedExpertReg === 1.U, io.expertGateWeight(1), io.expertGateWeight(0))
  val expertGateBias = Mux(selectedExpertReg === 1.U, io.expertGateBias(1), io.expertGateBias(0))
  val expertUpWeight = Mux(selectedExpertReg === 1.U, io.expertUpWeight(1), io.expertUpWeight(0))
  val expertUpBias = Mux(selectedExpertReg === 1.U, io.expertUpBias(1), io.expertUpBias(0))
  val expertDownWeight = Mux(selectedExpertReg === 1.U, io.expertDownWeight(1), io.expertDownWeight(0))
  val expertDownBias = Mux(selectedExpertReg === 1.U, io.expertDownBias(1), io.expertDownBias(0))

  slotX(s.MoERouter) := xReg
  slotWeight(s.MoERouter) := routerWeightAsProjection
  slotBias(s.MoERouter) := routerBiasAsProjection
  slotX(s.MoEExpertGate) := xReg
  slotWeight(s.MoEExpertGate) := expertGateWeight
  slotBias(s.MoEExpertGate) := expertGateBias
  slotX(s.MoEExpertUp) := xReg
  slotWeight(s.MoEExpertUp) := expertUpWeight
  slotBias(s.MoEExpertUp) := expertUpBias
  slotX(s.MoEExpertDown) := hiddenReg
  slotWeight(s.MoEExpertDown) := expertDownWeight
  slotBias(s.MoEExpertDown) := expertDownBias

  when(state === launchRouter) {
    slotEnable(s.MoERouter) := true.B
  }.elsewhen(state === launchExpertGateUp) {
    slotEnable(s.MoEExpertGate) := true.B
    slotEnable(s.MoEExpertUp) := true.B
  }.elsewhen(state === launchExpertDown) {
    slotEnable(s.MoEExpertDown) := true.B
  }

  scheduler.io.start := state === launchRouter || state === launchExpertGateUp || state === launchExpertDown
  scheduler.io.slotEnable := slotEnable
  scheduler.io.x := slotX
  scheduler.io.weight := slotWeight
  scheduler.io.bias := slotBias

  when(io.clear) {
    state := idle
    doneReg := false.B
    xReg := zeroData
    selectedExpertReg := 0.U
    routerScoresReg := VecInit(Seq.fill(numExperts)(0.S(accWidth.W)))
    gateReg := zeroAcc
    upReg := zeroAcc
    hiddenReg := zeroData
    yReg := zeroAcc
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      xReg := io.x
      state := launchRouter
    }
  }.elsewhen(state === launchRouter) {
    doneReg := false.B
    state := waitRouter
  }.elsewhen(state === waitRouter) {
    doneReg := false.B
    when(scheduler.io.done) {
      routerScoresReg(0) := scheduler.io.y(s.MoERouter)(0)
      routerScoresReg(1) := scheduler.io.y(s.MoERouter)(1)
      selectedExpertReg := Mux(scheduler.io.y(s.MoERouter)(1) > scheduler.io.y(s.MoERouter)(0), 1.U, 0.U)
      state := launchExpertGateUp
    }
  }.elsewhen(state === launchExpertGateUp) {
    doneReg := false.B
    state := waitExpertGateUp
  }.elsewhen(state === waitExpertGateUp) {
    doneReg := false.B
    when(scheduler.io.done) {
      gateReg := scheduler.io.y(s.MoEExpertGate)
      upReg := scheduler.io.y(s.MoEExpertUp)
      state := computeHidden
    }
  }.elsewhen(state === computeHidden) {
    doneReg := false.B
    for (lane <- 0 until lanes) {
      val activatedGate = Mux(gateReg(lane) < 0.S, 0.S, narrowToData(gateReg(lane)))
      hiddenReg(lane) := narrowToData(activatedGate * narrowToData(upReg(lane)))
    }
    state := launchExpertDown
  }.elsewhen(state === launchExpertDown) {
    doneReg := false.B
    state := waitExpertDown
  }.elsewhen(state === waitExpertDown) {
    doneReg := false.B
    when(scheduler.io.done) {
      yReg := scheduler.io.y(s.MoEExpertDown)
      state := doneState
    }
  }.elsewhen(state === doneState) {
    doneReg := true.B
    state := idle
  }

  io.ready := state === idle
  io.busy := state =/= idle
  io.done := doneReg
  io.dispatchValid := (state =/= idle) && io.dispatchReady
  io.combineValid := doneReg && io.combineReady
  io.selectedExpert := selectedExpertReg
  io.routerScores := routerScoresReg
  io.y := yReg
  io.projectionBusy := scheduler.io.busy
}
