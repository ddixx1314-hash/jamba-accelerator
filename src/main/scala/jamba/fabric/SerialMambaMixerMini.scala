package jamba.fabric

import chisel3._
import chisel3.util.Enum

/** Token-level serial Mamba mixer shell.
  *
  * All three compute stages are time-multiplexed through a single MAC lane:
  *   1. Serial projection  (SerialMambaProjectionGroup)
  *   2. Serial causal conv (SerialCausalConvMini)
  *   3. Serial selective scan (SerialSelectiveScanMini)
  *
  * Each stage fires sequentially; the module pulses `done` when all three
  * have completed for one token.
  */
class SerialMambaMixerMini(lanes: Int = 4, taps: Int = 4, dataWidth: Int = 8, stateWidth: Int = 32, accWidth: Int = 32)
    extends Module {
  require(lanes == 4, "SerialMambaMixerMini currently uses 4-lane serial projection groups")
  require(taps > 0, "SerialMambaMixerMini taps must be positive")

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))

    val inputWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val inputBias = Input(Vec(lanes, SInt(accWidth.W)))
    val bWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val bBias = Input(Vec(lanes, SInt(accWidth.W)))
    val cWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val cBias = Input(Vec(lanes, SInt(accWidth.W)))
    val a = Input(Vec(lanes, SInt(dataWidth.W)))
    val kernel = Input(Vec(taps, Vec(lanes, SInt(dataWidth.W))))

    val ready = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val valid = Output(Bool())
    val projected = Output(Vec(lanes, SInt(dataWidth.W)))
    val conv = Output(Vec(lanes, SInt(accWidth.W)))
    val b = Output(Vec(lanes, SInt(dataWidth.W)))
    val c = Output(Vec(lanes, SInt(dataWidth.W)))
    val stateOut = Output(Vec(lanes, SInt(stateWidth.W)))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
  })

  private def zeroData = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroAcc = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))

  val idle :: project :: launchConv :: waitConv :: launchScan :: waitScan :: doneState :: Nil = Enum(7)
  val state = RegInit(idle)
  val doneReg = RegInit(false.B)

  val projectedReg = RegInit(zeroData)
  val bReg = RegInit(zeroData)
  val cReg = RegInit(zeroData)
  val convReg = RegInit(zeroAcc)

  val projections = Module(new SerialMambaProjectionGroup(dataWidth, accWidth))
  projections.io.start := io.start && state === idle
  projections.io.clear := io.clear
  projections.io.x := io.x
  projections.io.inputWeight := io.inputWeight
  projections.io.inputBias := io.inputBias
  projections.io.bWeight := io.bWeight
  projections.io.bBias := io.bBias
  projections.io.cWeight := io.cWeight
  projections.io.cBias := io.cBias

  val conv = Module(new SerialCausalConvMini(lanes, taps, dataWidth, accWidth))
  conv.io.start       := state === launchConv
  conv.io.clear       := io.clear
  conv.io.x           := projectedReg
  conv.io.kernel      := io.kernel
  conv.io.loadHistory := false.B
  conv.io.historyIn   := VecInit(Seq.fill(taps - 1)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))

  val scan = Module(new SerialSelectiveScanMini(lanes, dataWidth, stateWidth, accWidth))
  scan.io.start     := state === launchScan
  scan.io.clear     := io.clear
  scan.io.a         := io.a
  scan.io.loadState := false.B
  scan.io.stateIn   := VecInit(Seq.fill(lanes)(0.S(stateWidth.W)))

  for (i <- 0 until lanes) {
    scan.io.x(i) := convReg(i)(dataWidth - 1, 0).asSInt
    scan.io.b(i) := bReg(i)
    scan.io.c(i) := cReg(i)
  }

  when(io.clear) {
    state := idle
    doneReg := false.B
    projectedReg := zeroData
    bReg := zeroData
    cReg := zeroData
    convReg := zeroAcc
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      state := project
    }
  }.elsewhen(state === project) {
    doneReg := false.B
    when(projections.io.done) {
      projectedReg := projections.io.projected
      bReg := projections.io.b
      cReg := projections.io.c
      state := launchConv
    }
  }.elsewhen(state === launchConv) {
    doneReg := false.B
    state := waitConv
  }.elsewhen(state === waitConv) {
    doneReg := false.B
    when(conv.io.done) {
      convReg := conv.io.y
      state := launchScan
    }
  }.elsewhen(state === launchScan) {
    doneReg := false.B
    state := waitScan
  }.elsewhen(state === waitScan) {
    doneReg := false.B
    when(scan.io.done) {
      state := doneState
    }
  }.elsewhen(state === doneState) {
    doneReg := true.B
    state := idle
  }

  io.ready := state === idle
  io.busy := state =/= idle
  io.done := doneReg
  io.valid := doneReg
  io.projected := projectedReg
  io.conv := convReg
  io.b := bReg
  io.c := cReg
  io.stateOut := scan.io.stateOut
  io.y := scan.io.y
}
