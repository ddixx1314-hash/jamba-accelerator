package jamba.fabric

import chisel3._
import chisel3.util.{Enum, switch, is}
import jamba.common.FixedPointMath

/** Semantic wrapper for serial attention Q/K/V/out projections.
  *
  * Q, K, and V consume the token input. The output projection consumes a
  * separate already-narrowed attention result, so this module schedules four
  * projections while selecting the correct input vector for each projection.
  */
class SerialAttentionProjectionGroup(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  val lanes = 4

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val outInput = Input(Vec(lanes, SInt(dataWidth.W)))

    val qWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val qBias = Input(Vec(lanes, SInt(accWidth.W)))
    val kWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val kBias = Input(Vec(lanes, SInt(accWidth.W)))
    val vWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val vBias = Input(Vec(lanes, SInt(accWidth.W)))
    val outWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val outBias = Input(Vec(lanes, SInt(accWidth.W)))

    val ready = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val projectionIndex = Output(UInt(2.W))
    val qRaw = Output(Vec(lanes, SInt(accWidth.W)))
    val kRaw = Output(Vec(lanes, SInt(accWidth.W)))
    val vRaw = Output(Vec(lanes, SInt(accWidth.W)))
    val q = Output(Vec(lanes, SInt(dataWidth.W)))
    val k = Output(Vec(lanes, SInt(dataWidth.W)))
    val v = Output(Vec(lanes, SInt(dataWidth.W)))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
  })

  private def zeroData = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroAcc = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  private def zeroWeight = VecInit(Seq.fill(4)(VecInit(Seq.fill(lanes)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))))
  private def zeroBias = VecInit(Seq.fill(4)(VecInit(Seq.fill(lanes)(0.S(accWidth.W)))))
  private def narrowToData(value: SInt): SInt = FixedPointMath.saturate(value, dataWidth)

  val idle :: launch :: waitLinear :: Nil = Enum(3)
  val state = RegInit(idle)
  val projection = RegInit(0.U(2.W))
  val doneReg = RegInit(false.B)

  val xReg = RegInit(zeroData)
  val outInputReg = RegInit(zeroData)
  val weightReg = RegInit(zeroWeight)
  val biasReg = RegInit(zeroBias)
  val qReg = RegInit(zeroAcc)
  val kReg = RegInit(zeroAcc)
  val vReg = RegInit(zeroAcc)
  val yReg = RegInit(zeroAcc)

  val selectedInput = Wire(Vec(lanes, SInt(dataWidth.W)))
  selectedInput := Mux(projection === 3.U, outInputReg, xReg)

  val linear = Module(new SerialSharedLinear4(dataWidth, accWidth))
  linear.io.start := state === launch
  linear.io.clear := io.clear
  linear.io.x := selectedInput
  linear.io.weight := weightReg(projection)
  linear.io.bias := biasReg(projection)

  when(io.clear) {
    state := idle
    projection := 0.U
    doneReg := false.B
    qReg := zeroAcc
    kReg := zeroAcc
    vReg := zeroAcc
    yReg := zeroAcc
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      xReg := io.x
      outInputReg := io.outInput
      weightReg(0) := io.qWeight
      weightReg(1) := io.kWeight
      weightReg(2) := io.vWeight
      weightReg(3) := io.outWeight
      biasReg(0) := io.qBias
      biasReg(1) := io.kBias
      biasReg(2) := io.vBias
      biasReg(3) := io.outBias
      qReg := zeroAcc
      kReg := zeroAcc
      vReg := zeroAcc
      yReg := zeroAcc
      projection := 0.U
      state := launch
    }
  }.elsewhen(state === launch) {
    doneReg := false.B
    state := waitLinear
  }.elsewhen(state === waitLinear) {
    doneReg := false.B
    when(linear.io.done) {
      switch(projection) {
        is(0.U) { qReg := linear.io.y }
        is(1.U) { kReg := linear.io.y }
        is(2.U) { vReg := linear.io.y }
        is(3.U) { yReg := linear.io.y }
      }

      when(projection === 3.U) {
        state := idle
        doneReg := true.B
      }.otherwise {
        projection := projection + 1.U
        state := launch
      }
    }
  }

  io.ready := state === idle
  io.busy := state =/= idle
  io.done := doneReg
  io.projectionIndex := projection
  io.qRaw := qReg
  io.kRaw := kReg
  io.vRaw := vReg
  io.y := yReg

  for (lane <- 0 until lanes) {
    io.q(lane) := narrowToData(qReg(lane))
    io.k(lane) := narrowToData(kReg(lane))
    io.v(lane) := narrowToData(vReg(lane))
  }
}
