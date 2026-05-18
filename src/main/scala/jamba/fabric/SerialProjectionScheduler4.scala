package jamba.fabric

import chisel3._
import chisel3.util.{Enum, log2Ceil}

/** Schedules several 4x4 projections onto one serial Linear4 fabric.
  *
  * This is the first multi-operator time-multiplexing block. It can represent
  * Mamba input/B/C projections with `numProjections = 3`, or attention
  * Q/K/V/out projections with `numProjections = 4`.
  */
class SerialProjectionScheduler4(numProjections: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(numProjections > 0, "SerialProjectionScheduler4 must schedule at least one projection")
  require(dataWidth > 0, "SerialProjectionScheduler4 dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "SerialProjectionScheduler4 accWidth should hold four products")

  private val lanes = 4
  private val projectionIndexWidth = math.max(1, log2Ceil(numProjections))

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val weight = Input(Vec(numProjections, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val bias = Input(Vec(numProjections, Vec(lanes, SInt(accWidth.W))))

    val ready = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val projectionIndex = Output(UInt(projectionIndexWidth.W))
    val y = Output(Vec(numProjections, Vec(lanes, SInt(accWidth.W))))
  })

  private def zeroX = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroWeight = VecInit(Seq.fill(numProjections)(VecInit(Seq.fill(lanes)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))))
  private def zeroBias = VecInit(Seq.fill(numProjections)(VecInit(Seq.fill(lanes)(0.S(accWidth.W)))))
  private def zeroY = VecInit(Seq.fill(numProjections)(VecInit(Seq.fill(lanes)(0.S(accWidth.W)))))

  val idle :: launch :: waitLinear :: Nil = Enum(3)
  val state = RegInit(idle)
  val projection = RegInit(0.U(projectionIndexWidth.W))
  val doneReg = RegInit(false.B)

  val xReg = RegInit(zeroX)
  val weightReg = RegInit(zeroWeight)
  val biasReg = RegInit(zeroBias)
  val yReg = RegInit(zeroY)

  val linear = Module(new SerialSharedLinear4(dataWidth, accWidth))
  linear.io.start := state === launch
  linear.io.clear := io.clear
  linear.io.x := xReg
  linear.io.weight := weightReg(projection)
  linear.io.bias := biasReg(projection)

  when(io.clear) {
    state := idle
    projection := 0.U
    doneReg := false.B
    yReg := zeroY
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      xReg := io.x
      weightReg := io.weight
      biasReg := io.bias
      yReg := zeroY
      projection := 0.U
      state := launch
    }
  }.elsewhen(state === launch) {
    doneReg := false.B
    state := waitLinear
  }.elsewhen(state === waitLinear) {
    doneReg := false.B
    when(linear.io.done) {
      yReg(projection) := linear.io.y
      when(projection === (numProjections - 1).U) {
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
  io.y := yReg
}
