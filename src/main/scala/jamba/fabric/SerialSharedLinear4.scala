package jamba.fabric

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.SignedMath

/** Time-multiplexed 4x4 linear projection using one reusable MAC lane.
  *
  * The combinational `SharedLinear4` computes four dot products in parallel.
  * This serial version latches inputs on `start`, then reuses one `MacLane`
  * across 16 cycles. It is the first explicit latency/resource tradeoff point
  * in the shared-fabric track.
  */
class SerialSharedLinear4(dataWidth: Int = 8, accWidth: Int = 32, zeroSkip: Boolean = false) extends Module {
  require(dataWidth > 0, "SerialSharedLinear4 dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "SerialSharedLinear4 accWidth should hold four products")

  private val lanes = 4
  private val indexWidth = log2Ceil(lanes)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val weight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val bias = Input(Vec(lanes, SInt(accWidth.W)))

    val ready = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val rowIndex = Output(UInt(indexWidth.W))
    val colIndex = Output(UInt(indexWidth.W))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
  })

  val xReg = Reg(Vec(lanes, SInt(dataWidth.W)))
  val weightReg = Reg(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
  val biasReg = Reg(Vec(lanes, SInt(accWidth.W)))
  val yReg = RegInit(VecInit(Seq.fill(lanes)(0.S(accWidth.W))))

  val row = RegInit(0.U(indexWidth.W))
  val col = RegInit(0.U(indexWidth.W))
  val acc = RegInit(0.S(accWidth.W))
  val busyReg = RegInit(false.B)
  val doneReg = RegInit(false.B)

  val mac = Module(new MacLane(dataWidth, accWidth, zeroSkip))
  mac.io.a := xReg(col)
  mac.io.b := weightReg(row)(col)
  mac.io.accIn := acc
  val nextAcc = SignedMath.resize(mac.io.accOut, accWidth)

  when(io.clear) {
    row := 0.U
    col := 0.U
    acc := 0.S
    yReg := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
    busyReg := false.B
    doneReg := false.B
  }.elsewhen(io.start && !busyReg) {
    xReg := io.x
    weightReg := io.weight
    biasReg := io.bias
    row := 0.U
    col := 0.U
    acc := io.bias(0)
    yReg := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
    busyReg := true.B
    doneReg := false.B
  }.elsewhen(busyReg) {
    doneReg := false.B

    when(col === (lanes - 1).U) {
      yReg(row) := nextAcc
      col := 0.U

      when(row === (lanes - 1).U) {
        busyReg := false.B
        doneReg := true.B
        acc := 0.S
      }.otherwise {
        row := row + 1.U
        acc := biasReg(row + 1.U)
      }
    }.otherwise {
      col := col + 1.U
      acc := nextAcc
    }
  }.otherwise {
    doneReg := false.B
  }

  io.ready := !busyReg
  io.busy := busyReg
  io.done := doneReg
  io.rowIndex := row
  io.colIndex := col
  io.y := yReg
}
