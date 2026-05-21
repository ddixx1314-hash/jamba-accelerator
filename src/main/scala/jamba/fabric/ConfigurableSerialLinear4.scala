package jamba.fabric

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.SignedMath

/** Time-multiplexed linear projection with configurable MAC parallelism.
  *
  * Data path:
  *
  *   x / W[row] -> column chunk scheduler -> macLanes partial accumulators
  *              -> combinational reduction adder -> + bias[row] -> y[row]
  *
  * Each MacLane owns one column position per cycle. For lanes=4:
  *   - macLanes=1: 4 chunks/row, 16 cycles total
  *   - macLanes=2: 2 chunks/row,  8 cycles total
  *   - macLanes=4: 1 chunk /row,  4 cycles total
  *
  * Bias is added once at the end of each row, after reducing all partial sums.
  */
class ConfigurableSerialLinear4(
    dataWidth: Int = 8,
    accWidth:  Int = 32,
    lanes:     Int = 4,
    macLanes:  Int = 1,
    zeroSkip:  Boolean = false)
    extends Module {
  require(dataWidth > 0, "ConfigurableSerialLinear4 dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "ConfigurableSerialLinear4 accWidth should hold four products")
  require(lanes > 0, "ConfigurableSerialLinear4 lanes must be positive")
  require(macLanes >= 1, "ConfigurableSerialLinear4 macLanes must be positive")
  require(macLanes <= lanes, "ConfigurableSerialLinear4 macLanes must be <= lanes")
  require(lanes % macLanes == 0, "ConfigurableSerialLinear4 lanes must be divisible by macLanes")

  private val chunksPerRow = lanes / macLanes
  private val indexWidth = math.max(1, log2Ceil(lanes))
  private val chunkWidth = math.max(1, log2Ceil(chunksPerRow))

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

  private def zeroY = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  private def zeroPartials = VecInit(Seq.fill(macLanes)(0.S(accWidth.W)))

  val xReg = Reg(Vec(lanes, SInt(dataWidth.W)))
  val weightReg = Reg(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
  val biasReg = Reg(Vec(lanes, SInt(accWidth.W)))
  val yReg = RegInit(zeroY)

  val row = RegInit(0.U(indexWidth.W))
  val chunk = RegInit(0.U(chunkWidth.W))
  val partialAcc = RegInit(zeroPartials)
  val busyReg = RegInit(false.B)
  val doneReg = RegInit(false.B)

  val macs = Seq.fill(macLanes)(Module(new MacLane(dataWidth, accWidth, zeroSkip)))
  val nextPartials = Wire(Vec(macLanes, SInt(accWidth.W)))
  for (lane <- 0 until macLanes) {
    val rawCol = chunk * macLanes.U + lane.U
    val col = rawCol(indexWidth - 1, 0).asUInt
    macs(lane).io.a := xReg(col)
    macs(lane).io.b := weightReg(row)(col)
    macs(lane).io.accIn := partialAcc(lane)
    nextPartials(lane) := SignedMath.resize(macs(lane).io.accOut, accWidth)
  }

  private def reduce(values: Seq[SInt]): SInt =
    values.reduce((a, b) => SignedMath.resize(a +& b, accWidth))

  val rowPartialSum = reduce((0 until macLanes).map(nextPartials(_)))
  val rowWithBias = SignedMath.resize(rowPartialSum +& biasReg(row), accWidth)
  val lastChunk = chunk === (chunksPerRow - 1).U

  when(io.clear) {
    row := 0.U
    chunk := 0.U
    partialAcc := zeroPartials
    yReg := zeroY
    busyReg := false.B
    doneReg := false.B
  }.elsewhen(io.start && !busyReg) {
    xReg := io.x
    weightReg := io.weight
    biasReg := io.bias
    row := 0.U
    chunk := 0.U
    partialAcc := zeroPartials
    yReg := zeroY
    busyReg := true.B
    doneReg := false.B
  }.elsewhen(busyReg) {
    doneReg := false.B

    when(lastChunk) {
      yReg(row) := rowWithBias
      chunk := 0.U
      partialAcc := zeroPartials

      when(row === (lanes - 1).U) {
        row := 0.U
        busyReg := false.B
        doneReg := true.B
      }.otherwise {
        row := row + 1.U
      }
    }.otherwise {
      chunk := chunk + 1.U
      partialAcc := nextPartials
    }
  }.otherwise {
    doneReg := false.B
  }

  io.ready := !busyReg
  io.busy := busyReg
  io.done := doneReg
  io.rowIndex := row
  io.colIndex := chunk * macLanes.U
  io.y := yReg
}
