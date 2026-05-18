package jamba.fabric

import chisel3._
import chisel3.util.log2Ceil

/** Time-multiplexed per-lane causal convolution using one reusable MAC lane. */
class SerialCausalConvMini(lanes: Int = 4, taps: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(lanes > 0, "SerialCausalConvMini lanes must be positive")
  require(taps > 1, "SerialCausalConvMini currently expects at least two taps")
  require(dataWidth > 0, "SerialCausalConvMini dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "SerialCausalConvMini accWidth should hold products")

  private val laneWidth = math.max(1, log2Ceil(lanes))
  private val tapWidth = math.max(1, log2Ceil(taps))

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val kernel = Input(Vec(taps, Vec(lanes, SInt(dataWidth.W))))

    val ready = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val laneIndex = Output(UInt(laneWidth.W))
    val tapIndex = Output(UInt(tapWidth.W))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
  })

  private def zeroInput = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroKernel = VecInit(Seq.fill(taps)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def zeroHistory = VecInit(Seq.fill(taps - 1)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def zeroOutput = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))

  val xReg = RegInit(zeroInput)
  val kernelReg = RegInit(zeroKernel)
  val history = RegInit(zeroHistory)
  val yReg = RegInit(zeroOutput)

  val lane = RegInit(0.U(laneWidth.W))
  val tap = RegInit(0.U(tapWidth.W))
  val acc = RegInit(0.S(accWidth.W))
  val busyReg = RegInit(false.B)
  val doneReg = RegInit(false.B)

  val samplesForLane = Wire(Vec(taps, SInt(dataWidth.W)))
  samplesForLane(0) := xReg(lane)
  for (historyTap <- 1 until taps) {
    samplesForLane(historyTap) := history(historyTap - 1)(lane)
  }

  val mac = Module(new MacLane(dataWidth, accWidth))
  mac.io.a := samplesForLane(tap)
  mac.io.b := kernelReg(tap)(lane)
  mac.io.accIn := acc

  when(io.clear) {
    xReg := zeroInput
    kernelReg := zeroKernel
    history := zeroHistory
    yReg := zeroOutput
    lane := 0.U
    tap := 0.U
    acc := 0.S
    busyReg := false.B
    doneReg := false.B
  }.elsewhen(io.start && !busyReg) {
    xReg := io.x
    kernelReg := io.kernel
    yReg := zeroOutput
    lane := 0.U
    tap := 0.U
    acc := 0.S
    busyReg := true.B
    doneReg := false.B
  }.elsewhen(busyReg) {
    doneReg := false.B
    val nextAcc = mac.io.accOut

    when(tap === (taps - 1).U) {
      yReg(lane) := nextAcc
      tap := 0.U
      acc := 0.S

      when(lane === (lanes - 1).U) {
        if (taps > 2) {
          for (historyTap <- (taps - 2) to 1 by -1) {
            history(historyTap) := history(historyTap - 1)
          }
        }
        history(0) := xReg
        lane := 0.U
        busyReg := false.B
        doneReg := true.B
      }.otherwise {
        lane := lane + 1.U
      }
    }.otherwise {
      tap := tap + 1.U
      acc := nextAcc
    }
  }.otherwise {
    doneReg := false.B
  }

  io.ready := !busyReg
  io.busy := busyReg
  io.done := doneReg
  io.laneIndex := lane
  io.tapIndex := tap
  io.y := yReg
}
