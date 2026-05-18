package jamba.fabric

import chisel3._

/** Three-tap per-channel causal convolution using reusable MAC lanes. */
class SharedCausalConv1D(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "SharedCausalConv1D length must be positive")
  require(dataWidth > 0, "SharedCausalConv1D dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "SharedCausalConv1D accWidth should hold three products")

  val io = IO(new Bundle {
    val en     = Input(Bool())
    val clear  = Input(Bool())
    val x      = Input(Vec(length, SInt(dataWidth.W)))
    val kernel = Input(Vec(3, Vec(length, SInt(dataWidth.W))))
    val y      = Output(Vec(length, SInt(accWidth.W)))
  })

  val delay1 = RegInit(VecInit(Seq.fill(length)(0.S(dataWidth.W))))
  val delay2 = RegInit(VecInit(Seq.fill(length)(0.S(dataWidth.W))))
  val macs = Seq.fill(length, 3)(Module(new MacLane(dataWidth, accWidth)))

  for (lane <- 0 until length) {
    macs(lane)(0).io.a := io.x(lane)
    macs(lane)(0).io.b := io.kernel(0)(lane)
    macs(lane)(0).io.accIn := 0.S(accWidth.W)

    macs(lane)(1).io.a := delay1(lane)
    macs(lane)(1).io.b := io.kernel(1)(lane)
    macs(lane)(1).io.accIn := macs(lane)(0).io.accOut

    macs(lane)(2).io.a := delay2(lane)
    macs(lane)(2).io.b := io.kernel(2)(lane)
    macs(lane)(2).io.accIn := macs(lane)(1).io.accOut

    io.y(lane) := macs(lane)(2).io.accOut
  }

  when(io.clear) {
    delay1 := VecInit(Seq.fill(length)(0.S(dataWidth.W)))
    delay2 := VecInit(Seq.fill(length)(0.S(dataWidth.W)))
  }.elsewhen(io.en) {
    delay2 := delay1
    delay1 := io.x
  }
}
