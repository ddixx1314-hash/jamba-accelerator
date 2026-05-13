package basic

import chisel3._

/** Three-tap per-channel causal convolution for streaming token vectors. */
class CausalConv1D(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "CausalConv1D length must be positive")
  require(dataWidth > 0, "CausalConv1D dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "CausalConv1D accWidth should hold three products")

  val io = IO(new Bundle {
    val en     = Input(Bool())
    val clear  = Input(Bool())
    val x      = Input(Vec(length, SInt(dataWidth.W)))
    val kernel = Input(Vec(3, Vec(length, SInt(dataWidth.W))))
    val y      = Output(Vec(length, SInt(accWidth.W)))
  })

  val delay1 = RegInit(VecInit(Seq.fill(length)(0.S(dataWidth.W))))
  val delay2 = RegInit(VecInit(Seq.fill(length)(0.S(dataWidth.W))))

  for (i <- 0 until length) {
    val currentTerm = io.x(i) * io.kernel(0)(i)
    val prevTerm = delay1(i) * io.kernel(1)(i)
    val prevPrevTerm = delay2(i) * io.kernel(2)(i)
    io.y(i) := SignedMath.resize(currentTerm +& prevTerm +& prevPrevTerm, accWidth)
  }

  when(io.clear) {
    delay1 := VecInit(Seq.fill(length)(0.S(dataWidth.W)))
    delay2 := VecInit(Seq.fill(length)(0.S(dataWidth.W)))
  }.elsewhen(io.en) {
    delay2 := delay1
    delay1 := io.x
  }
}
