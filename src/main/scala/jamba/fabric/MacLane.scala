package jamba.fabric

import chisel3._
import jamba.common.SignedMath

/** Shared signed multiply-accumulate lane: accOut = a * b + accIn.
  *
  * When zeroSkip is true, the multiply is bypassed (accIn passed through)
  * if either input is zero. This is an elaboration-time structural option
  * for weight-sparsity-aware resource analysis.
  */
class MacLane(dataWidth: Int = 8, accWidth: Int = 32, zeroSkip: Boolean = false) extends Module {
  require(dataWidth > 0, "MacLane dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "MacLane accWidth should hold the full product")

  val io = IO(new Bundle {
    val a      = Input(SInt(dataWidth.W))
    val b      = Input(SInt(dataWidth.W))
    val accIn  = Input(SInt(accWidth.W))
    val accOut = Output(SInt(accWidth.W))
  })

  if (zeroSkip) {
    val skipMul = io.a === 0.S || io.b === 0.S
    val product = Mux(skipMul, 0.S(accWidth.W), SignedMath.resize(io.a * io.b, accWidth))
    io.accOut := SignedMath.resize(product +& io.accIn, accWidth)
  } else {
    val product = SignedMath.resize(io.a * io.b, accWidth)
    io.accOut := SignedMath.resize(product +& io.accIn, accWidth)
  }
}
