package jamba.fabric

import chisel3._
import jamba.common.SignedMath

/** Shared signed multiply-accumulate lane: accOut = a * b + accIn. */
class MacLane(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(dataWidth > 0, "MacLane dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "MacLane accWidth should hold the full product")

  val io = IO(new Bundle {
    val a      = Input(SInt(dataWidth.W))
    val b      = Input(SInt(dataWidth.W))
    val accIn  = Input(SInt(accWidth.W))
    val accOut = Output(SInt(accWidth.W))
  })

  val product = SignedMath.resize(io.a * io.b, accWidth)
  io.accOut := SignedMath.resize(product +& io.accIn, accWidth)
}
