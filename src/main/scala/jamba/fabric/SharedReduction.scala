package jamba.fabric

import chisel3._
import jamba.common.SignedMath

/** Combinational signed reduction for shared arithmetic fabrics. */
class SharedReduction(length: Int = 4, accWidth: Int = 32) extends Module {
  require(length > 0, "SharedReduction length must be positive")
  require(accWidth > 0, "SharedReduction accWidth must be positive")

  val io = IO(new Bundle {
    val in = Input(Vec(length, SInt(accWidth.W)))
    val y  = Output(SInt(accWidth.W))
  })

  io.y := SignedMath.resize(io.in.reduce(_ +& _), accWidth)
}
