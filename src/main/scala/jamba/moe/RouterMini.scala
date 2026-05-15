package jamba.moe

import chisel3._
import jamba.common.SignedMath

/** Token-serial top-1 router for MoE-lite. */
class RouterMini(lanes: Int = 4, numExperts: Int = 2, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(lanes > 0, "RouterMini lanes must be positive")
  require(numExperts == 2, "RouterMini first implementation supports exactly 2 experts")

  val io = IO(new Bundle {
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val weight = Input(Vec(numExperts, Vec(lanes, SInt(dataWidth.W))))
    val bias = Input(Vec(numExperts, SInt(accWidth.W)))
    val scores = Output(Vec(numExperts, SInt(accWidth.W)))
    val selectedExpert = Output(UInt(1.W))
  })

  for (expert <- 0 until numExperts) {
    val products = Wire(Vec(lanes, SInt(accWidth.W)))
    for (lane <- 0 until lanes) {
      products(lane) := SignedMath.resize(io.x(lane) * io.weight(expert)(lane), accWidth)
    }
    io.scores(expert) := products.reduce(_ +& _) + io.bias(expert)
  }

  io.selectedExpert := Mux(io.scores(1) > io.scores(0), 1.U, 0.U)
}
