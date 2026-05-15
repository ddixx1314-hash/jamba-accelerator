package jamba.norm

import chisel3._
import jamba.common.SignedMath

/** Integer-friendly RMSNorm approximation using mean-square division. */
class RmsNormApprox(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "RmsNormApprox length must be positive")
  require(dataWidth > 0, "RmsNormApprox dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "RmsNormApprox accWidth should hold products")

  val io = IO(new Bundle {
    val x          = Input(Vec(length, SInt(dataWidth.W)))
    val weight     = Input(Vec(length, SInt(dataWidth.W)))
    val sumSquares = Output(SInt(accWidth.W))
    val meanSquare = Output(SInt(accWidth.W))
    val y          = Output(Vec(length, SInt(dataWidth.W)))
  })

  val stats = Module(new RmsNormStats(length, dataWidth, accWidth))
  stats.io.x := io.x

  val denominator = Mux(stats.io.meanSquare === 0.S, 1.S(accWidth.W), stats.io.meanSquare)

  for (i <- 0 until length) {
    val scaled = (io.x(i) * io.weight(i)) / denominator
    io.y(i) := SignedMath.resize(scaled, dataWidth)
  }

  io.sumSquares := stats.io.sumSquares
  io.meanSquare := stats.io.meanSquare
}
