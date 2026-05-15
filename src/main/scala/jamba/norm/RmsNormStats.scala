package jamba.norm

import chisel3._

/** Computes integer RMSNorm statistics: sum(x^2) and floor(sum(x^2) / length). */
class RmsNormStats(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "RmsNormStats length must be positive")
  require(dataWidth > 0, "RmsNormStats dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "RmsNormStats accWidth should hold squared values")

  val io = IO(new Bundle {
    val x          = Input(Vec(length, SInt(dataWidth.W)))
    val sumSquares = Output(SInt(accWidth.W))
    val meanSquare = Output(SInt(accWidth.W))
  })

  val squares = Wire(Vec(length, SInt(accWidth.W)))

  for (i <- 0 until length) {
    squares(i) := io.x(i) * io.x(i)
  }

  io.sumSquares := squares.reduce(_ +& _)
  io.meanSquare := io.sumSquares / length.S
}
