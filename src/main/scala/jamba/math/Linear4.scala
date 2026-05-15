package jamba.math

import chisel3._

/** Four-lane signed linear projection: y(row) = bias(row) + sum(weight(row)(col) * x(col)). */
class Linear4(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(dataWidth > 0, "Linear4 dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "Linear4 accWidth should hold four products")

  val io = IO(new Bundle {
    val x      = Input(Vec(4, SInt(dataWidth.W)))
    val weight = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val bias   = Input(Vec(4, SInt(accWidth.W)))
    val y      = Output(Vec(4, SInt(accWidth.W)))
  })

  for (row <- 0 until 4) {
    val products = Wire(Vec(4, SInt(accWidth.W)))

    for (col <- 0 until 4) {
      products(col) := io.x(col) * io.weight(row)(col)
    }

    io.y(row) := products.reduce(_ +& _) + io.bias(row)
  }
}
