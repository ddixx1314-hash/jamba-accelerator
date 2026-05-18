package jamba.fabric

import chisel3._
import jamba.common.SignedMath

/** Four-lane linear projection built from shared-style dot-product blocks. */
class SharedLinear4(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(dataWidth > 0, "SharedLinear4 dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "SharedLinear4 accWidth should hold four products")

  val io = IO(new Bundle {
    val x      = Input(Vec(4, SInt(dataWidth.W)))
    val weight = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val bias   = Input(Vec(4, SInt(accWidth.W)))
    val y      = Output(Vec(4, SInt(accWidth.W)))
  })

  val dots = Seq.fill(4)(Module(new SharedDotProduct(4, dataWidth, accWidth)))

  for (row <- 0 until 4) {
    dots(row).io.a := io.x
    dots(row).io.b := io.weight(row)
    io.y(row) := SignedMath.resize(dots(row).io.y +& io.bias(row), accWidth)
  }
}
