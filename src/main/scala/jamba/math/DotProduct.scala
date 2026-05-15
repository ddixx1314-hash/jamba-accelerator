package jamba.math

import chisel3._
import jamba.common.SignedMath

/** Combinational signed dot product: y = sum(a(i) * b(i)). */
class DotProduct(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  val io = IO(new Bundle {
    val a = Input(Vec(length, SInt(dataWidth.W)))
    val b = Input(Vec(length, SInt(dataWidth.W)))
    val y = Output(SInt(accWidth.W))
  })

  val products = Wire(Vec(length, SInt(accWidth.W)))

  for (i <- 0 until length) {
    products(i) := SignedMath.resize(io.a(i) * io.b(i), accWidth)
  }

  io.y := SignedMath.resize(products.reduce(_ +& _), accWidth)
}
