package jamba.math

import chisel3._

/** Combinational element-wise vector arithmetic used by later tiny blocks. */
class VectorOps(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "VectorOps length must be positive")
  require(dataWidth > 0, "VectorOps dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "VectorOps accWidth should hold products")

  val io = IO(new Bundle {
    val a    = Input(Vec(length, SInt(dataWidth.W)))
    val b    = Input(Vec(length, SInt(dataWidth.W)))
    val add  = Output(Vec(length, SInt(accWidth.W)))
    val sub  = Output(Vec(length, SInt(accWidth.W)))
    val mul  = Output(Vec(length, SInt(accWidth.W)))
    val relu = Output(Vec(length, SInt(dataWidth.W)))
  })

  for (i <- 0 until length) {
    io.add(i) := io.a(i) +& io.b(i)
    io.sub(i) := io.a(i) -& io.b(i)
    io.mul(i) := io.a(i) * io.b(i)
    io.relu(i) := Mux(io.a(i) < 0.S, 0.S, io.a(i))
  }
}
