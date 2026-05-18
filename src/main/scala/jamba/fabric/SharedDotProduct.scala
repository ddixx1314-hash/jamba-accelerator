package jamba.fabric

import chisel3._

/** Dot product built from reusable MAC lanes. */
class SharedDotProduct(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "SharedDotProduct length must be positive")
  require(dataWidth > 0, "SharedDotProduct dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "SharedDotProduct accWidth should hold accumulated products")

  val io = IO(new Bundle {
    val a = Input(Vec(length, SInt(dataWidth.W)))
    val b = Input(Vec(length, SInt(dataWidth.W)))
    val y = Output(SInt(accWidth.W))
  })

  val lanes = Seq.fill(length)(Module(new MacLane(dataWidth, accWidth)))

  for (i <- 0 until length) {
    lanes(i).io.a := io.a(i)
    lanes(i).io.b := io.b(i)
    lanes(i).io.accIn := {
      if (i == 0) 0.S(accWidth.W) else lanes(i - 1).io.accOut
    }
  }

  io.y := lanes.last.io.accOut
}
