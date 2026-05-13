package basic

import chisel3._

/** Combinational 4x4 signed matrix multiply: c = a * b. */
class SmallGemm4x4(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(dataWidth > 0, "SmallGemm4x4 dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "SmallGemm4x4 accWidth should hold four products")

  val io = IO(new Bundle {
    val a = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val b = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val c = Output(Vec(4, Vec(4, SInt(accWidth.W))))
  })

  for (row <- 0 until 4) {
    for (col <- 0 until 4) {
      val products = Wire(Vec(4, SInt(accWidth.W)))

      for (k <- 0 until 4) {
        products(k) := io.a(row)(k) * io.b(k)(col)
      }

      io.c(row)(col) := products.reduce(_ +& _)
    }
  }
}
