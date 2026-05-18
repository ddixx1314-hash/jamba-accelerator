package jamba.fabric

import chisel3._

/** Top-1 MoE router mapped onto shared dot-product lanes. */
class SharedRouterMini(lanes: Int = 4, numExperts: Int = 2, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(lanes == 4, "SharedRouterMini currently uses SharedDotProduct and requires lanes == 4")
  require(numExperts == 2, "SharedRouterMini first implementation supports exactly 2 experts")

  val io = IO(new Bundle {
    val x              = Input(Vec(lanes, SInt(dataWidth.W)))
    val weight         = Input(Vec(numExperts, Vec(lanes, SInt(dataWidth.W))))
    val bias           = Input(Vec(numExperts, SInt(accWidth.W)))
    val scores         = Output(Vec(numExperts, SInt(accWidth.W)))
    val selectedExpert = Output(UInt(1.W))
  })

  val scoreDots = Seq.fill(numExperts)(Module(new SharedDotProduct(lanes, dataWidth, accWidth)))
  for (expert <- 0 until numExperts) {
    scoreDots(expert).io.a := io.x
    scoreDots(expert).io.b := io.weight(expert)
    io.scores(expert) := scoreDots(expert).io.y + io.bias(expert)
  }

  io.selectedExpert := Mux(io.scores(1) > io.scores(0), 1.U, 0.U)
}
