package jamba.fabric

import chisel3._

/** Small selective-scan block using shared state-update and mixed-width gate MACs. */
class SharedSelectiveScanTiny(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "SharedSelectiveScanTiny length must be positive")
  require(dataWidth > 0, "SharedSelectiveScanTiny dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "SharedSelectiveScanTiny accWidth should hold products")

  val io = IO(new Bundle {
    val en       = Input(Bool())
    val clear    = Input(Bool())
    val x        = Input(Vec(length, SInt(dataWidth.W)))
    val a        = Input(Vec(length, SInt(dataWidth.W)))
    val b        = Input(Vec(length, SInt(dataWidth.W)))
    val gate     = Input(Vec(length, SInt(dataWidth.W)))
    val stateOut = Output(Vec(length, SInt(accWidth.W)))
    val y        = Output(Vec(length, SInt(accWidth.W)))
  })

  val update = Module(new SharedMambaStateUpdate(length, dataWidth, accWidth))
  update.io.en := io.en
  update.io.clear := io.clear
  update.io.x := io.x
  update.io.a := io.a
  update.io.b := io.b

  val gateMacs = Seq.fill(length)(Module(new MacLaneMixed(accWidth, dataWidth, accWidth)))
  for (lane <- 0 until length) {
    io.stateOut(lane) := update.io.stateOut(lane)
    gateMacs(lane).io.a := update.io.stateOut(lane)
    gateMacs(lane).io.b := io.gate(lane)
    gateMacs(lane).io.accIn := 0.S(accWidth.W)
    io.y(lane) := gateMacs(lane).io.accOut
  }
}
