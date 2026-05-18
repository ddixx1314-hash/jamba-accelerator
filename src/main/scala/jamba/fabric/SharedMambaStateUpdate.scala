package jamba.fabric

import chisel3._

/** Tiny integer SSM state update using shared mixed-width MAC lanes. */
class SharedMambaStateUpdate(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "SharedMambaStateUpdate length must be positive")
  require(dataWidth > 0, "SharedMambaStateUpdate dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "SharedMambaStateUpdate accWidth should hold products")

  val io = IO(new Bundle {
    val en       = Input(Bool())
    val clear    = Input(Bool())
    val x        = Input(Vec(length, SInt(dataWidth.W)))
    val a        = Input(Vec(length, SInt(dataWidth.W)))
    val b        = Input(Vec(length, SInt(dataWidth.W)))
    val stateOut = Output(Vec(length, SInt(accWidth.W)))
  })

  val state = RegInit(VecInit(Seq.fill(length)(0.S(accWidth.W))))
  val recurrentMacs = Seq.fill(length)(Module(new MacLaneMixed(accWidth, dataWidth, accWidth)))
  val inputMacs = Seq.fill(length)(Module(new MacLane(dataWidth, accWidth)))

  for (lane <- 0 until length) {
    recurrentMacs(lane).io.a := state(lane)
    recurrentMacs(lane).io.b := io.a(lane)
    recurrentMacs(lane).io.accIn := 0.S(accWidth.W)

    inputMacs(lane).io.a := io.x(lane)
    inputMacs(lane).io.b := io.b(lane)
    inputMacs(lane).io.accIn := recurrentMacs(lane).io.accOut
  }

  when(io.clear) {
    state := VecInit(Seq.fill(length)(0.S(accWidth.W)))
  }.elsewhen(io.en) {
    for (lane <- 0 until length) {
      state(lane) := inputMacs(lane).io.accOut
    }
  }

  io.stateOut := state
}
