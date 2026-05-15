package jamba.mamba

import chisel3._
import jamba.common.SignedMath

/** Tiny integer SSM state update: state := a * state + b * x. */
class MambaStateUpdate(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "MambaStateUpdate length must be positive")
  require(dataWidth > 0, "MambaStateUpdate dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "MambaStateUpdate accWidth should hold products")

  val io = IO(new Bundle {
    val en       = Input(Bool())
    val clear    = Input(Bool())
    val x        = Input(Vec(length, SInt(dataWidth.W)))
    val a        = Input(Vec(length, SInt(dataWidth.W)))
    val b        = Input(Vec(length, SInt(dataWidth.W)))
    val stateOut = Output(Vec(length, SInt(accWidth.W)))
  })

  val state = RegInit(VecInit(Seq.fill(length)(0.S(accWidth.W))))
  val nextState = Wire(Vec(length, SInt(accWidth.W)))

  for (i <- 0 until length) {
    val recurrent = state(i) * io.a(i)
    val inputTerm = io.x(i) * io.b(i)
    nextState(i) := SignedMath.resize(recurrent +& inputTerm, accWidth)
  }

  when(io.clear) {
    state := VecInit(Seq.fill(length)(0.S(accWidth.W)))
  }.elsewhen(io.en) {
    state := nextState
  }

  io.stateOut := state
}
