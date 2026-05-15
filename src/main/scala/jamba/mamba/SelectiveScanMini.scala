package jamba.mamba

import chisel3._
import jamba.common.SignedMath

/** Token-serial SSM scan: state := a * state + b * x, y := state * c. */
class SelectiveScanMini(lanes: Int = 4, dataWidth: Int = 8, stateWidth: Int = 32, accWidth: Int = 32) extends Module {
  require(lanes > 0, "SelectiveScanMini lanes must be positive")
  require(dataWidth > 0, "SelectiveScanMini dataWidth must be positive")
  require(stateWidth >= 2 * dataWidth, "SelectiveScanMini stateWidth should hold products")
  require(accWidth >= 2 * dataWidth, "SelectiveScanMini accWidth should hold products")

  val io = IO(new Bundle {
    val en = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val a = Input(Vec(lanes, SInt(dataWidth.W)))
    val b = Input(Vec(lanes, SInt(dataWidth.W)))
    val c = Input(Vec(lanes, SInt(dataWidth.W)))
    val stateOut = Output(Vec(lanes, SInt(stateWidth.W)))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
  })

  val state = RegInit(VecInit(Seq.fill(lanes)(0.S(stateWidth.W))))
  val nextState = Wire(Vec(lanes, SInt(stateWidth.W)))

  for (lane <- 0 until lanes) {
    val recurrent = state(lane) * io.a(lane)
    val inputTerm = io.x(lane) * io.b(lane)
    nextState(lane) := SignedMath.resize(recurrent +& inputTerm, stateWidth)
    io.stateOut(lane) := state(lane)
    io.y(lane) := SignedMath.resize(state(lane) * io.c(lane), accWidth)
  }

  when(io.clear) {
    state := VecInit(Seq.fill(lanes)(0.S(stateWidth.W)))
  }.elsewhen(io.en) {
    state := nextState
  }
}
