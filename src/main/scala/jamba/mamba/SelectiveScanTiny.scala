package jamba.mamba

import chisel3._
import jamba.common.SignedMath

/** Small selective-scan block: updates SSM state and applies an element-wise gate. */
class SelectiveScanTiny(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "SelectiveScanTiny length must be positive")
  require(dataWidth > 0, "SelectiveScanTiny dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "SelectiveScanTiny accWidth should hold products")

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

  val update = Module(new MambaStateUpdate(length, dataWidth, accWidth))
  update.io.en := io.en
  update.io.clear := io.clear
  update.io.x := io.x
  update.io.a := io.a
  update.io.b := io.b

  for (i <- 0 until length) {
    io.stateOut(i) := update.io.stateOut(i)
    io.y(i) := SignedMath.resize(update.io.stateOut(i) * io.gate(i), accWidth)
  }
}
