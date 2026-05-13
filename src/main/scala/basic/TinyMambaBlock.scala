package basic

import chisel3._

/** Minimal integer Mamba-like block: causal conv, selective scan, and residual gate. */
class TinyMambaBlock(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "TinyMambaBlock length must be positive")
  require(dataWidth > 0, "TinyMambaBlock dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "TinyMambaBlock accWidth should hold products")

  val io = IO(new Bundle {
    val en       = Input(Bool())
    val clear    = Input(Bool())
    val x        = Input(Vec(length, SInt(dataWidth.W)))
    val kernel   = Input(Vec(3, Vec(length, SInt(dataWidth.W))))
    val a        = Input(Vec(length, SInt(dataWidth.W)))
    val b        = Input(Vec(length, SInt(dataWidth.W)))
    val c        = Input(Vec(length, SInt(dataWidth.W)))
    val gate     = Input(Vec(length, SInt(dataWidth.W)))
    val stateOut = Output(Vec(length, SInt(accWidth.W)))
    val y        = Output(Vec(length, SInt(accWidth.W)))
  })

  val conv = Module(new CausalConv1D(length, dataWidth, accWidth))
  conv.io.en := io.en
  conv.io.clear := io.clear
  conv.io.x := io.x
  conv.io.kernel := io.kernel

  val scanInput = Wire(Vec(length, SInt(dataWidth.W)))
  for (i <- 0 until length) {
    scanInput(i) := SignedMath.resize(conv.io.y(i), dataWidth)
  }

  val scan = Module(new SelectiveScanTiny(length, dataWidth, accWidth))
  scan.io.en := io.en
  scan.io.clear := io.clear
  scan.io.x := scanInput
  scan.io.a := io.a
  scan.io.b := io.b
  scan.io.gate := io.gate

  for (i <- 0 until length) {
    val projectedState = scan.io.stateOut(i) * io.c(i)
    val residualGate = io.x(i) * io.gate(i)
    io.stateOut(i) := scan.io.stateOut(i)
    io.y(i) := SignedMath.resize(projectedState +& residualGate, accWidth)
  }
}
