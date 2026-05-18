package jamba.fabric

import chisel3._
import jamba.common.SignedMath

/** Minimal Mamba-like block composed from shared convolution, scan, and MAC resources. */
class SharedTinyMambaBlock(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "SharedTinyMambaBlock length must be positive")
  require(dataWidth > 0, "SharedTinyMambaBlock dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "SharedTinyMambaBlock accWidth should hold products")

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

  val conv = Module(new SharedCausalConv1D(length, dataWidth, accWidth))
  conv.io.en := io.en
  conv.io.clear := io.clear
  conv.io.x := io.x
  conv.io.kernel := io.kernel

  val scanInput = Wire(Vec(length, SInt(dataWidth.W)))
  for (lane <- 0 until length) {
    scanInput(lane) := SignedMath.resize(conv.io.y(lane), dataWidth)
  }

  val scan = Module(new SharedSelectiveScanTiny(length, dataWidth, accWidth))
  scan.io.en := io.en
  scan.io.clear := io.clear
  scan.io.x := scanInput
  scan.io.a := io.a
  scan.io.b := io.b
  scan.io.gate := io.gate

  val stateProjectionMacs = Seq.fill(length)(Module(new MacLaneMixed(accWidth, dataWidth, accWidth)))
  val residualGateMacs = Seq.fill(length)(Module(new MacLane(dataWidth, accWidth)))

  for (lane <- 0 until length) {
    io.stateOut(lane) := scan.io.stateOut(lane)

    stateProjectionMacs(lane).io.a := scan.io.stateOut(lane)
    stateProjectionMacs(lane).io.b := io.c(lane)
    stateProjectionMacs(lane).io.accIn := 0.S(accWidth.W)

    residualGateMacs(lane).io.a := io.x(lane)
    residualGateMacs(lane).io.b := io.gate(lane)
    residualGateMacs(lane).io.accIn := stateProjectionMacs(lane).io.accOut

    io.y(lane) := residualGateMacs(lane).io.accOut
  }
}
