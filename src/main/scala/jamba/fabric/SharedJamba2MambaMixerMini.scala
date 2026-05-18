package jamba.fabric

import chisel3._
import jamba.mamba.{CausalConvMini, SelectiveScanMini}

/** Jamba2 mini Mamba mixer with input/B/C projections mapped to shared linear fabric. */
class SharedJamba2MambaMixerMini(lanes: Int = 4, taps: Int = 4, dataWidth: Int = 8, stateWidth: Int = 32, accWidth: Int = 32)
    extends Module {
  require(lanes == 4, "SharedJamba2MambaMixerMini currently uses SharedLinear4 and requires lanes == 4")
  require(taps > 0, "SharedJamba2MambaMixerMini taps must be positive")

  val io = IO(new Bundle {
    val en    = Input(Bool())
    val clear = Input(Bool())
    val x     = Input(Vec(lanes, SInt(dataWidth.W)))

    val inputWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val inputBias   = Input(Vec(lanes, SInt(accWidth.W)))
    val bWeight     = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val bBias       = Input(Vec(lanes, SInt(accWidth.W)))
    val cWeight     = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val cBias       = Input(Vec(lanes, SInt(accWidth.W)))
    val a           = Input(Vec(lanes, SInt(dataWidth.W)))
    val kernel      = Input(Vec(taps, Vec(lanes, SInt(dataWidth.W))))

    val projected = Output(Vec(lanes, SInt(dataWidth.W)))
    val conv      = Output(Vec(lanes, SInt(accWidth.W)))
    val b         = Output(Vec(lanes, SInt(dataWidth.W)))
    val c         = Output(Vec(lanes, SInt(dataWidth.W)))
    val stateOut  = Output(Vec(lanes, SInt(stateWidth.W)))
    val y         = Output(Vec(lanes, SInt(accWidth.W)))
  })

  private def narrowToData(value: SInt): SInt = value(dataWidth - 1, 0).asSInt

  val inputProjection = Module(new SharedLinear4(dataWidth, accWidth))
  inputProjection.io.x := io.x
  inputProjection.io.weight := io.inputWeight
  inputProjection.io.bias := io.inputBias

  val bProjection = Module(new SharedLinear4(dataWidth, accWidth))
  bProjection.io.x := io.x
  bProjection.io.weight := io.bWeight
  bProjection.io.bias := io.bBias

  val cProjection = Module(new SharedLinear4(dataWidth, accWidth))
  cProjection.io.x := io.x
  cProjection.io.weight := io.cWeight
  cProjection.io.bias := io.cBias

  val conv = Module(new CausalConvMini(lanes, taps, dataWidth, accWidth))
  conv.io.en := io.en
  conv.io.clear := io.clear
  conv.io.kernel := io.kernel

  val scan = Module(new SelectiveScanMini(lanes, dataWidth, stateWidth, accWidth))
  scan.io.en := io.en
  scan.io.clear := io.clear
  scan.io.a := io.a

  for (lane <- 0 until lanes) {
    io.projected(lane) := narrowToData(inputProjection.io.y(lane))
    io.b(lane) := narrowToData(bProjection.io.y(lane))
    io.c(lane) := narrowToData(cProjection.io.y(lane))
    conv.io.x(lane) := io.projected(lane)
    scan.io.x(lane) := narrowToData(conv.io.y(lane))
    scan.io.b(lane) := io.b(lane)
    scan.io.c(lane) := io.c(lane)
    io.conv(lane) := conv.io.y(lane)
    io.stateOut(lane) := scan.io.stateOut(lane)
    io.y(lane) := scan.io.y(lane)
  }
}
