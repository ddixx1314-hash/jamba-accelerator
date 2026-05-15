package jamba.top

import chisel3._
import jamba.common.JambaMiniConfig
import jamba.stream.Jamba2MiniStream

/** Formal engineering top for the first-stage mini accelerator tile. */
class JambaMiniTile(config: JambaMiniConfig = JambaMiniConfig()) extends Module {
  require(config.lanes == 4, "JambaMiniTile currently supports 4 lanes")
  require(config.convTaps == 3, "JambaMiniTile currently supports 3 convolution taps")

  private val dataWidth = config.dataWidth
  private val accWidth = config.accWidth

  val io = IO(new Bundle {
    val clear           = Input(Bool())
    val useAttention    = Input(Bool())
    val inValid         = Input(Bool())
    val inReady         = Output(Bool())
    val in              = Input(Vec(4, SInt(dataWidth.W)))
    val outValid        = Output(Bool())
    val outReady        = Input(Bool())
    val out             = Output(Vec(4, SInt(accWidth.W)))
    val rmsWeight       = Input(Vec(4, SInt(dataWidth.W)))
    val inputWeight     = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val inputBias       = Input(Vec(4, SInt(accWidth.W)))
    val gateWeight      = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val gateBias        = Input(Vec(4, SInt(accWidth.W)))
    val bWeight         = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val bBias           = Input(Vec(4, SInt(accWidth.W)))
    val cWeight         = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val cBias           = Input(Vec(4, SInt(accWidth.W)))
    val outWeight       = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val outBias         = Input(Vec(4, SInt(accWidth.W)))
    val kernel          = Input(Vec(3, Vec(4, SInt(dataWidth.W))))
    val mambaA          = Input(Vec(4, SInt(dataWidth.W)))
    val attentionKeys   = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val attentionValues = Input(Vec(4, Vec(4, SInt(dataWidth.W))))
    val stateOut        = Output(Vec(4, SInt(accWidth.W)))
    val attentionScores = Output(Vec(4, SInt(accWidth.W)))
  })

  val stream = Module(new Jamba2MiniStream(dataWidth, accWidth))
  stream.io.clear := io.clear
  stream.io.useAttention := io.useAttention
  stream.io.inValid := io.inValid
  stream.io.in := io.in
  stream.io.outReady := io.outReady
  stream.io.rmsWeight := io.rmsWeight
  stream.io.inputWeight := io.inputWeight
  stream.io.inputBias := io.inputBias
  stream.io.gateWeight := io.gateWeight
  stream.io.gateBias := io.gateBias
  stream.io.bWeight := io.bWeight
  stream.io.bBias := io.bBias
  stream.io.cWeight := io.cWeight
  stream.io.cBias := io.cBias
  stream.io.outWeight := io.outWeight
  stream.io.outBias := io.outBias
  stream.io.kernel := io.kernel
  stream.io.mambaA := io.mambaA
  stream.io.attentionKeys := io.attentionKeys
  stream.io.attentionValues := io.attentionValues

  io.inReady := stream.io.inReady
  io.outValid := stream.io.outValid
  io.out := stream.io.out
  io.stateOut := stream.io.stateOut
  io.attentionScores := stream.io.attentionScores
}
