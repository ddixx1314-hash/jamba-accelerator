package jamba.stream

import chisel3._
import jamba.core.Jamba2MiniCore

/** Streaming wrapper around Jamba2MiniCore with a one-entry output buffer. */
class Jamba2MiniStream(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(dataWidth > 0, "Jamba2MiniStream dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "Jamba2MiniStream accWidth should hold products")

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

  val outputValid = RegInit(false.B)
  val outputReg = RegInit(VecInit(Seq.fill(4)(0.S(accWidth.W))))
  val willConsume = outputValid && io.outReady
  val canAccept = !outputValid || willConsume
  val fire = io.inValid && canAccept

  val core = Module(new Jamba2MiniCore(dataWidth, accWidth))
  core.io.en := fire
  core.io.clear := io.clear
  core.io.useAttention := io.useAttention
  core.io.x := io.in
  core.io.rmsWeight := io.rmsWeight
  core.io.inputWeight := io.inputWeight
  core.io.inputBias := io.inputBias
  core.io.gateWeight := io.gateWeight
  core.io.gateBias := io.gateBias
  core.io.bWeight := io.bWeight
  core.io.bBias := io.bBias
  core.io.cWeight := io.cWeight
  core.io.cBias := io.cBias
  core.io.outWeight := io.outWeight
  core.io.outBias := io.outBias
  core.io.kernel := io.kernel
  core.io.mambaA := io.mambaA
  core.io.attentionKeys := io.attentionKeys
  core.io.attentionValues := io.attentionValues

  when(io.clear) {
    outputValid := false.B
    outputReg := VecInit(Seq.fill(4)(0.S(accWidth.W)))
  }.elsewhen(fire) {
    outputValid := true.B
    outputReg := core.io.y
  }.elsewhen(willConsume) {
    outputValid := false.B
  }

  io.inReady := canAccept && !io.clear
  io.outValid := outputValid
  io.out := outputReg
  io.stateOut := core.io.stateOut
  io.attentionScores := core.io.attentionScores
}
