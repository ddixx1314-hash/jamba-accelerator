package jamba.fabric

import chisel3._
import jamba.common.SignedMath

/** Tiny Jamba-like block built from shared Mamba and attention paths. */
class SharedTinyJambaBlock(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "SharedTinyJambaBlock length must be positive")
  require(dataWidth > 0, "SharedTinyJambaBlock dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "SharedTinyJambaBlock accWidth should hold products")

  val io = IO(new Bundle {
    val en              = Input(Bool())
    val clear           = Input(Bool())
    val useAttention    = Input(Bool())
    val x               = Input(Vec(length, SInt(dataWidth.W)))
    val kernel          = Input(Vec(3, Vec(length, SInt(dataWidth.W))))
    val mambaA          = Input(Vec(length, SInt(dataWidth.W)))
    val mambaB          = Input(Vec(length, SInt(dataWidth.W)))
    val mambaC          = Input(Vec(length, SInt(dataWidth.W)))
    val gate            = Input(Vec(length, SInt(dataWidth.W)))
    val attentionKeys   = Input(Vec(length, Vec(length, SInt(dataWidth.W))))
    val attentionValues = Input(Vec(length, Vec(length, SInt(dataWidth.W))))
    val stateOut        = Output(Vec(length, SInt(accWidth.W)))
    val attentionScores = Output(Vec(length, SInt(accWidth.W)))
    val y               = Output(Vec(length, SInt(accWidth.W)))
  })

  val mamba = Module(new SharedTinyMambaBlock(length, dataWidth, accWidth))
  mamba.io.en := io.en
  mamba.io.clear := io.clear
  mamba.io.x := io.x
  mamba.io.kernel := io.kernel
  mamba.io.a := io.mambaA
  mamba.io.b := io.mambaB
  mamba.io.c := io.mambaC
  mamba.io.gate := io.gate

  val attention = Module(new SharedAttentionDecodeTiny(length, length, dataWidth, accWidth))
  attention.io.q := io.x
  attention.io.keys := io.attentionKeys
  attention.io.values := io.attentionValues

  for (lane <- 0 until length) {
    io.stateOut(lane) := mamba.io.stateOut(lane)
    io.attentionScores(lane) := attention.io.scores(lane)
    io.y(lane) := Mux(io.useAttention, SignedMath.resize(mamba.io.y(lane) +& attention.io.y(lane), accWidth), mamba.io.y(lane))
  }
}
