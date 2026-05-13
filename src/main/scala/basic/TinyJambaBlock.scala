package basic

import chisel3._

/** Tiny Jamba-like block that can mix a Mamba-style path with a tiny attention path. */
class TinyJambaBlock(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "TinyJambaBlock length must be positive")
  require(dataWidth > 0, "TinyJambaBlock dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "TinyJambaBlock accWidth should hold products")

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

  val mamba = Module(new TinyMambaBlock(length, dataWidth, accWidth))
  mamba.io.en := io.en
  mamba.io.clear := io.clear
  mamba.io.x := io.x
  mamba.io.kernel := io.kernel
  mamba.io.a := io.mambaA
  mamba.io.b := io.mambaB
  mamba.io.c := io.mambaC
  mamba.io.gate := io.gate

  val attention = Module(new AttentionDecodeTiny(length, length, dataWidth, accWidth))
  attention.io.q := io.x
  attention.io.keys := io.attentionKeys
  attention.io.values := io.attentionValues

  for (i <- 0 until length) {
    io.stateOut(i) := mamba.io.stateOut(i)
    io.attentionScores(i) := attention.io.scores(i)
    io.y(i) := Mux(io.useAttention, mamba.io.y(i) + attention.io.y(i), mamba.io.y(i))
  }
}
