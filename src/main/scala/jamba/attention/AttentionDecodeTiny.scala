package jamba.attention

import chisel3._
import jamba.common.SignedMath

/** Tiny attention-decode datapath without softmax: scores = q dot k, y = scores * values. */
class AttentionDecodeTiny(seqLen: Int = 4, dim: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(seqLen > 0, "AttentionDecodeTiny seqLen must be positive")
  require(dim > 0, "AttentionDecodeTiny dim must be positive")
  require(dataWidth > 0, "AttentionDecodeTiny dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "AttentionDecodeTiny accWidth should hold dot products")

  val io = IO(new Bundle {
    val q      = Input(Vec(dim, SInt(dataWidth.W)))
    val keys   = Input(Vec(seqLen, Vec(dim, SInt(dataWidth.W))))
    val values = Input(Vec(seqLen, Vec(dim, SInt(dataWidth.W))))
    val scores = Output(Vec(seqLen, SInt(accWidth.W)))
    val y      = Output(Vec(dim, SInt(accWidth.W)))
  })

  for (row <- 0 until seqLen) {
    val products = Wire(Vec(dim, SInt(accWidth.W)))

    for (col <- 0 until dim) {
      products(col) := io.q(col) * io.keys(row)(col)
    }

    io.scores(row) := products.reduce(_ +& _)
  }

  for (col <- 0 until dim) {
    val weightedValues = Wire(Vec(seqLen, SInt(accWidth.W)))

    for (row <- 0 until seqLen) {
      weightedValues(row) := SignedMath.resize(io.scores(row) * io.values(row)(col), accWidth)
    }

    io.y(col) := weightedValues.reduce(_ +& _)
  }
}
