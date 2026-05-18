package jamba.fabric

import chisel3._

/** Tiny attention decode using shared-fabric dot products and MAC chains. */
class SharedAttentionDecodeTiny(seqLen: Int = 4, dim: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(seqLen > 0, "SharedAttentionDecodeTiny seqLen must be positive")
  require(dim > 0, "SharedAttentionDecodeTiny dim must be positive")
  require(dataWidth > 0, "SharedAttentionDecodeTiny dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "SharedAttentionDecodeTiny accWidth should hold dot products")

  val io = IO(new Bundle {
    val q      = Input(Vec(dim, SInt(dataWidth.W)))
    val keys   = Input(Vec(seqLen, Vec(dim, SInt(dataWidth.W))))
    val values = Input(Vec(seqLen, Vec(dim, SInt(dataWidth.W))))
    val scores = Output(Vec(seqLen, SInt(accWidth.W)))
    val y      = Output(Vec(dim, SInt(accWidth.W)))
  })

  val scoreDots = Seq.fill(seqLen)(Module(new SharedDotProduct(dim, dataWidth, accWidth)))
  for (row <- 0 until seqLen) {
    scoreDots(row).io.a := io.q
    scoreDots(row).io.b := io.keys(row)
    io.scores(row) := scoreDots(row).io.y
  }

  val valueMacs = Seq.fill(dim, seqLen)(Module(new MacLaneMixed(accWidth, dataWidth, accWidth)))
  for (col <- 0 until dim) {
    for (row <- 0 until seqLen) {
      valueMacs(col)(row).io.a := io.scores(row)
      valueMacs(col)(row).io.b := io.values(row)(col)
      valueMacs(col)(row).io.accIn := {
        if (row == 0) 0.S(accWidth.W) else valueMacs(col)(row - 1).io.accOut
      }
    }
    io.y(col) := valueMacs(col).last.io.accOut
  }
}
