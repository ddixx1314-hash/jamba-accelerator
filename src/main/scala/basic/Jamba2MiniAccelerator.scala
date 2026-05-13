package basic

import chisel3._

/** Top-level fixed-point prototype for a tiny Jamba 2.0-style accelerator block. */
class Jamba2MiniAccelerator(length: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(length > 0, "Jamba2MiniAccelerator length must be positive")
  require(dataWidth > 0, "Jamba2MiniAccelerator dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "Jamba2MiniAccelerator accWidth should hold products")

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
    val y               = Output(Vec(length, SInt(accWidth.W)))
    val stateOut        = Output(Vec(length, SInt(accWidth.W)))
    val attentionScores = Output(Vec(length, SInt(accWidth.W)))
    val valid           = Output(Bool())
  })

  val block = Module(new TinyJambaBlock(length, dataWidth, accWidth))
  block.io.en := io.en
  block.io.clear := io.clear
  block.io.useAttention := io.useAttention
  block.io.x := io.x
  block.io.kernel := io.kernel
  block.io.mambaA := io.mambaA
  block.io.mambaB := io.mambaB
  block.io.mambaC := io.mambaC
  block.io.gate := io.gate
  block.io.attentionKeys := io.attentionKeys
  block.io.attentionValues := io.attentionValues

  val validReg = RegInit(false.B)
  when(io.clear) {
    validReg := false.B
  }.otherwise {
    validReg := io.en
  }

  io.y := block.io.y
  io.stateOut := block.io.stateOut
  io.attentionScores := block.io.attentionScores
  io.valid := validReg
}
