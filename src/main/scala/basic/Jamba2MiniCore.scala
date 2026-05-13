package basic

import chisel3._

/** More complete tiny Jamba 2.0-style core: norm, projections, sequence block, output projection. */
class Jamba2MiniCore(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(dataWidth > 0, "Jamba2MiniCore dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "Jamba2MiniCore accWidth should hold products")

  val io = IO(new Bundle {
    val en              = Input(Bool())
    val clear           = Input(Bool())
    val useAttention    = Input(Bool())
    val x               = Input(Vec(4, SInt(dataWidth.W)))
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
    val normMeanSquare  = Output(SInt(accWidth.W))
    val projectedX      = Output(Vec(4, SInt(dataWidth.W)))
    val blockY          = Output(Vec(4, SInt(accWidth.W)))
    val stateOut        = Output(Vec(4, SInt(accWidth.W)))
    val attentionScores = Output(Vec(4, SInt(accWidth.W)))
    val y               = Output(Vec(4, SInt(accWidth.W)))
    val valid           = Output(Bool())
  })

  val norm = Module(new RmsNormApprox(4, dataWidth, accWidth))
  norm.io.x := io.x
  norm.io.weight := io.rmsWeight

  val inputProj = Module(new Linear4(dataWidth, accWidth))
  inputProj.io.x := norm.io.y
  inputProj.io.weight := io.inputWeight
  inputProj.io.bias := io.inputBias

  val gateProj = Module(new Linear4(dataWidth, accWidth))
  gateProj.io.x := norm.io.y
  gateProj.io.weight := io.gateWeight
  gateProj.io.bias := io.gateBias

  val bProj = Module(new Linear4(dataWidth, accWidth))
  bProj.io.x := norm.io.y
  bProj.io.weight := io.bWeight
  bProj.io.bias := io.bBias

  val cProj = Module(new Linear4(dataWidth, accWidth))
  cProj.io.x := norm.io.y
  cProj.io.weight := io.cWeight
  cProj.io.bias := io.cBias

  val block = Module(new TinyJambaBlock(4, dataWidth, accWidth))
  block.io.en := io.en
  block.io.clear := io.clear
  block.io.useAttention := io.useAttention
  block.io.kernel := io.kernel
  block.io.mambaA := io.mambaA
  block.io.attentionKeys := io.attentionKeys
  block.io.attentionValues := io.attentionValues

  val outInput = Wire(Vec(4, SInt(dataWidth.W)))
  for (i <- 0 until 4) {
    block.io.x(i) := SignedMath.resize(inputProj.io.y(i), dataWidth)
    block.io.gate(i) := SignedMath.resize(gateProj.io.y(i), dataWidth)
    block.io.mambaB(i) := SignedMath.resize(bProj.io.y(i), dataWidth)
    block.io.mambaC(i) := SignedMath.resize(cProj.io.y(i), dataWidth)
    outInput(i) := SignedMath.resize(block.io.y(i), dataWidth)
  }

  val outProj = Module(new Linear4(dataWidth, accWidth))
  outProj.io.x := outInput
  outProj.io.weight := io.outWeight
  outProj.io.bias := io.outBias

  val validReg = RegInit(false.B)
  when(io.clear) {
    validReg := false.B
  }.otherwise {
    validReg := io.en
  }

  io.normMeanSquare := norm.io.meanSquare
  io.projectedX := block.io.x
  io.blockY := block.io.y
  io.stateOut := block.io.stateOut
  io.attentionScores := block.io.attentionScores
  io.y := outProj.io.y
  io.valid := validReg
}
