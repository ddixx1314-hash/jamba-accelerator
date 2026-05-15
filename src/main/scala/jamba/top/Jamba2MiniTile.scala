package jamba.top

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.Jamba2MiniConfig
import jamba.core.Jamba2MiniHybridCore
import jamba.memory.WeightStoreMini

/** Formal Jamba2 mini accelerator tile with token stream, command/status, and weight-load shell. */
class Jamba2MiniTile(config: Jamba2MiniConfig = Jamba2MiniConfig.debug, weightDepth: Int = 256) extends Module {
  require(config.lanes == 4, "Jamba2MiniTile currently supports 4 lanes")
  require(weightDepth > 0, "Jamba2MiniTile weightDepth must be positive")

  private val lanes = config.lanes
  private val dataWidth = config.dataWidth
  private val accWidth = config.accWidth
  private val stateWidth = config.ssmStateBits
  private val addrWidth = math.max(1, log2Ceil(weightDepth))

  val io = IO(new Bundle {
    val clear = Input(Bool())
    val start = Input(Bool())
    val enableMoE = Input(Bool())

    val inValid = Input(Bool())
    val inReady = Output(Bool())
    val in = Input(Vec(lanes, SInt(dataWidth.W)))

    val outValid = Output(Bool())
    val outReady = Input(Bool())
    val out = Output(Vec(lanes, SInt(accWidth.W)))

    val weightWriteValid = Input(Bool())
    val weightWriteReady = Output(Bool())
    val weightWriteAddr = Input(UInt(addrWidth.W))
    val weightWriteData = Input(SInt(accWidth.W))
    val weightReadAddr = Input(UInt(addrWidth.W))
    val weightReadData = Output(SInt(accWidth.W))

    val busy = Output(Bool())
    val done = Output(Bool())
    val error = Output(Bool())

    val debugLayerUsesAttention = Output(Vec(config.numLayers, Bool()))
    val debugLayerSelectedExpert = Output(Vec(config.numLayers, UInt(1.W)))
    val debugLayerStateOut = Output(Vec(config.numLayers, Vec(lanes, SInt(stateWidth.W))))
    val debugLayerOutputs = Output(Vec(config.numLayers, Vec(lanes, SInt(accWidth.W))))
  })

  val weightStore = Module(new WeightStoreMini(weightDepth, accWidth))
  weightStore.io.clear := io.clear
  weightStore.io.writeValid := io.weightWriteValid
  weightStore.io.writeAddr := io.weightWriteAddr
  weightStore.io.writeData := io.weightWriteData
  weightStore.io.readAddr := io.weightReadAddr

  io.weightWriteReady := weightStore.io.writeReady
  io.weightReadData := weightStore.io.readData

  val outputValid = RegInit(false.B)
  val outputReg = RegInit(VecInit(Seq.fill(lanes)(0.S(accWidth.W))))
  val willConsume = outputValid && io.outReady
  val canAccept = !outputValid || willConsume
  val fire = io.start && io.inValid && canAccept && !io.clear

  val core = Module(new Jamba2MiniHybridCore(config))
  core.io.en := fire
  core.io.clear := io.clear
  core.io.enableMoE := io.enableMoE
  core.io.x := io.in

  connectDemoWeights(core)

  when(io.clear) {
    outputValid := false.B
    outputReg := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  }.elsewhen(fire) {
    outputValid := true.B
    outputReg := core.io.y
  }.elsewhen(willConsume) {
    outputValid := false.B
  }

  io.inReady := io.start && canAccept && !io.clear
  io.outValid := outputValid
  io.out := outputReg

  io.busy := outputValid
  io.done := outputValid
  io.error := false.B

  io.debugLayerUsesAttention := core.io.layerUsesAttention
  io.debugLayerSelectedExpert := core.io.layerSelectedExpert
  io.debugLayerStateOut := core.io.layerStateOut
  io.debugLayerOutputs := core.io.layerOutputs

  private def connectDemoWeights(core: Jamba2MiniHybridCore): Unit = {
    for (lane <- 0 until lanes) {
      core.io.norm1Weight(lane) := 1.S(dataWidth.W)
      core.io.norm2Weight(lane) := 1.S(dataWidth.W)

      core.io.mambaInputBias(lane) := 0.S(accWidth.W)
      core.io.mambaBBias(lane) := 2.S(accWidth.W)
      core.io.mambaCBias(lane) := 1.S(accWidth.W)
      core.io.mambaA(lane) := 1.S(dataWidth.W)

      core.io.qBias(lane) := 0.S(accWidth.W)
      core.io.kBias(lane) := 0.S(accWidth.W)
      core.io.vBias(lane) := 0.S(accWidth.W)
      core.io.attentionOutBias(lane) := 0.S(accWidth.W)

      core.io.mlpGateBias(lane) := 1.S(accWidth.W)
      core.io.mlpUpBias(lane) := 0.S(accWidth.W)
      core.io.mlpDownBias(lane) := 0.S(accWidth.W)
    }

    for (row <- 0 until lanes) {
      for (col <- 0 until lanes) {
        val identity = if (row == col) 1 else 0
        val reverseIdentity = if (row == lanes - 1 - col) 1 else 0

        core.io.mambaInputWeight(row)(col) := identity.S(dataWidth.W)
        core.io.mambaBWeight(row)(col) := 0.S(dataWidth.W)
        core.io.mambaCWeight(row)(col) := 0.S(dataWidth.W)

        core.io.qWeight(row)(col) := identity.S(dataWidth.W)
        core.io.kWeight(row)(col) := identity.S(dataWidth.W)
        core.io.vWeight(row)(col) := identity.S(dataWidth.W)
        core.io.attentionOutWeight(row)(col) := identity.S(dataWidth.W)

        core.io.mlpGateWeight(row)(col) := identity.S(dataWidth.W)
        core.io.mlpUpWeight(row)(col) := reverseIdentity.S(dataWidth.W)
        core.io.mlpDownWeight(row)(col) := identity.S(dataWidth.W)
      }
    }

    for (tap <- 0 until config.convTaps) {
      for (lane <- 0 until lanes) {
        core.io.mambaKernel(tap)(lane) := 1.S(dataWidth.W)
      }
    }

    for (expert <- 0 until 2) {
      core.io.routerBias(expert) := (if (expert == 0) 0 else 1).S(accWidth.W)
      for (lane <- 0 until lanes) {
        val routerValue = if (expert == lane) 1 else 0
        core.io.routerWeight(expert)(lane) := routerValue.S(dataWidth.W)
        core.io.expertGateBias(expert)(lane) := 1.S(accWidth.W)
        core.io.expertUpBias(expert)(lane) := 0.S(accWidth.W)
        core.io.expertDownBias(expert)(lane) := expert.S(accWidth.W)

        for (col <- 0 until lanes) {
          val identity = if (lane == col) 1 else 0
          core.io.expertGateWeight(expert)(lane)(col) := identity.S(dataWidth.W)
          core.io.expertUpWeight(expert)(lane)(col) := identity.S(dataWidth.W)
          core.io.expertDownWeight(expert)(lane)(col) := identity.S(dataWidth.W)
        }
      }
    }
  }
}
