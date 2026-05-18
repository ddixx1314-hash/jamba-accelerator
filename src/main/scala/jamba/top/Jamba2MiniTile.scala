package jamba.top

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.{Jamba2MiniConfig, SignedMath}
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
    val useLoadedWeights = Input(Bool())

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
    val debugLayerKvWriteIndex = Output(Vec(config.numLayers, UInt(math.max(1, log2Ceil(config.contextLength)).W)))
    val debugLayerKvValidCount = Output(Vec(config.numLayers, UInt(log2Ceil(config.contextLength + 1).W)))
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
  when(io.useLoadedWeights) {
    connectLoadedWeights(core, weightStore.io.readAll)
  }

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
  io.debugLayerKvWriteIndex := core.io.layerKvWriteIndex
  io.debugLayerKvValidCount := core.io.layerKvValidCount

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

  private object WeightMap {
    val Norm1Weight = 0
    val Norm2Weight = 4

    val MambaInputWeight = 16
    val MambaInputBias = 32
    val MambaBWeight = 36
    val MambaBBias = 52
    val MambaCWeight = 56
    val MambaCBias = 72
    val MambaA = 76
    val MambaKernel = 80

    val QWeight = 96
    val QBias = 112
    val KWeight = 116
    val KBias = 132
    val VWeight = 136
    val VBias = 152
    val AttentionOutWeight = 156
    val AttentionOutBias = 172

    val MlpGateWeight = 176
    val MlpGateBias = 192
    val MlpUpWeight = 196
    val MlpUpBias = 212
    val MlpDownWeight = 216
    val MlpDownBias = 232

    val RouterWeight = 236
    val RouterBias = 244
  }

  private def connectLoadedWeights(core: Jamba2MiniHybridCore, weights: Vec[SInt]): Unit = {
    def accAt(addr: Int): SInt =
      if (addr < weightDepth) weights(addr) else 0.S(accWidth.W)

    def dataAt(addr: Int): SInt =
      SignedMath.resize(accAt(addr), dataWidth)

    def matrixAt(base: Int, row: Int, col: Int): SInt =
      dataAt(base + row * lanes + col)

    def vectorAccAt(base: Int, lane: Int): SInt =
      accAt(base + lane)

    def vectorDataAt(base: Int, lane: Int): SInt =
      dataAt(base + lane)

    for (lane <- 0 until lanes) {
      core.io.norm1Weight(lane) := vectorDataAt(WeightMap.Norm1Weight, lane)
      core.io.norm2Weight(lane) := vectorDataAt(WeightMap.Norm2Weight, lane)

      core.io.mambaInputBias(lane) := vectorAccAt(WeightMap.MambaInputBias, lane)
      core.io.mambaBBias(lane) := vectorAccAt(WeightMap.MambaBBias, lane)
      core.io.mambaCBias(lane) := vectorAccAt(WeightMap.MambaCBias, lane)
      core.io.mambaA(lane) := vectorDataAt(WeightMap.MambaA, lane)

      core.io.qBias(lane) := vectorAccAt(WeightMap.QBias, lane)
      core.io.kBias(lane) := vectorAccAt(WeightMap.KBias, lane)
      core.io.vBias(lane) := vectorAccAt(WeightMap.VBias, lane)
      core.io.attentionOutBias(lane) := vectorAccAt(WeightMap.AttentionOutBias, lane)

      core.io.mlpGateBias(lane) := vectorAccAt(WeightMap.MlpGateBias, lane)
      core.io.mlpUpBias(lane) := vectorAccAt(WeightMap.MlpUpBias, lane)
      core.io.mlpDownBias(lane) := vectorAccAt(WeightMap.MlpDownBias, lane)
    }

    for (row <- 0 until lanes) {
      for (col <- 0 until lanes) {
        core.io.mambaInputWeight(row)(col) := matrixAt(WeightMap.MambaInputWeight, row, col)
        core.io.mambaBWeight(row)(col) := matrixAt(WeightMap.MambaBWeight, row, col)
        core.io.mambaCWeight(row)(col) := matrixAt(WeightMap.MambaCWeight, row, col)

        core.io.qWeight(row)(col) := matrixAt(WeightMap.QWeight, row, col)
        core.io.kWeight(row)(col) := matrixAt(WeightMap.KWeight, row, col)
        core.io.vWeight(row)(col) := matrixAt(WeightMap.VWeight, row, col)
        core.io.attentionOutWeight(row)(col) := matrixAt(WeightMap.AttentionOutWeight, row, col)

        core.io.mlpGateWeight(row)(col) := matrixAt(WeightMap.MlpGateWeight, row, col)
        core.io.mlpUpWeight(row)(col) := matrixAt(WeightMap.MlpUpWeight, row, col)
        core.io.mlpDownWeight(row)(col) := matrixAt(WeightMap.MlpDownWeight, row, col)
      }
    }

    for (tap <- 0 until config.convTaps) {
      for (lane <- 0 until lanes) {
        core.io.mambaKernel(tap)(lane) := dataAt(WeightMap.MambaKernel + tap * lanes + lane)
      }
    }

    for (expert <- 0 until 2) {
      core.io.routerBias(expert) := accAt(WeightMap.RouterBias + expert)
      for (lane <- 0 until lanes) {
        core.io.routerWeight(expert)(lane) := dataAt(WeightMap.RouterWeight + expert * lanes + lane)
      }
    }
  }
}
