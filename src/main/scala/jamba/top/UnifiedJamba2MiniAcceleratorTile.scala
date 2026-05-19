package jamba.top

import chisel3._
import chisel3.util.{Enum, log2Ceil}
import jamba.common.{Jamba2MiniConfig, SignedMath}
import jamba.fabric.UnifiedJamba2MiniLayer
import jamba.memory.WeightStoreMini

/** Multi-cycle top shell around the unified Jamba2 mini layer.
  *
  * This is the first accelerator-shaped wrapper for the unified layer: it has
  * token valid/ready, output backpressure, a tiny weight store, mode selection,
  * and status/debug outputs. It is intentionally one layer for now; later work
  * can add a tile-level layer scheduler around the same shell.
  */
class UnifiedJamba2MiniAcceleratorTile(config: Jamba2MiniConfig = Jamba2MiniConfig.debug, weightDepth: Int = 256)
    extends Module {
  require(config.lanes == 4, "UnifiedJamba2MiniAcceleratorTile currently supports 4 lanes")
  require(weightDepth > 0, "UnifiedJamba2MiniAcceleratorTile weightDepth must be positive")

  private val lanes = config.lanes
  private val dataWidth = config.dataWidth
  private val accWidth = config.accWidth
  private val stateWidth = config.ssmStateBits
  private val addrWidth = math.max(1, log2Ceil(weightDepth))
  private val indexWidth = math.max(1, log2Ceil(config.contextLength))
  private val countWidth = log2Ceil(config.contextLength + 1)

  val io = IO(new Bundle {
    val clear = Input(Bool())
    val start = Input(Bool())
    val enableMoE = Input(Bool())
    val useLoadedWeights = Input(Bool())
    val mode = Input(UInt(2.W)) // 0/1 Mamba, 2 Attention, 3 config-selected hybrid demo

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

    val debugUsesAttention = Output(Bool())
    val debugSelectedExpert = Output(UInt(1.W))
    val debugStateOut = Output(Vec(lanes, SInt(stateWidth.W)))
    val debugLayerOutput = Output(Vec(lanes, SInt(accWidth.W)))
    val debugKvWriteIndex = Output(UInt(indexWidth.W))
    val debugKvValidCount = Output(UInt(countWidth.W))
    val debugProjectionBusy = Output(Bool())
  })

  val idle :: launchLayer :: waitLayer :: Nil = Enum(3)
  val state = RegInit(idle)
  val outputValid = RegInit(false.B)
  val outputReg = RegInit(VecInit(Seq.fill(lanes)(0.S(accWidth.W))))
  val xReg = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))
  val useAttentionReg = RegInit(false.B)
  val enableMoEReg = RegInit(false.B)
  val doneReg = RegInit(false.B)

  val weightStore = Module(new WeightStoreMini(weightDepth, accWidth))
  weightStore.io.clear := io.clear
  weightStore.io.writeValid := io.weightWriteValid
  weightStore.io.writeAddr := io.weightWriteAddr
  weightStore.io.writeData := io.weightWriteData
  weightStore.io.readAddr := io.weightReadAddr

  io.weightWriteReady := weightStore.io.writeReady
  io.weightReadData := weightStore.io.readData

  val layer = Module(new UnifiedJamba2MiniLayer(
    lanes = lanes,
    taps = config.convTaps,
    contextLength = config.contextLength,
    dataWidth = dataWidth,
    stateWidth = stateWidth,
    accWidth = accWidth
  ))
  layer.io.start := state === launchLayer
  layer.io.clear := io.clear
  layer.io.useAttention := useAttentionReg
  layer.io.enableMoE := enableMoEReg
  layer.io.x := xReg

  connectDemoWeights(layer)
  when(io.useLoadedWeights) {
    connectLoadedWeights(layer, weightStore.io.readAll)
  }

  val willConsume = outputValid && io.outReady
  val canAccept = state === idle && (!outputValid || willConsume)
  val fire = io.start && io.inValid && canAccept && !io.clear
  val modeUsesAttention = io.mode === 2.U || (io.mode === 3.U && config.isAttentionLayer(0).B)

  when(io.clear) {
    state := idle
    outputValid := false.B
    outputReg := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
    xReg := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    useAttentionReg := false.B
    enableMoEReg := false.B
    doneReg := false.B
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(fire) {
      xReg := io.in
      useAttentionReg := modeUsesAttention
      enableMoEReg := io.enableMoE
      outputValid := false.B
      state := launchLayer
    }.elsewhen(willConsume) {
      outputValid := false.B
    }
  }.elsewhen(state === launchLayer) {
    doneReg := false.B
    state := waitLayer
  }.elsewhen(state === waitLayer) {
    doneReg := false.B
    when(layer.io.done) {
      outputReg := layer.io.y
      outputValid := true.B
      doneReg := true.B
      state := idle
    }
  }

  io.inReady := io.start && canAccept && !io.clear
  io.outValid := outputValid
  io.out := outputReg
  io.busy := state =/= idle || outputValid
  io.done := doneReg || outputValid
  io.error := false.B

  io.debugUsesAttention := layer.io.mixerType
  io.debugSelectedExpert := layer.io.selectedExpert
  io.debugStateOut := layer.io.stateOut
  io.debugLayerOutput := layer.io.y
  io.debugKvWriteIndex := layer.io.kvWriteIndex
  io.debugKvValidCount := layer.io.kvValidCount
  io.debugProjectionBusy := layer.io.projectionBusy

  private def connectDemoWeights(layer: UnifiedJamba2MiniLayer): Unit = {
    for (lane <- 0 until lanes) {
      layer.io.norm1Weight(lane) := 1.S(dataWidth.W)
      layer.io.norm2Weight(lane) := 1.S(dataWidth.W)

      layer.io.mambaInputBias(lane) := 0.S(accWidth.W)
      layer.io.mambaBBias(lane) := 2.S(accWidth.W)
      layer.io.mambaCBias(lane) := 1.S(accWidth.W)
      layer.io.mambaA(lane) := 1.S(dataWidth.W)

      layer.io.qBias(lane) := 0.S(accWidth.W)
      layer.io.kBias(lane) := 0.S(accWidth.W)
      layer.io.vBias(lane) := 0.S(accWidth.W)
      layer.io.attentionOutBias(lane) := 0.S(accWidth.W)

      layer.io.mlpGateBias(lane) := 1.S(accWidth.W)
      layer.io.mlpUpBias(lane) := 0.S(accWidth.W)
      layer.io.mlpDownBias(lane) := 0.S(accWidth.W)
    }

    for (row <- 0 until lanes) {
      for (col <- 0 until lanes) {
        val identity = if (row == col) 1 else 0
        val reverseIdentity = if (row == lanes - 1 - col) 1 else 0

        layer.io.mambaInputWeight(row)(col) := identity.S(dataWidth.W)
        layer.io.mambaBWeight(row)(col) := 0.S(dataWidth.W)
        layer.io.mambaCWeight(row)(col) := 0.S(dataWidth.W)

        layer.io.qWeight(row)(col) := identity.S(dataWidth.W)
        layer.io.kWeight(row)(col) := identity.S(dataWidth.W)
        layer.io.vWeight(row)(col) := identity.S(dataWidth.W)
        layer.io.attentionOutWeight(row)(col) := identity.S(dataWidth.W)

        layer.io.mlpGateWeight(row)(col) := identity.S(dataWidth.W)
        layer.io.mlpUpWeight(row)(col) := reverseIdentity.S(dataWidth.W)
        layer.io.mlpDownWeight(row)(col) := identity.S(dataWidth.W)
      }
    }

    for (tap <- 0 until config.convTaps) {
      for (lane <- 0 until lanes) {
        layer.io.mambaKernel(tap)(lane) := 1.S(dataWidth.W)
      }
    }

    connectDemoExpertWeights(layer)
  }

  private def connectDemoExpertWeights(layer: UnifiedJamba2MiniLayer): Unit = {
    for (expert <- 0 until 2) {
      layer.io.routerBias(expert) := (if (expert == 0) 0 else 1).S(accWidth.W)
      for (lane <- 0 until lanes) {
        val routerValue = if (expert == lane) 1 else 0
        layer.io.routerWeight(expert)(lane) := routerValue.S(dataWidth.W)
        layer.io.expertGateBias(expert)(lane) := 1.S(accWidth.W)
        layer.io.expertUpBias(expert)(lane) := 0.S(accWidth.W)
        layer.io.expertDownBias(expert)(lane) := expert.S(accWidth.W)

        for (col <- 0 until lanes) {
          val identity = if (lane == col) 1 else 0
          layer.io.expertGateWeight(expert)(lane)(col) := identity.S(dataWidth.W)
          layer.io.expertUpWeight(expert)(lane)(col) := identity.S(dataWidth.W)
          layer.io.expertDownWeight(expert)(lane)(col) := identity.S(dataWidth.W)
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

  private def connectLoadedWeights(layer: UnifiedJamba2MiniLayer, weights: Vec[SInt]): Unit = {
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
      layer.io.norm1Weight(lane) := vectorDataAt(WeightMap.Norm1Weight, lane)
      layer.io.norm2Weight(lane) := vectorDataAt(WeightMap.Norm2Weight, lane)

      layer.io.mambaInputBias(lane) := vectorAccAt(WeightMap.MambaInputBias, lane)
      layer.io.mambaBBias(lane) := vectorAccAt(WeightMap.MambaBBias, lane)
      layer.io.mambaCBias(lane) := vectorAccAt(WeightMap.MambaCBias, lane)
      layer.io.mambaA(lane) := vectorDataAt(WeightMap.MambaA, lane)

      layer.io.qBias(lane) := vectorAccAt(WeightMap.QBias, lane)
      layer.io.kBias(lane) := vectorAccAt(WeightMap.KBias, lane)
      layer.io.vBias(lane) := vectorAccAt(WeightMap.VBias, lane)
      layer.io.attentionOutBias(lane) := vectorAccAt(WeightMap.AttentionOutBias, lane)

      layer.io.mlpGateBias(lane) := vectorAccAt(WeightMap.MlpGateBias, lane)
      layer.io.mlpUpBias(lane) := vectorAccAt(WeightMap.MlpUpBias, lane)
      layer.io.mlpDownBias(lane) := vectorAccAt(WeightMap.MlpDownBias, lane)
    }

    for (row <- 0 until lanes) {
      for (col <- 0 until lanes) {
        layer.io.mambaInputWeight(row)(col) := matrixAt(WeightMap.MambaInputWeight, row, col)
        layer.io.mambaBWeight(row)(col) := matrixAt(WeightMap.MambaBWeight, row, col)
        layer.io.mambaCWeight(row)(col) := matrixAt(WeightMap.MambaCWeight, row, col)

        layer.io.qWeight(row)(col) := matrixAt(WeightMap.QWeight, row, col)
        layer.io.kWeight(row)(col) := matrixAt(WeightMap.KWeight, row, col)
        layer.io.vWeight(row)(col) := matrixAt(WeightMap.VWeight, row, col)
        layer.io.attentionOutWeight(row)(col) := matrixAt(WeightMap.AttentionOutWeight, row, col)

        layer.io.mlpGateWeight(row)(col) := matrixAt(WeightMap.MlpGateWeight, row, col)
        layer.io.mlpUpWeight(row)(col) := matrixAt(WeightMap.MlpUpWeight, row, col)
        layer.io.mlpDownWeight(row)(col) := matrixAt(WeightMap.MlpDownWeight, row, col)
      }
    }

    for (tap <- 0 until config.convTaps) {
      for (lane <- 0 until lanes) {
        layer.io.mambaKernel(tap)(lane) := dataAt(WeightMap.MambaKernel + tap * lanes + lane)
      }
    }

    for (expert <- 0 until 2) {
      layer.io.routerBias(expert) := accAt(WeightMap.RouterBias + expert)
      for (lane <- 0 until lanes) {
        layer.io.routerWeight(expert)(lane) := dataAt(WeightMap.RouterWeight + expert * lanes + lane)
      }
    }
  }
}
