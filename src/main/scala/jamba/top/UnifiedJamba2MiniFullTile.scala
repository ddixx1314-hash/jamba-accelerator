package jamba.top

import chisel3._
import chisel3.util.{Enum, MuxLookup, log2Ceil}
import jamba.common.{Jamba2MiniConfig, SignedMath}
import jamba.memory.WeightStoreMini

/** Accelerator-shaped shell around the multi-layer unified Jamba2 mini scheduler.
  *
  * Unlike UnifiedJamba2MiniAcceleratorTile, this top does not expose a runtime
  * mixer mode. Attention placement is fixed by Jamba2MiniConfig so the token
  * runs through a mini hybrid layer stack.
  */
class UnifiedJamba2MiniFullTile(config: Jamba2MiniConfig = Jamba2MiniConfig.debug, weightDepth: Int = 256)
    extends Module {
  require(config.lanes == 4, "UnifiedJamba2MiniFullTile currently supports 4 lanes")
  require(weightDepth > 0, "UnifiedJamba2MiniFullTile weightDepth must be positive")

  private val lanes = config.lanes
  private val dataWidth = config.dataWidth
  private val accWidth = config.accWidth
  private val stateWidth = config.ssmStateBits
  private val addrWidth = math.max(1, log2Ceil(weightDepth))
  private val layerIndexWidth = math.max(1, log2Ceil(config.numLayers))
  private val kvIndexWidth = math.max(1, log2Ceil(config.contextLength))
  private val kvCountWidth = log2Ceil(config.contextLength + 1)

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

    val debugActiveLayer = Output(UInt(layerIndexWidth.W))
    val debugLayerUsesAttention = Output(Vec(config.numLayers, Bool()))
    val debugLayerOutput = Output(Vec(config.numLayers, Vec(lanes, SInt(accWidth.W))))
    val debugLayerStateOut = Output(Vec(config.numLayers, Vec(lanes, SInt(stateWidth.W))))
    val debugLayerKvWriteIndex = Output(Vec(config.numLayers, UInt(kvIndexWidth.W)))
    val debugLayerKvValidCount = Output(Vec(config.numLayers, UInt(kvCountWidth.W)))
    val debugLayerSelectedExpert = Output(Vec(config.numLayers, UInt(1.W)))
    val debugSchedulerBusy = Output(Bool())
  })

  private def zeroData = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroAcc = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))

  val idle :: launchScheduler :: waitScheduler :: Nil = Enum(3)
  val state = RegInit(idle)
  val xReg = RegInit(zeroData)
  val outputValid = RegInit(false.B)
  val outputReg = RegInit(zeroAcc)
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

  val scheduler = Module(new UnifiedJamba2MiniTileScheduler(config))
  scheduler.io.start := state === launchScheduler
  scheduler.io.clear := io.clear
  scheduler.io.enableMoE := enableMoEReg
  scheduler.io.x := xReg

  connectDemoWeights(scheduler)
  when(io.useLoadedWeights) {
    connectLoadedWeights(scheduler, weightStore.io.readAll)
  }

  val willConsume = outputValid && io.outReady
  val canAccept = state === idle && (!outputValid || willConsume)
  val fire = io.start && io.inValid && canAccept && !io.clear

  when(io.clear) {
    state := idle
    xReg := zeroData
    outputValid := false.B
    outputReg := zeroAcc
    enableMoEReg := false.B
    doneReg := false.B
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(fire) {
      xReg := io.in
      enableMoEReg := io.enableMoE
      outputValid := false.B
      state := launchScheduler
    }.elsewhen(willConsume) {
      outputValid := false.B
    }
  }.elsewhen(state === launchScheduler) {
    doneReg := false.B
    state := waitScheduler
  }.elsewhen(state === waitScheduler) {
    doneReg := false.B
    when(scheduler.io.done) {
      outputReg := scheduler.io.y
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

  io.debugActiveLayer := scheduler.io.activeLayer
  io.debugLayerUsesAttention := scheduler.io.layerUsesAttention
  io.debugLayerOutput := scheduler.io.layerOutputs
  io.debugLayerStateOut := scheduler.io.layerStateOut
  io.debugLayerKvWriteIndex := scheduler.io.layerKvWriteIndex
  io.debugLayerKvValidCount := scheduler.io.layerKvValidCount
  io.debugLayerSelectedExpert := scheduler.io.layerSelectedExpert
  io.debugSchedulerBusy := scheduler.io.busy

  private def connectDemoWeights(scheduler: UnifiedJamba2MiniTileScheduler): Unit = {
    for (lane <- 0 until lanes) {
      scheduler.io.norm1Weight(lane) := 1.S(dataWidth.W)
      scheduler.io.norm2Weight(lane) := 1.S(dataWidth.W)

      scheduler.io.mambaInputBias(lane) := 0.S(accWidth.W)
      scheduler.io.mambaBBias(lane) := 2.S(accWidth.W)
      scheduler.io.mambaCBias(lane) := 1.S(accWidth.W)
      scheduler.io.mambaA(lane) := 1.S(dataWidth.W)

      scheduler.io.qBias(lane) := 0.S(accWidth.W)
      scheduler.io.kBias(lane) := 0.S(accWidth.W)
      scheduler.io.vBias(lane) := 0.S(accWidth.W)
      scheduler.io.attentionOutBias(lane) := 0.S(accWidth.W)

      scheduler.io.mlpGateBias(lane) := 1.S(accWidth.W)
      scheduler.io.mlpUpBias(lane) := 0.S(accWidth.W)
      scheduler.io.mlpDownBias(lane) := 0.S(accWidth.W)
    }

    for (row <- 0 until lanes) {
      for (col <- 0 until lanes) {
        val identity = if (row == col) 1 else 0
        val reverseIdentity = if (row == lanes - 1 - col) 1 else 0

        scheduler.io.mambaInputWeight(row)(col) := identity.S(dataWidth.W)
        scheduler.io.mambaBWeight(row)(col) := 0.S(dataWidth.W)
        scheduler.io.mambaCWeight(row)(col) := 0.S(dataWidth.W)

        scheduler.io.qWeight(row)(col) := identity.S(dataWidth.W)
        scheduler.io.kWeight(row)(col) := identity.S(dataWidth.W)
        scheduler.io.vWeight(row)(col) := identity.S(dataWidth.W)
        scheduler.io.attentionOutWeight(row)(col) := identity.S(dataWidth.W)

        scheduler.io.mlpGateWeight(row)(col) := identity.S(dataWidth.W)
        scheduler.io.mlpUpWeight(row)(col) := reverseIdentity.S(dataWidth.W)
        scheduler.io.mlpDownWeight(row)(col) := identity.S(dataWidth.W)
      }
    }

    for (tap <- 0 until config.convTaps) {
      for (lane <- 0 until lanes) {
        scheduler.io.mambaKernel(tap)(lane) := 1.S(dataWidth.W)
      }
    }

    connectDemoExpertWeights(scheduler)
  }

  private def connectDemoExpertWeights(scheduler: UnifiedJamba2MiniTileScheduler): Unit = {
    for (expert <- 0 until 2) {
      scheduler.io.routerBias(expert) := (if (expert == 0) 0 else 1).S(accWidth.W)
      for (lane <- 0 until lanes) {
        val routerValue = if (expert == lane) 1 else 0
        scheduler.io.routerWeight(expert)(lane) := routerValue.S(dataWidth.W)
        scheduler.io.expertGateBias(expert)(lane) := 1.S(accWidth.W)
        scheduler.io.expertUpBias(expert)(lane) := 0.S(accWidth.W)
        scheduler.io.expertDownBias(expert)(lane) := expert.S(accWidth.W)

        for (col <- 0 until lanes) {
          val identity = if (lane == col) 1 else 0
          scheduler.io.expertGateWeight(expert)(lane)(col) := identity.S(dataWidth.W)
          scheduler.io.expertUpWeight(expert)(lane)(col) := identity.S(dataWidth.W)
          scheduler.io.expertDownWeight(expert)(lane)(col) := identity.S(dataWidth.W)
        }
      }
    }
  }

  private object WeightMap {
    val LayerStride = 256

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

  private def connectLoadedWeights(scheduler: UnifiedJamba2MiniTileScheduler, weights: Vec[SInt]): Unit = {
    val dynamicAddrWidth =
      math.max(1, log2Ceil(math.max(weightDepth, WeightMap.LayerStride * config.numLayers) + 1))

    def accAtDynamic(addr: UInt): SInt =
      MuxLookup(addr, 0.S(accWidth.W))(Seq.tabulate(weightDepth) { index =>
        index.U(dynamicAddrWidth.W) -> weights(index)
      })

    def layerAddress(offset: Int): UInt = {
      val layerBase = scheduler.io.activeLayer * WeightMap.LayerStride.U(dynamicAddrWidth.W)
      layerBase + offset.U(dynamicAddrWidth.W)
    }

    def accAt(addr: Int): SInt =
      accAtDynamic(layerAddress(addr))

    def dataAt(addr: Int): SInt =
      SignedMath.resize(accAt(addr), dataWidth)

    def matrixAt(base: Int, row: Int, col: Int): SInt =
      dataAt(base + row * lanes + col)

    def vectorAccAt(base: Int, lane: Int): SInt =
      accAt(base + lane)

    def vectorDataAt(base: Int, lane: Int): SInt =
      dataAt(base + lane)

    for (lane <- 0 until lanes) {
      scheduler.io.norm1Weight(lane) := vectorDataAt(WeightMap.Norm1Weight, lane)
      scheduler.io.norm2Weight(lane) := vectorDataAt(WeightMap.Norm2Weight, lane)

      scheduler.io.mambaInputBias(lane) := vectorAccAt(WeightMap.MambaInputBias, lane)
      scheduler.io.mambaBBias(lane) := vectorAccAt(WeightMap.MambaBBias, lane)
      scheduler.io.mambaCBias(lane) := vectorAccAt(WeightMap.MambaCBias, lane)
      scheduler.io.mambaA(lane) := vectorDataAt(WeightMap.MambaA, lane)

      scheduler.io.qBias(lane) := vectorAccAt(WeightMap.QBias, lane)
      scheduler.io.kBias(lane) := vectorAccAt(WeightMap.KBias, lane)
      scheduler.io.vBias(lane) := vectorAccAt(WeightMap.VBias, lane)
      scheduler.io.attentionOutBias(lane) := vectorAccAt(WeightMap.AttentionOutBias, lane)

      scheduler.io.mlpGateBias(lane) := vectorAccAt(WeightMap.MlpGateBias, lane)
      scheduler.io.mlpUpBias(lane) := vectorAccAt(WeightMap.MlpUpBias, lane)
      scheduler.io.mlpDownBias(lane) := vectorAccAt(WeightMap.MlpDownBias, lane)
    }

    for (row <- 0 until lanes) {
      for (col <- 0 until lanes) {
        scheduler.io.mambaInputWeight(row)(col) := matrixAt(WeightMap.MambaInputWeight, row, col)
        scheduler.io.mambaBWeight(row)(col) := matrixAt(WeightMap.MambaBWeight, row, col)
        scheduler.io.mambaCWeight(row)(col) := matrixAt(WeightMap.MambaCWeight, row, col)

        scheduler.io.qWeight(row)(col) := matrixAt(WeightMap.QWeight, row, col)
        scheduler.io.kWeight(row)(col) := matrixAt(WeightMap.KWeight, row, col)
        scheduler.io.vWeight(row)(col) := matrixAt(WeightMap.VWeight, row, col)
        scheduler.io.attentionOutWeight(row)(col) := matrixAt(WeightMap.AttentionOutWeight, row, col)

        scheduler.io.mlpGateWeight(row)(col) := matrixAt(WeightMap.MlpGateWeight, row, col)
        scheduler.io.mlpUpWeight(row)(col) := matrixAt(WeightMap.MlpUpWeight, row, col)
        scheduler.io.mlpDownWeight(row)(col) := matrixAt(WeightMap.MlpDownWeight, row, col)
      }
    }

    for (tap <- 0 until config.convTaps) {
      for (lane <- 0 until lanes) {
        scheduler.io.mambaKernel(tap)(lane) := dataAt(WeightMap.MambaKernel + tap * lanes + lane)
      }
    }

    for (expert <- 0 until 2) {
      scheduler.io.routerBias(expert) := accAt(WeightMap.RouterBias + expert)
      for (lane <- 0 until lanes) {
        scheduler.io.routerWeight(expert)(lane) := dataAt(WeightMap.RouterWeight + expert * lanes + lane)
      }
    }
  }
}
