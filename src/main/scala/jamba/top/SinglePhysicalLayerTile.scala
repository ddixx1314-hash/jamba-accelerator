package jamba.top

import chisel3._
import chisel3.util.{Enum, log2Ceil}
import jamba.common.{Jamba2MiniConfig, SignedMath}
import jamba.fabric.UnifiedJamba2MiniLayer
import jamba.memory.LayeredWeightStoreMini

/** Single-physical-layer tile: L logical layers share ONE UnifiedJamba2MiniLayer.
  *
  * This is M7-A (structure proof). The tile sequences through all numLayers logical
  * layers by re-using a single UnifiedJamba2MiniLayer instance. LayeredWeightStoreMini
  * selects the active layer's weights combinatorially via activeLayer. The output of
  * each logical layer is narrowed to dataWidth and fed as the input to the next.
  *
  * Resource effect: instance-weighted mul-proxy ≈ 92 (constant), regardless of
  * numLayers. Compare to UnifiedJamba2MiniFullTile where the proxy grows as ~92L.
  *
  * Acknowledged limitation (M7-A): SSM hidden state and KV cache inside the single
  * physical layer are NOT saved/restored between logical layers. Each token's logical
  * layers share the same running SSM state. Full per-layer state virtualization is
  * deferred to M7-B.
  */
class SinglePhysicalLayerTile(
    config:      Jamba2MiniConfig = Jamba2MiniConfig.debug,
    weightDepth: Int              = 256)
    extends Module {
  require(config.lanes == 4, "SinglePhysicalLayerTile currently supports 4 lanes")
  require(config.numLayers > 0, "SinglePhysicalLayerTile needs at least one layer")
  require(weightDepth > 0, "SinglePhysicalLayerTile weightDepth must be positive")

  private val lanes           = config.lanes
  private val dataWidth       = config.dataWidth
  private val accWidth        = config.accWidth
  private val stateWidth      = config.ssmStateBits
  private val numLayers       = config.numLayers
  private val layerIndexWidth = math.max(1, log2Ceil(numLayers))
  private val addrWidth       = math.max(1, log2Ceil(weightDepth))

  val io = IO(new Bundle {
    val clear          = Input(Bool())
    val start          = Input(Bool())
    val enableMoE      = Input(Bool())
    val useLoadedWeights = Input(Bool())

    val inValid  = Input(Bool())
    val inReady  = Output(Bool())
    val in       = Input(Vec(lanes, SInt(dataWidth.W)))

    val outValid = Output(Bool())
    val outReady = Input(Bool())
    val out      = Output(Vec(lanes, SInt(accWidth.W)))

    val weightWriteValid = Input(Bool())
    val weightWriteReady = Output(Bool())
    val weightWriteAddr  = Input(UInt(addrWidth.W))
    val weightWriteData  = Input(SInt(accWidth.W))
    val weightReadAddr   = Input(UInt(addrWidth.W))
    val weightReadData   = Output(SInt(accWidth.W))

    val busy  = Output(Bool())
    val done  = Output(Bool())
    val error = Output(Bool())

    val debugActiveLayer        = Output(UInt(layerIndexWidth.W))
    val debugLayerUsesAttention = Output(Vec(numLayers, Bool()))
    val debugLayerOutput        = Output(Vec(numLayers, Vec(lanes, SInt(accWidth.W))))
  })

  private def zeroData = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroAcc  = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))

  // Compile-time attention-layer lookup ROM
  private val attentionLayerMap =
    VecInit((0 until numLayers).map(i => config.isAttentionLayer(i).B))

  val idle :: launchLayer :: waitLayer :: Nil = Enum(3)
  val state          = RegInit(idle)
  val activeLayerReg = RegInit(0.U(layerIndexWidth.W))
  val currentX       = RegInit(zeroData)
  val outputValid    = RegInit(false.B)
  val outputReg      = RegInit(zeroAcc)
  val doneReg        = RegInit(false.B)
  val enableMoEReg   = RegInit(false.B)
  val layerOutputsReg = RegInit(VecInit(Seq.fill(numLayers)(zeroAcc)))

  // ── ONE physical layer ─────────────────────────────────────────────────────
  val physLayer = Module(new UnifiedJamba2MiniLayer(
    lanes         = lanes,
    taps          = config.convTaps,
    contextLength = config.contextLength,
    dataWidth     = dataWidth,
    stateWidth    = stateWidth,
    accWidth      = accWidth
  ))
  physLayer.io.start        := state === launchLayer
  physLayer.io.clear        := io.clear
  physLayer.io.x            := currentX
  physLayer.io.useAttention := attentionLayerMap(activeLayerReg)
  physLayer.io.enableMoE    := enableMoEReg

  // ── Weight store (per-layer bank, selected by activeLayer) ─────────────────
  val weightStore = Module(new LayeredWeightStoreMini(config, weightDepth))
  weightStore.io.clear       := io.clear
  weightStore.io.writeValid  := io.weightWriteValid
  weightStore.io.writeAddr   := io.weightWriteAddr
  weightStore.io.writeData   := io.weightWriteData
  weightStore.io.readAddr    := io.weightReadAddr
  weightStore.io.activeLayer := activeLayerReg

  io.weightWriteReady := weightStore.io.writeReady
  io.weightReadData   := weightStore.io.readData

  // Default: demo identity-matrix weights; override when useLoadedWeights
  connectDemoWeights(physLayer)
  when(io.useLoadedWeights) {
    connectLoadedWeights(physLayer, weightStore)
  }

  // ── Token handshaking ──────────────────────────────────────────────────────
  val willConsume = outputValid && io.outReady
  val canAccept   = state === idle && (!outputValid || willConsume)
  val fire        = io.start && io.inValid && canAccept && !io.clear

  // ── FSM ────────────────────────────────────────────────────────────────────
  when(io.clear) {
    state          := idle
    activeLayerReg := 0.U
    currentX       := zeroData
    outputValid    := false.B
    outputReg      := zeroAcc
    enableMoEReg   := false.B
    doneReg        := false.B
    layerOutputsReg := VecInit(Seq.fill(numLayers)(zeroAcc))
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(fire) {
      currentX       := io.in
      activeLayerReg := 0.U
      enableMoEReg   := io.enableMoE
      outputValid    := false.B
      state          := launchLayer
    }.elsewhen(willConsume) {
      outputValid := false.B
    }
  }.elsewhen(state === launchLayer) {
    doneReg := false.B
    state   := waitLayer
  }.elsewhen(state === waitLayer) {
    doneReg := false.B
    when(physLayer.io.done) {
      // Record this logical layer's output
      layerOutputsReg(activeLayerReg) := physLayer.io.y
      // Narrow output to dataWidth for the next logical layer's input
      for (lane <- 0 until lanes) {
        currentX(lane) := SignedMath.resize(physLayer.io.y(lane), dataWidth)
      }
      when(activeLayerReg === (numLayers - 1).U) {
        // All logical layers complete
        outputReg   := physLayer.io.y
        outputValid := true.B
        doneReg     := true.B
        state       := idle
      }.otherwise {
        activeLayerReg := activeLayerReg + 1.U
        state          := launchLayer
      }
    }
  }

  io.inReady  := io.start && canAccept && !io.clear
  io.outValid := outputValid
  io.out      := outputReg
  io.busy     := state =/= idle || outputValid
  io.done     := doneReg || outputValid
  io.error    := false.B

  io.debugActiveLayer        := activeLayerReg
  io.debugLayerUsesAttention := attentionLayerMap
  io.debugLayerOutput        := layerOutputsReg

  // ── Weight helpers (mirrors UnifiedJamba2MiniFullTile) ─────────────────────

  private def connectDemoWeights(layer: UnifiedJamba2MiniLayer): Unit = {
    for (lane <- 0 until lanes) {
      layer.io.norm1Weight(lane) := 1.S(dataWidth.W)
      layer.io.norm2Weight(lane) := 1.S(dataWidth.W)

      layer.io.mambaInputBias(lane)    := 0.S(accWidth.W)
      layer.io.mambaBBias(lane)        := 2.S(accWidth.W)
      layer.io.mambaCBias(lane)        := 1.S(accWidth.W)
      layer.io.mambaA(lane)            := 1.S(dataWidth.W)
      layer.io.qBias(lane)             := 0.S(accWidth.W)
      layer.io.kBias(lane)             := 0.S(accWidth.W)
      layer.io.vBias(lane)             := 0.S(accWidth.W)
      layer.io.attentionOutBias(lane)  := 0.S(accWidth.W)
      layer.io.mlpGateBias(lane)       := 1.S(accWidth.W)
      layer.io.mlpUpBias(lane)         := 0.S(accWidth.W)
      layer.io.mlpDownBias(lane)       := 0.S(accWidth.W)
    }
    for (row <- 0 until lanes) {
      for (col <- 0 until lanes) {
        val identity        = if (row == col) 1 else 0
        val reverseIdentity = if (row == lanes - 1 - col) 1 else 0
        layer.io.mambaInputWeight(row)(col)    := identity.S(dataWidth.W)
        layer.io.mambaBWeight(row)(col)        := 0.S(dataWidth.W)
        layer.io.mambaCWeight(row)(col)        := 0.S(dataWidth.W)
        layer.io.qWeight(row)(col)             := identity.S(dataWidth.W)
        layer.io.kWeight(row)(col)             := identity.S(dataWidth.W)
        layer.io.vWeight(row)(col)             := identity.S(dataWidth.W)
        layer.io.attentionOutWeight(row)(col)  := identity.S(dataWidth.W)
        layer.io.mlpGateWeight(row)(col)       := identity.S(dataWidth.W)
        layer.io.mlpUpWeight(row)(col)         := reverseIdentity.S(dataWidth.W)
        layer.io.mlpDownWeight(row)(col)       := identity.S(dataWidth.W)
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
        layer.io.routerWeight(expert)(lane)    := routerValue.S(dataWidth.W)
        layer.io.expertGateBias(expert)(lane)  := 1.S(accWidth.W)
        layer.io.expertUpBias(expert)(lane)    := 0.S(accWidth.W)
        layer.io.expertDownBias(expert)(lane)  := expert.S(accWidth.W)
        for (col <- 0 until lanes) {
          val identity = if (lane == col) 1 else 0
          layer.io.expertGateWeight(expert)(lane)(col)  := identity.S(dataWidth.W)
          layer.io.expertUpWeight(expert)(lane)(col)    := identity.S(dataWidth.W)
          layer.io.expertDownWeight(expert)(lane)(col)  := identity.S(dataWidth.W)
        }
      }
    }
  }

  private def connectLoadedWeights(
      layer: UnifiedJamba2MiniLayer,
      store: LayeredWeightStoreMini): Unit = {
    layer.io.norm1Weight       := store.io.norm1Weight
    layer.io.norm2Weight       := store.io.norm2Weight
    layer.io.mambaInputWeight  := store.io.mambaInputWeight
    layer.io.mambaInputBias    := store.io.mambaInputBias
    layer.io.mambaBWeight      := store.io.mambaBWeight
    layer.io.mambaBBias        := store.io.mambaBBias
    layer.io.mambaCWeight      := store.io.mambaCWeight
    layer.io.mambaCBias        := store.io.mambaCBias
    layer.io.mambaA            := store.io.mambaA
    layer.io.mambaKernel       := store.io.mambaKernel
    layer.io.qWeight           := store.io.qWeight
    layer.io.qBias             := store.io.qBias
    layer.io.kWeight           := store.io.kWeight
    layer.io.kBias             := store.io.kBias
    layer.io.vWeight           := store.io.vWeight
    layer.io.vBias             := store.io.vBias
    layer.io.attentionOutWeight := store.io.attentionOutWeight
    layer.io.attentionOutBias  := store.io.attentionOutBias
    layer.io.mlpGateWeight     := store.io.mlpGateWeight
    layer.io.mlpGateBias       := store.io.mlpGateBias
    layer.io.mlpUpWeight       := store.io.mlpUpWeight
    layer.io.mlpUpBias         := store.io.mlpUpBias
    layer.io.mlpDownWeight     := store.io.mlpDownWeight
    layer.io.mlpDownBias       := store.io.mlpDownBias
    layer.io.routerWeight      := store.io.routerWeight
    layer.io.routerBias        := store.io.routerBias
    layer.io.expertGateWeight  := store.io.expertGateWeight
    layer.io.expertGateBias    := store.io.expertGateBias
    layer.io.expertUpWeight    := store.io.expertUpWeight
    layer.io.expertUpBias      := store.io.expertUpBias
    layer.io.expertDownWeight  := store.io.expertDownWeight
    layer.io.expertDownBias    := store.io.expertDownBias
  }
}
