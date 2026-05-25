package jamba.top

import chisel3._
import chisel3.util.{Enum, log2Ceil}
import jamba.common.{Jamba2MiniConfig, SignedMath}
import jamba.fabric.UnifiedJamba2MiniLayer
import jamba.memory.LayeredWeightStoreMini

/** Single-physical-layer tile with per-layer state virtualization (M7-B).
  *
  * L logical layers share ONE UnifiedJamba2MiniLayer instance. Before each
  * logical layer the tile restores that layer's SSM hidden state, causal-conv
  * history, and KV cache from the per-layer state file. After the layer
  * completes the tile saves the updated state back.
  *
  * Resource effect: instance-weighted mul-proxy ≈ 92 (constant) regardless of
  * numLayers. The state file adds O(L × stateSize) registers at the tile level
  * rather than replicated compute fabric.
  *
  * M7-A structural limitation resolved: SSM state and KV cache are now
  * correctly virtualized per logical layer across tokens.
  */
class SinglePhysicalLayerTile(
    config:        Jamba2MiniConfig = Jamba2MiniConfig.debug,
    weightDepth:   Int              = 256,
    vectorBypass:  Boolean          = false)
    extends Module {
  require(config.lanes > 0, "SinglePhysicalLayerTile lanes must be positive")
  require(config.numLayers > 0, "SinglePhysicalLayerTile needs at least one layer")
  require(config.convTaps > 1, "SinglePhysicalLayerTile requires convTaps > 1 for conv history")
  require(weightDepth > 0, "SinglePhysicalLayerTile weightDepth must be positive")

  private val lanes           = config.lanes
  private val dataWidth       = config.dataWidth
  private val accWidth        = config.accWidth
  private val stateWidth      = config.ssmStateBits
  private val numLayers       = config.numLayers
  private val contextLength   = config.contextLength
  private val convTaps        = config.convTaps
  private val historyDepth    = convTaps - 1
  private val layerIndexWidth = math.max(1, log2Ceil(numLayers))
  private val indexWidth      = math.max(1, log2Ceil(contextLength))
  private val countWidth      = log2Ceil(contextLength + 1)
  private val addrWidth       = math.max(1, log2Ceil(weightDepth))

  val io = IO(new Bundle {
    val clear            = Input(Bool())
    val start            = Input(Bool())
    val enableMoE        = Input(Bool())
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
    // M10-D: total projection slots bypassed across all logical layers for the last token.
    // Always 0 when vectorBypass = false.
    val debugProjectionBypassCount = Output(UInt(8.W))
  })

  private def zeroData    = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroAcc     = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  private def zeroState   = VecInit(Seq.fill(lanes)(0.S(stateWidth.W)))
  private def zeroHistory = VecInit(Seq.fill(historyDepth)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def zeroCache   = VecInit(Seq.fill(contextLength)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))

  // Compile-time attention-layer lookup ROM
  private val attentionLayerMap =
    VecInit((0 until numLayers).map(i => config.isAttentionLayer(i).B))

  // ── FSM ────────────────────────────────────────────────────────────────────
  val idle :: restoreState :: launchLayer :: waitLayer :: Nil = Enum(4)
  val state               = RegInit(idle)
  val activeLayerReg      = RegInit(0.U(layerIndexWidth.W))
  val currentX            = RegInit(zeroData)
  val outputValid         = RegInit(false.B)
  val outputReg           = RegInit(zeroAcc)
  val doneReg             = RegInit(false.B)
  val enableMoEReg        = RegInit(false.B)
  val useLoadedWeightsReg = RegInit(false.B)
  val layerOutputsReg     = RegInit(VecInit(Seq.fill(numLayers)(zeroAcc)))

  // ── M10-D: token-level bypass accumulator ─────────────────────────────────
  // Accumulates projectionBypassCount across all logical layers for one token.
  // Resets to 0 when a new token fires.  Saturates at 255.
  val tokenBypassReg = RegInit(0.U(8.W))

  // ── Per-layer state file (M7-B) ─────────────────────────────────────────────
  val ssmStateFile   = RegInit(VecInit(Seq.fill(numLayers)(zeroState)))
  val convHistFile   = RegInit(VecInit(Seq.fill(numLayers)(zeroHistory)))
  val keyCacheFile   = RegInit(VecInit(Seq.fill(numLayers)(zeroCache)))
  val valueCacheFile = RegInit(VecInit(Seq.fill(numLayers)(zeroCache)))
  val kvWriteIdxFile = RegInit(VecInit(Seq.fill(numLayers)(0.U(indexWidth.W))))
  val kvValidCntFile = RegInit(VecInit(Seq.fill(numLayers)(0.U(countWidth.W))))

  // ── ONE physical layer ─────────────────────────────────────────────────────
  val physLayer = Module(new UnifiedJamba2MiniLayer(
    lanes              = lanes,
    taps               = convTaps,
    contextLength      = contextLength,
    dataWidth          = dataWidth,
    stateWidth         = stateWidth,
    accWidth           = accWidth,
    projectionMacLanes = config.projectionMacLanes,
    vectorBypass       = vectorBypass
  ))
  physLayer.io.start        := state === launchLayer
  physLayer.io.clear        := io.clear
  physLayer.io.x            := currentX
  // Static branch avoids "dynamic index too wide for Vec of size 1" when numLayers==1
  physLayer.io.useAttention :=
    (if (numLayers == 1) attentionLayerMap(0) else attentionLayerMap(activeLayerReg))
  physLayer.io.enableMoE    := enableMoEReg

  // State restore: driven in restoreState cycle; index safe for numLayers==1
  physLayer.io.loadState  := state === restoreState
  physLayer.io.loadHistory := state === restoreState
  physLayer.io.loadKvState := state === restoreState
  physLayer.io.stateIn    :=
    (if (numLayers == 1) ssmStateFile(0) else ssmStateFile(activeLayerReg))
  physLayer.io.historyIn  :=
    (if (numLayers == 1) convHistFile(0) else convHistFile(activeLayerReg))
  physLayer.io.keyCacheIn :=
    (if (numLayers == 1) keyCacheFile(0) else keyCacheFile(activeLayerReg))
  physLayer.io.valueCacheIn :=
    (if (numLayers == 1) valueCacheFile(0) else valueCacheFile(activeLayerReg))
  physLayer.io.kvWriteIndexIn :=
    (if (numLayers == 1) kvWriteIdxFile(0) else kvWriteIdxFile(activeLayerReg))
  physLayer.io.kvValidCountIn :=
    (if (numLayers == 1) kvValidCntFile(0) else kvValidCntFile(activeLayerReg))

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

  // Latched at token-fire time so weight source is stable across all logical layers
  connectDemoWeights(physLayer)
  when(useLoadedWeightsReg) {
    connectLoadedWeights(physLayer, weightStore)
  }

  // ── Token handshaking ──────────────────────────────────────────────────────
  val willConsume = outputValid && io.outReady
  val canAccept   = state === idle && (!outputValid || willConsume)
  val fire        = io.start && io.inValid && canAccept && !io.clear

  // ── FSM logic ──────────────────────────────────────────────────────────────
  when(io.clear) {
    state               := idle
    activeLayerReg      := 0.U
    currentX            := zeroData
    outputValid         := false.B
    outputReg           := zeroAcc
    enableMoEReg        := false.B
    useLoadedWeightsReg := false.B
    doneReg             := false.B
    tokenBypassReg      := 0.U
    layerOutputsReg     := VecInit(Seq.fill(numLayers)(zeroAcc))
    ssmStateFile        := VecInit(Seq.fill(numLayers)(zeroState))
    convHistFile        := VecInit(Seq.fill(numLayers)(zeroHistory))
    keyCacheFile        := VecInit(Seq.fill(numLayers)(zeroCache))
    valueCacheFile      := VecInit(Seq.fill(numLayers)(zeroCache))
    kvWriteIdxFile      := VecInit(Seq.fill(numLayers)(0.U(indexWidth.W)))
    kvValidCntFile      := VecInit(Seq.fill(numLayers)(0.U(countWidth.W)))
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(fire) {
      currentX            := io.in
      activeLayerReg      := 0.U
      enableMoEReg        := io.enableMoE
      useLoadedWeightsReg := io.useLoadedWeights
      outputValid         := false.B
      tokenBypassReg      := 0.U   // M10-D: reset per-token accumulator
      state               := restoreState      // restore layer 0 state before launching
    }.elsewhen(willConsume) {
      outputValid := false.B
    }
  }.elsewhen(state === restoreState) {
    // One cycle: load signals are driven combinatorially above; state registers
    // in scan/conv/layer update at this clock edge; next cycle is launchLayer.
    doneReg := false.B
    state   := launchLayer
  }.elsewhen(state === launchLayer) {
    doneReg := false.B
    state   := waitLayer
  }.elsewhen(state === waitLayer) {
    doneReg := false.B
    when(physLayer.io.done) {
      // Record this logical layer's output
      if (numLayers == 1) {
        layerOutputsReg(0) := physLayer.io.y
      } else {
        layerOutputsReg(activeLayerReg) := physLayer.io.y
      }

      // Save current layer's state back to state file
      if (numLayers == 1) {
        ssmStateFile(0)   := physLayer.io.stateOut
        convHistFile(0)   := physLayer.io.historyOut
        keyCacheFile(0)   := physLayer.io.keyCacheOut
        valueCacheFile(0) := physLayer.io.valueCacheOut
        kvWriteIdxFile(0) := physLayer.io.kvWriteIndex
        kvValidCntFile(0) := physLayer.io.kvValidCount
      } else {
        ssmStateFile(activeLayerReg)   := physLayer.io.stateOut
        convHistFile(activeLayerReg)   := physLayer.io.historyOut
        keyCacheFile(activeLayerReg)   := physLayer.io.keyCacheOut
        valueCacheFile(activeLayerReg) := physLayer.io.valueCacheOut
        kvWriteIdxFile(activeLayerReg) := physLayer.io.kvWriteIndex
        kvValidCntFile(activeLayerReg) := physLayer.io.kvValidCount
      }

      // M10-D: accumulate bypass count from this logical layer (saturating add)
      tokenBypassReg := Mux(
        tokenBypassReg +& physLayer.io.projectionBypassCount >= 255.U,
        255.U,
        tokenBypassReg + physLayer.io.projectionBypassCount
      )

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
        state          := restoreState    // restore next layer before launching it
      }
    }
  }

  io.inReady  := io.start && canAccept && !io.clear
  io.outValid := outputValid
  io.out      := outputReg
  io.busy     := state =/= idle || outputValid
  io.done     := doneReg || outputValid
  io.error    := false.B

  io.debugActiveLayer            := activeLayerReg
  io.debugLayerUsesAttention     := attentionLayerMap
  io.debugLayerOutput            := layerOutputsReg
  io.debugProjectionBypassCount  := tokenBypassReg

  // ── Weight helpers ──────────────────────────────────────────────────────────

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
    for (tap <- 0 until convTaps) {
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
