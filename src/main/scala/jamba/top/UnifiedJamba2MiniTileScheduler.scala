package jamba.top

import chisel3._
import chisel3.util.{Enum, log2Ceil}
import jamba.common.{Jamba2MiniConfig, SignedMath}
import jamba.fabric.UnifiedJamba2MiniLayer

/** Sequential multi-layer scheduler for UnifiedJamba2MiniLayer.
  *
  * This module proves that the unified layer can execute a mini Jamba-style
  * layer stack. Each logical layer owns its own state/cache, and the scheduler
  * launches one layer at a time.
  */
class UnifiedJamba2MiniTileScheduler(config: Jamba2MiniConfig = Jamba2MiniConfig.debug) extends Module {
  require(config.lanes == 4, "UnifiedJamba2MiniTileScheduler currently supports 4 lanes")
  require(config.numLayers > 0, "UnifiedJamba2MiniTileScheduler needs at least one layer")

  private val lanes = config.lanes
  private val dataWidth = config.dataWidth
  private val accWidth = config.accWidth
  private val stateWidth = config.ssmStateBits
  private val numLayers = config.numLayers
  private val layerIndexWidth = math.max(1, log2Ceil(numLayers))
  private val kvIndexWidth = math.max(1, log2Ceil(config.contextLength))
  private val kvCountWidth = log2Ceil(config.contextLength + 1)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val enableMoE = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))

    val norm1Weight = Input(Vec(lanes, SInt(dataWidth.W)))
    val norm2Weight = Input(Vec(lanes, SInt(dataWidth.W)))

    val mambaInputWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaInputBias = Input(Vec(lanes, SInt(accWidth.W)))
    val mambaBWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaBBias = Input(Vec(lanes, SInt(accWidth.W)))
    val mambaCWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaCBias = Input(Vec(lanes, SInt(accWidth.W)))
    val mambaA = Input(Vec(lanes, SInt(dataWidth.W)))
    val mambaKernel = Input(Vec(config.convTaps, Vec(lanes, SInt(dataWidth.W))))

    val qWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val qBias = Input(Vec(lanes, SInt(accWidth.W)))
    val kWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val kBias = Input(Vec(lanes, SInt(accWidth.W)))
    val vWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val vBias = Input(Vec(lanes, SInt(accWidth.W)))
    val attentionOutWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val attentionOutBias = Input(Vec(lanes, SInt(accWidth.W)))

    val mlpGateWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpGateBias = Input(Vec(lanes, SInt(accWidth.W)))
    val mlpUpWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpUpBias = Input(Vec(lanes, SInt(accWidth.W)))
    val mlpDownWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpDownBias = Input(Vec(lanes, SInt(accWidth.W)))
    val routerWeight = Input(Vec(2, Vec(lanes, SInt(dataWidth.W))))
    val routerBias = Input(Vec(2, SInt(accWidth.W)))
    val expertGateWeight = Input(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertGateBias = Input(Vec(2, Vec(lanes, SInt(accWidth.W))))
    val expertUpWeight = Input(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertUpBias = Input(Vec(2, Vec(lanes, SInt(accWidth.W))))
    val expertDownWeight = Input(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertDownBias = Input(Vec(2, Vec(lanes, SInt(accWidth.W))))

    val ready = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val y = Output(Vec(lanes, SInt(accWidth.W)))
    val activeLayer = Output(UInt(layerIndexWidth.W))
    val layerUsesAttention = Output(Vec(numLayers, Bool()))
    val layerOutputs = Output(Vec(numLayers, Vec(lanes, SInt(accWidth.W))))
    val layerStateOut = Output(Vec(numLayers, Vec(lanes, SInt(stateWidth.W))))
    val layerKvWriteIndex = Output(Vec(numLayers, UInt(kvIndexWidth.W)))
    val layerKvValidCount = Output(Vec(numLayers, UInt(kvCountWidth.W)))
    val layerSelectedExpert = Output(Vec(numLayers, UInt(1.W)))
  })

  private def zeroData = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroAcc = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))

  val idle :: launchLayer :: waitLayer :: doneState :: Nil = Enum(4)
  val state = RegInit(idle)
  val layerIndex = RegInit(0.U(layerIndexWidth.W))
  val currentX = RegInit(zeroData)
  val yReg = RegInit(zeroAcc)
  val doneReg = RegInit(false.B)
  val layerOutputsReg = RegInit(VecInit(Seq.fill(numLayers)(zeroAcc)))

  val layers = Seq.tabulate(numLayers) { _ =>
    Module(new UnifiedJamba2MiniLayer(
      lanes = lanes,
      taps = config.convTaps,
      contextLength = config.contextLength,
      dataWidth = dataWidth,
      stateWidth = stateWidth,
      accWidth = accWidth
    ))
  }

  for (layerIdx <- 0 until numLayers) {
    val layer = layers(layerIdx)

    layer.io.start := state === launchLayer && layerIndex === layerIdx.U
    layer.io.clear := io.clear
    layer.io.useAttention := config.isAttentionLayer(layerIdx).B
    layer.io.enableMoE := io.enableMoE
    layer.io.x := currentX

    layer.io.norm1Weight := io.norm1Weight
    layer.io.norm2Weight := io.norm2Weight
    layer.io.mambaInputWeight := io.mambaInputWeight
    layer.io.mambaInputBias := io.mambaInputBias
    layer.io.mambaBWeight := io.mambaBWeight
    layer.io.mambaBBias := io.mambaBBias
    layer.io.mambaCWeight := io.mambaCWeight
    layer.io.mambaCBias := io.mambaCBias
    layer.io.mambaA := io.mambaA
    layer.io.mambaKernel := io.mambaKernel
    layer.io.qWeight := io.qWeight
    layer.io.qBias := io.qBias
    layer.io.kWeight := io.kWeight
    layer.io.kBias := io.kBias
    layer.io.vWeight := io.vWeight
    layer.io.vBias := io.vBias
    layer.io.attentionOutWeight := io.attentionOutWeight
    layer.io.attentionOutBias := io.attentionOutBias
    layer.io.mlpGateWeight := io.mlpGateWeight
    layer.io.mlpGateBias := io.mlpGateBias
    layer.io.mlpUpWeight := io.mlpUpWeight
    layer.io.mlpUpBias := io.mlpUpBias
    layer.io.mlpDownWeight := io.mlpDownWeight
    layer.io.mlpDownBias := io.mlpDownBias
    layer.io.routerWeight := io.routerWeight
    layer.io.routerBias := io.routerBias
    layer.io.expertGateWeight := io.expertGateWeight
    layer.io.expertGateBias := io.expertGateBias
    layer.io.expertUpWeight := io.expertUpWeight
    layer.io.expertUpBias := io.expertUpBias
    layer.io.expertDownWeight := io.expertDownWeight
    layer.io.expertDownBias := io.expertDownBias

    io.layerUsesAttention(layerIdx) := config.isAttentionLayer(layerIdx).B
    io.layerOutputs(layerIdx) := layerOutputsReg(layerIdx)
    io.layerStateOut(layerIdx) := layer.io.stateOut
    io.layerKvWriteIndex(layerIdx) := layer.io.kvWriteIndex
    io.layerKvValidCount(layerIdx) := layer.io.kvValidCount
    io.layerSelectedExpert(layerIdx) := layer.io.selectedExpert
  }

  when(io.clear) {
    state := idle
    layerIndex := 0.U
    currentX := zeroData
    yReg := zeroAcc
    doneReg := false.B
    layerOutputsReg := VecInit(Seq.fill(numLayers)(zeroAcc))
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      currentX := io.x
      layerIndex := 0.U
      state := launchLayer
    }
  }.elsewhen(state === launchLayer) {
    doneReg := false.B
    state := waitLayer
  }.elsewhen(state === waitLayer) {
    doneReg := false.B
    for (layerIdx <- 0 until numLayers) {
      when(layerIndex === layerIdx.U && layers(layerIdx).io.done) {
        layerOutputsReg(layerIdx) := layers(layerIdx).io.y
        yReg := layers(layerIdx).io.y
        for (lane <- 0 until lanes) {
          currentX(lane) := SignedMath.resize(layers(layerIdx).io.y(lane), dataWidth)
        }
        when(layerIndex === (numLayers - 1).U) {
          state := doneState
        }.otherwise {
          layerIndex := layerIndex + 1.U
          state := launchLayer
        }
      }
    }
  }.elsewhen(state === doneState) {
    doneReg := true.B
    state := idle
  }

  io.ready := state === idle
  io.busy := state =/= idle
  io.done := doneReg
  io.y := yReg
  io.activeLayer := layerIndex
}
