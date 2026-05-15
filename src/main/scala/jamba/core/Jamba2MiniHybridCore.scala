package jamba.core

import chisel3._
import jamba.common.{Jamba2MiniConfig, SignedMath}

/** Multi-layer Jamba2 mini core scheduler with sparse Attention layers and Mamba-major layers. */
class Jamba2MiniHybridCore(config: Jamba2MiniConfig = Jamba2MiniConfig.debug) extends Module {
  require(config.lanes == 4, "Jamba2MiniHybridCore currently requires lanes == 4")
  require(config.convTaps > 0, "Jamba2MiniHybridCore convTaps must be positive")

  private val lanes = config.lanes
  private val dataWidth = config.dataWidth
  private val accWidth = config.accWidth
  private val stateWidth = config.ssmStateBits
  private val numLayers = config.numLayers

  val io = IO(new Bundle {
    val en = Input(Bool())
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

    val y = Output(Vec(lanes, SInt(accWidth.W)))
    val valid = Output(Bool())
    val layerUsesAttention = Output(Vec(numLayers, Bool()))
    val layerOutputs = Output(Vec(numLayers, Vec(lanes, SInt(accWidth.W))))
    val layerStateOut = Output(Vec(numLayers, Vec(lanes, SInt(stateWidth.W))))
    val layerSelectedExpert = Output(Vec(numLayers, UInt(1.W)))
  })

  val layers = Seq.tabulate(numLayers) { layerIndex =>
    Module(new Jamba2MiniLayer(lanes, config.convTaps, config.contextLength, dataWidth, stateWidth, accWidth))
  }

  for (layerIndex <- 0 until numLayers) {
    val layer = layers(layerIndex)
    val useAttention = config.isAttentionLayer(layerIndex).B

    layer.io.en := io.en
    layer.io.clear := io.clear
    layer.io.useAttention := useAttention
    layer.io.enableMoE := io.enableMoE

    if (layerIndex == 0) {
      layer.io.x := io.x
    } else {
      for (lane <- 0 until lanes) {
        layer.io.x(lane) := SignedMath.resize(layers(layerIndex - 1).io.y(lane), dataWidth)
      }
    }

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

    io.layerUsesAttention(layerIndex) := useAttention
    io.layerOutputs(layerIndex) := layer.io.y
    io.layerStateOut(layerIndex) := layer.io.stateOut
    io.layerSelectedExpert(layerIndex) := layer.io.selectedExpert
  }

  io.y := layers.last.io.y

  val validReg = RegInit(false.B)
  when(io.clear) {
    validReg := false.B
  }.otherwise {
    validReg := io.en
  }
  io.valid := validReg
}
