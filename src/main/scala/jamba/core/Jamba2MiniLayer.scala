package jamba.core

import chisel3._
import jamba.attention.AttentionMixerMini
import jamba.mamba.Jamba2MambaMixerMini
import jamba.norm.RmsNormApprox

/** One formal Jamba2 mini layer: RMSNorm -> Mixer -> residual -> RMSNorm -> MLP -> residual. */
class Jamba2MiniLayer(lanes: Int = 4, taps: Int = 4, contextLength: Int = 4, dataWidth: Int = 8, stateWidth: Int = 32, accWidth: Int = 32)
    extends Module {
  require(lanes == 4, "Jamba2MiniLayer currently uses Linear4-based children and requires lanes == 4")

  val io = IO(new Bundle {
    val en = Input(Bool())
    val clear = Input(Bool())
    val useAttention = Input(Bool())
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
    val mambaKernel = Input(Vec(taps, Vec(lanes, SInt(dataWidth.W))))

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
    val mixerY = Output(Vec(lanes, SInt(accWidth.W)))
    val firstResidual = Output(Vec(lanes, SInt(dataWidth.W)))
    val mlpY = Output(Vec(lanes, SInt(accWidth.W)))
    val stateOut = Output(Vec(lanes, SInt(stateWidth.W)))
    val kvWriteIndex = Output(UInt(math.max(1, chisel3.util.log2Ceil(contextLength)).W))
    val kvValidCount = Output(UInt(chisel3.util.log2Ceil(contextLength + 1).W))
    val mixerType = Output(Bool())
    val dispatchValid = Output(Bool())
    val combineValid = Output(Bool())
    val selectedExpert = Output(UInt(1.W))
  })

  private def narrowToData(value: SInt): SInt = value(dataWidth - 1, 0).asSInt

  val norm1 = Module(new RmsNormApprox(lanes, dataWidth, accWidth))
  norm1.io.x := io.x
  norm1.io.weight := io.norm1Weight

  val mamba = Module(new Jamba2MambaMixerMini(lanes, taps, dataWidth, stateWidth, accWidth))
  mamba.io.en := io.en && !io.useAttention
  mamba.io.clear := io.clear
  mamba.io.x := norm1.io.y
  mamba.io.inputWeight := io.mambaInputWeight
  mamba.io.inputBias := io.mambaInputBias
  mamba.io.bWeight := io.mambaBWeight
  mamba.io.bBias := io.mambaBBias
  mamba.io.cWeight := io.mambaCWeight
  mamba.io.cBias := io.mambaCBias
  mamba.io.a := io.mambaA
  mamba.io.kernel := io.mambaKernel

  val attention = Module(new AttentionMixerMini(lanes, contextLength, dataWidth, accWidth))
  attention.io.en := io.en && io.useAttention
  attention.io.clear := io.clear
  attention.io.x := norm1.io.y
  attention.io.qWeight := io.qWeight
  attention.io.qBias := io.qBias
  attention.io.kWeight := io.kWeight
  attention.io.kBias := io.kBias
  attention.io.vWeight := io.vWeight
  attention.io.vBias := io.vBias
  attention.io.outWeight := io.attentionOutWeight
  attention.io.outBias := io.attentionOutBias

  for (lane <- 0 until lanes) {
    io.mixerY(lane) := Mux(io.useAttention, attention.io.y(lane), mamba.io.y(lane))
    io.firstResidual(lane) := narrowToData(io.x(lane) + io.mixerY(lane))
  }

  val norm2 = Module(new RmsNormApprox(lanes, dataWidth, accWidth))
  norm2.io.x := io.firstResidual
  norm2.io.weight := io.norm2Weight

  val mlp = Module(new MlpPathMini(lanes, 2, dataWidth, accWidth))
  mlp.io.enableMoE := io.enableMoE
  mlp.io.x := norm2.io.y
  mlp.io.gateWeight := io.mlpGateWeight
  mlp.io.gateBias := io.mlpGateBias
  mlp.io.upWeight := io.mlpUpWeight
  mlp.io.upBias := io.mlpUpBias
  mlp.io.downWeight := io.mlpDownWeight
  mlp.io.downBias := io.mlpDownBias
  mlp.io.routerWeight := io.routerWeight
  mlp.io.routerBias := io.routerBias
  mlp.io.expertGateWeight := io.expertGateWeight
  mlp.io.expertGateBias := io.expertGateBias
  mlp.io.expertUpWeight := io.expertUpWeight
  mlp.io.expertUpBias := io.expertUpBias
  mlp.io.expertDownWeight := io.expertDownWeight
  mlp.io.expertDownBias := io.expertDownBias
  mlp.io.dispatchReady := true.B
  mlp.io.combineReady := true.B

  for (lane <- 0 until lanes) {
    io.mlpY(lane) := mlp.io.y(lane)
    io.y(lane) := io.firstResidual(lane) + io.mlpY(lane)
  }

  io.stateOut := mamba.io.stateOut
  io.kvWriteIndex := attention.io.kvWriteIndex
  io.kvValidCount := attention.io.kvValidCount
  io.mixerType := io.useAttention
  io.dispatchValid := mlp.io.dispatchValid
  io.combineValid := mlp.io.combineValid
  io.selectedExpert := mlp.io.selectedExpert
}
