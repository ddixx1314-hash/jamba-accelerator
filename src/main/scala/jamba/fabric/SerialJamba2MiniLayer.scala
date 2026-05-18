package jamba.fabric

import chisel3._
import chisel3.util.{Enum, log2Ceil}
import jamba.common.SignedMath
import jamba.norm.RmsNormApprox

/** Full Jamba2 mini layer with serial time-multiplexed mixer paths.
  *
  * The mixer (Mamba or Attention) is fully serial (one MAC lane reused across
  * all projections, conv, and scan). The MLP path reuses shared fabric
  * (SharedMlpPathMini). Norm layers are combinational (RmsNormApprox).
  *
  * FSM: idle → launchMixer → waitMixer → runMlp → doneState
  */
class SerialJamba2MiniLayer(
    lanes:         Int = 4,
    taps:          Int = 4,
    contextLength: Int = 4,
    dataWidth:     Int = 8,
    stateWidth:    Int = 32,
    accWidth:      Int = 32)
    extends Module {
  require(lanes == 4, "SerialJamba2MiniLayer requires lanes == 4")

  private val indexWidth = math.max(1, log2Ceil(contextLength))
  private val countWidth = log2Ceil(contextLength + 1)

  val io = IO(new Bundle {
    val start        = Input(Bool())
    val clear        = Input(Bool())
    val useAttention = Input(Bool())
    val enableMoE    = Input(Bool())
    val x            = Input(Vec(lanes, SInt(dataWidth.W)))

    val norm1Weight = Input(Vec(lanes, SInt(dataWidth.W)))
    val norm2Weight = Input(Vec(lanes, SInt(dataWidth.W)))

    val mambaInputWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaInputBias   = Input(Vec(lanes, SInt(accWidth.W)))
    val mambaBWeight     = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaBBias       = Input(Vec(lanes, SInt(accWidth.W)))
    val mambaCWeight     = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaCBias       = Input(Vec(lanes, SInt(accWidth.W)))
    val mambaA           = Input(Vec(lanes, SInt(dataWidth.W)))
    val mambaKernel      = Input(Vec(taps, Vec(lanes, SInt(dataWidth.W))))

    val qWeight            = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val qBias              = Input(Vec(lanes, SInt(accWidth.W)))
    val kWeight            = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val kBias              = Input(Vec(lanes, SInt(accWidth.W)))
    val vWeight            = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val vBias              = Input(Vec(lanes, SInt(accWidth.W)))
    val attentionOutWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val attentionOutBias   = Input(Vec(lanes, SInt(accWidth.W)))

    val mlpGateWeight    = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpGateBias      = Input(Vec(lanes, SInt(accWidth.W)))
    val mlpUpWeight      = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpUpBias        = Input(Vec(lanes, SInt(accWidth.W)))
    val mlpDownWeight    = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpDownBias      = Input(Vec(lanes, SInt(accWidth.W)))
    val routerWeight     = Input(Vec(2, Vec(lanes, SInt(dataWidth.W))))
    val routerBias       = Input(Vec(2, SInt(accWidth.W)))
    val expertGateWeight = Input(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertGateBias   = Input(Vec(2, Vec(lanes, SInt(accWidth.W))))
    val expertUpWeight   = Input(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertUpBias     = Input(Vec(2, Vec(lanes, SInt(accWidth.W))))
    val expertDownWeight = Input(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertDownBias   = Input(Vec(2, Vec(lanes, SInt(accWidth.W))))

    val ready          = Output(Bool())
    val busy           = Output(Bool())
    val done           = Output(Bool())
    val y              = Output(Vec(lanes, SInt(accWidth.W)))
    val mixerY         = Output(Vec(lanes, SInt(accWidth.W)))
    val firstResidual  = Output(Vec(lanes, SInt(dataWidth.W)))
    val mlpY           = Output(Vec(lanes, SInt(accWidth.W)))
    val stateOut       = Output(Vec(lanes, SInt(stateWidth.W)))
    val kvWriteIndex   = Output(UInt(indexWidth.W))
    val kvValidCount   = Output(UInt(countWidth.W))
    val mixerType      = Output(Bool())
    val dispatchValid  = Output(Bool())
    val combineValid   = Output(Bool())
    val selectedExpert = Output(UInt(1.W))
  })

  private def narrowToData(v: SInt): SInt = v(dataWidth - 1, 0).asSInt

  val idle :: launchMixer :: waitMixer :: runMlp :: doneState :: Nil = Enum(5)
  val state   = RegInit(idle)
  val doneReg = RegInit(false.B)

  val xReg             = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))
  val useAttentionReg  = RegInit(false.B)
  val firstResidualReg = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))
  val mixerYReg        = RegInit(VecInit(Seq.fill(lanes)(0.S(accWidth.W))))
  val mlpYReg          = RegInit(VecInit(Seq.fill(lanes)(0.S(accWidth.W))))
  val yReg             = RegInit(VecInit(Seq.fill(lanes)(0.S(accWidth.W))))

  // Norm1 is combinational: x → norm1.y (driven from io.x when computing)
  val norm1 = Module(new RmsNormApprox(lanes, dataWidth, accWidth))
  norm1.io.x      := io.x
  norm1.io.weight := io.norm1Weight

  // Serial Mamba mixer
  val mamba = Module(new SerialMambaMixerMini(lanes, taps, dataWidth, stateWidth, accWidth))
  mamba.io.start := (state === launchMixer) && !io.useAttention
  mamba.io.clear := io.clear
  mamba.io.x := norm1.io.y
  mamba.io.inputWeight := io.mambaInputWeight
  mamba.io.inputBias   := io.mambaInputBias
  mamba.io.bWeight     := io.mambaBWeight
  mamba.io.bBias       := io.mambaBBias
  mamba.io.cWeight     := io.mambaCWeight
  mamba.io.cBias       := io.mambaCBias
  mamba.io.a           := io.mambaA
  mamba.io.kernel      := io.mambaKernel

  // Serial Attention mixer
  val attention = Module(new SerialAttentionMixerMini(lanes, contextLength, dataWidth, accWidth))
  attention.io.start := (state === launchMixer) && io.useAttention
  attention.io.clear := io.clear
  attention.io.x := norm1.io.y
  attention.io.qWeight   := io.qWeight
  attention.io.qBias     := io.qBias
  attention.io.kWeight   := io.kWeight
  attention.io.kBias     := io.kBias
  attention.io.vWeight   := io.vWeight
  attention.io.vBias     := io.vBias
  attention.io.outWeight := io.attentionOutWeight
  attention.io.outBias   := io.attentionOutBias

  val mixerDone = Mux(useAttentionReg, attention.io.done, mamba.io.done)
  val mixerY    = Mux(useAttentionReg,
    VecInit(attention.io.y.map(_.asTypeOf(0.S(accWidth.W)))),
    mamba.io.y)

  // Norm2 is combinational: firstResidual → norm2.y
  val norm2 = Module(new RmsNormApprox(lanes, dataWidth, accWidth))
  norm2.io.x      := firstResidualReg
  norm2.io.weight := io.norm2Weight

  // MLP path (shared fabric, en-style control)
  val mlp = Module(new SharedMlpPathMini(lanes, 2, dataWidth, accWidth))
  mlp.io.enableMoE    := io.enableMoE
  mlp.io.x            := norm2.io.y
  mlp.io.gateWeight   := io.mlpGateWeight
  mlp.io.gateBias     := io.mlpGateBias
  mlp.io.upWeight     := io.mlpUpWeight
  mlp.io.upBias       := io.mlpUpBias
  mlp.io.downWeight   := io.mlpDownWeight
  mlp.io.downBias     := io.mlpDownBias
  mlp.io.routerWeight := io.routerWeight
  mlp.io.routerBias   := io.routerBias
  mlp.io.expertGateWeight := io.expertGateWeight
  mlp.io.expertGateBias   := io.expertGateBias
  mlp.io.expertUpWeight   := io.expertUpWeight
  mlp.io.expertUpBias     := io.expertUpBias
  mlp.io.expertDownWeight := io.expertDownWeight
  mlp.io.expertDownBias   := io.expertDownBias
  mlp.io.dispatchReady := true.B
  mlp.io.combineReady  := true.B

  when(io.clear) {
    state            := idle
    doneReg          := false.B
    xReg             := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    useAttentionReg  := false.B
    firstResidualReg := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    mixerYReg        := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
    mlpYReg          := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
    yReg             := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      xReg            := io.x
      useAttentionReg := io.useAttention
      state           := launchMixer
    }
  }.elsewhen(state === launchMixer) {
    doneReg := false.B
    state   := waitMixer
  }.elsewhen(state === waitMixer) {
    doneReg := false.B
    when(mixerDone) {
      mixerYReg := mixerY
      for (lane <- 0 until lanes) {
        firstResidualReg(lane) := narrowToData(xReg(lane) + mixerY(lane))
      }
      state := runMlp
    }
  }.elsewhen(state === runMlp) {
    // MLP is combinational (en-style SharedMlpPathMini fires one cycle)
    mlpYReg := mlp.io.y
    for (lane <- 0 until lanes) {
      yReg(lane) := firstResidualReg(lane) + mlp.io.y(lane)
    }
    state := doneState
  }.elsewhen(state === doneState) {
    doneReg := true.B
    state   := idle
  }

  io.ready         := state === idle
  io.busy          := state =/= idle
  io.done          := doneReg
  io.y             := yReg
  io.mixerY        := mixerYReg
  io.firstResidual := firstResidualReg
  io.mlpY          := mlpYReg
  io.stateOut      := mamba.io.stateOut
  io.kvWriteIndex  := attention.io.kvWriteIndex
  io.kvValidCount  := attention.io.kvValidCount
  io.mixerType     := useAttentionReg
  io.dispatchValid := mlp.io.dispatchValid
  io.combineValid  := mlp.io.combineValid
  io.selectedExpert := mlp.io.selectedExpert
}
