package jamba.memory

import chisel3._
import chisel3.util.{Enum, log2Ceil}
import jamba.common.Jamba2MiniConfig

/** Accumulates a complete weight field from an element stream into typed registers.
  *
  * Accepts element-by-element input from SequentialWeightCaptureMini and routes
  * each element into the correct typed register slot using the position metadata
  * (row/col/lane/tap/expert). When numElements elements have been received,
  * asserts done for one cycle and holds the registers stable until the next start.
  *
  * Only one field is buffered at a time. The fieldId input selects which typed
  * output register group is populated; all other groups retain their previous
  * values or zero after reset.
  */
class FieldWeightBufferMini(
    config: Jamba2MiniConfig = Jamba2MiniConfig.debug
) extends Module {
  require(config.lanes == 4, "FieldWeightBufferMini currently supports 4 lanes")

  private val lanes = config.lanes
  private val dataWidth = config.dataWidth
  private val accWidth = config.accWidth
  private val fieldWidth = math.max(1, log2Ceil(WeightAddressGenMini.NumFields + 1))
  private val maxElements = Seq(lanes * lanes, config.convTaps * lanes, 2 * lanes, 2 * lanes * lanes).max
  private val elementWidth = math.max(1, log2Ceil(maxElements))
  private val countWidth = math.max(1, log2Ceil(maxElements + 1))
  private val laneWidth = math.max(1, log2Ceil(lanes))
  private val tapWidth = math.max(1, log2Ceil(config.convTaps))

  val io = IO(new Bundle {
    val clear = Input(Bool())
    val start = Input(Bool())
    val fieldId = Input(UInt(fieldWidth.W))

    val inValid = Input(Bool())
    val inReady = Output(Bool())
    val inData = Input(SInt(accWidth.W))
    val inIsAcc = Input(Bool())
    val inElementIndex = Input(UInt(elementWidth.W))
    val inNumElements = Input(UInt(countWidth.W))
    val inRow = Input(UInt(laneWidth.W))
    val inCol = Input(UInt(laneWidth.W))
    val inLane = Input(UInt(laneWidth.W))
    val inTap = Input(UInt(tapWidth.W))
    val inExpert = Input(UInt(1.W))

    val done = Output(Bool())
    val busy = Output(Bool())

    // Typed output registers — always valid after done
    val dataVec = Output(Vec(lanes, SInt(dataWidth.W)))
    val accVec = Output(Vec(lanes, SInt(accWidth.W)))
    val dataMatrix = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val kernelOut = Output(Vec(config.convTaps, Vec(lanes, SInt(dataWidth.W))))
    val routerWeightOut = Output(Vec(2, Vec(lanes, SInt(dataWidth.W))))
    val routerBiasOut = Output(Vec(2, SInt(accWidth.W)))
    val expertMatrix = Output(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertAccVec = Output(Vec(2, Vec(lanes, SInt(accWidth.W))))
  })

  private def zeroDataVec = VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
  private def zeroAccVec = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  private def zeroDataMatrix = VecInit(Seq.fill(lanes)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def zeroKernel =
    VecInit(Seq.fill(config.convTaps)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def zeroRouterWeight = VecInit(Seq.fill(2)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def zeroRouterBias = VecInit(Seq.fill(2)(0.S(accWidth.W)))
  private def zeroExpertMatrix =
    VecInit(Seq.fill(2)(VecInit(Seq.fill(lanes)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))))
  private def zeroExpertAccVec = VecInit(Seq.fill(2)(VecInit(Seq.fill(lanes)(0.S(accWidth.W)))))

  val idle :: filling :: doneState :: Nil = Enum(3)
  val state = RegInit(idle)
  val doneReg = RegInit(false.B)
  val fieldIdReg = RegInit(0.U(fieldWidth.W))
  val received = RegInit(0.U(countWidth.W))
  val numElements = RegInit(0.U(countWidth.W))

  val dataVecReg = RegInit(zeroDataVec)
  val accVecReg = RegInit(zeroAccVec)
  val dataMatrixReg = RegInit(zeroDataMatrix)
  val kernelReg = RegInit(zeroKernel)
  val routerWeightReg = RegInit(zeroRouterWeight)
  val routerBiasReg = RegInit(zeroRouterBias)
  val expertMatrixReg = RegInit(zeroExpertMatrix)
  val expertAccVecReg = RegInit(zeroExpertAccVec)

  private def dataWrite(raw: SInt): SInt = raw(dataWidth - 1, 0).asSInt

  val fire = io.inValid && io.inReady

  when(io.clear) {
    state := idle
    doneReg := false.B
    received := 0.U
    numElements := 0.U
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      fieldIdReg := io.fieldId
      received := 0.U
      state := filling
    }
  }.elsewhen(state === filling) {
    doneReg := false.B
    when(fire) {
      numElements := io.inNumElements
      received := received + 1.U

      // Use fieldIdReg (latched at start) so a changing io.fieldId mid-fill
      // cannot redirect writes to the wrong register bank.
      val isVec = fieldIdReg === WeightAddressGenMini.Norm1Weight.U ||
        fieldIdReg === WeightAddressGenMini.Norm2Weight.U ||
        fieldIdReg === WeightAddressGenMini.MambaA.U

      val isAccVec = fieldIdReg === WeightAddressGenMini.MambaInputBias.U ||
        fieldIdReg === WeightAddressGenMini.MambaBBias.U ||
        fieldIdReg === WeightAddressGenMini.MambaCBias.U ||
        fieldIdReg === WeightAddressGenMini.QBias.U ||
        fieldIdReg === WeightAddressGenMini.KBias.U ||
        fieldIdReg === WeightAddressGenMini.VBias.U ||
        fieldIdReg === WeightAddressGenMini.AttentionOutBias.U ||
        fieldIdReg === WeightAddressGenMini.MlpGateBias.U ||
        fieldIdReg === WeightAddressGenMini.MlpUpBias.U ||
        fieldIdReg === WeightAddressGenMini.MlpDownBias.U

      val isMatrix = fieldIdReg === WeightAddressGenMini.MambaInputWeight.U ||
        fieldIdReg === WeightAddressGenMini.MambaBWeight.U ||
        fieldIdReg === WeightAddressGenMini.MambaCWeight.U ||
        fieldIdReg === WeightAddressGenMini.QWeight.U ||
        fieldIdReg === WeightAddressGenMini.KWeight.U ||
        fieldIdReg === WeightAddressGenMini.VWeight.U ||
        fieldIdReg === WeightAddressGenMini.AttentionOutWeight.U ||
        fieldIdReg === WeightAddressGenMini.MlpGateWeight.U ||
        fieldIdReg === WeightAddressGenMini.MlpUpWeight.U ||
        fieldIdReg === WeightAddressGenMini.MlpDownWeight.U

      val isKernel = fieldIdReg === WeightAddressGenMini.MambaKernel.U
      val isRouterWeight = fieldIdReg === WeightAddressGenMini.RouterWeight.U
      val isRouterBias = fieldIdReg === WeightAddressGenMini.RouterBias.U

      val isExpertMatrix = fieldIdReg === WeightAddressGenMini.ExpertGateWeight.U ||
        fieldIdReg === WeightAddressGenMini.ExpertUpWeight.U ||
        fieldIdReg === WeightAddressGenMini.ExpertDownWeight.U

      val isExpertAccVec = fieldIdReg === WeightAddressGenMini.ExpertGateBias.U ||
        fieldIdReg === WeightAddressGenMini.ExpertUpBias.U ||
        fieldIdReg === WeightAddressGenMini.ExpertDownBias.U

      when(isVec) {
        dataVecReg(io.inLane) := dataWrite(io.inData)
      }.elsewhen(isAccVec) {
        accVecReg(io.inLane) := io.inData
      }.elsewhen(isMatrix) {
        dataMatrixReg(io.inRow)(io.inCol) := dataWrite(io.inData)
      }.elsewhen(isKernel) {
        kernelReg(io.inTap)(io.inLane) := dataWrite(io.inData)
      }.elsewhen(isRouterWeight) {
        routerWeightReg(io.inExpert)(io.inLane) := dataWrite(io.inData)
      }.elsewhen(isRouterBias) {
        routerBiasReg(io.inExpert) := io.inData
      }.elsewhen(isExpertMatrix) {
        expertMatrixReg(io.inExpert)(io.inRow)(io.inCol) := dataWrite(io.inData)
      }.elsewhen(isExpertAccVec) {
        expertAccVecReg(io.inExpert)(io.inLane) := io.inData
      }

      when(received + 1.U >= io.inNumElements) {
        state := doneState
      }
    }
  }.elsewhen(state === doneState) {
    doneReg := true.B
    state := idle
  }

  io.inReady := state === filling
  io.done := doneReg
  io.busy := state =/= idle

  io.dataVec := dataVecReg
  io.accVec := accVecReg
  io.dataMatrix := dataMatrixReg
  io.kernelOut := kernelReg
  io.routerWeightOut := routerWeightReg
  io.routerBiasOut := routerBiasReg
  io.expertMatrix := expertMatrixReg
  io.expertAccVec := expertAccVecReg
}
