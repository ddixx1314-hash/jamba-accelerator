package jamba.memory

import chisel3._
import chisel3.util.{Enum, MuxLookup, log2Ceil}
import jamba.common.Jamba2MiniConfig

/** Sequential field-element address walker for mini Jamba weights.
  *
  * This module does not store the weights yet. It emits the address stream for
  * one requested field so a later BRAM-backed loader can read one element per
  * accepted cycle and fill a small typed buffer.
  */
class SequentialWeightLoaderMini(
    config: Jamba2MiniConfig = Jamba2MiniConfig.debug,
    depth: Int = 2048,
    layerStride: Int = LayeredWeightStoreMini.LayerStride
) extends Module {
  require(config.lanes == 4, "SequentialWeightLoaderMini currently supports 4 lanes")
  require(config.numLayers > 0, "SequentialWeightLoaderMini needs at least one layer")

  private val lanes = config.lanes
  private val maxElements = Seq(lanes * lanes, config.convTaps * lanes, 2 * lanes, 2 * lanes * lanes).max
  private val addrWidth = math.max(1, log2Ceil(depth))
  private val layerIndexWidth = math.max(1, log2Ceil(config.numLayers))
  private val laneWidth = math.max(1, log2Ceil(lanes))
  private val tapWidth = math.max(1, log2Ceil(config.convTaps))
  private val fieldWidth = math.max(1, log2Ceil(WeightAddressGenMini.NumFields + 1))
  private val elementWidth = math.max(1, log2Ceil(maxElements))
  private val countWidth = math.max(1, log2Ceil(maxElements + 1))

  val io = IO(new Bundle {
    val clear = Input(Bool())
    val start = Input(Bool())
    val layer = Input(UInt(layerIndexWidth.W))
    val field = Input(UInt(fieldWidth.W))

    val outReady = Input(Bool())
    val outValid = Output(Bool())
    val addr = Output(UInt(addrWidth.W))
    val localAddr = Output(UInt(addrWidth.W))
    val isAcc = Output(Bool())
    val currentField = Output(UInt(fieldWidth.W))
    val elementIndex = Output(UInt(elementWidth.W))
    val numElements = Output(UInt(countWidth.W))
    val row = Output(UInt(laneWidth.W))
    val col = Output(UInt(laneWidth.W))
    val lane = Output(UInt(laneWidth.W))
    val tap = Output(UInt(tapWidth.W))
    val expert = Output(UInt(1.W))

    val busy = Output(Bool())
    val done = Output(Bool())
    val error = Output(Bool())
  })

  val idle :: running :: doneState :: Nil = Enum(3)
  val state = RegInit(idle)
  val layerReg = RegInit(0.U(layerIndexWidth.W))
  val fieldReg = RegInit(0.U(fieldWidth.W))
  val elementIndex = RegInit(0.U(elementWidth.W))
  val doneReg = RegInit(false.B)
  val errorReg = RegInit(false.B)

  private def isField(value: Int): Bool =
    fieldReg === value.U(fieldWidth.W)

  private val isMatrixField =
    isField(WeightAddressGenMini.MambaInputWeight) ||
      isField(WeightAddressGenMini.MambaBWeight) ||
      isField(WeightAddressGenMini.MambaCWeight) ||
      isField(WeightAddressGenMini.QWeight) ||
      isField(WeightAddressGenMini.KWeight) ||
      isField(WeightAddressGenMini.VWeight) ||
      isField(WeightAddressGenMini.AttentionOutWeight) ||
      isField(WeightAddressGenMini.MlpGateWeight) ||
      isField(WeightAddressGenMini.MlpUpWeight) ||
      isField(WeightAddressGenMini.MlpDownWeight)

  private val isKernelField = isField(WeightAddressGenMini.MambaKernel)
  private val isRouterWeightField = isField(WeightAddressGenMini.RouterWeight)
  private val isRouterBiasField = isField(WeightAddressGenMini.RouterBias)
  private val isExpertMatrixField =
    isField(WeightAddressGenMini.ExpertGateWeight) ||
      isField(WeightAddressGenMini.ExpertUpWeight) ||
      isField(WeightAddressGenMini.ExpertDownWeight)
  private val isExpertVectorField =
    isField(WeightAddressGenMini.ExpertGateBias) ||
      isField(WeightAddressGenMini.ExpertUpBias) ||
      isField(WeightAddressGenMini.ExpertDownBias)

  private val expertMatrixElements = (2 * lanes * lanes).U(countWidth.W)
  private val expertVectorElements = (2 * lanes).U(countWidth.W)

  private val numElements = MuxLookup(fieldReg, 0.U(countWidth.W))(Seq(
    WeightAddressGenMini.Norm1Weight.U -> lanes.U,
    WeightAddressGenMini.Norm2Weight.U -> lanes.U,
    WeightAddressGenMini.MambaInputWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.MambaInputBias.U -> lanes.U,
    WeightAddressGenMini.MambaBWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.MambaBBias.U -> lanes.U,
    WeightAddressGenMini.MambaCWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.MambaCBias.U -> lanes.U,
    WeightAddressGenMini.MambaA.U -> lanes.U,
    WeightAddressGenMini.MambaKernel.U -> (config.convTaps * lanes).U,
    WeightAddressGenMini.QWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.QBias.U -> lanes.U,
    WeightAddressGenMini.KWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.KBias.U -> lanes.U,
    WeightAddressGenMini.VWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.VBias.U -> lanes.U,
    WeightAddressGenMini.AttentionOutWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.AttentionOutBias.U -> lanes.U,
    WeightAddressGenMini.MlpGateWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.MlpGateBias.U -> lanes.U,
    WeightAddressGenMini.MlpUpWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.MlpUpBias.U -> lanes.U,
    WeightAddressGenMini.MlpDownWeight.U -> (lanes * lanes).U,
    WeightAddressGenMini.MlpDownBias.U -> lanes.U,
    WeightAddressGenMini.RouterWeight.U -> (2 * lanes).U,
    WeightAddressGenMini.RouterBias.U -> 2.U,
    WeightAddressGenMini.ExpertGateWeight.U -> expertMatrixElements,
    WeightAddressGenMini.ExpertGateBias.U -> expertVectorElements,
    WeightAddressGenMini.ExpertUpWeight.U -> expertMatrixElements,
    WeightAddressGenMini.ExpertUpBias.U -> expertVectorElements,
    WeightAddressGenMini.ExpertDownWeight.U -> expertMatrixElements,
    WeightAddressGenMini.ExpertDownBias.U -> expertVectorElements
  ))

  // Expert matrix: expert = index / (lanes*lanes), within-expert pos = index % (lanes*lanes)
  private val expertMatrixWithin = (elementIndex % (lanes * lanes).U)(laneWidth * 2 - 1, 0)
  private val expertMatrixRow = (expertMatrixWithin / lanes.U)(laneWidth - 1, 0)
  private val expertMatrixCol = (expertMatrixWithin % lanes.U)(laneWidth - 1, 0)
  private val expertMatrixExpert = (elementIndex / (lanes * lanes).U)(0)
  // Expert vector: expert = index / lanes, lane = index % lanes
  private val expertVectorExpert = (elementIndex / lanes.U)(0)
  private val expertVectorLane = (elementIndex % lanes.U)(laneWidth - 1, 0)

  private val row = Mux(isExpertMatrixField, expertMatrixRow,
    Mux(isMatrixField, (elementIndex / lanes.U)(laneWidth - 1, 0), 0.U))
  private val col = Mux(isExpertMatrixField, expertMatrixCol,
    Mux(isMatrixField, (elementIndex % lanes.U)(laneWidth - 1, 0), 0.U))
  private val vectorLane = Mux(isExpertMatrixField, expertMatrixCol,
    Mux(isExpertVectorField, expertVectorLane,
      Mux(isMatrixField, (elementIndex % lanes.U)(laneWidth - 1, 0),
        (elementIndex % lanes.U)(laneWidth - 1, 0))))
  private val tap = Mux(isKernelField, (elementIndex / lanes.U)(tapWidth - 1, 0), 0.U)
  private val expert = Mux(isExpertMatrixField, expertMatrixExpert,
    Mux(isExpertVectorField, expertVectorExpert,
      Mux(isRouterBiasField, elementIndex(0),
        Mux(isRouterWeightField, (elementIndex / lanes.U)(0), 0.U))))

  val addressGen = Module(new WeightAddressGenMini(config, depth, layerStride))
  addressGen.io.layer := layerReg
  addressGen.io.field := fieldReg
  addressGen.io.row := row
  addressGen.io.col := col
  addressGen.io.lane := vectorLane
  addressGen.io.tap := tap
  addressGen.io.expert := expert

  val fire = state === running && io.outReady && addressGen.io.valid
  val lastElement = elementIndex === numElements - 1.U
  val startValid = io.field < WeightAddressGenMini.NumFields.U

  when(io.clear) {
    state := idle
    layerReg := 0.U
    fieldReg := 0.U
    elementIndex := 0.U
    doneReg := false.B
    errorReg := false.B
  }.elsewhen(state === idle) {
    doneReg := false.B
    errorReg := false.B
    when(io.start) {
      layerReg := io.layer
      fieldReg := io.field
      elementIndex := 0.U
      when(startValid) {
        state := running
      }.otherwise {
        state := doneState
        errorReg := true.B
      }
    }
  }.elsewhen(state === running) {
    doneReg := false.B
    when(fire) {
      when(lastElement) {
        state := doneState
      }.otherwise {
        elementIndex := elementIndex + 1.U
      }
    }
  }.elsewhen(state === doneState) {
    doneReg := true.B
    state := idle
  }

  io.outValid := state === running && addressGen.io.valid
  io.addr := addressGen.io.addr
  io.localAddr := addressGen.io.localAddr
  io.isAcc := addressGen.io.isAcc
  io.currentField := fieldReg
  io.elementIndex := elementIndex
  io.numElements := numElements
  io.row := row
  io.col := col
  io.lane := vectorLane
  io.tap := tap
  io.expert := expert
  io.busy := state =/= idle
  io.done := doneReg
  io.error := errorReg
}
