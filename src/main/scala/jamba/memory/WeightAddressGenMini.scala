package jamba.memory

import chisel3._
import chisel3.util.{MuxLookup, log2Ceil}
import jamba.common.Jamba2MiniConfig

/** Address generator for sequential mini weight loading.
  *
  * This module is the BRAM-style counterpart to the field-banked weight store:
  * instead of exposing every typed field in parallel, a later loader can request
  * one field element at a time and use this module to produce the flat address.
  */
class WeightAddressGenMini(
    config: Jamba2MiniConfig = Jamba2MiniConfig.debug,
    depth: Int = 2048,
    layerStride: Int = LayeredWeightStoreMini.LayerStride
) extends Module {
  require(config.lanes == 4, "WeightAddressGenMini currently supports 4 lanes")
  require(config.numLayers > 0, "WeightAddressGenMini needs at least one layer")
  require(depth >= config.numLayers * layerStride, "depth must cover every configured layer segment")
  require(layerStride > LayeredWeightStoreMini.ExpertDownBias + 7, "layerStride must cover all weight fields including expert MoE")

  private val lanes = config.lanes
  private val addrWidth = math.max(1, log2Ceil(depth))
  private val layerIndexWidth = math.max(1, log2Ceil(config.numLayers))
  private val laneWidth = math.max(1, log2Ceil(lanes))
  private val tapWidth = math.max(1, log2Ceil(config.convTaps))
  private val fieldWidth = math.max(1, log2Ceil(WeightAddressGenMini.NumFields + 1))

  val io = IO(new Bundle {
    val layer = Input(UInt(layerIndexWidth.W))
    val field = Input(UInt(fieldWidth.W))
    val row = Input(UInt(laneWidth.W))
    val col = Input(UInt(laneWidth.W))
    val lane = Input(UInt(laneWidth.W))
    val tap = Input(UInt(tapWidth.W))
    val expert = Input(UInt(1.W))

    val addr = Output(UInt(addrWidth.W))
    val localAddr = Output(UInt(addrWidth.W))
    val valid = Output(Bool())
    val isAcc = Output(Bool())
  })

  private def u(value: Int): UInt = value.U(addrWidth.W)
  private def matrixOffset(base: Int): UInt =
    u(base) + io.row * lanes.U(addrWidth.W) + io.col
  private def vectorOffset(base: Int): UInt =
    u(base) + io.lane
  private def routerWeightOffset: UInt =
    u(LayeredWeightStoreMini.RouterWeight) + io.expert * lanes.U(addrWidth.W) + io.lane
  private def routerBiasOffset: UInt =
    u(LayeredWeightStoreMini.RouterBias) + io.expert
  private def kernelOffset: UInt =
    u(LayeredWeightStoreMini.MambaKernel) + io.tap * lanes.U(addrWidth.W) + io.lane
  private def expertMatrixOffset(base: Int): UInt =
    u(base) + io.expert * (lanes * lanes).U(addrWidth.W) + io.row * lanes.U(addrWidth.W) + io.col
  private def expertVectorOffset(base: Int): UInt =
    u(base) + io.expert * lanes.U(addrWidth.W) + io.lane

  val localAddr = MuxLookup(io.field, 0.U(addrWidth.W))(Seq(
    WeightAddressGenMini.Norm1Weight.U -> vectorOffset(LayeredWeightStoreMini.Norm1Weight),
    WeightAddressGenMini.Norm2Weight.U -> vectorOffset(LayeredWeightStoreMini.Norm2Weight),
    WeightAddressGenMini.MambaInputWeight.U -> matrixOffset(LayeredWeightStoreMini.MambaInputWeight),
    WeightAddressGenMini.MambaInputBias.U -> vectorOffset(LayeredWeightStoreMini.MambaInputBias),
    WeightAddressGenMini.MambaBWeight.U -> matrixOffset(LayeredWeightStoreMini.MambaBWeight),
    WeightAddressGenMini.MambaBBias.U -> vectorOffset(LayeredWeightStoreMini.MambaBBias),
    WeightAddressGenMini.MambaCWeight.U -> matrixOffset(LayeredWeightStoreMini.MambaCWeight),
    WeightAddressGenMini.MambaCBias.U -> vectorOffset(LayeredWeightStoreMini.MambaCBias),
    WeightAddressGenMini.MambaA.U -> vectorOffset(LayeredWeightStoreMini.MambaA),
    WeightAddressGenMini.MambaKernel.U -> kernelOffset,
    WeightAddressGenMini.QWeight.U -> matrixOffset(LayeredWeightStoreMini.QWeight),
    WeightAddressGenMini.QBias.U -> vectorOffset(LayeredWeightStoreMini.QBias),
    WeightAddressGenMini.KWeight.U -> matrixOffset(LayeredWeightStoreMini.KWeight),
    WeightAddressGenMini.KBias.U -> vectorOffset(LayeredWeightStoreMini.KBias),
    WeightAddressGenMini.VWeight.U -> matrixOffset(LayeredWeightStoreMini.VWeight),
    WeightAddressGenMini.VBias.U -> vectorOffset(LayeredWeightStoreMini.VBias),
    WeightAddressGenMini.AttentionOutWeight.U -> matrixOffset(LayeredWeightStoreMini.AttentionOutWeight),
    WeightAddressGenMini.AttentionOutBias.U -> vectorOffset(LayeredWeightStoreMini.AttentionOutBias),
    WeightAddressGenMini.MlpGateWeight.U -> matrixOffset(LayeredWeightStoreMini.MlpGateWeight),
    WeightAddressGenMini.MlpGateBias.U -> vectorOffset(LayeredWeightStoreMini.MlpGateBias),
    WeightAddressGenMini.MlpUpWeight.U -> matrixOffset(LayeredWeightStoreMini.MlpUpWeight),
    WeightAddressGenMini.MlpUpBias.U -> vectorOffset(LayeredWeightStoreMini.MlpUpBias),
    WeightAddressGenMini.MlpDownWeight.U -> matrixOffset(LayeredWeightStoreMini.MlpDownWeight),
    WeightAddressGenMini.MlpDownBias.U -> vectorOffset(LayeredWeightStoreMini.MlpDownBias),
    WeightAddressGenMini.RouterWeight.U -> routerWeightOffset,
    WeightAddressGenMini.RouterBias.U -> routerBiasOffset,
    WeightAddressGenMini.ExpertGateWeight.U -> expertMatrixOffset(LayeredWeightStoreMini.ExpertGateWeight),
    WeightAddressGenMini.ExpertGateBias.U -> expertVectorOffset(LayeredWeightStoreMini.ExpertGateBias),
    WeightAddressGenMini.ExpertUpWeight.U -> expertMatrixOffset(LayeredWeightStoreMini.ExpertUpWeight),
    WeightAddressGenMini.ExpertUpBias.U -> expertVectorOffset(LayeredWeightStoreMini.ExpertUpBias),
    WeightAddressGenMini.ExpertDownWeight.U -> expertMatrixOffset(LayeredWeightStoreMini.ExpertDownWeight),
    WeightAddressGenMini.ExpertDownBias.U -> expertVectorOffset(LayeredWeightStoreMini.ExpertDownBias)
  ))

  val layerBase = io.layer * layerStride.U(addrWidth.W)
  val physicalAddr = layerBase + localAddr
  val fieldValid = io.field < WeightAddressGenMini.NumFields.U

  io.localAddr := localAddr
  io.addr := physicalAddr
  io.valid := fieldValid && physicalAddr < depth.U
  io.isAcc := WeightAddressGenMini.isAccField(io.field)
}

object WeightAddressGenMini {
  val Norm1Weight = 0
  val Norm2Weight = 1
  val MambaInputWeight = 2
  val MambaInputBias = 3
  val MambaBWeight = 4
  val MambaBBias = 5
  val MambaCWeight = 6
  val MambaCBias = 7
  val MambaA = 8
  val MambaKernel = 9
  val QWeight = 10
  val QBias = 11
  val KWeight = 12
  val KBias = 13
  val VWeight = 14
  val VBias = 15
  val AttentionOutWeight = 16
  val AttentionOutBias = 17
  val MlpGateWeight = 18
  val MlpGateBias = 19
  val MlpUpWeight = 20
  val MlpUpBias = 21
  val MlpDownWeight = 22
  val MlpDownBias = 23
  val RouterWeight = 24
  val RouterBias = 25
  val ExpertGateWeight = 26
  val ExpertGateBias = 27
  val ExpertUpWeight = 28
  val ExpertUpBias = 29
  val ExpertDownWeight = 30
  val ExpertDownBias = 31
  val NumFields = 32

  def isAccField(field: UInt): Bool =
    field === MambaInputBias.U ||
      field === MambaBBias.U ||
      field === MambaCBias.U ||
      field === QBias.U ||
      field === KBias.U ||
      field === VBias.U ||
      field === AttentionOutBias.U ||
      field === MlpGateBias.U ||
      field === MlpUpBias.U ||
      field === MlpDownBias.U ||
      field === RouterBias.U ||
      field === ExpertGateBias.U ||
      field === ExpertUpBias.U ||
      field === ExpertDownBias.U
}
