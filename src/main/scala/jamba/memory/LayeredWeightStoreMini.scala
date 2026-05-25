package jamba.memory

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.{Jamba2MiniConfig, SignedMath}

/** Field-banked mini weight store for the unified multi-layer tile.
  *
  * Software still writes a flat address space, but hardware stores the decoded
  * fields in per-parameter banks. The active layer selects a small per-field
  * vector instead of fanning out a full readAll bus into every weight port.
  */
class LayeredWeightStoreMini(
    config: Jamba2MiniConfig = Jamba2MiniConfig.debug,
    depth: Int = 2048,
    layerStride: Int = LayeredWeightStoreMini.LayerStride
) extends Module {
  require(config.lanes > 0, "LayeredWeightStoreMini lanes must be positive")
  require(config.numLayers > 0, "LayeredWeightStoreMini needs at least one layer")
  require(depth > 0, "LayeredWeightStoreMini depth must be positive")
  require(layerStride > LayeredWeightStoreMini.ExpertDownBias + 7, "layerStride must cover all weight fields including expert MoE")

  private val lanes = config.lanes
  private val dataWidth = config.dataWidth
  private val accWidth = config.accWidth
  private val numLayers = config.numLayers
  private val addrWidth = math.max(1, log2Ceil(depth))
  private val layerIndexWidth = math.max(1, log2Ceil(numLayers))

  val io = IO(new Bundle {
    val clear = Input(Bool())
    val writeValid = Input(Bool())
    val writeReady = Output(Bool())
    val writeAddr = Input(UInt(addrWidth.W))
    val writeData = Input(SInt(accWidth.W))
    val readAddr = Input(UInt(addrWidth.W))
    val readData = Output(SInt(accWidth.W))
    val activeLayer = Input(UInt(layerIndexWidth.W))

    val norm1Weight = Output(Vec(lanes, SInt(dataWidth.W)))
    val norm2Weight = Output(Vec(lanes, SInt(dataWidth.W)))

    val mambaInputWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaInputBias = Output(Vec(lanes, SInt(accWidth.W)))
    val mambaBWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaBBias = Output(Vec(lanes, SInt(accWidth.W)))
    val mambaCWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mambaCBias = Output(Vec(lanes, SInt(accWidth.W)))
    val mambaA = Output(Vec(lanes, SInt(dataWidth.W)))
    val mambaKernel = Output(Vec(config.convTaps, Vec(lanes, SInt(dataWidth.W))))

    val qWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val qBias = Output(Vec(lanes, SInt(accWidth.W)))
    val kWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val kBias = Output(Vec(lanes, SInt(accWidth.W)))
    val vWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val vBias = Output(Vec(lanes, SInt(accWidth.W)))
    val attentionOutWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val attentionOutBias = Output(Vec(lanes, SInt(accWidth.W)))

    val mlpGateWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpGateBias = Output(Vec(lanes, SInt(accWidth.W)))
    val mlpUpWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpUpBias = Output(Vec(lanes, SInt(accWidth.W)))
    val mlpDownWeight = Output(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val mlpDownBias = Output(Vec(lanes, SInt(accWidth.W)))

    val routerWeight = Output(Vec(2, Vec(lanes, SInt(dataWidth.W))))
    val routerBias = Output(Vec(2, SInt(accWidth.W)))

    val expertGateWeight = Output(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertGateBias   = Output(Vec(2, Vec(lanes, SInt(accWidth.W))))
    val expertUpWeight   = Output(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertUpBias     = Output(Vec(2, Vec(lanes, SInt(accWidth.W))))
    val expertDownWeight = Output(Vec(2, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertDownBias   = Output(Vec(2, Vec(lanes, SInt(accWidth.W))))
  })

  private def zeroDataVec = VecInit(Seq.fill(numLayers)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  private def zeroAccVec = VecInit(Seq.fill(numLayers)(VecInit(Seq.fill(lanes)(0.S(accWidth.W)))))
  private def zeroDataMatrix =
    VecInit(Seq.fill(numLayers)(VecInit(Seq.fill(lanes)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))))

  val rawMem = RegInit(VecInit(Seq.fill(depth)(0.S(accWidth.W))))

  val norm1Weight = RegInit(zeroDataVec)
  val norm2Weight = RegInit(zeroDataVec)
  val mambaInputWeight = RegInit(zeroDataMatrix)
  val mambaInputBias = RegInit(zeroAccVec)
  val mambaBWeight = RegInit(zeroDataMatrix)
  val mambaBBias = RegInit(zeroAccVec)
  val mambaCWeight = RegInit(zeroDataMatrix)
  val mambaCBias = RegInit(zeroAccVec)
  val mambaA = RegInit(zeroDataVec)
  val mambaKernel =
    RegInit(VecInit(Seq.fill(numLayers)(VecInit(Seq.fill(config.convTaps)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))))))

  val qWeight = RegInit(zeroDataMatrix)
  val qBias = RegInit(zeroAccVec)
  val kWeight = RegInit(zeroDataMatrix)
  val kBias = RegInit(zeroAccVec)
  val vWeight = RegInit(zeroDataMatrix)
  val vBias = RegInit(zeroAccVec)
  val attentionOutWeight = RegInit(zeroDataMatrix)
  val attentionOutBias = RegInit(zeroAccVec)

  val mlpGateWeight = RegInit(zeroDataMatrix)
  val mlpGateBias = RegInit(zeroAccVec)
  val mlpUpWeight = RegInit(zeroDataMatrix)
  val mlpUpBias = RegInit(zeroAccVec)
  val mlpDownWeight = RegInit(zeroDataMatrix)
  val mlpDownBias = RegInit(zeroAccVec)

  val routerWeight =
    RegInit(VecInit(Seq.fill(numLayers)(VecInit(Seq.fill(2)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))))))
  val routerBias = RegInit(VecInit(Seq.fill(numLayers)(VecInit(Seq.fill(2)(0.S(accWidth.W))))))

  private def zeroExpertMatrix =
    VecInit(Seq.fill(numLayers)(VecInit(Seq.fill(2)(VecInit(Seq.fill(lanes)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))))))
  private def zeroExpertAccVec =
    VecInit(Seq.fill(numLayers)(VecInit(Seq.fill(2)(VecInit(Seq.fill(lanes)(0.S(accWidth.W)))))))

  val expertGateWeight = RegInit(zeroExpertMatrix)
  val expertGateBias   = RegInit(zeroExpertAccVec)
  val expertUpWeight   = RegInit(zeroExpertMatrix)
  val expertUpBias     = RegInit(zeroExpertAccVec)
  val expertDownWeight = RegInit(zeroExpertMatrix)
  val expertDownBias   = RegInit(zeroExpertAccVec)

  io.writeReady := true.B
  val writeFire = io.writeValid && io.writeReady

  when(writeFire) {
    rawMem(io.writeAddr) := io.writeData
  }

  private def dataWrite: SInt =
    SignedMath.resize(io.writeData, dataWidth)

  private def whenWriteLocal(localAddr: Int)(body: Int => Unit): Unit = {
    for (layer <- 0 until numLayers) {
      val physicalAddr = layer * layerStride + localAddr
      if (physicalAddr < depth) {
        when(writeFire && io.writeAddr === physicalAddr.U(addrWidth.W)) {
          body(layer)
        }
      }
    }
  }

  private def writeDataVector(base: Int, target: Vec[Vec[SInt]]): Unit = {
    for (lane <- 0 until lanes) {
      whenWriteLocal(base + lane) { layer =>
        target(layer)(lane) := dataWrite
      }
    }
  }

  private def writeAccVector(base: Int, target: Vec[Vec[SInt]]): Unit = {
    for (lane <- 0 until lanes) {
      whenWriteLocal(base + lane) { layer =>
        target(layer)(lane) := io.writeData
      }
    }
  }

  private def writeDataMatrix(base: Int, target: Vec[Vec[Vec[SInt]]]): Unit = {
    for (row <- 0 until lanes) {
      for (col <- 0 until lanes) {
        whenWriteLocal(base + row * lanes + col) { layer =>
          target(layer)(row)(col) := dataWrite
        }
      }
    }
  }

  writeDataVector(LayeredWeightStoreMini.Norm1Weight, norm1Weight)
  writeDataVector(LayeredWeightStoreMini.Norm2Weight, norm2Weight)

  writeDataMatrix(LayeredWeightStoreMini.MambaInputWeight, mambaInputWeight)
  writeAccVector(LayeredWeightStoreMini.MambaInputBias, mambaInputBias)
  writeDataMatrix(LayeredWeightStoreMini.MambaBWeight, mambaBWeight)
  writeAccVector(LayeredWeightStoreMini.MambaBBias, mambaBBias)
  writeDataMatrix(LayeredWeightStoreMini.MambaCWeight, mambaCWeight)
  writeAccVector(LayeredWeightStoreMini.MambaCBias, mambaCBias)
  writeDataVector(LayeredWeightStoreMini.MambaA, mambaA)
  for (tap <- 0 until config.convTaps) {
    for (lane <- 0 until lanes) {
      whenWriteLocal(LayeredWeightStoreMini.MambaKernel + tap * lanes + lane) { layer =>
        mambaKernel(layer)(tap)(lane) := dataWrite
      }
    }
  }

  writeDataMatrix(LayeredWeightStoreMini.QWeight, qWeight)
  writeAccVector(LayeredWeightStoreMini.QBias, qBias)
  writeDataMatrix(LayeredWeightStoreMini.KWeight, kWeight)
  writeAccVector(LayeredWeightStoreMini.KBias, kBias)
  writeDataMatrix(LayeredWeightStoreMini.VWeight, vWeight)
  writeAccVector(LayeredWeightStoreMini.VBias, vBias)
  writeDataMatrix(LayeredWeightStoreMini.AttentionOutWeight, attentionOutWeight)
  writeAccVector(LayeredWeightStoreMini.AttentionOutBias, attentionOutBias)

  writeDataMatrix(LayeredWeightStoreMini.MlpGateWeight, mlpGateWeight)
  writeAccVector(LayeredWeightStoreMini.MlpGateBias, mlpGateBias)
  writeDataMatrix(LayeredWeightStoreMini.MlpUpWeight, mlpUpWeight)
  writeAccVector(LayeredWeightStoreMini.MlpUpBias, mlpUpBias)
  writeDataMatrix(LayeredWeightStoreMini.MlpDownWeight, mlpDownWeight)
  writeAccVector(LayeredWeightStoreMini.MlpDownBias, mlpDownBias)

  for (expert <- 0 until 2) {
    for (lane <- 0 until lanes) {
      whenWriteLocal(LayeredWeightStoreMini.RouterWeight + expert * lanes + lane) { layer =>
        routerWeight(layer)(expert)(lane) := dataWrite
      }
    }
    whenWriteLocal(LayeredWeightStoreMini.RouterBias + expert) { layer =>
      routerBias(layer)(expert) := io.writeData
    }
  }

  private def writeExpertDataMatrix(base: Int, target: Vec[Vec[Vec[Vec[SInt]]]]): Unit = {
    for (expert <- 0 until 2) {
      for (row <- 0 until lanes) {
        for (col <- 0 until lanes) {
          whenWriteLocal(base + expert * lanes * lanes + row * lanes + col) { layer =>
            target(layer)(expert)(row)(col) := dataWrite
          }
        }
      }
    }
  }

  private def writeExpertAccVector(base: Int, target: Vec[Vec[Vec[SInt]]]): Unit = {
    for (expert <- 0 until 2) {
      for (lane <- 0 until lanes) {
        whenWriteLocal(base + expert * lanes + lane) { layer =>
          target(layer)(expert)(lane) := io.writeData
        }
      }
    }
  }

  writeExpertDataMatrix(LayeredWeightStoreMini.ExpertGateWeight, expertGateWeight)
  writeExpertAccVector(LayeredWeightStoreMini.ExpertGateBias, expertGateBias)
  writeExpertDataMatrix(LayeredWeightStoreMini.ExpertUpWeight, expertUpWeight)
  writeExpertAccVector(LayeredWeightStoreMini.ExpertUpBias, expertUpBias)
  writeExpertDataMatrix(LayeredWeightStoreMini.ExpertDownWeight, expertDownWeight)
  writeExpertAccVector(LayeredWeightStoreMini.ExpertDownBias, expertDownBias)

  private def selectLayer[T <: Data](bank: Vec[T]): T =
    if (numLayers == 1) bank(0) else bank(Mux(io.activeLayer < numLayers.U, io.activeLayer, 0.U))

  io.readData := rawMem(io.readAddr)
  io.norm1Weight := selectLayer(norm1Weight)
  io.norm2Weight := selectLayer(norm2Weight)
  io.mambaInputWeight := selectLayer(mambaInputWeight)
  io.mambaInputBias := selectLayer(mambaInputBias)
  io.mambaBWeight := selectLayer(mambaBWeight)
  io.mambaBBias := selectLayer(mambaBBias)
  io.mambaCWeight := selectLayer(mambaCWeight)
  io.mambaCBias := selectLayer(mambaCBias)
  io.mambaA := selectLayer(mambaA)
  io.mambaKernel := selectLayer(mambaKernel)
  io.qWeight := selectLayer(qWeight)
  io.qBias := selectLayer(qBias)
  io.kWeight := selectLayer(kWeight)
  io.kBias := selectLayer(kBias)
  io.vWeight := selectLayer(vWeight)
  io.vBias := selectLayer(vBias)
  io.attentionOutWeight := selectLayer(attentionOutWeight)
  io.attentionOutBias := selectLayer(attentionOutBias)
  io.mlpGateWeight := selectLayer(mlpGateWeight)
  io.mlpGateBias := selectLayer(mlpGateBias)
  io.mlpUpWeight := selectLayer(mlpUpWeight)
  io.mlpUpBias := selectLayer(mlpUpBias)
  io.mlpDownWeight := selectLayer(mlpDownWeight)
  io.mlpDownBias := selectLayer(mlpDownBias)
  io.routerWeight := selectLayer(routerWeight)
  io.routerBias := selectLayer(routerBias)
  io.expertGateWeight := selectLayer(expertGateWeight)
  io.expertGateBias := selectLayer(expertGateBias)
  io.expertUpWeight := selectLayer(expertUpWeight)
  io.expertUpBias := selectLayer(expertUpBias)
  io.expertDownWeight := selectLayer(expertDownWeight)
  io.expertDownBias := selectLayer(expertDownBias)
}

object LayeredWeightStoreMini {
  val LayerStride = 512

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

  // Expert MoE weight fields (2 experts × lanes × lanes each)
  val ExpertGateWeight = 246  // 32 slots: 246..277
  val ExpertGateBias   = 278  // 8 slots:  278..285
  val ExpertUpWeight   = 286  // 32 slots: 286..317
  val ExpertUpBias     = 318  // 8 slots:  318..325
  val ExpertDownWeight = 326  // 32 slots: 326..357
  val ExpertDownBias   = 358  // 8 slots:  358..365
}
