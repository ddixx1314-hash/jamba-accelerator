package jamba.memory

import chisel3._
import chisel3.util.{Enum, log2Ceil}
import jamba.common.Jamba2MiniConfig

/** Connects SequentialWeightLoaderMini to a memory read port and captures one
  * element per accepted cycle.
  *
  * The module drives io.readAddr from the loader's address stream and samples
  * io.readData each cycle the loader signals outValid. The captured element is
  * forwarded downstream via outValid/outData with the associated metadata.
  *
  * The memory is assumed to return readData combinationally on the same cycle
  * that readAddr is presented (register-file style), matching LayeredWeightStoreMini.
  */
class SequentialWeightCaptureMini(
    config: Jamba2MiniConfig = Jamba2MiniConfig.debug,
    depth: Int = 2048,
    layerStride: Int = LayeredWeightStoreMini.LayerStride
) extends Module {
  require(config.lanes == 4, "SequentialWeightCaptureMini currently supports 4 lanes")

  private val addrWidth = math.max(1, log2Ceil(depth))
  private val fieldWidth = math.max(1, log2Ceil(WeightAddressGenMini.NumFields + 1))
  private val layerIndexWidth = math.max(1, log2Ceil(config.numLayers))
  private val maxElements = Seq(config.lanes * config.lanes, config.convTaps * config.lanes, 2 * config.lanes, 2 * config.lanes * config.lanes).max
  private val elementWidth = math.max(1, log2Ceil(maxElements))
  private val countWidth = math.max(1, log2Ceil(maxElements + 1))
  private val laneWidth = math.max(1, log2Ceil(config.lanes))
  private val tapWidth = math.max(1, log2Ceil(config.convTaps))

  val io = IO(new Bundle {
    val clear = Input(Bool())
    val start = Input(Bool())
    val layer = Input(UInt(layerIndexWidth.W))
    val field = Input(UInt(fieldWidth.W))

    val readAddr = Output(UInt(addrWidth.W))
    val readData = Input(SInt(config.accWidth.W))

    val outValid = Output(Bool())
    val outReady = Input(Bool())
    val outData = Output(SInt(config.accWidth.W))
    val outIsAcc = Output(Bool())
    val outElementIndex = Output(UInt(elementWidth.W))
    val outNumElements = Output(UInt(countWidth.W))
    val outRow = Output(UInt(laneWidth.W))
    val outCol = Output(UInt(laneWidth.W))
    val outLane = Output(UInt(laneWidth.W))
    val outTap = Output(UInt(tapWidth.W))
    val outExpert = Output(UInt(1.W))

    val busy = Output(Bool())
    val done = Output(Bool())
    val error = Output(Bool())
  })

  val loader = Module(new SequentialWeightLoaderMini(config, depth, layerStride))
  loader.io.clear := io.clear
  loader.io.start := io.start
  loader.io.layer := io.layer
  loader.io.field := io.field
  loader.io.outReady := io.outReady

  // Memory address comes directly from the loader
  io.readAddr := loader.io.addr

  // When the loader presents a valid address and downstream is ready, forward readData
  io.outValid := loader.io.outValid
  io.outData := io.readData
  io.outIsAcc := loader.io.isAcc
  io.outElementIndex := loader.io.elementIndex
  io.outNumElements := loader.io.numElements
  io.outRow := loader.io.row
  io.outCol := loader.io.col
  io.outLane := loader.io.lane
  io.outTap := loader.io.tap
  io.outExpert := loader.io.expert

  io.busy := loader.io.busy
  io.done := loader.io.done
  io.error := loader.io.error
}
