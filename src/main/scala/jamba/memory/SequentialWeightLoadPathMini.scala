package jamba.memory

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.Jamba2MiniConfig

/** End-to-end sequential weight loading path for one field.
  *
  * Chains SequentialWeightCaptureMini (address walker + memory read) and
  * FieldWeightBufferMini (element-to-typed-register accumulator). A single
  * start pulse with layer and field selects which layer's field to load;
  * done is asserted when all elements have been captured and stored.
  *
  * Comparison with LayeredWeightStoreMini:
  *   LayeredWeightStoreMini   — field-banked parallel decode; all typed ports
  *                              always driven from a large register file.
  *   SequentialWeightLoadPathMini — one field loaded per request via a narrow
  *                              address stream; typed output valid after done.
  */
class SequentialWeightLoadPathMini(
    config: Jamba2MiniConfig = Jamba2MiniConfig.debug,
    depth: Int = 2048,
    layerStride: Int = LayeredWeightStoreMini.LayerStride
) extends Module {
  require(config.lanes == 4, "SequentialWeightLoadPathMini currently supports 4 lanes")

  private val addrWidth = math.max(1, log2Ceil(depth))
  private val fieldWidth = math.max(1, log2Ceil(WeightAddressGenMini.NumFields + 1))
  private val layerIndexWidth = math.max(1, log2Ceil(config.numLayers))
  private val maxElements = Seq(config.lanes * config.lanes, config.convTaps * config.lanes, 2 * config.lanes, 2 * config.lanes * config.lanes).max
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

    val busy = Output(Bool())
    val done = Output(Bool())
    val error = Output(Bool())

    // Typed output registers — valid after done
    val dataVec = Output(Vec(config.lanes, SInt(config.dataWidth.W)))
    val accVec = Output(Vec(config.lanes, SInt(config.accWidth.W)))
    val dataMatrix = Output(Vec(config.lanes, Vec(config.lanes, SInt(config.dataWidth.W))))
    val kernelOut = Output(Vec(config.convTaps, Vec(config.lanes, SInt(config.dataWidth.W))))
    val routerWeightOut = Output(Vec(2, Vec(config.lanes, SInt(config.dataWidth.W))))
    val routerBiasOut = Output(Vec(2, SInt(config.accWidth.W)))
    val expertMatrix = Output(Vec(2, Vec(config.lanes, Vec(config.lanes, SInt(config.dataWidth.W)))))
    val expertAccVec = Output(Vec(2, Vec(config.lanes, SInt(config.accWidth.W))))
  })

  val capture = Module(new SequentialWeightCaptureMini(config, depth, layerStride))
  capture.io.clear := io.clear
  capture.io.start := io.start
  capture.io.layer := io.layer
  capture.io.field := io.field
  capture.io.readData := io.readData
  io.readAddr := capture.io.readAddr

  val buffer = Module(new FieldWeightBufferMini(config))
  buffer.io.clear := io.clear
  buffer.io.start := io.start
  buffer.io.fieldId := io.field
  buffer.io.inValid := capture.io.outValid
  capture.io.outReady := buffer.io.inReady
  buffer.io.inData := capture.io.outData
  buffer.io.inIsAcc := capture.io.outIsAcc
  buffer.io.inElementIndex := capture.io.outElementIndex
  buffer.io.inNumElements := capture.io.outNumElements
  buffer.io.inRow := capture.io.outRow
  buffer.io.inCol := capture.io.outCol
  buffer.io.inLane := capture.io.outLane
  buffer.io.inTap := capture.io.outTap
  buffer.io.inExpert := capture.io.outExpert

  io.busy := capture.io.busy || buffer.io.busy
  io.done := buffer.io.done
  io.error := capture.io.error

  io.dataVec := buffer.io.dataVec
  io.accVec := buffer.io.accVec
  io.dataMatrix := buffer.io.dataMatrix
  io.kernelOut := buffer.io.kernelOut
  io.routerWeightOut := buffer.io.routerWeightOut
  io.routerBiasOut := buffer.io.routerBiasOut
  io.expertMatrix := buffer.io.expertMatrix
  io.expertAccVec := buffer.io.expertAccVec
}
