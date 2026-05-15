package jamba.attention

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.SignedMath
import jamba.math.Linear4

/** Jamba2 mini attention mixer with Q/K/V projection and a circular KV cache. */
class AttentionMixerMini(
    lanes: Int = 4,
    contextLength: Int = 4,
    dataWidth: Int = 8,
    accWidth: Int = 32,
    normShift: Int = 2)
    extends Module {
  require(lanes == 4, "AttentionMixerMini currently uses Linear4 and requires lanes == 4")
  require(contextLength > 0, "AttentionMixerMini contextLength must be positive")
  require(dataWidth > 0, "AttentionMixerMini dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "AttentionMixerMini accWidth should hold dot products")
  require(normShift >= 0, "AttentionMixerMini normShift must be non-negative")

  private val indexWidth = math.max(1, log2Ceil(contextLength))
  private val countWidth = log2Ceil(contextLength + 1)

  val io = IO(new Bundle {
    val en = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))

    val qWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val qBias = Input(Vec(lanes, SInt(accWidth.W)))
    val kWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val kBias = Input(Vec(lanes, SInt(accWidth.W)))
    val vWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val vBias = Input(Vec(lanes, SInt(accWidth.W)))
    val outWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val outBias = Input(Vec(lanes, SInt(accWidth.W)))

    val q = Output(Vec(lanes, SInt(dataWidth.W)))
    val k = Output(Vec(lanes, SInt(dataWidth.W)))
    val v = Output(Vec(lanes, SInt(dataWidth.W)))
    val scores = Output(Vec(contextLength, SInt(accWidth.W)))
    val weights = Output(Vec(contextLength, SInt(accWidth.W)))
    val rawY = Output(Vec(lanes, SInt(accWidth.W)))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
    val kvWriteIndex = Output(UInt(indexWidth.W))
    val kvValidCount = Output(UInt(countWidth.W))
  })

  private def narrowToData(value: SInt): SInt = value(dataWidth - 1, 0).asSInt

  val qProjection = Module(new Linear4(dataWidth, accWidth))
  qProjection.io.x := io.x
  qProjection.io.weight := io.qWeight
  qProjection.io.bias := io.qBias

  val kProjection = Module(new Linear4(dataWidth, accWidth))
  kProjection.io.x := io.x
  kProjection.io.weight := io.kWeight
  kProjection.io.bias := io.kBias

  val vProjection = Module(new Linear4(dataWidth, accWidth))
  vProjection.io.x := io.x
  vProjection.io.weight := io.vWeight
  vProjection.io.bias := io.vBias

  val nextQ = Wire(Vec(lanes, SInt(dataWidth.W)))
  val nextK = Wire(Vec(lanes, SInt(dataWidth.W)))
  val nextV = Wire(Vec(lanes, SInt(dataWidth.W)))

  for (lane <- 0 until lanes) {
    nextQ(lane) := narrowToData(qProjection.io.y(lane))
    nextK(lane) := narrowToData(kProjection.io.y(lane))
    nextV(lane) := narrowToData(vProjection.io.y(lane))
    io.q(lane) := nextQ(lane)
    io.k(lane) := nextK(lane)
    io.v(lane) := nextV(lane)
  }

  val keyCache = RegInit(VecInit(Seq.fill(contextLength)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))))
  val valueCache = RegInit(VecInit(Seq.fill(contextLength)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))))
  val writeIndex = RegInit(0.U(indexWidth.W))
  val validCount = RegInit(0.U(countWidth.W))

  val effectiveKeys = Wire(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))
  val effectiveValues = Wire(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))
  val nextWriteIndex = Wire(UInt(indexWidth.W))
  val nextValidCount = Wire(UInt(countWidth.W))

  val advancedWriteIndex = Mux(writeIndex === (contextLength - 1).U, 0.U, writeIndex + 1.U)
  val advancedValidCount = Mux(validCount === contextLength.U, validCount, validCount + 1.U)
  nextWriteIndex := Mux(io.en, advancedWriteIndex, writeIndex)
  nextValidCount := Mux(io.en, advancedValidCount, validCount)

  for (row <- 0 until contextLength) {
    for (lane <- 0 until lanes) {
      effectiveKeys(row)(lane) := keyCache(row)(lane)
      effectiveValues(row)(lane) := valueCache(row)(lane)
    }
  }

  when(io.en) {
    for (lane <- 0 until lanes) {
      effectiveKeys(writeIndex)(lane) := nextK(lane)
      effectiveValues(writeIndex)(lane) := nextV(lane)
    }
  }

  val fullAfterWrite = nextValidCount === contextLength.U
  val physicalRows = Wire(Vec(contextLength, UInt(indexWidth.W)))
  val rowValid = Wire(Vec(contextLength, Bool()))

  for (row <- 0 until contextLength) {
    val shifted = nextWriteIndex + row.U
    val wrappedRow = Mux(shifted >= contextLength.U, shifted - contextLength.U, shifted)
    physicalRows(row) := Mux(fullAfterWrite, wrappedRow, row.U)
    rowValid(row) := row.U < nextValidCount

    val products = Wire(Vec(lanes, SInt(accWidth.W)))
    for (lane <- 0 until lanes) {
      products(lane) := SignedMath.resize(nextQ(lane) * effectiveKeys(physicalRows(row))(lane), accWidth)
    }

    io.scores(row) := Mux(rowValid(row), products.reduce(_ +& _), 0.S)
    io.weights(row) := io.scores(row) >> normShift
  }

  for (lane <- 0 until lanes) {
    val weightedValues = Wire(Vec(contextLength, SInt(accWidth.W)))
    for (row <- 0 until contextLength) {
      weightedValues(row) := Mux(
        rowValid(row),
        SignedMath.resize(io.weights(row) * effectiveValues(physicalRows(row))(lane), accWidth),
        0.S
      )
    }

    io.rawY(lane) := weightedValues.reduce(_ +& _)
  }

  val outProjection = Module(new Linear4(dataWidth, accWidth))
  for (lane <- 0 until lanes) {
    outProjection.io.x(lane) := narrowToData(io.rawY(lane))
  }
  outProjection.io.weight := io.outWeight
  outProjection.io.bias := io.outBias
  io.y := outProjection.io.y

  when(io.clear) {
    keyCache := VecInit(Seq.fill(contextLength)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
    valueCache := VecInit(Seq.fill(contextLength)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
    writeIndex := 0.U
    validCount := 0.U
  }.elsewhen(io.en) {
    keyCache(writeIndex) := nextK
    valueCache(writeIndex) := nextV
    writeIndex := nextWriteIndex
    validCount := nextValidCount
  }

  io.kvWriteIndex := writeIndex
  io.kvValidCount := validCount
}
