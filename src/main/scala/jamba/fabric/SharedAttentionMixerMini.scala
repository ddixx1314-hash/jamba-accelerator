package jamba.fabric

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.{FixedPointMath, SignedMath}

/** Jamba2 mini attention mixer with Q/K/V/out projections mapped to shared linear fabric. */
class SharedAttentionMixerMini(
    lanes:         Int = 4,
    contextLength: Int = 4,
    dataWidth:     Int = 8,
    accWidth:      Int = 32,
    normShift:     Int = 2)
    extends Module {
  require(lanes == 4, "SharedAttentionMixerMini currently uses SharedLinear4 and requires lanes == 4")
  require(contextLength > 0, "SharedAttentionMixerMini contextLength must be positive")
  require(dataWidth > 0, "SharedAttentionMixerMini dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "SharedAttentionMixerMini accWidth should hold dot products")
  require(normShift >= 0, "SharedAttentionMixerMini normShift must be non-negative")

  private val indexWidth = math.max(1, log2Ceil(contextLength))
  private val countWidth = log2Ceil(contextLength + 1)

  val io = IO(new Bundle {
    val en    = Input(Bool())
    val clear = Input(Bool())
    val x     = Input(Vec(lanes, SInt(dataWidth.W)))

    val qWeight   = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val qBias     = Input(Vec(lanes, SInt(accWidth.W)))
    val kWeight   = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val kBias     = Input(Vec(lanes, SInt(accWidth.W)))
    val vWeight   = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val vBias     = Input(Vec(lanes, SInt(accWidth.W)))
    val outWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val outBias   = Input(Vec(lanes, SInt(accWidth.W)))

    val q            = Output(Vec(lanes, SInt(dataWidth.W)))
    val k            = Output(Vec(lanes, SInt(dataWidth.W)))
    val v            = Output(Vec(lanes, SInt(dataWidth.W)))
    val scores       = Output(Vec(contextLength, SInt(accWidth.W)))
    val weights      = Output(Vec(contextLength, SInt(accWidth.W)))
    val rawY         = Output(Vec(lanes, SInt(accWidth.W)))
    val y            = Output(Vec(lanes, SInt(accWidth.W)))
    val kvWriteIndex = Output(UInt(indexWidth.W))
    val kvValidCount = Output(UInt(countWidth.W))
  })

  private def narrowToData(value: SInt): SInt = FixedPointMath.saturate(value, dataWidth)

  private def zeroCache = VecInit(Seq.fill(contextLength)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))

  val qProjection = Module(new SharedLinear4(dataWidth, accWidth))
  qProjection.io.x := io.x
  qProjection.io.weight := io.qWeight
  qProjection.io.bias := io.qBias

  val kProjection = Module(new SharedLinear4(dataWidth, accWidth))
  kProjection.io.x := io.x
  kProjection.io.weight := io.kWeight
  kProjection.io.bias := io.kBias

  val vProjection = Module(new SharedLinear4(dataWidth, accWidth))
  vProjection.io.x := io.x
  vProjection.io.weight := io.vWeight
  vProjection.io.bias := io.vBias

  val projQ = Wire(Vec(lanes, SInt(dataWidth.W)))
  val projK = Wire(Vec(lanes, SInt(dataWidth.W)))
  val projV = Wire(Vec(lanes, SInt(dataWidth.W)))

  for (lane <- 0 until lanes) {
    projQ(lane) := narrowToData(qProjection.io.y(lane))
    projK(lane) := narrowToData(kProjection.io.y(lane))
    projV(lane) := narrowToData(vProjection.io.y(lane))
    io.q(lane) := projQ(lane)
    io.k(lane) := projK(lane)
    io.v(lane) := projV(lane)
  }

  val keyCache = RegInit(zeroCache)
  val valueCache = RegInit(zeroCache)
  val writeIndex = RegInit(0.U(indexWidth.W))
  val validCount = RegInit(0.U(countWidth.W))

  val effectiveKeys = Wire(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))
  val effectiveValues = Wire(Vec(contextLength, Vec(lanes, SInt(dataWidth.W))))
  val nextWriteIndex = Wire(UInt(indexWidth.W))
  val nextValidCount = Wire(UInt(countWidth.W))

  val advancedWriteIndex = Mux(writeIndex === (contextLength - 1).U, 0.U, writeIndex + 1.U)
  val advancedValidCount = Mux(validCount === contextLength.U, validCount, validCount + 1.U)
  val cacheWillWrite = io.en && !io.clear
  nextWriteIndex := Mux(io.clear, 0.U, Mux(cacheWillWrite, advancedWriteIndex, writeIndex))
  nextValidCount := Mux(io.clear, 0.U, Mux(cacheWillWrite, advancedValidCount, validCount))

  for (row <- 0 until contextLength) {
    for (lane <- 0 until lanes) {
      effectiveKeys(row)(lane) := keyCache(row)(lane)
      effectiveValues(row)(lane) := valueCache(row)(lane)
    }
  }

  when(cacheWillWrite) {
    for (lane <- 0 until lanes) {
      effectiveKeys(writeIndex)(lane) := projK(lane)
      effectiveValues(writeIndex)(lane) := projV(lane)
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
      products(lane) := SignedMath.resize(projQ(lane) * effectiveKeys(physicalRows(row))(lane), accWidth)
    }

    io.scores(row) := Mux(rowValid(row), products.reduce(_ +& _), 0.S)
    io.weights(row) := FixedPointMath.roundedShiftRight(io.scores(row), normShift)
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

  val outProjection = Module(new SharedLinear4(dataWidth, accWidth))
  for (lane <- 0 until lanes) {
    outProjection.io.x(lane) := narrowToData(io.rawY(lane))
  }
  outProjection.io.weight := io.outWeight
  outProjection.io.bias := io.outBias
  io.y := outProjection.io.y

  when(io.clear) {
    keyCache := zeroCache
    valueCache := zeroCache
    writeIndex := 0.U
    validCount := 0.U
  }.elsewhen(io.en) {
    keyCache(writeIndex) := projK
    valueCache(writeIndex) := projV
    writeIndex := nextWriteIndex
    validCount := nextValidCount
  }

  io.kvWriteIndex := writeIndex
  io.kvValidCount := validCount
}
