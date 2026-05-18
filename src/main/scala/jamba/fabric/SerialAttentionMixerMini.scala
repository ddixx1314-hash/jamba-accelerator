package jamba.fabric

import chisel3._
import chisel3.util.{Enum, MuxCase, log2Ceil}
import jamba.common.{FixedPointMath, SignedMath}

/** Token-level serial attention mixer.
  *
  * One SerialSharedLinear4 (one MAC) is reused four times in sequence:
  *   Q projection → K projection → V projection → score/value compute → out projection
  *
  * Score and value accumulation are combinational after K/V are written to the
  * KV cache (same write-through semantics as AttentionMixerMini).
  */
class SerialAttentionMixerMini(
    lanes:         Int = 4,
    contextLength: Int = 4,
    dataWidth:     Int = 8,
    accWidth:      Int = 32,
    normShift:     Int = 2)
    extends Module {
  require(lanes == 4, "SerialAttentionMixerMini requires lanes == 4")
  require(contextLength > 0, "contextLength must be positive")
  require(dataWidth > 0, "dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "accWidth should hold dot products")
  require(normShift >= 0, "normShift must be non-negative")

  private val indexWidth = math.max(1, log2Ceil(contextLength))
  private val countWidth = log2Ceil(contextLength + 1)

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))

    val qWeight   = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val qBias     = Input(Vec(lanes, SInt(accWidth.W)))
    val kWeight   = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val kBias     = Input(Vec(lanes, SInt(accWidth.W)))
    val vWeight   = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val vBias     = Input(Vec(lanes, SInt(accWidth.W)))
    val outWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val outBias   = Input(Vec(lanes, SInt(accWidth.W)))

    val ready = Output(Bool())
    val busy  = Output(Bool())
    val done  = Output(Bool())

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

  private def narrowToData(v: SInt): SInt = FixedPointMath.saturate(v, dataWidth)
  private def zeroCache = VecInit(Seq.fill(contextLength)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))

  val idle :: launchQ :: waitQ :: launchK :: waitK :: launchV :: waitV :: computeScores :: launchOut :: waitOut :: doneState :: Nil = Enum(11)
  val state   = RegInit(idle)
  val doneReg = RegInit(false.B)

  val xReg     = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))
  val projQReg = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))
  val projKReg = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))
  val projVReg = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))

  val scoresReg  = RegInit(VecInit(Seq.fill(contextLength)(0.S(accWidth.W))))
  val weightsReg = RegInit(VecInit(Seq.fill(contextLength)(0.S(accWidth.W))))
  val rawYReg    = RegInit(VecInit(Seq.fill(lanes)(0.S(accWidth.W))))
  val yReg       = RegInit(VecInit(Seq.fill(lanes)(0.S(accWidth.W))))

  val keyCache   = RegInit(zeroCache)
  val valueCache = RegInit(zeroCache)
  val writeIndex = RegInit(0.U(indexWidth.W))
  val validCount = RegInit(0.U(countWidth.W))

  // Single shared serial linear reused across all four projections
  val linear = Module(new SerialSharedLinear4(dataWidth, accWidth))
  linear.io.clear := io.clear
  linear.io.start := (state === launchQ || state === launchK || state === launchV || state === launchOut)

  linear.io.x := Mux(state === launchOut,
    VecInit(rawYReg.map(narrowToData)),
    xReg)

  linear.io.weight := MuxCase(io.qWeight, Seq(
    (state === launchK)   -> io.kWeight,
    (state === launchV)   -> io.vWeight,
    (state === launchOut) -> io.outWeight
  ))
  linear.io.bias := MuxCase(io.qBias, Seq(
    (state === launchK)   -> io.kBias,
    (state === launchV)   -> io.vBias,
    (state === launchOut) -> io.outBias
  ))

  // Combinational score/value computation (driven from projQReg and cache).
  // After waitV, writeIndex/validCount are already advanced; cache has new K and V.
  val fullCache = validCount === contextLength.U
  val physicalRows = Wire(Vec(contextLength, UInt(indexWidth.W)))
  val rowValid     = Wire(Vec(contextLength, Bool()))
  for (row <- 0 until contextLength) {
    val shifted    = writeIndex + row.U
    val wrappedRow = Mux(shifted >= contextLength.U, shifted - contextLength.U, shifted)
    physicalRows(row) := Mux(fullCache, wrappedRow, row.U)
    rowValid(row) := row.U < validCount
  }

  val scoresWire  = Wire(Vec(contextLength, SInt(accWidth.W)))
  val weightsWire = Wire(Vec(contextLength, SInt(accWidth.W)))
  val rawYWire    = Wire(Vec(lanes, SInt(accWidth.W)))

  for (row <- 0 until contextLength) {
    val products = Wire(Vec(lanes, SInt(accWidth.W)))
    for (lane <- 0 until lanes) {
      products(lane) := SignedMath.resize(projQReg(lane) * keyCache(physicalRows(row))(lane), accWidth)
    }
    scoresWire(row)  := Mux(rowValid(row), products.reduce(_ +& _), 0.S)
    weightsWire(row) := FixedPointMath.roundedShiftRight(scoresWire(row), normShift)
  }
  for (lane <- 0 until lanes) {
    val weighted = Wire(Vec(contextLength, SInt(accWidth.W)))
    for (row <- 0 until contextLength) {
      weighted(row) := Mux(
        rowValid(row),
        SignedMath.resize(weightsWire(row) * valueCache(physicalRows(row))(lane), accWidth),
        0.S
      )
    }
    rawYWire(lane) := weighted.reduce(_ +& _)
  }

  when(io.clear) {
    state      := idle
    doneReg    := false.B
    xReg       := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    projQReg   := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    projKReg   := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    projVReg   := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    scoresReg  := VecInit(Seq.fill(contextLength)(0.S(accWidth.W)))
    weightsReg := VecInit(Seq.fill(contextLength)(0.S(accWidth.W)))
    rawYReg    := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
    yReg       := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
    keyCache   := zeroCache
    valueCache := zeroCache
    writeIndex := 0.U
    validCount := 0.U
  }.elsewhen(state === idle) {
    doneReg := false.B
    when(io.start) {
      xReg  := io.x
      state := launchQ
    }
  }.elsewhen(state === launchQ) {
    doneReg := false.B
    state   := waitQ
  }.elsewhen(state === waitQ) {
    doneReg := false.B
    when(linear.io.done) {
      for (lane <- 0 until lanes) { projQReg(lane) := narrowToData(linear.io.y(lane)) }
      state := launchK
    }
  }.elsewhen(state === launchK) {
    doneReg := false.B
    state   := waitK
  }.elsewhen(state === waitK) {
    doneReg := false.B
    when(linear.io.done) {
      for (lane <- 0 until lanes) { projKReg(lane) := narrowToData(linear.io.y(lane)) }
      state := launchV
    }
  }.elsewhen(state === launchV) {
    doneReg := false.B
    state   := waitV
  }.elsewhen(state === waitV) {
    doneReg := false.B
    when(linear.io.done) {
      val newProjV = VecInit(linear.io.y.map(narrowToData))
      for (lane <- 0 until lanes) {
        projVReg(lane) := newProjV(lane)
        keyCache(writeIndex)(lane)   := projKReg(lane)
        valueCache(writeIndex)(lane) := newProjV(lane)
      }
      writeIndex := Mux(writeIndex === (contextLength - 1).U, 0.U, writeIndex + 1.U)
      validCount := Mux(validCount === contextLength.U, validCount, validCount + 1.U)
      state := computeScores
    }
  }.elsewhen(state === computeScores) {
    doneReg    := false.B
    scoresReg  := scoresWire
    weightsReg := weightsWire
    rawYReg    := rawYWire
    state      := launchOut
  }.elsewhen(state === launchOut) {
    doneReg := false.B
    state   := waitOut
  }.elsewhen(state === waitOut) {
    doneReg := false.B
    when(linear.io.done) {
      yReg  := linear.io.y
      state := doneState
    }
  }.elsewhen(state === doneState) {
    doneReg := true.B
    state   := idle
  }

  io.ready := state === idle
  io.busy  := state =/= idle
  io.done  := doneReg

  io.q            := projQReg
  io.k            := projKReg
  io.v            := projVReg
  io.scores       := scoresReg
  io.weights      := weightsReg
  io.rawY         := rawYReg
  io.y            := yReg
  io.kvWriteIndex := writeIndex
  io.kvValidCount := validCount
}
