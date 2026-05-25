package jamba.fabric

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.SignedMath

/** Time-multiplexed linear projection with configurable MAC parallelism.
  *
  * Data path:
  *
  *   x / W[row] -> column chunk scheduler -> macLanes partial accumulators
  *              -> combinational reduction adder -> + bias[row] -> y[row]
  *
  * Each MacLane owns one column position per cycle. For lanes=4:
  *   - macLanes=1: 4 chunks/row, 16 cycles total
  *   - macLanes=2: 2 chunks/row,  8 cycles total
  *   - macLanes=4: 1 chunk /row,  4 cycles total
  *
  * Bias is added once at the end of each row, after reducing all partial sums.
  *
  * M12-A Column Skip (columnSkip=true, macLanes==1 only):
  *   Switches to column-major sparse iteration.  A combinational priority encoder
  *   finds the next non-zero input column at each column boundary, so zero-valued
  *   columns are skipped in zero additional cycles (look-ahead rather than check-cycle).
  *
  *   Cycles = k × lanes  where k = number of non-zero input columns.
  *   Special case k=0: 1 cycle (all-zero commit).
  *   Compare standard: lanes² = 16 cycles for lanes=4.
  *     k=0: 1  (vs 16)   k=2: 8  (vs 16)   k=4: 16  (vs 16 — identical, zero overhead).
  *
  *   For macLanes > 1, columnSkip is silently ignored (falls through to standard path).
  */
class ConfigurableSerialLinear4(
    dataWidth:  Int     = 8,
    accWidth:   Int     = 32,
    lanes:      Int     = 4,
    macLanes:   Int     = 1,
    zeroSkip:   Boolean = false,
    columnSkip: Boolean = false)   // M12-A: column-major sparse iteration (macLanes=1 only)
    extends Module {
  require(dataWidth > 0, "ConfigurableSerialLinear4 dataWidth must be positive")
  require(accWidth >= 2 * dataWidth + 2, "ConfigurableSerialLinear4 accWidth should hold four products")
  require(lanes > 0, "ConfigurableSerialLinear4 lanes must be positive")
  require(macLanes >= 1, "ConfigurableSerialLinear4 macLanes must be positive")
  require(macLanes <= lanes, "ConfigurableSerialLinear4 macLanes must be <= lanes")
  require(lanes % macLanes == 0, "ConfigurableSerialLinear4 lanes must be divisible by macLanes")

  // Is the sparse column-major path active?
  private val sparseActive = columnSkip && (macLanes == 1)

  private val chunksPerRow = lanes / macLanes
  private val indexWidth   = math.max(1, log2Ceil(lanes))
  private val chunkWidth   = math.max(1, log2Ceil(chunksPerRow))
  // Width sufficient to hold values 0..lanes (inclusive)
  private val colPlusWidth = math.max(1, log2Ceil(lanes + 1))

  val io = IO(new Bundle {
    val start    = Input(Bool())
    val clear    = Input(Bool())
    val x        = Input(Vec(lanes, SInt(dataWidth.W)))
    val weight   = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val bias     = Input(Vec(lanes, SInt(accWidth.W)))
    val ready    = Output(Bool())
    val busy     = Output(Bool())
    val done     = Output(Bool())
    val rowIndex = Output(UInt(indexWidth.W))
    val colIndex = Output(UInt(indexWidth.W))
    val y        = Output(Vec(lanes, SInt(accWidth.W)))
  })

  private def zeroY        = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
  private def zeroPartials = VecInit(Seq.fill(macLanes)(0.S(accWidth.W)))
  private def zeroRowAcc   = VecInit(Seq.fill(lanes)(0.S(accWidth.W)))

  // -----------------------------------------------------------------------
  // Shared registers (used in both paths)
  // -----------------------------------------------------------------------
  val xReg      = Reg(Vec(lanes, SInt(dataWidth.W)))
  val weightReg = Reg(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
  val biasReg   = Reg(Vec(lanes, SInt(accWidth.W)))
  val yReg      = RegInit(zeroY)
  val busyReg   = RegInit(false.B)
  val doneReg   = RegInit(false.B)

  // -----------------------------------------------------------------------
  // Standard path registers (row-major)
  // -----------------------------------------------------------------------
  val row        = RegInit(0.U(indexWidth.W))
  val chunk      = RegInit(0.U(chunkWidth.W))
  val partialAcc = RegInit(zeroPartials)

  // -----------------------------------------------------------------------
  // Sparse path registers (column-major, M12-A; declared unconditionally
  // but only wired/driven when sparseActive=true)
  // -----------------------------------------------------------------------
  val spCol     = RegInit(0.U(colPlusWidth.W))   // current non-zero column; lanes means "commit"
  val spRow     = RegInit(0.U(indexWidth.W))
  val rowAccReg = RegInit(zeroRowAcc)            // per-row accumulator (indexed by spRow)

  // -----------------------------------------------------------------------
  // Single MAC bank
  // -----------------------------------------------------------------------
  val macs         = Seq.fill(macLanes)(Module(new MacLane(dataWidth, accWidth, zeroSkip)))
  val nextPartials = Wire(Vec(macLanes, SInt(accWidth.W)))

  // -----------------------------------------------------------------------
  // Combinational: priority encoders for sparse path
  // -----------------------------------------------------------------------
  // first non-zero col in io.x (used at start, before xReg latched)
  val firstNonZeroColWire = Wire(UInt(colPlusWidth.W))
  firstNonZeroColWire := lanes.U
  if (sparseActive) {
    for (c <- (0 until lanes).reverse) {
      when(io.x(c) =/= 0.S) { firstNonZeroColWire := c.U }
    }
  }

  // next non-zero col in xReg after current spCol (used during sparse FSM)
  val nextNonZeroColWire = Wire(UInt(colPlusWidth.W))
  nextNonZeroColWire := lanes.U
  if (sparseActive) {
    for (c <- (0 until lanes).reverse) {
      when(c.U > spCol && xReg(c) =/= 0.S) { nextNonZeroColWire := c.U }
    }
  }

  // Safe index into xReg/weightReg for the sparse path (clips spCol to valid range)
  val spColIdx = spCol(indexWidth - 1, 0)

  // -----------------------------------------------------------------------
  // MAC combinational wiring
  // -----------------------------------------------------------------------
  if (sparseActive) {
    // Sparse column-major: one MAC, indexed by (spRow, spCol)
    macs(0).io.a    := xReg(spColIdx)
    macs(0).io.b    := weightReg(spRow)(spColIdx)
    macs(0).io.accIn := rowAccReg(spRow)
    nextPartials(0) := SignedMath.resize(macs(0).io.accOut, accWidth)
  } else {
    // Standard row-major: macLanes MACs process macLanes columns in parallel
    for (lane <- 0 until macLanes) {
      val rawCol = chunk * macLanes.U + lane.U
      val col    = rawCol(indexWidth - 1, 0).asUInt
      macs(lane).io.a    := xReg(col)
      macs(lane).io.b    := weightReg(row)(col)
      macs(lane).io.accIn := partialAcc(lane)
      nextPartials(lane) := SignedMath.resize(macs(lane).io.accOut, accWidth)
    }
  }

  private def reduce(values: Seq[SInt]): SInt =
    values.reduce((a, b) => SignedMath.resize(a +& b, accWidth))

  val rowPartialSum = reduce((0 until macLanes).map(nextPartials(_)))
  val rowWithBias   = SignedMath.resize(rowPartialSum +& biasReg(row), accWidth)
  val lastChunk     = chunk === (chunksPerRow - 1).U

  // -----------------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------------
  when(io.clear) {
    row        := 0.U
    chunk      := 0.U
    partialAcc := zeroPartials
    yReg       := zeroY
    spCol      := 0.U
    spRow      := 0.U
    rowAccReg  := zeroRowAcc
    busyReg    := false.B
    doneReg    := false.B
  }.elsewhen(io.start && !busyReg) {
    xReg      := io.x
    weightReg := io.weight
    biasReg   := io.bias
    yReg      := zeroY
    busyReg   := true.B
    doneReg   := false.B
    if (sparseActive) {
      // Sparse path: jump directly to the first non-zero column (combinational look-ahead)
      spCol     := firstNonZeroColWire
      spRow     := 0.U
      rowAccReg := zeroRowAcc
    } else {
      row        := 0.U
      chunk      := 0.U
      partialAcc := zeroPartials
    }
  }.elsewhen(busyReg) {
    doneReg := false.B

    if (sparseActive) {
      // ---- M12-A Sparse column-major FSM ----
      when(spCol >= lanes.U) {
        // All non-zero columns processed (or k=0): commit y = rowAccReg + bias
        for (r <- 0 until lanes) {
          yReg(r) := SignedMath.resize(rowAccReg(r) +& biasReg(r), accWidth)
        }
        busyReg := false.B
        doneReg := true.B
      }.otherwise {
        // COMPUTE_ROW: accumulate x[spCol]*w[spRow][spCol] into rowAccReg[spRow]
        rowAccReg(spRow) := nextPartials(0)  // = x[spCol]*w[spRow][spCol] + rowAccReg[spRow]

        when(spRow === (lanes - 1).U) {
          // Last row of current column: look ahead for next non-zero column
          spRow := 0.U
          spCol := nextNonZeroColWire
          // If no more columns, next busyReg cycle will trigger COMMIT
        }.otherwise {
          spRow := spRow + 1.U
        }
      }
    } else {
      // ---- Standard row-major FSM ----
      when(lastChunk) {
        yReg(row) := rowWithBias
        chunk      := 0.U
        partialAcc := zeroPartials

        when(row === (lanes - 1).U) {
          row     := 0.U
          busyReg := false.B
          doneReg := true.B
        }.otherwise {
          row := row + 1.U
        }
      }.otherwise {
        chunk      := chunk + 1.U
        partialAcc := nextPartials
      }
    }
  }.otherwise {
    doneReg := false.B
  }

  io.ready    := !busyReg
  io.busy     := busyReg
  io.done     := doneReg
  io.rowIndex := Mux(busyReg && sparseActive.B, spRow, row)
  io.colIndex := Mux(busyReg && sparseActive.B, spColIdx, chunk * macLanes.U)
  io.y        := yReg
}
