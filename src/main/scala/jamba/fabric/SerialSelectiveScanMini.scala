package jamba.fabric

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.SignedMath

/** Time-multiplexed selective scan: state = a*state + b*x, y = state_new*c.
  *
  * One MacLaneMixed(aWidth=stateWidth, bWidth=dataWidth) is reused across all
  * lanes and three operations per lane, taking 3*lanes cycles per token.
  *
  * Op schedule per lane (matches parallel SelectiveScanMini FSM semantics):
  *   op=0: recurrent = state(lane) * a(lane)
  *   op=1: nextState = recurrent + x(lane) * b(lane)   [stored in nextStateReg]
  *   op=2: y(lane)   = nextState(lane) * c(lane)       [new state, then commit]
  *         state(lane) := nextStateReg(lane)
  */
class SerialSelectiveScanMini(
    lanes:      Int = 4,
    dataWidth:  Int = 8,
    stateWidth: Int = 32,
    accWidth:   Int = 32,
    zeroSkip:   Boolean = false)
    extends Module {
  require(lanes > 0, "SerialSelectiveScanMini lanes must be positive")
  require(dataWidth > 0, "SerialSelectiveScanMini dataWidth must be positive")
  require(stateWidth >= 2 * dataWidth, "SerialSelectiveScanMini stateWidth should hold products")
  require(accWidth >= 2 * dataWidth, "SerialSelectiveScanMini accWidth should hold products")

  private val laneWidth = math.max(1, log2Ceil(lanes))

  val io = IO(new Bundle {
    val start     = Input(Bool())
    val clear     = Input(Bool())
    val loadState = Input(Bool())
    val stateIn   = Input(Vec(lanes, SInt(stateWidth.W)))
    val x         = Input(Vec(lanes, SInt(dataWidth.W)))
    val a         = Input(Vec(lanes, SInt(dataWidth.W)))
    val b         = Input(Vec(lanes, SInt(dataWidth.W)))
    val c         = Input(Vec(lanes, SInt(dataWidth.W)))
    val ready     = Output(Bool())
    val busy      = Output(Bool())
    val done      = Output(Bool())
    val laneIndex = Output(UInt(laneWidth.W))
    val stateOut  = Output(Vec(lanes, SInt(stateWidth.W)))
    val y         = Output(Vec(lanes, SInt(accWidth.W)))
  })

  // Persistent state across tokens (only cleared by io.clear)
  val stateReg = RegInit(VecInit(Seq.fill(lanes)(0.S(stateWidth.W))))

  // Input snapshots latched on start
  val xReg = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))
  val aReg = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))
  val bReg = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))
  val cReg = RegInit(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))

  // Output registers
  val yReg         = RegInit(VecInit(Seq.fill(lanes)(0.S(accWidth.W))))
  val nextStateReg = RegInit(VecInit(Seq.fill(lanes)(0.S(stateWidth.W))))

  // FSM counters
  val lane    = RegInit(0.U(laneWidth.W))
  val op      = RegInit(0.U(2.W))   // 0, 1, 2
  val acc     = RegInit(0.S(stateWidth.W))
  val busyReg = RegInit(false.B)
  val doneReg = RegInit(false.B)

  // Single shared MAC: wide × narrow → wide
  val mac = Module(new MacLaneMixed(stateWidth, dataWidth, stateWidth, zeroSkip))

  // op=0: state(lane) * a(lane)
  // op=1: x(lane)     * b(lane) + acc  (acc holds recurrent from op=0)
  // op=2: nextState(lane) * c(lane)
  mac.io.a := Mux(op === 1.U,
    SignedMath.resize(xReg(lane), stateWidth),
    Mux(op === 2.U, nextStateReg(lane), stateReg(lane)))
  mac.io.b := Mux(op === 0.U, aReg(lane),
               Mux(op === 1.U, bReg(lane), cReg(lane)))
  mac.io.accIn := Mux(op === 1.U, acc, 0.S)

  when(io.clear) {
    stateReg     := VecInit(Seq.fill(lanes)(0.S(stateWidth.W)))
    xReg         := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    aReg         := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    bReg         := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    cReg         := VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))
    yReg         := VecInit(Seq.fill(lanes)(0.S(accWidth.W)))
    nextStateReg := VecInit(Seq.fill(lanes)(0.S(stateWidth.W)))
    lane         := 0.U
    op           := 0.U
    acc          := 0.S
    busyReg      := false.B
    doneReg      := false.B
  }.elsewhen(io.loadState && !busyReg) {
    stateReg := io.stateIn
  }.elsewhen(io.start && !busyReg) {
    xReg    := io.x
    aReg    := io.a
    bReg    := io.b
    cReg    := io.c
    lane    := 0.U
    op      := 0.U
    acc     := 0.S
    busyReg := true.B
    doneReg := false.B
  }.elsewhen(busyReg) {
    doneReg := false.B
    val macResult = mac.io.accOut

    when(op === 0.U) {
      // recurrent = stateReg(lane) * a(lane)
      acc := macResult
      op  := 1.U
    }.elsewhen(op === 1.U) {
      // nextState = recurrent + x(lane) * b(lane)
      nextStateReg(lane) := macResult
      acc := 0.S
      op  := 2.U
    }.otherwise {
      // y(lane) = nextState(lane) * c(lane); commit state
      yReg(lane)      := SignedMath.resize(macResult, accWidth)
      stateReg(lane)  := nextStateReg(lane)
      op              := 0.U
      acc             := 0.S

      when(lane === (lanes - 1).U) {
        lane    := 0.U
        busyReg := false.B
        doneReg := true.B
      }.otherwise {
        lane := lane + 1.U
      }
    }
  }.otherwise {
    doneReg := false.B
  }

  io.ready     := !busyReg
  io.busy      := busyReg
  io.done      := doneReg
  io.laneIndex := lane
  io.stateOut  := stateReg
  io.y         := yReg
}
