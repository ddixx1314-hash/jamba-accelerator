package jamba.fabric

import chisel3._
import chisel3.util.log2Ceil
import jamba.common.SignedMath

/** Time-multiplexed selective scan: state = a*state + b*x, y = state_new*c.
  *
  * One MacLaneMixed(aWidth=stateWidth, bWidth=dataWidth) is reused across all
  * lanes and three operations per lane, taking 3*lanes cycles per token.
  *
  * Op schedule per lane (standard, useShiftA=false):
  *   op=0: recurrent = state(lane) * a(lane)
  *   op=1: nextState = recurrent + x(lane) * b(lane)   [stored in nextStateReg]
  *   op=2: y(lane)   = nextState(lane) * c(lane)       [new state, then commit]
  *         state(lane) := nextStateReg(lane)
  *
  * M12-P Power-of-Two A (useShiftA=true):
  *   io.a(lane) is reinterpreted as a right-shift amount (non-negative integer).
  *   state * a  ≡  state >> shiftAmt  (arithmetic right shift, power-of-two A).
  *   The shift is folded into the MAC accIn of op=0, eliminating one MAC per lane:
  *   op=0: nextState = mac(x*b + (state>>a))   [accIn = state >> a]
  *   op=1: y = nextState * c; commit state
  *   Total: 2*lanes cycles per token (vs 3*lanes cycles standard).
  *   For lanes=4: 8 cycles vs 12 cycles — saves lanes cycles per token.
  *
  * Note: useShiftA=true assumes io.a values are non-negative (shift amounts 0,1,2,...).
  * Negative values in io.a are reinterpreted as large UInt shifts and produce
  * implementation-defined results (hardware guard not added for area reasons).
  */
class SerialSelectiveScanMini(
    lanes:      Int = 4,
    dataWidth:  Int = 8,
    stateWidth: Int = 32,
    accWidth:   Int = 32,
    zeroSkip:   Boolean = false,
    useShiftA:  Boolean = false)   // M12-P: power-of-two A, shift instead of multiply
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
    // useShiftA=false: a is the recurrent decay coefficient (multiplied with state).
    // useShiftA=true:  a is interpreted as right-shift amount (non-negative integer).
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
  val op      = RegInit(0.U(2.W))   // standard: 0,1,2; useShiftA: 0,1
  val acc     = RegInit(0.S(stateWidth.W))  // used only when !useShiftA
  val busyReg = RegInit(false.B)
  val doneReg = RegInit(false.B)

  // Single shared MAC: wide × narrow → wide
  val mac = Module(new MacLaneMixed(stateWidth, dataWidth, stateWidth, zeroSkip))

  // -----------------------------------------------------------------------
  // MAC combinational wiring (compile-time branch)
  // -----------------------------------------------------------------------
  if (useShiftA) {
    // M12-P schedule (2 ops per lane):
    //   op=0: mac(x, b, accIn=state>>a) → nextState = x*b + (state>>a)
    //   op=1: mac(nextState, c, accIn=0) → y = nextState*c
    mac.io.a    := Mux(op === 0.U,
                       SignedMath.resize(xReg(lane), stateWidth),
                       nextStateReg(lane))
    mac.io.b    := Mux(op === 0.U, bReg(lane), cReg(lane))
    mac.io.accIn := Mux(op === 0.U,
                        // Arithmetic right shift by shift amount in aReg(lane).
                        // aReg(lane).asUInt reinterprets SInt bits as UInt shift amount.
                        stateReg(lane) >> aReg(lane).asUInt,
                        0.S(stateWidth.W))
  } else {
    // Standard 3-op schedule:
    //   op=0: mac(state, a, 0) → recurrent = state*a
    //   op=1: mac(x, b, acc)   → nextState = recurrent + x*b
    //   op=2: mac(nextState, c, 0) → y = nextState*c
    mac.io.a := Mux(op === 1.U,
      SignedMath.resize(xReg(lane), stateWidth),
      Mux(op === 2.U, nextStateReg(lane), stateReg(lane)))
    mac.io.b := Mux(op === 0.U, aReg(lane),
                 Mux(op === 1.U, bReg(lane), cReg(lane)))
    mac.io.accIn := Mux(op === 1.U, acc, 0.S)
  }

  // -----------------------------------------------------------------------
  // FSM
  // -----------------------------------------------------------------------
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

    if (useShiftA) {
      // ---- M12-P: 2-op schedule ----
      when(op === 0.U) {
        // nextState = x*b + (state >> a)  [accIn carries the shifted state]
        nextStateReg(lane) := macResult
        op := 1.U
      }.otherwise { // op === 1
        // y = nextState * c; commit state
        yReg(lane)      := SignedMath.resize(macResult, accWidth)
        stateReg(lane)  := nextStateReg(lane)
        op              := 0.U
        when(lane === (lanes - 1).U) {
          lane    := 0.U
          busyReg := false.B
          doneReg := true.B
        }.otherwise {
          lane := lane + 1.U
        }
      }
    } else {
      // ---- Standard: 3-op schedule ----
      when(op === 0.U) {
        // recurrent = state * a
        acc := macResult
        op  := 1.U
      }.elsewhen(op === 1.U) {
        // nextState = recurrent + x * b
        nextStateReg(lane) := macResult
        acc := 0.S
        op  := 2.U
      }.otherwise {
        // y = nextState * c; commit state
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
