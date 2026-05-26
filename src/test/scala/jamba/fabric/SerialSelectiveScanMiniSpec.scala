package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SerialSelectiveScanMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SerialSelectiveScanMini"

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).poke(values(i).S)
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).expect(values(i).S)
  }

  private def runToDone(dut: SerialSelectiveScanMini, maxCycles: Int = 20): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"SerialSelectiveScanMini did not finish within $maxCycles cycles")
  }

  // Golden: nextState = state*a + x*b, y = nextState*c (new state)
  it should "compute one token: state update and output with identity coefficients" in {
    test(new SerialSelectiveScanMini()) { dut =>
      dut.io.clear.poke(false.B)
      // Initial state = [0,0,0,0], x=[1,2,3,4], a=b=c=[1,1,1,1]
      // nextState = 0*1 + [1,2,3,4]*1 = [1,2,3,4]
      // y = nextState * c = [1,2,3,4]
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(1, 1, 1, 1))
      pokeVector(dut.io.c, Seq(1, 1, 1, 1))

      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.stateOut, Seq(1, 2, 3, 4))
      expectVector(dut.io.y, Seq(1, 2, 3, 4))
    }
  }

  it should "accumulate state across two tokens" in {
    test(new SerialSelectiveScanMini()) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(1, 1, 1, 1))
      pokeVector(dut.io.c, Seq(1, 1, 1, 1))

      // Token 1: x=[1,0,0,0], state=[0,0,0,0]
      // nextState = [1,0,0,0], y = [1,0,0,0]
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      expectVector(dut.io.stateOut, Seq(1, 0, 0, 0))
      expectVector(dut.io.y, Seq(1, 0, 0, 0))

      // Token 2: x=[2,0,0,0], state=[1,0,0,0]
      // nextState = 1*1 + 2*1 = 3 (lane 0), 0 others
      // y = nextState * c = [3,0,0,0]
      pokeVector(dut.io.x, Seq(2, 0, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      expectVector(dut.io.stateOut, Seq(3, 0, 0, 0))
      expectVector(dut.io.y, Seq(3, 0, 0, 0))
    }
  }

  // ---- M12-P: Power-of-Two A matrix (useShiftA=true) ----

  it should "compute correct output with useShiftA=true (shift-A golden model)" in {
    // Golden: nextState[i] = (state[i] >> a[i]) + x[i]*b[i]
    //         y[i]          = nextState[i] * c[i]
    // With state=[0,0,0,0], a=[1,2,0,3], x=[4,8,2,6], b=c=[1,1,1,1]
    // nextState[0] = (0>>1) + 4*1 = 0+4 = 4
    // nextState[1] = (0>>2) + 8*1 = 0+8 = 8
    // nextState[2] = (0>>0) + 2*1 = 0+2 = 2
    // nextState[3] = (0>>3) + 6*1 = 0+6 = 6
    // y = [4, 8, 2, 6]
    test(new SerialSelectiveScanMini(useShiftA = true)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(4, 8, 2, 6))
      pokeVector(dut.io.a, Seq(1, 2, 0, 3))  // shift amounts
      pokeVector(dut.io.b, Seq(1, 1, 1, 1))
      pokeVector(dut.io.c, Seq(1, 1, 1, 1))

      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      expectVector(dut.io.stateOut, Seq(4, 8, 2, 6))
      expectVector(dut.io.y, Seq(4, 8, 2, 6))
    }
  }

  it should "accumulate state correctly across two tokens with useShiftA=true" in {
    // Token 1: state=[0,0,0,0], a=[1,1,0,0], x=[4,8,0,0], b=c=[1,1,1,1]
    //   nextState[0] = (0>>1) + 4 = 4,  nextState[1] = (0>>1) + 8 = 8
    //   y = [4, 8, 0, 0]
    // Token 2: state=[4,8,0,0], a=[1,1,0,0], x=[2,1,0,0], b=c=[1,1,1,1]
    //   nextState[0] = (4>>1) + 2 = 2+2 = 4,  nextState[1] = (8>>1) + 1 = 4+1 = 5
    //   y = [4, 5, 0, 0]
    test(new SerialSelectiveScanMini(useShiftA = true)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.a, Seq(1, 1, 0, 0))
      pokeVector(dut.io.b, Seq(1, 1, 1, 1))
      pokeVector(dut.io.c, Seq(1, 1, 1, 1))

      // Token 1
      pokeVector(dut.io.x, Seq(4, 8, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      expectVector(dut.io.stateOut, Seq(4, 8, 0, 0))
      expectVector(dut.io.y, Seq(4, 8, 0, 0))

      // Token 2
      pokeVector(dut.io.x, Seq(2, 1, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      expectVector(dut.io.stateOut, Seq(4, 5, 0, 0))
      expectVector(dut.io.y, Seq(4, 5, 0, 0))
    }
  }

  it should "run 2*lanes cycles (not 3*lanes) per token with useShiftA=true" in {
    // useShiftA=false: 3*lanes = 12 cycles for lanes=4
    // useShiftA=true:  2*lanes = 8 cycles for lanes=4  → saves lanes=4 cycles
    def countCyclesToDone(dut: SerialSelectiveScanMini): Int = {
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      var cycles = 0
      while (!dut.io.done.peek().litToBoolean && cycles < 30) {
        dut.clock.step()
        cycles += 1
      }
      assert(dut.io.done.peek().litToBoolean, s"SSM did not finish (cycles=$cycles)")
      cycles
    }

    val lanesN = 4
    val expectedStandard = 3 * lanesN  // 12
    val expectedShiftA   = 2 * lanesN  // 8

    val x = Seq(1, 2, 3, 4)
    val a = Seq(1, 1, 1, 1)
    val b = Seq(1, 1, 1, 1)
    val c = Seq(1, 1, 1, 1)

    var cyclesStandard = 0
    var cyclesShiftA   = 0

    test(new SerialSelectiveScanMini(lanes = lanesN, useShiftA = false)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x); pokeVector(dut.io.a, a)
      pokeVector(dut.io.b, b); pokeVector(dut.io.c, c)
      cyclesStandard = countCyclesToDone(dut)
    }

    test(new SerialSelectiveScanMini(lanes = lanesN, useShiftA = true)) { dut =>
      dut.io.clear.poke(false.B)
      // shift by 0 = multiply by 1 (same as a=1 in standard mode for first token, state=0)
      pokeVector(dut.io.x, x); pokeVector(dut.io.a, Seq(0, 0, 0, 0))
      pokeVector(dut.io.b, b); pokeVector(dut.io.c, c)
      cyclesShiftA = countCyclesToDone(dut)
    }

    assert(cyclesStandard == expectedStandard,
      s"Standard SSM should take $expectedStandard cycles, got $cyclesStandard")
    assert(cyclesShiftA == expectedShiftA,
      s"ShiftA SSM should take $expectedShiftA cycles, got $cyclesShiftA")
    assert(cyclesStandard - cyclesShiftA == lanesN,
      s"useShiftA should save exactly lanes=$lanesN cycles: $cyclesStandard - $cyclesShiftA")
  }

  it should "produce identical output for first token when a=0 (shiftA) ↔ a=1 (standard)" in {
    // state=0 initially, so:
    //   standard:  nextState = state*1 + x*b = 0 + x*b = x*b
    //   shiftA a=0: nextState = (state>>0) + x*b = state + x*b = 0 + x*b = x*b
    // Outputs must be identical.
    val x = Seq(3, 7, 1, 5)
    val b = Seq(2, 1, 3, 1)
    val c = Seq(1, 2, 1, 1)

    var stdOut   = Seq.empty[Long]
    var shiftOut = Seq.empty[Long]

    test(new SerialSelectiveScanMini(useShiftA = false)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x); pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, b); pokeVector(dut.io.c, c)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      runToDone(dut)
      stdOut = dut.io.y.map(_.peek().litValue.toLong)
    }

    test(new SerialSelectiveScanMini(useShiftA = true)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x); pokeVector(dut.io.a, Seq(0, 0, 0, 0))  // shift by 0 = ×1
      pokeVector(dut.io.b, b); pokeVector(dut.io.c, c)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      runToDone(dut)
      shiftOut = dut.io.y.map(_.peek().litValue.toLong)
    }

    assert(stdOut == shiftOut,
      s"First-token output must match: standard=$stdOut vs shiftA=$shiftOut")
  }

  // ---- M14-C: lanes=8 correctness and cycle-count tests ----

  it should "produce correct output for lanes=8 with identity coefficients (standard mode)" in {
    // state=[0]*8, x=[1..8], a=b=c=[1]*8
    // nextState = 0*1 + x*1 = x; y = nextState*c = x
    val x8 = Seq(1, 2, 3, 4, 5, 6, 7, 8)
    test(new SerialSelectiveScanMini(lanes = 8)) { dut =>
      dut.io.clear.poke(false.B)
      for (i <- 0 until 8) dut.io.x(i).poke(x8(i).S)
      for (i <- 0 until 8) { dut.io.a(i).poke(1.S); dut.io.b(i).poke(1.S); dut.io.c(i).poke(1.S) }
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut, maxCycles = 30)   // lanes=8 standard: 3*8=24 cycles
      dut.io.done.expect(true.B)
      for (i <- 0 until 8) dut.io.y(i).expect(x8(i).S)
      for (i <- 0 until 8) dut.io.stateOut(i).expect(x8(i).S)
      println(s"[M14-C] lanes=8 standard: y=${x8} ✓")
    }
  }

  it should "produce correct output for lanes=8 with useShiftA=true" in {
    // useShiftA: state >> a(lane) instead of state*a; first token state=0, a[i]=0 (shift by 0)
    // nextState = x*b + (0>>0) = x*b = x (b=[1]*8)
    // y = nextState*c = x
    val x8 = Seq(2, 4, 1, 3, 7, 5, 6, 8)
    test(new SerialSelectiveScanMini(lanes = 8, useShiftA = true)) { dut =>
      dut.io.clear.poke(false.B)
      for (i <- 0 until 8) dut.io.x(i).poke(x8(i).S)
      for (i <- 0 until 8) { dut.io.a(i).poke(0.S); dut.io.b(i).poke(1.S); dut.io.c(i).poke(1.S) }
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut, maxCycles = 25)   // lanes=8 useShiftA: 2*8=16 cycles
      dut.io.done.expect(true.B)
      for (i <- 0 until 8) dut.io.y(i).expect(x8(i).S)
      println(s"[M14-C] lanes=8 useShiftA: y=${x8} ✓")
    }
  }

  it should "finish in fewer cycles for lanes=8 useShiftA vs standard (2-op vs 3-op Pareto)" in {
    // Standard lanes=8: 3*8=24 cycles; useShiftA lanes=8: 2*8=16 cycles
    val x8 = Seq(1, 2, 3, 4, 5, 6, 7, 8)

    def measure(useShift: Boolean): Int = {
      var cycles = 0
      test(new SerialSelectiveScanMini(lanes = 8, useShiftA = useShift)) { dut =>
        dut.io.clear.poke(false.B)
        for (i <- 0 until 8) dut.io.x(i).poke(x8(i).S)
        for (i <- 0 until 8) { dut.io.a(i).poke(0.S); dut.io.b(i).poke(1.S); dut.io.c(i).poke(1.S) }
        dut.io.start.poke(true.B)
        dut.clock.step(); cycles += 1
        dut.io.start.poke(false.B)
        var limit = 30
        while (!dut.io.done.peek().litToBoolean && limit > 0) {
          dut.clock.step(); cycles += 1; limit -= 1
        }
        assert(dut.io.done.peek().litToBoolean, s"lanes=8 useShiftA=$useShift did not finish in 30 cycles")
      }
      cycles
    }

    val cyclesStd   = measure(useShift = false)
    val cyclesShift = measure(useShift = true)
    println(s"[M14-C] lanes=8 cycles: standard=$cyclesStd  useShiftA=$cyclesShift  saved=${cyclesStd - cyclesShift}")
    assert(cyclesShift < cyclesStd,
      s"useShiftA should be faster for lanes=8: shift=$cyclesShift std=$cyclesStd")
    assert(cyclesStd - cyclesShift >= 8,
      s"useShiftA should save at least lanes=8 cycles: saved=${cyclesStd - cyclesShift}")
  }

  it should "clear resets persistent state to zero" in {
    test(new SerialSelectiveScanMini()) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(5, 5, 5, 5))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(1, 1, 1, 1))
      pokeVector(dut.io.c, Seq(1, 1, 1, 1))

      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.clear.poke(false.B)

      dut.io.ready.expect(true.B)
      dut.io.busy.expect(false.B)
      dut.io.done.expect(false.B)
      expectVector(dut.io.stateOut, Seq(0, 0, 0, 0))
      expectVector(dut.io.y, Seq(0, 0, 0, 0))
    }
  }
}
