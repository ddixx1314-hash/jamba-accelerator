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
