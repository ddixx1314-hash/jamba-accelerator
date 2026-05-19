package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class LatencyBudgetSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Serial module cycle latencies"

  private def stepsUntilDone(isDone: => Boolean, step: => Unit, max: Int): Int = {
    var n = 0
    while (n < max && !isDone) { step; n += 1 }
    assert(n < max, s"did not complete within $max cycles")
    n
  }

  it should "SerialSharedLinear4 take exactly 16 cycles from start to done" in {
    test(new SerialSharedLinear4()) { dut =>
      for (row <- 0 until 4; col <- 0 until 4)
        dut.io.weight(row)(col).poke((if (row == col) 1 else 0).S)
      for (i <- 0 until 4) {
        dut.io.x(i).poke(0.S)
        dut.io.bias(i).poke(0.S)
      }
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val cycles = stepsUntilDone(
        dut.io.done.peek().litToBoolean,
        dut.clock.step(),
        30
      )
      assert(cycles == 16, s"Expected 16 cycles, got $cycles")
    }
  }

  it should "SerialCausalConvMini take exactly 16 cycles from start to done" in {
    test(new SerialCausalConvMini()) { dut =>
      for (i <- 0 until 4) dut.io.x(i).poke(0.S)
      for (t <- 0 until 4; i <- 0 until 4) dut.io.kernel(t)(i).poke(1.S)
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val cycles = stepsUntilDone(
        dut.io.done.peek().litToBoolean,
        dut.clock.step(),
        30
      )
      assert(cycles == 16, s"Expected 16 cycles, got $cycles")
    }
  }

  it should "SerialSelectiveScanMini take exactly 12 cycles from start to done" in {
    test(new SerialSelectiveScanMini()) { dut =>
      for (i <- 0 until 4) {
        dut.io.x(i).poke(0.S)
        dut.io.a(i).poke(1.S)
        dut.io.b(i).poke(0.S)
        dut.io.c(i).poke(0.S)
      }
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val cycles = stepsUntilDone(
        dut.io.done.peek().litToBoolean,
        dut.clock.step(),
        30
      )
      assert(cycles == 12, s"Expected 12 cycles, got $cycles")
    }
  }
}
