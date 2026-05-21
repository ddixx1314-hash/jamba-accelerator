package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ConfigurableSerialLinear4Spec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "ConfigurableSerialLinear4"

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit =
    for (i <- values.indices) port(i).poke(values(i).S)

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) {
        port(row)(col).poke(values(row)(col).S)
      }
    }
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit =
    for (i <- values.indices) port(i).expect(values(i).S)

  private def runToDone(dut: ConfigurableSerialLinear4, maxCycles: Int = 30): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"ConfigurableSerialLinear4 did not finish within $maxCycles cycles")
  }

  private def runConfig(macLanes: Int, x: Seq[Int], weight: Seq[Seq[Int]], bias: Seq[Int], expected: Seq[Int]): Unit = {
    test(new ConfigurableSerialLinear4(macLanes = macLanes)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x)
      pokeMatrix(dut.io.weight, weight)
      pokeVector(dut.io.bias, bias)
      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.y, expected)
    }
  }

  it should "match identity, reverse, bias, zero, and signed cases for all MAC-lane counts" in {
    val cases = Seq(
      (
        "identity",
        Seq(1, 2, 3, 4),
        Seq(Seq(1, 0, 0, 0), Seq(0, 1, 0, 0), Seq(0, 0, 1, 0), Seq(0, 0, 0, 1)),
        Seq(0, 0, 0, 0),
        Seq(1, 2, 3, 4)
      ),
      (
        "reverse identity",
        Seq(1, 2, 3, 4),
        Seq(Seq(0, 0, 0, 1), Seq(0, 0, 1, 0), Seq(0, 1, 0, 0), Seq(1, 0, 0, 0)),
        Seq(0, 0, 0, 0),
        Seq(4, 3, 2, 1)
      ),
      (
        "bias",
        Seq(1, 2, 3, 4),
        Seq.fill(4)(Seq(1, 1, 1, 1)),
        Seq(1, 2, 3, 4),
        Seq(11, 12, 13, 14)
      ),
      (
        "zero",
        Seq(0, 0, 0, 0),
        Seq.fill(4)(Seq(7, -2, 5, 1)),
        Seq(0, 0, 0, 0),
        Seq(0, 0, 0, 0)
      ),
      (
        "mixed signed",
        Seq(1, -2, 3, -4),
        Seq(Seq(1, 0, 0, 0), Seq(0, 1, 0, 0), Seq(1, 1, 1, 1), Seq(2, 0, -1, 1)),
        Seq(10, 20, 30, 40),
        Seq(11, 18, 28, 35)
      )
    )

    for ((_, x, weight, bias, expected) <- cases; macLanes <- Seq(1, 2, 4)) {
      runConfig(macLanes, x, weight, bias, expected)
    }
  }

  it should "produce identical output to SerialSharedLinear4 when macLanes is 1" in {
    val x = Seq(2, -3, 4, -5)
    val weight = Seq(
      Seq(1, 2, 0, -1),
      Seq(0, -1, 3, 1),
      Seq(2, 0, -2, 1),
      Seq(-1, 1, 1, 0)
    )
    val bias = Seq(7, -4, 5, 2)
    var legacyOut = Seq.fill(4)(0L)
    var configurableOut = Seq.fill(4)(0L)

    test(new SerialSharedLinear4()) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x)
      pokeMatrix(dut.io.weight, weight)
      pokeVector(dut.io.bias, bias)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      var seenDone = false
      for (_ <- 0 until 30) {
        if (!seenDone) {
          dut.clock.step()
          seenDone = dut.io.done.peek().litToBoolean
        }
      }
      assert(seenDone, "SerialSharedLinear4 should finish")
      legacyOut = (0 until 4).map(i => dut.io.y(i).peek().litValue.toLong)
    }

    test(new ConfigurableSerialLinear4(macLanes = 1)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x)
      pokeMatrix(dut.io.weight, weight)
      pokeVector(dut.io.bias, bias)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      configurableOut = (0 until 4).map(i => dut.io.y(i).peek().litValue.toLong)
    }

    assert(configurableOut == legacyOut,
      s"macLanes=1 changed legacy behavior: configurable=$configurableOut legacy=$legacyOut")
  }

  it should "clear busy state and output registers" in {
    test(new ConfigurableSerialLinear4(macLanes = 2)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeMatrix(dut.io.weight, Seq.fill(4)(Seq(1, 1, 1, 1)))
      pokeVector(dut.io.bias, Seq(0, 0, 0, 0))
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
      expectVector(dut.io.y, Seq(0, 0, 0, 0))
    }
  }
}
