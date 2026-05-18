package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SerialSharedLinear4Spec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SerialSharedLinear4"

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).poke(values(i).S)
    }
  }

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) {
        port(row)(col).poke(values(row)(col).S)
      }
    }
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).expect(values(i).S)
    }
  }

  private def runToDone(dut: SerialSharedLinear4, maxCycles: Int = 20): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        if (dut.io.done.peek().litToBoolean) {
          seenDone = true
        }
      }
    }
    assert(seenDone, s"SerialSharedLinear4 did not finish within $maxCycles cycles")
  }

  it should "compute the same hand-checked projection after 16 MAC cycles" in {
    test(new SerialSharedLinear4()) { dut =>
      pokeVector(dut.io.x, Seq(1, -2, 3, -4))
      pokeMatrix(
        dut.io.weight,
        Seq(
          Seq(1, 0, 0, 0),
          Seq(0, 1, 0, 0),
          Seq(1, 1, 1, 1),
          Seq(2, 0, -1, 1)
        )
      )
      pokeVector(dut.io.bias, Seq(10, 20, 30, 40))
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()

      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.y, Seq(11, 18, 28, 35))
    }
  }

  it should "ignore a second start while busy and use the latched operands" in {
    test(new SerialSharedLinear4()) { dut =>
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeMatrix(
        dut.io.weight,
        Seq(
          Seq(1, 0, 0, 0),
          Seq(0, 1, 0, 0),
          Seq(0, 0, 1, 0),
          Seq(0, 0, 0, 1)
        )
      )
      pokeVector(dut.io.bias, Seq(0, 0, 0, 0))
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()

      pokeVector(dut.io.x, Seq(9, 9, 9, 9))
      dut.io.start.poke(true.B)
      dut.clock.step(2)
      dut.io.start.poke(false.B)
      runToDone(dut)

      expectVector(dut.io.y, Seq(1, 2, 3, 4))
    }
  }

  it should "clear busy state and output registers" in {
    test(new SerialSharedLinear4()) { dut =>
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeMatrix(dut.io.weight, Seq.fill(4)(Seq(1, 1, 1, 1)))
      pokeVector(dut.io.bias, Seq(0, 0, 0, 0))
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()

      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)
      dut.io.clear.poke(true.B)
      dut.clock.step()

      dut.io.clear.poke(false.B)
      dut.io.busy.expect(false.B)
      dut.io.done.expect(false.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.y, Seq(0, 0, 0, 0))
    }
  }
}
