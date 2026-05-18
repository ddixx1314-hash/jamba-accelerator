package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SerialProjectionScheduler4Spec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SerialProjectionScheduler4"

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).poke(values(i).S)
    }
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).expect(values(i).S)
    }
  }

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) {
        port(row)(col).poke(values(row)(col).S)
      }
    }
  }

  private def pokeIdentity(port: Vec[Vec[SInt]]): Unit = {
    pokeMatrix(
      port,
      Seq(
        Seq(1, 0, 0, 0),
        Seq(0, 1, 0, 0),
        Seq(0, 0, 1, 0),
        Seq(0, 0, 0, 1)
      )
    )
  }

  private def pokeReverseIdentity(port: Vec[Vec[SInt]]): Unit = {
    pokeMatrix(
      port,
      Seq(
        Seq(0, 0, 0, 1),
        Seq(0, 0, 1, 0),
        Seq(0, 1, 0, 0),
        Seq(1, 0, 0, 0)
      )
    )
  }

  private def runToDone(dut: SerialProjectionScheduler4, maxCycles: Int = 100): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        if (dut.io.done.peek().litToBoolean) {
          seenDone = true
        }
      }
    }
    assert(seenDone, s"SerialProjectionScheduler4 did not finish within $maxCycles cycles")
  }

  it should "schedule three Mamba-style projections through one serial Linear4" in {
    test(new SerialProjectionScheduler4(numProjections = 3)) { dut =>
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))

      pokeIdentity(dut.io.weight(0))
      pokeVector(dut.io.bias(0), Seq(0, 0, 0, 0))
      pokeReverseIdentity(dut.io.weight(1))
      pokeVector(dut.io.bias(1), Seq(10, 10, 10, 10))
      pokeMatrix(dut.io.weight(2), Seq.fill(4)(Seq(1, 1, 1, 1)))
      pokeVector(dut.io.bias(2), Seq(0, 1, 2, 3))

      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()

      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.y(0), Seq(1, 2, 3, 4))
      expectVector(dut.io.y(1), Seq(14, 13, 12, 11))
      expectVector(dut.io.y(2), Seq(10, 11, 12, 13))
    }
  }

  it should "schedule four attention-style projections and keep latched inputs while busy" in {
    test(new SerialProjectionScheduler4(numProjections = 4)) { dut =>
      pokeVector(dut.io.x, Seq(2, -1, 0, 3))
      pokeIdentity(dut.io.weight(0))
      pokeVector(dut.io.bias(0), Seq(0, 0, 0, 0))
      pokeReverseIdentity(dut.io.weight(1))
      pokeVector(dut.io.bias(1), Seq(0, 0, 0, 0))
      pokeMatrix(dut.io.weight(2), Seq.fill(4)(Seq(1, 0, 0, 0)))
      pokeVector(dut.io.bias(2), Seq(0, 1, 2, 3))
      pokeMatrix(dut.io.weight(3), Seq.fill(4)(Seq(0, 0, 0, 1)))
      pokeVector(dut.io.bias(3), Seq(4, 3, 2, 1))

      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()

      pokeVector(dut.io.x, Seq(9, 9, 9, 9))
      dut.io.start.poke(true.B)
      dut.clock.step(2)
      dut.io.start.poke(false.B)
      runToDone(dut)

      expectVector(dut.io.y(0), Seq(2, -1, 0, 3))
      expectVector(dut.io.y(1), Seq(3, 0, -1, 2))
      expectVector(dut.io.y(2), Seq(2, 3, 4, 5))
      expectVector(dut.io.y(3), Seq(7, 6, 5, 4))
    }
  }

  it should "clear an active projection schedule" in {
    test(new SerialProjectionScheduler4(numProjections = 3)) { dut =>
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      for (projection <- 0 until 3) {
        pokeIdentity(dut.io.weight(projection))
        pokeVector(dut.io.bias(projection), Seq(0, 0, 0, 0))
      }

      dut.io.clear.poke(false.B)
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
      for (projection <- 0 until 3) {
        expectVector(dut.io.y(projection), Seq(0, 0, 0, 0))
      }
    }
  }
}
