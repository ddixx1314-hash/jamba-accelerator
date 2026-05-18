package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SerialSemanticProjectionGroupSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "semantic serial projection groups"

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

  private def runMambaToDone(dut: SerialMambaProjectionGroup, maxCycles: Int = 80): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"SerialMambaProjectionGroup did not finish within $maxCycles cycles")
  }

  private def runAttentionToDone(dut: SerialAttentionProjectionGroup, maxCycles: Int = 110): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"SerialAttentionProjectionGroup did not finish within $maxCycles cycles")
  }

  it should "name the three serial Mamba projection outputs" in {
    test(new SerialMambaProjectionGroup()) { dut =>
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeIdentity(dut.io.inputWeight)
      pokeVector(dut.io.inputBias, Seq(0, 0, 0, 0))
      pokeReverseIdentity(dut.io.bWeight)
      pokeVector(dut.io.bBias, Seq(10, 10, 10, 10))
      pokeMatrix(dut.io.cWeight, Seq.fill(4)(Seq(1, 1, 1, 1)))
      pokeVector(dut.io.cBias, Seq(0, 1, 2, 3))

      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runMambaToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.projectedRaw, Seq(1, 2, 3, 4))
      expectVector(dut.io.bRaw, Seq(14, 13, 12, 11))
      expectVector(dut.io.cRaw, Seq(10, 11, 12, 13))
      expectVector(dut.io.projected, Seq(1, 2, 3, 4))
      expectVector(dut.io.b, Seq(14, 13, 12, 11))
      expectVector(dut.io.c, Seq(10, 11, 12, 13))
    }
  }

  it should "schedule attention Q/K/V from token input and out projection from separate input" in {
    test(new SerialAttentionProjectionGroup()) { dut =>
      pokeVector(dut.io.x, Seq(2, -1, 0, 3))
      pokeVector(dut.io.outInput, Seq(4, 5, 6, 7))
      pokeIdentity(dut.io.qWeight)
      pokeVector(dut.io.qBias, Seq(0, 0, 0, 0))
      pokeReverseIdentity(dut.io.kWeight)
      pokeVector(dut.io.kBias, Seq(0, 0, 0, 0))
      pokeMatrix(dut.io.vWeight, Seq.fill(4)(Seq(1, 0, 0, 0)))
      pokeVector(dut.io.vBias, Seq(0, 1, 2, 3))
      pokeMatrix(dut.io.outWeight, Seq.fill(4)(Seq(0, 0, 0, 1)))
      pokeVector(dut.io.outBias, Seq(4, 3, 2, 1))

      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()

      pokeVector(dut.io.x, Seq(9, 9, 9, 9))
      pokeVector(dut.io.outInput, Seq(1, 1, 1, 1))
      dut.io.start.poke(true.B)
      dut.clock.step(2)
      dut.io.start.poke(false.B)
      runAttentionToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.qRaw, Seq(2, -1, 0, 3))
      expectVector(dut.io.kRaw, Seq(3, 0, -1, 2))
      expectVector(dut.io.vRaw, Seq(2, 3, 4, 5))
      expectVector(dut.io.q, Seq(2, -1, 0, 3))
      expectVector(dut.io.k, Seq(3, 0, -1, 2))
      expectVector(dut.io.v, Seq(2, 3, 4, 5))
      expectVector(dut.io.y, Seq(11, 10, 9, 8))
    }
  }

  it should "saturate serial attention q/k/v outputs" in {
    test(new SerialAttentionProjectionGroup()) { dut =>
      pokeVector(dut.io.x, Seq(100, 0, 0, 0))
      pokeVector(dut.io.outInput, Seq(0, 0, 0, 0))
      pokeMatrix(dut.io.qWeight, Seq.fill(4)(Seq(2, 0, 0, 0)))
      pokeVector(dut.io.qBias, Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.kWeight)
      pokeVector(dut.io.kBias, Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.vWeight)
      pokeVector(dut.io.vBias, Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.outWeight)
      pokeVector(dut.io.outBias, Seq(0, 0, 0, 0))

      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runAttentionToDone(dut)

      expectVector(dut.io.qRaw, Seq(200, 200, 200, 200))
      expectVector(dut.io.q, Seq(127, 127, 127, 127))
    }
  }
}
