package jamba.top

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class JambaMiniTileSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "JambaMiniTile"

  private val identity = Seq(
    Seq(1, 0, 0, 0),
    Seq(0, 1, 0, 0),
    Seq(0, 0, 1, 0),
    Seq(0, 0, 0, 1)
  )

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

  private def pokeCommon(dut: JambaMiniTile): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.useAttention.poke(false.B)
    dut.io.inValid.poke(false.B)
    dut.io.outReady.poke(false.B)
    pokeVector(dut.io.in, Seq(1, 2, 3, 4))
    pokeVector(dut.io.rmsWeight, Seq(7, 7, 7, 7))
    pokeMatrix(dut.io.inputWeight, identity)
    pokeVector(dut.io.inputBias, Seq(0, 0, 0, 0))
    pokeMatrix(dut.io.gateWeight, identity)
    pokeVector(dut.io.gateBias, Seq(0, 0, 0, 0))
    pokeMatrix(dut.io.bWeight, Seq.fill(4)(Seq(0, 0, 0, 0)))
    pokeVector(dut.io.bBias, Seq(2, 2, 2, 2))
    pokeMatrix(dut.io.cWeight, Seq.fill(4)(Seq(0, 0, 0, 0)))
    pokeVector(dut.io.cBias, Seq(3, 3, 3, 3))
    pokeMatrix(dut.io.outWeight, identity)
    pokeVector(dut.io.outBias, Seq(0, 0, 0, 0))
    pokeMatrix(dut.io.kernel, Seq(Seq(1, 1, 1, 1), Seq(0, 0, 0, 0), Seq(0, 0, 0, 0)))
    pokeVector(dut.io.mambaA, Seq(1, 1, 1, 1))
    pokeMatrix(dut.io.attentionKeys, Seq.fill(4)(Seq(1, 0, 0, 0)))
    pokeMatrix(dut.io.attentionValues, Seq.fill(4)(Seq(1, 1, 1, 1)))
  }

  it should "clear output valid and deassert input ready while clearing" in {
    test(new JambaMiniTile()) { dut =>
      pokeCommon(dut)

      dut.io.inValid.poke(true.B)
      dut.clock.step()
      dut.io.outValid.expect(true.B)

      dut.io.clear.poke(true.B)
      dut.io.inReady.expect(false.B)
      dut.clock.step()

      dut.io.outValid.expect(false.B)
      expectVector(dut.io.out, Seq(0, 0, 0, 0))
    }
  }

  it should "accept one token and hold output under backpressure" in {
    test(new JambaMiniTile()) { dut =>
      pokeCommon(dut)

      dut.io.inReady.expect(true.B)
      dut.io.outValid.expect(false.B)

      dut.io.inValid.poke(true.B)
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      dut.io.inReady.expect(false.B)
      expectVector(dut.io.out, Seq(1, 4, 9, 16))

      pokeVector(dut.io.in, Seq(4, 3, 2, 1))
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      dut.io.inReady.expect(false.B)
      expectVector(dut.io.out, Seq(1, 4, 9, 16))
    }
  }

  it should "consume and replace output in the same cycle" in {
    test(new JambaMiniTile()) { dut =>
      pokeCommon(dut)

      dut.io.inValid.poke(true.B)
      dut.clock.step()
      expectVector(dut.io.out, Seq(1, 4, 9, 16))

      pokeVector(dut.io.in, Seq(4, 3, 2, 1))
      dut.io.outReady.poke(true.B)
      dut.io.inReady.expect(true.B)
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      expectVector(dut.io.out, Seq(22, 21, 22, 25))
    }
  }

  it should "mix attention output when enabled" in {
    test(new JambaMiniTile()) { dut =>
      pokeCommon(dut)
      dut.io.useAttention.poke(true.B)
      dut.io.inValid.poke(true.B)
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      expectVector(dut.io.attentionScores, Seq(1, 1, 1, 1))
      expectVector(dut.io.out, Seq(5, 8, 13, 20))
    }
  }
}
