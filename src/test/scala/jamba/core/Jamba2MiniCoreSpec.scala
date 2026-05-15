package jamba.core

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class Jamba2MiniCoreSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Jamba2MiniCore"

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

  private def pokeCommon(dut: Jamba2MiniCore): Unit = {
    dut.io.en.poke(true.B)
    dut.io.clear.poke(false.B)
    dut.io.useAttention.poke(false.B)
    pokeVector(dut.io.x, Seq(1, 2, 3, 4))
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

  it should "run norm, projections, tiny block, and output projection" in {
    test(new Jamba2MiniCore()) { dut =>
      pokeCommon(dut)

      dut.io.valid.expect(false.B)
      dut.io.normMeanSquare.expect(7.S)
      expectVector(dut.io.projectedX, Seq(1, 2, 3, 4))
      expectVector(dut.io.blockY, Seq(1, 4, 9, 16))
      expectVector(dut.io.y, Seq(1, 4, 9, 16))

      dut.clock.step()
      dut.io.valid.expect(true.B)
      expectVector(dut.io.stateOut, Seq(2, 4, 6, 8))
      expectVector(dut.io.blockY, Seq(7, 16, 27, 40))
      expectVector(dut.io.y, Seq(7, 16, 27, 40))
    }
  }

  it should "mix in attention when enabled" in {
    test(new Jamba2MiniCore()) { dut =>
      pokeCommon(dut)
      dut.io.useAttention.poke(true.B)

      expectVector(dut.io.attentionScores, Seq(1, 1, 1, 1))
      expectVector(dut.io.blockY, Seq(5, 8, 13, 20))
      expectVector(dut.io.y, Seq(5, 8, 13, 20))
    }
  }
}
