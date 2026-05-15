package jamba.attention

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class AttentionMixerMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "AttentionMixerMini"

  // Expected values are generated from python.golden.mamba_ops.attention_mixer_step
  // using jamba2_mini_fixture(context_length = 2).

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

  private def pokeDefaultWeights(dut: AttentionMixerMini): Unit = {
    pokeIdentity(dut.io.qWeight)
    pokeVector(dut.io.qBias, Seq(0, 0, 0, 0))
    pokeIdentity(dut.io.kWeight)
    pokeVector(dut.io.kBias, Seq(0, 0, 0, 0))
    pokeIdentity(dut.io.vWeight)
    pokeVector(dut.io.vBias, Seq(0, 0, 0, 0))
    pokeIdentity(dut.io.outWeight)
    pokeVector(dut.io.outBias, Seq(0, 0, 0, 0))
  }

  it should "write the current token into KV cache and decode one token" in {
    test(new AttentionMixerMini(contextLength = 2)) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeDefaultWeights(dut)
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))

      expectVector(dut.io.q, Seq(4, 0, 0, 0))
      expectVector(dut.io.k, Seq(4, 0, 0, 0))
      expectVector(dut.io.v, Seq(4, 0, 0, 0))
      expectVector(dut.io.scores, Seq(16, 0))
      expectVector(dut.io.weights, Seq(4, 0))
      expectVector(dut.io.rawY, Seq(16, 0, 0, 0))
      expectVector(dut.io.y, Seq(16, 0, 0, 0))
      dut.io.kvWriteIndex.expect(0.U)
      dut.io.kvValidCount.expect(0.U)

      dut.clock.step()
      dut.io.kvWriteIndex.expect(1.U)
      dut.io.kvValidCount.expect(1.U)
    }
  }

  it should "match Python golden values across circular cache wrap" in {
    test(new AttentionMixerMini(contextLength = 2)) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeDefaultWeights(dut)

      pokeVector(dut.io.x, Seq(4, 0, 0, 0))
      expectVector(dut.io.scores, Seq(16, 0))
      expectVector(dut.io.weights, Seq(4, 0))
      expectVector(dut.io.y, Seq(16, 0, 0, 0))
      dut.clock.step()

      pokeVector(dut.io.x, Seq(0, 4, 0, 0))
      expectVector(dut.io.scores, Seq(0, 16))
      expectVector(dut.io.weights, Seq(0, 4))
      expectVector(dut.io.y, Seq(0, 16, 0, 0))
      dut.clock.step()
      dut.io.kvWriteIndex.expect(0.U)
      dut.io.kvValidCount.expect(2.U)

      pokeVector(dut.io.x, Seq(4, 4, 0, 0))
      expectVector(dut.io.scores, Seq(16, 32))
      expectVector(dut.io.weights, Seq(4, 8))
      expectVector(dut.io.y, Seq(32, 48, 0, 0))
      dut.clock.step()
      dut.io.kvWriteIndex.expect(1.U)
      dut.io.kvValidCount.expect(2.U)
    }
  }

  it should "clear KV cache state" in {
    test(new AttentionMixerMini(contextLength = 2)) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeDefaultWeights(dut)
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))
      dut.clock.step()
      dut.io.kvWriteIndex.expect(1.U)
      dut.io.kvValidCount.expect(1.U)

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.clear.poke(false.B)
      dut.io.kvWriteIndex.expect(0.U)
      dut.io.kvValidCount.expect(0.U)
      expectVector(dut.io.scores, Seq(16, 0))
      expectVector(dut.io.weights, Seq(4, 0))
    }
  }
}
