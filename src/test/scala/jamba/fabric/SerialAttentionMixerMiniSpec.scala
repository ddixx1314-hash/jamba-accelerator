package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SerialAttentionMixerMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SerialAttentionMixerMini"

  // Expected values match AttentionMixerMiniSpec (same semantics, serial timing).

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).poke(values(i).S)
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).expect(values(i).S)
  }

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) port(row)(col).poke(values(row)(col).S)
    }
  }

  private def pokeIdentity(port: Vec[Vec[SInt]]): Unit =
    pokeMatrix(port, Seq(Seq(1,0,0,0), Seq(0,1,0,0), Seq(0,0,1,0), Seq(0,0,0,1)))

  private def pokeDefaultWeights(dut: SerialAttentionMixerMini): Unit = {
    pokeIdentity(dut.io.qWeight);   pokeVector(dut.io.qBias,   Seq(0,0,0,0))
    pokeIdentity(dut.io.kWeight);   pokeVector(dut.io.kBias,   Seq(0,0,0,0))
    pokeIdentity(dut.io.vWeight);   pokeVector(dut.io.vBias,   Seq(0,0,0,0))
    pokeIdentity(dut.io.outWeight); pokeVector(dut.io.outBias, Seq(0,0,0,0))
  }

  private def runToDone(dut: SerialAttentionMixerMini, maxCycles: Int = 120): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"SerialAttentionMixerMini did not finish within $maxCycles cycles")
  }

  it should "write token to KV cache and decode one token with identity weights" in {
    // contextLength=2 to match AttentionMixerMiniSpec
    test(new SerialAttentionMixerMini(contextLength = 2)) { dut =>
      dut.io.clear.poke(false.B)
      pokeDefaultWeights(dut)
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))

      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.q,       Seq(4, 0, 0, 0))
      expectVector(dut.io.k,       Seq(4, 0, 0, 0))
      expectVector(dut.io.v,       Seq(4, 0, 0, 0))
      expectVector(dut.io.scores,  Seq(16, 0))
      expectVector(dut.io.weights, Seq(4, 0))
      expectVector(dut.io.rawY,    Seq(16, 0, 0, 0))
      expectVector(dut.io.y,       Seq(16, 0, 0, 0))
      dut.io.kvWriteIndex.expect(1.U)
      dut.io.kvValidCount.expect(1.U)
    }
  }

  it should "accumulate KV cache across two tokens" in {
    test(new SerialAttentionMixerMini(contextLength = 2)) { dut =>
      dut.io.clear.poke(false.B)
      pokeDefaultWeights(dut)

      // Token 1: x=[4,0,0,0]
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      dut.io.kvWriteIndex.expect(1.U)
      dut.io.kvValidCount.expect(1.U)

      // Token 2: x=[0,4,0,0]; cache full after this (contextLength=2)
      // Q=[0,4,0,0], K/V at index 1=[0,4,0,0]
      // physicalRows[0]=0(=[4,0,0,0]), physicalRows[1]=1(=[0,4,0,0])
      // scores[0]=Q·K[0]=0, scores[1]=Q·K[1]=16
      // weights[0]=0, weights[1]=4
      // rawY[1]=4*4=16, rawY rest=0 -> y=[0,16,0,0]
      pokeVector(dut.io.x, Seq(0, 4, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)

      expectVector(dut.io.q,       Seq(0, 4, 0, 0))
      expectVector(dut.io.k,       Seq(0, 4, 0, 0))
      expectVector(dut.io.v,       Seq(0, 4, 0, 0))
      expectVector(dut.io.scores,  Seq(0, 16))
      expectVector(dut.io.weights, Seq(0, 4))
      expectVector(dut.io.rawY,    Seq(0, 16, 0, 0))
      expectVector(dut.io.y,       Seq(0, 16, 0, 0))
      dut.io.kvWriteIndex.expect(0.U)
      dut.io.kvValidCount.expect(2.U)
    }
  }

  it should "clear resets KV cache and state" in {
    test(new SerialAttentionMixerMini(contextLength = 2)) { dut =>
      dut.io.clear.poke(false.B)
      pokeDefaultWeights(dut)
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))

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
      dut.io.kvWriteIndex.expect(0.U)
      dut.io.kvValidCount.expect(0.U)
      expectVector(dut.io.q,      Seq(0, 0, 0, 0))
      expectVector(dut.io.scores, Seq(0, 0))
      expectVector(dut.io.y,      Seq(0, 0, 0, 0))
    }
  }

  it should "saturate K/V projections and out projection input" in {
    // x=[100,0,0,0], K/V weight doubles lane 0 → 200 → saturates to 127
    // Q=[100,0,0,0], K=[127,0,0,0] (saturated), V=[127,0,0,0]
    // scores[0] = 100*127 = 12700, weights[0] = 12700>>2 = 3175
    // rawY[0] = 3175 * 127 = 403225 → saturated to 127 for out projection
    // out (identity) → y=[127,0,0,0]
    test(new SerialAttentionMixerMini(contextLength = 2)) { dut =>
      dut.io.clear.poke(false.B)
      pokeIdentity(dut.io.qWeight); pokeVector(dut.io.qBias, Seq(0,0,0,0))
      pokeMatrix(dut.io.kWeight, Seq(Seq(2,0,0,0), Seq(0,1,0,0), Seq(0,0,1,0), Seq(0,0,0,1)))
      pokeVector(dut.io.kBias, Seq(0,0,0,0))
      pokeMatrix(dut.io.vWeight, Seq(Seq(2,0,0,0), Seq(0,1,0,0), Seq(0,0,1,0), Seq(0,0,0,1)))
      pokeVector(dut.io.vBias, Seq(0,0,0,0))
      pokeIdentity(dut.io.outWeight); pokeVector(dut.io.outBias, Seq(0,0,0,0))
      pokeVector(dut.io.x, Seq(100, 0, 0, 0))

      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)

      expectVector(dut.io.q,       Seq(100, 0, 0, 0))
      expectVector(dut.io.k,       Seq(127, 0, 0, 0))
      expectVector(dut.io.v,       Seq(127, 0, 0, 0))
      expectVector(dut.io.scores,  Seq(12700, 0))
      expectVector(dut.io.weights, Seq(3175, 0))
      expectVector(dut.io.rawY,    Seq(403225, 0, 0, 0))
      expectVector(dut.io.y,       Seq(127, 0, 0, 0))
    }
  }
}
