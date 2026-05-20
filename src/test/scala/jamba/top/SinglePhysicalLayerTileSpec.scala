package jamba.top

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

/** Tests for SinglePhysicalLayerTile (M7-B: state virtualization).
  *
  * M7-A tests: FSM, layer-sequencing, and handshaking.
  * M7-B tests: functional equivalence with UnifiedJamba2MiniFullTile across
  *   multiple tokens, confirming per-layer SSM state, conv history, and KV
  *   cache are correctly saved/restored between logical layers.
  */
class SinglePhysicalLayerTileSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SinglePhysicalLayerTile"

  // 2-layer config: layer 0 = Mamba, layer 1 = Attention
  private val twoLayerConfig = Jamba2MiniConfig.debug.copy(
    numLayers            = 2,
    attentionLayerPeriod = 2,
    attentionLayerOffset = 1,
    contextLength        = 2,
    convTaps             = 2
  )
  private val testWeightDepth = 16

  // 1-layer config: Mamba only (sanity baseline)
  private val oneLayerConfig = Jamba2MiniConfig.debug.copy(
    numLayers            = 1,
    attentionLayerPeriod = 2,
    attentionLayerOffset = 1,
    contextLength        = 2,
    convTaps             = 2
  )

  // 3-layer config: Mamba(0) / Mamba(1) / Attention(2)
  // layer i is attention iff i % attentionLayerPeriod == attentionLayerOffset
  private val threeLayerConfig = Jamba2MiniConfig.debug.copy(
    numLayers            = 3,
    attentionLayerPeriod = 3,
    attentionLayerOffset = 2,
    contextLength        = 4,
    convTaps             = 2
  )

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit =
    for (i <- values.indices) port(i).poke(values(i).S)

  private def pokeIdle(dut: SinglePhysicalLayerTile): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.start.poke(true.B)
    dut.io.enableMoE.poke(false.B)
    dut.io.useLoadedWeights.poke(false.B)
    dut.io.inValid.poke(false.B)
    dut.io.outReady.poke(false.B)
    pokeVector(dut.io.in, Seq(1, 0, 0, 0))
    dut.io.weightWriteValid.poke(false.B)
    dut.io.weightWriteAddr.poke(0.U)
    dut.io.weightWriteData.poke(0.S)
    dut.io.weightReadAddr.poke(0.U)
  }

  private def pokeIdleFull(dut: UnifiedJamba2MiniFullTile): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.start.poke(true.B)
    dut.io.enableMoE.poke(false.B)
    dut.io.useLoadedWeights.poke(false.B)
    dut.io.inValid.poke(false.B)
    dut.io.outReady.poke(false.B)
    for (i <- 0 until 4) dut.io.in(i).poke(0.S)
    dut.io.weightWriteValid.poke(false.B)
    dut.io.weightWriteAddr.poke(0.U)
    dut.io.weightWriteData.poke(0.S)
    dut.io.weightReadAddr.poke(0.U)
  }

  private def runToOutput(dut: SinglePhysicalLayerTile, maxCycles: Int = 800): Boolean = {
    var seenValid = false
    for (_ <- 0 until maxCycles) {
      if (!seenValid) {
        dut.clock.step()
        seenValid = dut.io.outValid.peek().litToBoolean
      }
    }
    seenValid
  }

  private def runToOutputFull(dut: UnifiedJamba2MiniFullTile, maxCycles: Int = 800): Boolean = {
    var seenValid = false
    for (_ <- 0 until maxCycles) {
      if (!seenValid) {
        dut.clock.step()
        seenValid = dut.io.outValid.peek().litToBoolean
      }
    }
    seenValid
  }

  // ── M7-A: FSM, sequencing, and handshaking ─────────────────────────────────

  it should "complete one token through a single Mamba layer" in {
    test(new SinglePhysicalLayerTile(oneLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)
      dut.io.debugLayerUsesAttention(0).expect(false.B)
      dut.io.inReady.expect(true.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)

      assert(runToOutput(dut), "should produce output within 800 cycles")
      dut.io.outValid.expect(true.B)
      dut.io.done.expect(true.B)
      dut.io.debugActiveLayer.expect(0.U)
    }
  }

  it should "sequence through both layers (Mamba then Attention) for a 2-layer config" in {
    test(new SinglePhysicalLayerTile(twoLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)
      dut.io.debugLayerUsesAttention(0).expect(false.B)
      dut.io.debugLayerUsesAttention(1).expect(true.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)

      var sawLayer0 = false
      var sawLayer1 = false
      for (_ <- 0 until 800) {
        if (!dut.io.outValid.peek().litToBoolean) {
          dut.clock.step()
          val active = dut.io.debugActiveLayer.peek().litValue.toInt
          if (active == 0) sawLayer0 = true
          if (active == 1) sawLayer1 = true
        }
      }
      assert(sawLayer0, "debugActiveLayer should have been 0 (first logical layer)")
      assert(sawLayer1, "debugActiveLayer should have been 1 (second logical layer)")
      dut.io.outValid.expect(true.B)
      dut.io.done.expect(true.B)
    }
  }

  it should "process a second token after the first completes" in {
    test(new SinglePhysicalLayerTile(twoLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)
      dut.io.outReady.poke(true.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut), "first token should complete")
      dut.io.outValid.expect(true.B)
      dut.clock.step()

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(2, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut), "second token should complete")
      dut.io.outValid.expect(true.B)
      dut.io.done.expect(true.B)
    }
  }

  it should "clear all state and restart correctly" in {
    test(new SinglePhysicalLayerTile(twoLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.clear.poke(false.B)
      dut.io.inValid.poke(false.B)

      dut.io.inReady.expect(true.B)
      dut.io.outValid.expect(false.B)
      dut.io.busy.expect(false.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut), "post-clear token should complete")
      dut.io.outValid.expect(true.B)
    }
  }

  it should "hold output under backpressure and still accept a new token when consumer is ready" in {
    test(new SinglePhysicalLayerTile(twoLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)
      dut.io.outReady.poke(false.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut), "should produce output")
      dut.io.outValid.expect(true.B)

      dut.io.outReady.poke(true.B)
      dut.clock.step()
      dut.io.inReady.expect(true.B)
    }
  }

  it should "report inReady false while processing" in {
    test(new SinglePhysicalLayerTile(twoLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      dut.io.inReady.expect(false.B)

      runToOutput(dut)
    }
  }

  // ── M7-B: State virtualization correctness ─────────────────────────────────

  it should "produce the same token-1 output as UnifiedJamba2MiniFullTile" in {
    var sptOut = Seq.fill(4)(0L)
    var refOut = Seq.fill(4)(0L)

    test(new SinglePhysicalLayerTile(twoLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)
      dut.io.outReady.poke(true.B)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut), "SPT token 1 should complete")
      sptOut = (0 until 4).map(i => dut.io.out(i).peek().litValue.toLong)
    }

    test(new UnifiedJamba2MiniFullTile(twoLayerConfig, testWeightDepth)) { ref =>
      pokeIdleFull(ref)
      ref.io.outReady.poke(true.B)
      ref.io.inValid.poke(true.B)
      for (i <- 0 until 4) ref.io.in(i).poke((if (i == 0) 1 else 0).S)
      ref.clock.step()
      ref.io.inValid.poke(false.B)
      assert(runToOutputFull(ref), "FullTile token 1 should complete")
      refOut = (0 until 4).map(i => ref.io.out(i).peek().litValue.toLong)
    }

    assert(sptOut == refOut,
      s"Token-1 output mismatch: SinglePhysicalLayerTile=$sptOut UnifiedFullTile=$refOut")
  }

  it should "match UnifiedJamba2MiniFullTile output for token 2 (M7-B state virtualization)" in {
    var sptOut1 = Seq.fill(4)(0L)
    var sptOut2 = Seq.fill(4)(0L)
    var refOut1 = Seq.fill(4)(0L)
    var refOut2 = Seq.fill(4)(0L)

    // Run SinglePhysicalLayerTile for 2 tokens
    test(new SinglePhysicalLayerTile(twoLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)
      dut.io.outReady.poke(true.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut), "SPT token 1 should complete")
      sptOut1 = (0 until 4).map(i => dut.io.out(i).peek().litValue.toLong)
      dut.clock.step()

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(2, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut), "SPT token 2 should complete")
      sptOut2 = (0 until 4).map(i => dut.io.out(i).peek().litValue.toLong)
    }

    // Run UnifiedJamba2MiniFullTile for 2 tokens (reference)
    test(new UnifiedJamba2MiniFullTile(twoLayerConfig, testWeightDepth)) { ref =>
      pokeIdleFull(ref)
      ref.io.outReady.poke(true.B)

      ref.io.inValid.poke(true.B)
      for (i <- 0 until 4) ref.io.in(i).poke((if (i == 0) 1 else 0).S)
      ref.clock.step()
      ref.io.inValid.poke(false.B)
      assert(runToOutputFull(ref), "FullTile token 1 should complete")
      refOut1 = (0 until 4).map(i => ref.io.out(i).peek().litValue.toLong)
      ref.clock.step()

      ref.io.inValid.poke(true.B)
      for (i <- 0 until 4) ref.io.in(i).poke((if (i == 0) 2 else 0).S)
      ref.clock.step()
      ref.io.inValid.poke(false.B)
      assert(runToOutputFull(ref), "FullTile token 2 should complete")
      refOut2 = (0 until 4).map(i => ref.io.out(i).peek().litValue.toLong)
    }

    assert(sptOut1 == refOut1,
      s"Token-1 mismatch: SPT=$sptOut1 Full=$refOut1")
    assert(sptOut2 == refOut2,
      s"Token-2 mismatch (state virtualization error?): SPT=$sptOut2 Full=$refOut2")
  }

  // ── M7-B: 3-layer (Mamba / Mamba / Attention) state isolation ────────────
  // Proves that layer-0 and layer-1 SSM state / conv history do not bleed
  // into each other, and that layer-2 KV cache is independent of both Mamba layers.

  it should "match UnifiedJamba2MiniFullTile for 2 tokens in a 3-layer Mamba/Mamba/Attention config" in {
    var sptOut1 = Seq.fill(4)(0L)
    var sptOut2 = Seq.fill(4)(0L)
    var refOut1 = Seq.fill(4)(0L)
    var refOut2 = Seq.fill(4)(0L)

    test(new SinglePhysicalLayerTile(threeLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)
      dut.io.outReady.poke(true.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(3, 1, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut, maxCycles = 1200), "SPT 3L token 1 should complete")
      sptOut1 = (0 until 4).map(i => dut.io.out(i).peek().litValue.toLong)
      dut.clock.step()

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 2, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut, maxCycles = 1200), "SPT 3L token 2 should complete")
      sptOut2 = (0 until 4).map(i => dut.io.out(i).peek().litValue.toLong)
    }

    test(new UnifiedJamba2MiniFullTile(threeLayerConfig, testWeightDepth)) { ref =>
      pokeIdleFull(ref)
      ref.io.outReady.poke(true.B)

      ref.io.inValid.poke(true.B)
      for (i <- 0 until 4) ref.io.in(i).poke((Seq(3, 1, 0, 0)(i)).S)
      ref.clock.step()
      ref.io.inValid.poke(false.B)
      assert(runToOutputFull(ref, maxCycles = 1200), "FullTile 3L token 1 should complete")
      refOut1 = (0 until 4).map(i => ref.io.out(i).peek().litValue.toLong)
      ref.clock.step()

      ref.io.inValid.poke(true.B)
      for (i <- 0 until 4) ref.io.in(i).poke((Seq(1, 2, 0, 0)(i)).S)
      ref.clock.step()
      ref.io.inValid.poke(false.B)
      assert(runToOutputFull(ref, maxCycles = 1200), "FullTile 3L token 2 should complete")
      refOut2 = (0 until 4).map(i => ref.io.out(i).peek().litValue.toLong)
    }

    assert(sptOut1 == refOut1,
      s"3L Token-1 mismatch: SPT=$sptOut1 Full=$refOut1")
    assert(sptOut2 == refOut2,
      s"3L Token-2 mismatch (state isolation failure?): SPT=$sptOut2 Full=$refOut2")
  }
}
