package jamba.top

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

/** Tests for SinglePhysicalLayerTile (M7-A structure proof).
  *
  * These tests verify the tile-level FSM, layer-sequencing, and handshaking.
  * They do NOT check bit-exact outputs against UnifiedJamba2MiniFullTile because
  * M7-A shares SSM state and KV cache across logical layers (no state virtualization).
  * Functional equivalence is deferred to M7-B.
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

  it should "complete one token through a single Mamba layer" in {
    test(new SinglePhysicalLayerTile(oneLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)
      dut.io.debugLayerUsesAttention(0).expect(false.B)
      dut.io.inReady.expect(true.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)

      assert(runToOutput(dut), "should produce output within 2800 cycles")
      dut.io.outValid.expect(true.B)
      dut.io.done.expect(true.B)
      // debugActiveLayer holds last-processed layer index (0 for 1-layer config)
      dut.io.debugActiveLayer.expect(0.U)
    }
  }

  it should "sequence through both layers (Mamba then Attention) for a 2-layer config" in {
    test(new SinglePhysicalLayerTile(twoLayerConfig, testWeightDepth)) { dut =>
      pokeIdle(dut)
      // Compile-time attention map: layer0=Mamba, layer1=Attention
      dut.io.debugLayerUsesAttention(0).expect(false.B)
      dut.io.debugLayerUsesAttention(1).expect(true.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)

      // Monitor debugActiveLayer during processing
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

      // First token
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut), "first token should complete")
      dut.io.outValid.expect(true.B)
      dut.clock.step()  // consume output (outReady=true)

      // Second token
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

      // Start a token but immediately clear
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

      // Now run a fresh token to completion
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
      dut.io.outReady.poke(false.B)  // backpressure: consumer not ready

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      assert(runToOutput(dut), "should produce output")
      dut.io.outValid.expect(true.B)

      // Output held; consumer now ready
      dut.io.outReady.poke(true.B)
      dut.clock.step()
      // After consumption, tile should be ready to accept again
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
      // After accepting the token, tile should be busy
      dut.io.inReady.expect(false.B)

      runToOutput(dut)
    }
  }
}
