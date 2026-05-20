package jamba.top

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

/** Verifies the full weight-loading pipeline:
  *
  *   writePort → LayeredWeightStoreMini → TileScheduler → UnifiedJamba2MiniLayer → output
  *
  * Loads the same weights as connectDemoWeights into the flat address space, then runs
  * inference with useLoadedWeights=true and expects identical output to the demo path.
  *
  * Address layout from docs/weight_layout.md (LayerStride=512, layer 0 base=0).
  * Only non-zero weights are written; reset initialises the store to zero.
  */
class UnifiedJamba2MiniFullTileWeightLoadSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "UnifiedJamba2MiniFullTile weight loading"

  private val config = Jamba2MiniConfig.debug.copy(
    numLayers            = 1,
    attentionLayerPeriod = 2,
    attentionLayerOffset = 1,
    contextLength        = 2,
    convTaps             = 2
  )
  private val weightDepth = 512

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit =
    for (i <- values.indices) port(i).poke(values(i).S)

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit =
    for (i <- values.indices) port(i).expect(values(i).S)

  private def writeWeight(dut: UnifiedJamba2MiniFullTile, addr: Int, data: Int): Unit = {
    dut.io.weightWriteValid.poke(true.B)
    dut.io.weightWriteAddr.poke(addr.U)
    dut.io.weightWriteData.poke(data.S)
    dut.clock.step()
    dut.io.weightWriteValid.poke(false.B)
  }

  private def loadDemoWeights(dut: UnifiedJamba2MiniFullTile): Unit = {
    // ---- Norm weights (dataWidth, stored as accWidth) ----
    // norm1Weight[0..3] = 1 → addrs 0..3
    for (i <- 0 until 4) writeWeight(dut, i, 1)
    // norm2Weight[0..3] = 1 → addrs 4..7
    for (i <- 0 until 4) writeWeight(dut, 4 + i, 1)

    // ---- Mamba input projection: identity 4x4, row-major ----
    // mambaInputWeight → addrs 16..31, only diagonal=1
    for (i <- 0 until 4) writeWeight(dut, 16 + i * 4 + i, 1)

    // mambaInputBias = 0 → skip (already zero in store)

    // mambaBWeight = zeros → skip
    // mambaBBias[0..3] = 2 → addrs 52..55
    for (i <- 0 until 4) writeWeight(dut, 52 + i, 2)

    // mambaCWeight = zeros → skip
    // mambaCBias[0..3] = 1 → addrs 72..75
    for (i <- 0 until 4) writeWeight(dut, 72 + i, 1)

    // mambaA[0..3] = 1 → addrs 76..79
    for (i <- 0 until 4) writeWeight(dut, 76 + i, 1)

    // mambaKernel: convTaps=2, all-ones, tap-major, 4 lanes
    // tap=0: addrs 80..83, tap=1: addrs 84..87
    for (i <- 0 until 8) writeWeight(dut, 80 + i, 1)

    // Attention weights (qWeight, kWeight, vWeight, attentionOutWeight): identity
    // Layer 0 is Mamba so these are unused, but connectLoadedWeights wires them.
    // Writing zeros (store default) is fine.

    // ---- MLP gate: identity 4x4 ----
    for (i <- 0 until 4) writeWeight(dut, 176 + i * 4 + i, 1)
    // mlpGateBias[0..3] = 1 → addrs 192..195
    for (i <- 0 until 4) writeWeight(dut, 192 + i, 1)

    // ---- MLP up: anti-diagonal (row == 3-col) ----
    for (row <- 0 until 4) writeWeight(dut, 196 + row * 4 + (3 - row), 1)
    // mlpUpBias = 0 → skip

    // ---- MLP down: identity 4x4 ----
    for (i <- 0 until 4) writeWeight(dut, 216 + i * 4 + i, 1)
    // mlpDownBias = 0 → skip

    // ---- Router (not used with enableMoE=false, but wired by connectLoadedWeights) ----
    // routerWeight[0][0]=1 → addr 236, routerWeight[1][1]=1 → addr 241
    writeWeight(dut, 236, 1)
    writeWeight(dut, 241, 1)
    // routerBias[1]=1 → addr 245
    writeWeight(dut, 245, 1)
  }

  private def runToOutput(dut: UnifiedJamba2MiniFullTile, maxCycles: Int = 1400): Unit = {
    var seen = false
    for (_ <- 0 until maxCycles) {
      if (!seen) {
        dut.clock.step()
        seen = dut.io.outValid.peek().litToBoolean
      }
    }
    assert(seen, s"UnifiedJamba2MiniFullTile did not produce output within $maxCycles cycles")
  }

  it should "produce the same output as demo weights when weights are loaded via the write port" in {
    test(new UnifiedJamba2MiniFullTile(config, weightDepth)) { dut =>
      // Initial idle state
      dut.io.clear.poke(false.B)
      dut.io.start.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      dut.io.useLoadedWeights.poke(false.B)
      dut.io.inValid.poke(false.B)
      dut.io.outReady.poke(false.B)
      pokeVector(dut.io.in, Seq(0, 0, 0, 0))
      dut.io.weightWriteValid.poke(false.B)
      dut.io.weightWriteAddr.poke(0.U)
      dut.io.weightWriteData.poke(0.S)
      dut.io.weightReadAddr.poke(0.U)

      // Load weights that exactly mirror connectDemoWeights
      loadDemoWeights(dut)

      // Switch to loaded-weight mode and fire one token
      dut.io.useLoadedWeights.poke(true.B)
      dut.io.start.poke(true.B)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)

      runToOutput(dut)

      dut.io.outValid.expect(true.B)
      dut.io.done.expect(true.B)
      // Same expected output as UnifiedJamba2MiniFullTileSpec demo-weight run:
      // input [1,0,0,0] → [3,0,0,1] through the 1-layer Mamba + MLP path
      expectVector(dut.io.out, Seq(3, 0, 0, 1))
    }
  }

  it should "verify weightReadData reflects written values" in {
    test(new UnifiedJamba2MiniFullTile(config, weightDepth)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.weightWriteValid.poke(false.B)
      dut.io.weightWriteAddr.poke(0.U)
      dut.io.weightWriteData.poke(0.S)

      // Write a sentinel value at addr 192 (mlpGateBias[0])
      writeWeight(dut, 192, 42)

      dut.io.weightReadAddr.poke(192.U)
      dut.io.weightReadData.expect(42.S)
    }
  }
}
