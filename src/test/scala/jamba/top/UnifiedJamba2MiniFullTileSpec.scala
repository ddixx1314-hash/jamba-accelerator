package jamba.top

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class UnifiedJamba2MiniFullTileSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "UnifiedJamba2MiniFullTile"

  private val config = Jamba2MiniConfig.debug.copy(
    numLayers = 1,
    attentionLayerPeriod = 2,
    attentionLayerOffset = 1,
    contextLength = 2,
    convTaps = 2
  )
  private val testWeightDepth = 16

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).poke(values(i).S)
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).expect(values(i).S)
  }

  private def pokeIdle(dut: UnifiedJamba2MiniFullTile): Unit = {
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

  private def runToOutput(dut: UnifiedJamba2MiniFullTile, maxCycles: Int = 1400): Unit = {
    var seenValid = false
    for (_ <- 0 until maxCycles) {
      if (!seenValid) {
        dut.clock.step()
        seenValid = dut.io.outValid.peek().litToBoolean
      }
    }
    assert(seenValid, s"UnifiedJamba2MiniFullTile did not produce output within $maxCycles cycles")
  }

  it should "run one token through the scheduler shell" in {
    test(new UnifiedJamba2MiniFullTile(config, weightDepth = testWeightDepth)) { dut =>
      pokeIdle(dut)

      dut.io.debugLayerUsesAttention(0).expect(false.B)
      dut.io.inReady.expect(true.B)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      dut.io.inReady.expect(false.B)

      runToOutput(dut)
      dut.io.outValid.expect(true.B)
      dut.io.done.expect(true.B)
      dut.io.debugSchedulerBusy.expect(false.B)
      expectVector(dut.io.debugLayerOutput(0), Seq(3, 0, 0, 1))
      dut.io.debugLayerKvWriteIndex(0).expect(0.U)
      dut.io.debugLayerKvValidCount(0).expect(0.U)
      expectVector(dut.io.out, Seq(3, 0, 0, 1))
    }
  }

  it should "hold output under backpressure and preserve loaded weights across clear" in {
    test(new UnifiedJamba2MiniFullTile(config, weightDepth = testWeightDepth)) { dut =>
      pokeIdle(dut)

      dut.io.weightWriteValid.poke(true.B)
      dut.io.weightWriteAddr.poke(8.U)
      dut.io.weightWriteData.poke(5.S)
      dut.io.weightWriteReady.expect(true.B)
      dut.clock.step()
      dut.io.weightWriteValid.poke(false.B)

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.clear.poke(false.B)
      dut.io.weightReadAddr.poke(8.U)
      dut.io.weightReadData.expect(5.S)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      runToOutput(dut)

      dut.io.outValid.expect(true.B)
      dut.io.inReady.expect(false.B)
      dut.clock.step(3)
      dut.io.outValid.expect(true.B)
      expectVector(dut.io.out, Seq(3, 0, 0, 1))

      dut.io.outReady.poke(true.B)
      dut.clock.step()
      dut.io.outValid.expect(false.B)
      dut.io.busy.expect(false.B)
    }
  }
}
