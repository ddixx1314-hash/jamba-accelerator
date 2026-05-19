package jamba.top

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class UnifiedJamba2MiniAcceleratorTileSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "UnifiedJamba2MiniAcceleratorTile"

  private val config = Jamba2MiniConfig.debug

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).poke(values(i).S)
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).expect(values(i).S)
  }

  private def pokeIdle(dut: UnifiedJamba2MiniAcceleratorTile): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.start.poke(true.B)
    dut.io.enableMoE.poke(false.B)
    dut.io.useLoadedWeights.poke(false.B)
    dut.io.mode.poke(1.U)
    dut.io.inValid.poke(false.B)
    dut.io.outReady.poke(false.B)
    pokeVector(dut.io.in, Seq(1, 0, 0, 0))
    dut.io.weightWriteValid.poke(false.B)
    dut.io.weightWriteAddr.poke(0.U)
    dut.io.weightWriteData.poke(0.S)
    dut.io.weightReadAddr.poke(0.U)
  }

  private def runToOutput(dut: UnifiedJamba2MiniAcceleratorTile, maxCycles: Int = 420): Unit = {
    var seenValid = false
    for (_ <- 0 until maxCycles) {
      if (!seenValid) {
        dut.clock.step()
        seenValid = dut.io.outValid.peek().litToBoolean
      }
    }
    assert(seenValid, s"UnifiedJamba2MiniAcceleratorTile did not produce output within $maxCycles cycles")
  }

  it should "run one Mamba-mode token through the unified accelerator shell" in {
    test(new UnifiedJamba2MiniAcceleratorTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.inReady.expect(true.B)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)
      dut.io.inReady.expect(false.B)

      runToOutput(dut)
      dut.io.outValid.expect(true.B)
      dut.io.done.expect(true.B)
      expectVector(dut.io.out, Seq(3, 0, 0, 1))
      dut.io.debugUsesAttention.expect(false.B)
      expectVector(dut.io.debugStateOut, Seq(2, 0, 0, 0))
    }
  }

  it should "run one Attention-mode token and update KV debug state" in {
    test(new UnifiedJamba2MiniAcceleratorTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.mode.poke(2.U)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(4, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)

      runToOutput(dut)
      dut.io.outValid.expect(true.B)
      expectVector(dut.io.out, Seq(4, 0, 0, 1))
      dut.io.debugUsesAttention.expect(true.B)
      dut.io.debugKvWriteIndex.expect(1.U)
      dut.io.debugKvValidCount.expect(1.U)
    }
  }

  it should "run MoE mode through the unified MoE path" in {
    test(new UnifiedJamba2MiniAcceleratorTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.enableMoE.poke(true.B)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)

      runToOutput(dut, maxCycles = 520)
      dut.io.outValid.expect(true.B)
      expectVector(dut.io.out, Seq(5, 0, 0, 0))
      dut.io.debugSelectedExpert.expect(0.U)
    }
  }

  it should "preserve loaded weights across clear and use them when selected" in {
    test(new UnifiedJamba2MiniAcceleratorTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.weightWriteValid.poke(true.B)
      dut.io.weightWriteAddr.poke(232.U)
      dut.io.weightWriteData.poke(5.S)
      dut.io.weightWriteReady.expect(true.B)
      dut.clock.step()
      dut.io.weightWriteValid.poke(false.B)

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.clear.poke(false.B)
      dut.io.weightReadAddr.poke(232.U)
      dut.io.weightReadData.expect(5.S)

      dut.io.useLoadedWeights.poke(true.B)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.inValid.poke(false.B)

      runToOutput(dut)
      dut.io.outValid.expect(true.B)
      expectVector(dut.io.out, Seq(6, 0, 0, 0))
    }
  }

  it should "hold output under backpressure and clear the output buffer" in {
    test(new UnifiedJamba2MiniAcceleratorTile(config)) { dut =>
      pokeIdle(dut)

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
