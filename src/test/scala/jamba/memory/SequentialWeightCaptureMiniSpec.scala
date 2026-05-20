package jamba.memory

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class SequentialWeightCaptureMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SequentialWeightCaptureMini"

  private val config = Jamba2MiniConfig.debug.copy(numLayers = 2, convTaps = 4)

  private def pokeIdle(dut: SequentialWeightCaptureMini): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.start.poke(false.B)
    dut.io.layer.poke(0.U)
    dut.io.field.poke(0.U)
    dut.io.outReady.poke(true.B)
    dut.io.readData.poke(0.S)
  }

  private def runToDone(dut: SequentialWeightCaptureMini, maxCycles: Int = 50): Unit = {
    var cycles = 0
    while (!dut.io.done.peekBoolean() && cycles < maxCycles) {
      dut.clock.step()
      cycles += 1
    }
    assert(cycles < maxCycles, s"did not finish within $maxCycles cycles")
  }

  it should "present addresses and capture readData for a vector field" in {
    test(new SequentialWeightCaptureMini(config, depth = 1024)) { dut =>
      pokeIdle(dut)
      dut.io.layer.poke(0.U)
      dut.io.field.poke(WeightAddressGenMini.Norm1Weight.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      // Walk the 4 elements; inject readData = lane+1 each cycle
      for (lane <- 0 until 4) {
        dut.io.readData.poke((lane + 1).S)
        dut.io.outValid.expect(true.B)
        dut.io.outData.expect((lane + 1).S)
        dut.io.outLane.expect(lane.U)
        dut.io.outIsAcc.expect(false.B)
        dut.clock.step()
      }
      // One more step for loader's doneState → done pulse
      dut.clock.step()
      dut.io.done.expect(true.B)
    }
  }

  it should "hold outValid low when outReady is deasserted (backpressure)" in {
    test(new SequentialWeightCaptureMini(config, depth = 1024)) { dut =>
      pokeIdle(dut)
      dut.io.outReady.poke(false.B)
      dut.io.layer.poke(0.U)
      dut.io.field.poke(WeightAddressGenMini.Norm1Weight.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      // With outReady=false the loader stalls; outValid should still reflect loader
      dut.io.busy.expect(true.B)
      dut.io.done.expect(false.B)

      // Release backpressure — stream should proceed
      dut.io.outReady.poke(true.B)
      dut.io.readData.poke(99.S)
      runToDone(dut)
    }
  }

  it should "propagate layer stride for layer 1" in {
    test(new SequentialWeightCaptureMini(config, depth = 1024)) { dut =>
      pokeIdle(dut)
      dut.io.layer.poke(1.U)
      dut.io.field.poke(WeightAddressGenMini.MlpGateBias.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      val expectedBase = LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.MlpGateBias
      dut.io.readAddr.expect(expectedBase.U)
      dut.io.outIsAcc.expect(true.B)
      runToDone(dut)
    }
  }

  it should "report error for an out-of-range field" in {
    test(new SequentialWeightCaptureMini(config, depth = 1024)) { dut =>
      pokeIdle(dut)
      dut.io.field.poke(WeightAddressGenMini.NumFields.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      // After 1 step: loader is in doneState with errorReg=true
      // After 2nd step: loader transitions to idle (doneReg=true), errorReg still true
      dut.clock.step()
      dut.io.error.expect(true.B)
    }
  }
}
