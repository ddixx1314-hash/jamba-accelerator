package jamba.memory

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class SequentialWeightLoadPathMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SequentialWeightLoadPathMini"

  private val config = Jamba2MiniConfig.debug.copy(numLayers = 2, convTaps = 4)

  private def pokeIdle(dut: SequentialWeightLoadPathMini): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.start.poke(false.B)
    dut.io.layer.poke(0.U)
    dut.io.field.poke(0.U)
    dut.io.readData.poke(0.S)
  }

  private def runToDone(dut: SequentialWeightLoadPathMini, maxCycles: Int = 100): Unit = {
    var cycles = 0
    while (!dut.io.done.peekBoolean() && cycles < maxCycles) {
      dut.clock.step()
      cycles += 1
    }
    assert(cycles < maxCycles, s"did not finish within $maxCycles cycles")
  }

  it should "load a vector field end-to-end with readData injected per cycle" in {
    test(new SequentialWeightLoadPathMini(config, depth = 1024)) { dut =>
      pokeIdle(dut)
      dut.io.layer.poke(0.U)
      dut.io.field.poke(WeightAddressGenMini.Norm2Weight.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      // Inject readData values as the capture presents addresses
      for (lane <- 0 until 4) {
        dut.io.readData.poke((lane * 5 + 2).S)
        dut.clock.step()
      }
      // One more step for buffer's doneState
      dut.clock.step()
      dut.io.done.expect(true.B)
      for (lane <- 0 until 4) {
        dut.io.dataVec(lane).expect((lane * 5 + 2).S)
      }
    }
  }

  it should "load a matrix field end-to-end" in {
    test(new SequentialWeightLoadPathMini(config, depth = 1024)) { dut =>
      pokeIdle(dut)
      dut.io.layer.poke(1.U)
      dut.io.field.poke(WeightAddressGenMini.MlpDownWeight.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var data = 1
      for (_ <- 0 until 16) {
        dut.io.readData.poke(data.S)
        data += 1
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.done.expect(true.B)
      // Matrix starts at 1 and goes row-major
      for (row <- 0 until 4; col <- 0 until 4) {
        dut.io.dataMatrix(row)(col).expect((row * 4 + col + 1).S)
      }
    }
  }

  it should "report done only after all elements buffered" in {
    test(new SequentialWeightLoadPathMini(config, depth = 1024)) { dut =>
      pokeIdle(dut)
      dut.io.layer.poke(0.U)
      dut.io.field.poke(WeightAddressGenMini.MambaInputBias.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      // Midway through (2 of 4 elements), done should not be asserted
      for (_ <- 0 until 2) {
        dut.io.readData.poke(7.S)
        dut.clock.step()
      }
      dut.io.done.expect(false.B)

      // Feed remaining elements
      for (_ <- 0 until 2) {
        dut.io.readData.poke(7.S)
        dut.clock.step()
      }
      dut.clock.step()
      dut.io.done.expect(true.B)
    }
  }
}
