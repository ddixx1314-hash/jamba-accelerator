package jamba.memory

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class SequentialWeightLoaderMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SequentialWeightLoaderMini"

  private val config = Jamba2MiniConfig.debug.copy(numLayers = 2, convTaps = 4)

  private def pokeIdle(dut: SequentialWeightLoaderMini): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.start.poke(false.B)
    dut.io.layer.poke(0.U)
    dut.io.field.poke(0.U)
    dut.io.outReady.poke(false.B)
  }

  private def launch(dut: SequentialWeightLoaderMini, layer: Int, field: Int): Unit = {
    dut.io.layer.poke(layer.U)
    dut.io.field.poke(field.U)
    dut.io.start.poke(true.B)
    dut.clock.step()
    dut.io.start.poke(false.B)
  }

  it should "walk a vector field and hold when downstream is not ready" in {
    test(new SequentialWeightLoaderMini(config, depth = 512)) { dut =>
      pokeIdle(dut)
      launch(dut, layer = 0, field = WeightAddressGenMini.Norm1Weight)

      dut.io.outValid.expect(true.B)
      dut.io.addr.expect(LayeredWeightStoreMini.Norm1Weight.U)
      dut.io.elementIndex.expect(0.U)
      dut.io.lane.expect(0.U)
      dut.io.numElements.expect(4.U)
      dut.io.isAcc.expect(false.B)

      dut.clock.step(2)
      dut.io.elementIndex.expect(0.U)
      dut.io.addr.expect(LayeredWeightStoreMini.Norm1Weight.U)

      dut.io.outReady.poke(true.B)
      dut.clock.step()
      dut.io.elementIndex.expect(1.U)
      dut.io.addr.expect((LayeredWeightStoreMini.Norm1Weight + 1).U)

      dut.clock.step(3)
      dut.io.outValid.expect(false.B)
      dut.io.busy.expect(true.B)
      dut.io.done.expect(false.B)
      dut.clock.step()
      dut.io.done.expect(true.B)
      dut.io.busy.expect(false.B)
    }
  }

  it should "walk a matrix field in row-major order with layer stride" in {
    test(new SequentialWeightLoaderMini(config, depth = 512)) { dut =>
      pokeIdle(dut)
      launch(dut, layer = 1, field = WeightAddressGenMini.QWeight)
      dut.io.outReady.poke(true.B)

      for (idx <- 0 until 16) {
        val row = idx / 4
        val col = idx % 4
        dut.io.outValid.expect(true.B)
        dut.io.elementIndex.expect(idx.U)
        dut.io.row.expect(row.U)
        dut.io.col.expect(col.U)
        dut.io.lane.expect(col.U)
        dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.QWeight + idx).U)
        dut.io.isAcc.expect(false.B)
        dut.clock.step()
      }

      dut.io.outValid.expect(false.B)
      dut.clock.step()
      dut.io.done.expect(true.B)
    }
  }

  it should "walk kernel and router fields with specialized indices" in {
    test(new SequentialWeightLoaderMini(config, depth = 512)) { dut =>
      pokeIdle(dut)
      launch(dut, layer = 1, field = WeightAddressGenMini.MambaKernel)
      dut.io.outReady.poke(true.B)

      for (idx <- 0 until 16) {
        val tap = idx / 4
        val lane = idx % 4
        dut.io.tap.expect(tap.U)
        dut.io.lane.expect(lane.U)
        dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.MambaKernel + idx).U)
        dut.clock.step()
      }

      dut.clock.step()
      dut.io.done.expect(true.B)

      launch(dut, layer = 1, field = WeightAddressGenMini.RouterWeight)
      for (idx <- 0 until 8) {
        val expert = idx / 4
        val lane = idx % 4
        dut.io.outValid.expect(true.B)
        dut.io.expert.expect(expert.U)
        dut.io.lane.expect(lane.U)
        dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.RouterWeight + idx).U)
        dut.clock.step()
      }

      dut.clock.step()
      dut.io.done.expect(true.B)
    }
  }

  it should "flag accumulator-width bias fields and reject invalid fields" in {
    test(new SequentialWeightLoaderMini(config, depth = 512)) { dut =>
      pokeIdle(dut)
      launch(dut, layer = 1, field = WeightAddressGenMini.RouterBias)
      dut.io.outReady.poke(true.B)

      dut.io.outValid.expect(true.B)
      dut.io.expert.expect(0.U)
      dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.RouterBias).U)
      dut.io.isAcc.expect(true.B)
      dut.clock.step()
      dut.io.expert.expect(1.U)
      dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.RouterBias + 1).U)
      dut.io.isAcc.expect(true.B)
      dut.clock.step()
      dut.clock.step()
      dut.io.done.expect(true.B)

      dut.clock.step()
      launch(dut, layer = 0, field = WeightAddressGenMini.NumFields)
      dut.io.error.expect(true.B)
      dut.io.busy.expect(true.B)
      dut.clock.step()
      dut.io.done.expect(true.B)
      dut.io.error.expect(true.B)
      dut.io.busy.expect(false.B)
    }
  }
}
