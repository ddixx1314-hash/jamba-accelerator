package jamba.memory

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class LayeredWeightStoreMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "LayeredWeightStoreMini"

  private val config = Jamba2MiniConfig.debug.copy(numLayers = 2, convTaps = 4)

  private def pokeIdle(dut: LayeredWeightStoreMini): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.writeValid.poke(false.B)
    dut.io.writeAddr.poke(0.U)
    dut.io.writeData.poke(0.S)
    dut.io.readAddr.poke(0.U)
    dut.io.activeLayer.poke(0.U)
  }

  private def write(dut: LayeredWeightStoreMini, addr: Int, data: Int): Unit = {
    dut.io.writeValid.poke(true.B)
    dut.io.writeAddr.poke(addr.U)
    dut.io.writeData.poke(data.S)
    dut.io.writeReady.expect(true.B)
    dut.clock.step()
    dut.io.writeValid.poke(false.B)
  }

  it should "write and read back the flat address space" in {
    test(new LayeredWeightStoreMini(config, depth = 512)) { dut =>
      pokeIdle(dut)

      write(dut, 37, -3)
      dut.io.readAddr.poke(37.U)
      dut.io.readData.expect((-3).S)

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.clear.poke(false.B)
      dut.io.readData.expect((-3).S)
    }
  }

  it should "decode per-layer vector and bias fields" in {
    test(new LayeredWeightStoreMini(config, depth = 512)) { dut =>
      pokeIdle(dut)

      write(dut, LayeredWeightStoreMini.Norm1Weight + 1, 7)
      write(dut, LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.Norm1Weight + 1, -2)
      write(dut, LayeredWeightStoreMini.MlpDownBias, 5)
      write(dut, LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.MlpDownBias, 9)

      dut.io.activeLayer.poke(0.U)
      dut.io.norm1Weight(1).expect(7.S)
      dut.io.mlpDownBias(0).expect(5.S)

      dut.io.activeLayer.poke(1.U)
      dut.io.norm1Weight(1).expect((-2).S)
      dut.io.mlpDownBias(0).expect(9.S)
    }
  }

  it should "decode matrix, kernel, and router fields for the active layer" in {
    test(new LayeredWeightStoreMini(config, depth = 512)) { dut =>
      pokeIdle(dut)

      write(dut, LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.QWeight + 2 * 4 + 3, 4)
      write(dut, LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.MambaKernel + 3 * 4 + 2, -5)
      write(dut, LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.RouterWeight + 1 * 4 + 2, 6)
      write(dut, LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.RouterBias + 1, 11)

      dut.io.activeLayer.poke(0.U)
      dut.io.qWeight(2)(3).expect(0.S)
      dut.io.mambaKernel(3)(2).expect(0.S)
      dut.io.routerWeight(1)(2).expect(0.S)
      dut.io.routerBias(1).expect(0.S)

      dut.io.activeLayer.poke(1.U)
      dut.io.qWeight(2)(3).expect(4.S)
      dut.io.mambaKernel(3)(2).expect((-5).S)
      dut.io.routerWeight(1)(2).expect(6.S)
      dut.io.routerBias(1).expect(11.S)
    }
  }
}
