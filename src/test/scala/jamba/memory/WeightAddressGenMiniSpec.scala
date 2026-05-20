package jamba.memory

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class WeightAddressGenMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "WeightAddressGenMini"

  private val config = Jamba2MiniConfig.debug.copy(numLayers = 2, convTaps = 4)

  private def pokeDefaults(dut: WeightAddressGenMini): Unit = {
    dut.io.layer.poke(0.U)
    dut.io.field.poke(0.U)
    dut.io.row.poke(0.U)
    dut.io.col.poke(0.U)
    dut.io.lane.poke(0.U)
    dut.io.tap.poke(0.U)
    dut.io.expert.poke(0.U)
  }

  it should "generate vector and matrix addresses inside layer 0" in {
    test(new WeightAddressGenMini(config, depth = 1024)) { dut =>
      pokeDefaults(dut)

      dut.io.field.poke(WeightAddressGenMini.Norm1Weight.U)
      dut.io.lane.poke(2.U)
      dut.io.localAddr.expect((LayeredWeightStoreMini.Norm1Weight + 2).U)
      dut.io.addr.expect((LayeredWeightStoreMini.Norm1Weight + 2).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(false.B)

      dut.io.field.poke(WeightAddressGenMini.QWeight.U)
      dut.io.row.poke(2.U)
      dut.io.col.poke(3.U)
      dut.io.localAddr.expect((LayeredWeightStoreMini.QWeight + 2 * 4 + 3).U)
      dut.io.addr.expect((LayeredWeightStoreMini.QWeight + 2 * 4 + 3).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(false.B)
    }
  }

  it should "add the layer stride for layer 1" in {
    test(new WeightAddressGenMini(config, depth = 1024)) { dut =>
      pokeDefaults(dut)

      dut.io.layer.poke(1.U)
      dut.io.field.poke(WeightAddressGenMini.MlpDownBias.U)
      dut.io.lane.poke(0.U)
      dut.io.localAddr.expect(LayeredWeightStoreMini.MlpDownBias.U)
      dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.MlpDownBias).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(true.B)
    }
  }

  it should "generate kernel and router addresses" in {
    test(new WeightAddressGenMini(config, depth = 1024)) { dut =>
      pokeDefaults(dut)

      dut.io.layer.poke(1.U)
      dut.io.field.poke(WeightAddressGenMini.MambaKernel.U)
      dut.io.tap.poke(3.U)
      dut.io.lane.poke(2.U)
      dut.io.localAddr.expect((LayeredWeightStoreMini.MambaKernel + 3 * 4 + 2).U)
      dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.MambaKernel + 3 * 4 + 2).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(false.B)

      dut.io.field.poke(WeightAddressGenMini.RouterWeight.U)
      dut.io.expert.poke(1.U)
      dut.io.lane.poke(2.U)
      dut.io.localAddr.expect((LayeredWeightStoreMini.RouterWeight + 1 * 4 + 2).U)
      dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.RouterWeight + 1 * 4 + 2).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(false.B)

      dut.io.field.poke(WeightAddressGenMini.RouterBias.U)
      dut.io.expert.poke(1.U)
      dut.io.localAddr.expect((LayeredWeightStoreMini.RouterBias + 1).U)
      dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + LayeredWeightStoreMini.RouterBias + 1).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(true.B)
    }
  }

  it should "generate expert matrix and bias addresses" in {
    test(new WeightAddressGenMini(config, depth = 1024)) { dut =>
      pokeDefaults(dut)

      // ExpertGateWeight: expert=0, row=1, col=2 → base + 0*16 + 1*4 + 2 = 246+6 = 252
      dut.io.field.poke(WeightAddressGenMini.ExpertGateWeight.U)
      dut.io.expert.poke(0.U)
      dut.io.row.poke(1.U)
      dut.io.col.poke(2.U)
      dut.io.localAddr.expect((LayeredWeightStoreMini.ExpertGateWeight + 0 * 16 + 1 * 4 + 2).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(false.B)

      // ExpertGateWeight: expert=1, row=3, col=3 → base + 1*16 + 3*4 + 3 = 246+31 = 277
      dut.io.expert.poke(1.U)
      dut.io.row.poke(3.U)
      dut.io.col.poke(3.U)
      dut.io.localAddr.expect((LayeredWeightStoreMini.ExpertGateWeight + 1 * 16 + 3 * 4 + 3).U)
      dut.io.valid.expect(true.B)

      // ExpertGateBias: expert=1, lane=2 → base + 1*4 + 2 = 278+6 = 284; isAcc=true
      dut.io.field.poke(WeightAddressGenMini.ExpertGateBias.U)
      dut.io.expert.poke(1.U)
      dut.io.lane.poke(2.U)
      dut.io.localAddr.expect((LayeredWeightStoreMini.ExpertGateBias + 1 * 4 + 2).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(true.B)

      // ExpertDownBias: expert=1, lane=3 → base + 1*4 + 3 = 358+7 = 365
      dut.io.field.poke(WeightAddressGenMini.ExpertDownBias.U)
      dut.io.expert.poke(1.U)
      dut.io.lane.poke(3.U)
      dut.io.localAddr.expect((LayeredWeightStoreMini.ExpertDownBias + 1 * 4 + 3).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(true.B)
    }
  }

  it should "add layer stride for expert fields in layer 1" in {
    test(new WeightAddressGenMini(config, depth = 1024)) { dut =>
      pokeDefaults(dut)

      dut.io.layer.poke(1.U)
      dut.io.field.poke(WeightAddressGenMini.ExpertUpWeight.U)
      dut.io.expert.poke(0.U)
      dut.io.row.poke(2.U)
      dut.io.col.poke(1.U)
      val expectedLocal = LayeredWeightStoreMini.ExpertUpWeight + 0 * 16 + 2 * 4 + 1
      dut.io.localAddr.expect(expectedLocal.U)
      dut.io.addr.expect((LayeredWeightStoreMini.LayerStride + expectedLocal).U)
      dut.io.valid.expect(true.B)
      dut.io.isAcc.expect(false.B)
    }
  }

  it should "reject unsupported field ids" in {
    test(new WeightAddressGenMini(config, depth = 1024)) { dut =>
      pokeDefaults(dut)

      dut.io.field.poke(WeightAddressGenMini.NumFields.U)
      dut.io.valid.expect(false.B)
      dut.io.addr.expect(0.U)
    }
  }
}
