package jamba.memory

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class WeightStoreMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "WeightStoreMini"

  it should "write and read weights by address" in {
    test(new WeightStoreMini(depth = 8, dataWidth = 16)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.writeValid.poke(true.B)
      dut.io.writeAddr.poke(3.U)
      dut.io.writeData.poke(42.S)
      dut.io.writeReady.expect(true.B)
      dut.clock.step()

      dut.io.writeValid.poke(false.B)
      dut.io.readAddr.poke(3.U)
      dut.io.readData.expect(42.S)
    }
  }

  it should "overwrite an existing weight" in {
    test(new WeightStoreMini(depth = 8, dataWidth = 16)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.writeValid.poke(true.B)
      dut.io.writeAddr.poke(2.U)
      dut.io.writeData.poke(10.S)
      dut.clock.step()

      dut.io.writeAddr.poke(2.U)
      dut.io.writeData.poke((-7).S)
      dut.clock.step()

      dut.io.writeValid.poke(false.B)
      dut.io.readAddr.poke(2.U)
      dut.io.readData.expect((-7).S)
    }
  }

  it should "preserve weights across clear" in {
    test(new WeightStoreMini(depth = 8, dataWidth = 16)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.writeValid.poke(true.B)
      dut.io.writeAddr.poke(1.U)
      dut.io.writeData.poke(99.S)
      dut.clock.step()

      dut.io.writeValid.poke(false.B)
      dut.io.clear.poke(true.B)
      dut.clock.step()

      dut.io.clear.poke(false.B)
      dut.io.readAddr.poke(1.U)
      dut.io.readData.expect(99.S)
    }
  }
}
