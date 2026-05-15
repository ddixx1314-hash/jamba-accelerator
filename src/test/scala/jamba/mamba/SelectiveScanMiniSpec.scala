package jamba.mamba

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SelectiveScanMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SelectiveScanMini"

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).poke(values(i).S)
    }
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).expect(values(i).S)
    }
  }

  it should "update token-serial SSM state and emit gated visible state" in {
    test(new SelectiveScanMini()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, -2, 3, -4))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(2, 2, 2, 2))
      pokeVector(dut.io.c, Seq(3, 3, 3, 3))

      expectVector(dut.io.stateOut, Seq(0, 0, 0, 0))
      expectVector(dut.io.y, Seq(0, 0, 0, 0))
      dut.clock.step()

      expectVector(dut.io.stateOut, Seq(2, -4, 6, -8))
      expectVector(dut.io.y, Seq(6, -12, 18, -24))
    }
  }

  it should "hold state when disabled and clear state" in {
    test(new SelectiveScanMini()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 1, 1, 1))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(2, 2, 2, 2))
      pokeVector(dut.io.c, Seq(1, 1, 1, 1))
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(2, 2, 2, 2))

      dut.io.en.poke(false.B)
      pokeVector(dut.io.x, Seq(5, 5, 5, 5))
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(2, 2, 2, 2))

      dut.io.clear.poke(true.B)
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(0, 0, 0, 0))
    }
  }
}
