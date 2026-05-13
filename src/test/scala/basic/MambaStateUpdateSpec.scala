package basic

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class MambaStateUpdateSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MambaStateUpdate"

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

  it should "update state when enabled" in {
    test(new MambaStateUpdate()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(2, 2, 2, 2))

      expectVector(dut.io.stateOut, Seq(0, 0, 0, 0))
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(2, 4, 6, 8))
    }
  }

  it should "hold state when disabled and clear when requested" in {
    test(new MambaStateUpdate()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, -2, 3, -4))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(3, 3, 3, 3))
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(3, -6, 9, -12))

      dut.io.en.poke(false.B)
      pokeVector(dut.io.x, Seq(7, 7, 7, 7))
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(3, -6, 9, -12))

      dut.io.clear.poke(true.B)
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(0, 0, 0, 0))
    }
  }
}
