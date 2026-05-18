package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SharedSelectiveScanTinySpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SharedSelectiveScanTiny"

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

  it should "gate the visible SSM state" in {
    test(new SharedSelectiveScanTiny()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, -2, 3, -4))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(2, 2, 2, 2))
      pokeVector(dut.io.gate, Seq(3, 3, 3, 3))

      expectVector(dut.io.y, Seq(0, 0, 0, 0))
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(2, -4, 6, -8))
      expectVector(dut.io.y, Seq(6, -12, 18, -24))
    }
  }
}
