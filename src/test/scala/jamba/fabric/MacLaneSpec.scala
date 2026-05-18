package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class MacLaneSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MacLane"

  it should "compute a positive multiply-accumulate" in {
    test(new MacLane()) { dut =>
      dut.io.a.poke(3.S)
      dut.io.b.poke(4.S)
      dut.io.accIn.poke(5.S)
      dut.io.accOut.expect(17.S)
    }
  }

  it should "compute a negative multiply-accumulate" in {
    test(new MacLane()) { dut =>
      dut.io.a.poke((-3).S)
      dut.io.b.poke(4.S)
      dut.io.accIn.poke(5.S)
      dut.io.accOut.expect((-7).S)
    }
  }

  it should "pass through the accumulator for a zero product" in {
    test(new MacLane()) { dut =>
      dut.io.a.poke(0.S)
      dut.io.b.poke((-9).S)
      dut.io.accIn.poke(11.S)
      dut.io.accOut.expect(11.S)
    }
  }
}
