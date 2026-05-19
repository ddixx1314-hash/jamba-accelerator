package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class MacLaneZeroSkipSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MacLane with zeroSkip"

  it should "bypass multiply and pass accIn through when a is zero" in {
    test(new MacLane(zeroSkip = true)) { dut =>
      dut.io.a.poke(0.S)
      dut.io.b.poke(5.S)
      dut.io.accIn.poke(7.S)
      dut.io.accOut.expect(7.S)
    }
  }

  it should "bypass multiply and pass accIn through when b is zero" in {
    test(new MacLane(zeroSkip = true)) { dut =>
      dut.io.a.poke(3.S)
      dut.io.b.poke(0.S)
      dut.io.accIn.poke(11.S)
      dut.io.accOut.expect(11.S)
    }
  }

  it should "compute normally when both inputs are non-zero" in {
    test(new MacLane(zeroSkip = true)) { dut =>
      dut.io.a.poke(3.S)
      dut.io.b.poke(2.S)
      dut.io.accIn.poke(1.S)
      dut.io.accOut.expect(7.S)
    }
  }
}
