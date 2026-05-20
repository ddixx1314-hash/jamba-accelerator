package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class MacLaneMixedZeroSkipSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MacLaneMixed with zeroSkip"

  it should "bypass multiply and pass accIn through when a is zero" in {
    test(new MacLaneMixed(aWidth = 32, bWidth = 8, accWidth = 32, zeroSkip = true)) { dut =>
      dut.io.a.poke(0.S)
      dut.io.b.poke(5.S)
      dut.io.accIn.poke(42.S)
      dut.io.accOut.expect(42.S)
    }
  }

  it should "bypass multiply and pass accIn through when b is zero" in {
    test(new MacLaneMixed(aWidth = 32, bWidth = 8, accWidth = 32, zeroSkip = true)) { dut =>
      dut.io.a.poke(100.S)
      dut.io.b.poke(0.S)
      dut.io.accIn.poke(13.S)
      dut.io.accOut.expect(13.S)
    }
  }

  it should "compute a*b + accIn normally when both inputs are non-zero" in {
    test(new MacLaneMixed(aWidth = 32, bWidth = 8, accWidth = 32, zeroSkip = true)) { dut =>
      dut.io.a.poke(7.S)
      dut.io.b.poke(3.S)
      dut.io.accIn.poke(5.S)
      dut.io.accOut.expect(26.S)  // 7*3 + 5 = 26
    }
  }

  it should "bypass when a is zero, regardless of accIn value" in {
    test(new MacLaneMixed(aWidth = 32, bWidth = 8, accWidth = 32, zeroSkip = true)) { dut =>
      dut.io.a.poke(0.S)
      dut.io.b.poke(127.S)
      dut.io.accIn.poke(-99.S)
      dut.io.accOut.expect(-99.S)
    }
  }

  it should "match dense MAC when zeroSkip=false and one operand is zero" in {
    // Without zeroSkip, a=0 still computes 0*b+accIn = accIn, so outputs match.
    // This test verifies that the dense variant also gives the correct sum.
    test(new MacLaneMixed(aWidth = 32, bWidth = 8, accWidth = 32, zeroSkip = false)) { dut =>
      dut.io.a.poke(0.S)
      dut.io.b.poke(7.S)
      dut.io.accIn.poke(11.S)
      dut.io.accOut.expect(11.S)  // 0*7 + 11 = 11
    }
  }

  it should "handle negative operands with zeroSkip enabled" in {
    test(new MacLaneMixed(aWidth = 32, bWidth = 8, accWidth = 32, zeroSkip = true)) { dut =>
      dut.io.a.poke((-4).S)
      dut.io.b.poke(3.S)
      dut.io.accIn.poke(2.S)
      dut.io.accOut.expect(-10.S)  // -4*3 + 2 = -10
    }
  }
}
