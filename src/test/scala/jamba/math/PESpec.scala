package jamba.math

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class PESpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Processing Element"

  it should "compute multiply-accumulate correctly" in {
    test(new PE(16, 32)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.io.a.poke(3.S)
      dut.io.b.poke(4.S)
      dut.io.acc_in.poke(10.S)
      dut.clock.step()
      dut.io.acc_out.expect(22.S)  // 3*4 + 10 = 22
      dut.io.valid.expect(true.B)
    }
  }

  it should "preserve a full-width product in the accumulator" in {
    test(new PE(8, 32)) { dut =>
      dut.io.a.poke(100.S)
      dut.io.b.poke(100.S)
      dut.io.acc_in.poke(7.S)

      dut.io.acc_out.expect(10007.S)
      dut.io.valid.expect(true.B)
    }
  }

  it should "compute multiply-accumulate with negative inputs" in {
    test(new PE(8, 32)) { dut =>
      dut.io.a.poke((-3).S)
      dut.io.b.poke(4.S)
      dut.io.acc_in.poke(10.S)

      dut.io.acc_out.expect((-2).S)
      dut.io.valid.expect(true.B)
    }
  }
}
