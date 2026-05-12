package basic

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class PESpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Processing Element"

  it should "compute multiply-accumulate correctly" in {
    test(new PE(16)).withAnnotations(Seq(WriteVcdAnnotation)) { dut =>
      dut.io.a.poke(3.S)
      dut.io.b.poke(4.S)
      dut.io.acc_in.poke(10.S)
      dut.clock.step()
      dut.io.acc_out.expect(22.S)  // 3*4 + 10 = 22
      dut.io.valid.expect(true.B)
    }
  }
}
