package jamba.common

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class CounterTest extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Counter"

  it should "count when enable is high" in {
    test(new Counter(8)) { dut =>
      dut.io.en.poke(true.B)

      for (i <- 1 until 6) {
        dut.clock.step()
        dut.io.out.expect(i.U)
      }
    }
  }

  it should "hold its value when enable is low" in {
    test(new Counter(8)) { dut =>
      dut.io.en.poke(true.B)
      dut.clock.step(3)
      dut.io.out.expect(3.U)

      dut.io.en.poke(false.B)
      dut.clock.step(2)
      dut.io.out.expect(3.U)
    }
  }

  it should "wrap around when the counter overflows" in {
    test(new Counter(2)) { dut =>
      dut.io.en.poke(true.B)

      dut.clock.step()
      dut.io.out.expect(1.U)
      dut.clock.step()
      dut.io.out.expect(2.U)
      dut.clock.step()
      dut.io.out.expect(3.U)
      dut.clock.step()
      dut.io.out.expect(0.U)
    }
  }
}
