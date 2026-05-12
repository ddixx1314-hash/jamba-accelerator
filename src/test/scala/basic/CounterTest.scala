package basic

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
}
