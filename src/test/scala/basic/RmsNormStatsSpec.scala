package basic

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class RmsNormStatsSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "RmsNormStats"

  private def pokeVector(dut: RmsNormStats, values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      dut.io.x(i).poke(values(i).S)
    }
  }

  it should "compute sum of squares and integer mean square" in {
    test(new RmsNormStats()) { dut =>
      pokeVector(dut, Seq(1, 2, 3, 4))

      dut.io.sumSquares.expect(30.S)
      dut.io.meanSquare.expect(7.S)
    }
  }

  it should "handle negative inputs" in {
    test(new RmsNormStats()) { dut =>
      pokeVector(dut, Seq(1, -2, 3, -4))

      dut.io.sumSquares.expect(30.S)
      dut.io.meanSquare.expect(7.S)
    }
  }

  it should "output zeros for zero inputs" in {
    test(new RmsNormStats()) { dut =>
      pokeVector(dut, Seq(0, 0, 0, 0))

      dut.io.sumSquares.expect(0.S)
      dut.io.meanSquare.expect(0.S)
    }
  }
}
