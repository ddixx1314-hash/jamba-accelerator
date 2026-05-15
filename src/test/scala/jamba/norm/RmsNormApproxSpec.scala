package jamba.norm

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class RmsNormApproxSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "RmsNormApprox"

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

  it should "scale by integer mean square with zero protection" in {
    test(new RmsNormApprox()) { dut =>
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeVector(dut.io.weight, Seq(7, 7, 7, 7))

      dut.io.sumSquares.expect(30.S)
      dut.io.meanSquare.expect(7.S)
      expectVector(dut.io.y, Seq(1, 2, 3, 4))
    }
  }

  it should "handle all-zero input" in {
    test(new RmsNormApprox()) { dut =>
      pokeVector(dut.io.x, Seq(0, 0, 0, 0))
      pokeVector(dut.io.weight, Seq(7, 7, 7, 7))

      dut.io.meanSquare.expect(0.S)
      expectVector(dut.io.y, Seq(0, 0, 0, 0))
    }
  }
}
