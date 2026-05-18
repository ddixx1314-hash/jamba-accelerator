package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SharedDotProductSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SharedDotProduct"

  private def pokeVector(dut: SharedDotProduct, a: Seq[Int], b: Seq[Int]): Unit = {
    for (i <- a.indices) {
      dut.io.a(i).poke(a(i).S)
      dut.io.b(i).poke(b(i).S)
    }
  }

  it should "compute a positive dot product" in {
    test(new SharedDotProduct()) { dut =>
      pokeVector(dut, Seq(1, 2, 3, 4), Seq(5, 6, 7, 8))
      dut.io.y.expect(70.S)
    }
  }

  it should "compute a dot product with negative inputs" in {
    test(new SharedDotProduct()) { dut =>
      pokeVector(dut, Seq(1, -2, 3, -4), Seq(5, 6, -7, -8))
      dut.io.y.expect(4.S)
    }
  }

  it should "output zero for all zero inputs" in {
    test(new SharedDotProduct()) { dut =>
      pokeVector(dut, Seq(0, 0, 0, 0), Seq(0, 0, 0, 0))
      dut.io.y.expect(0.S)
    }
  }
}
