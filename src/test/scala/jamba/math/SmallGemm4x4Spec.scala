package jamba.math

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SmallGemm4x4Spec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SmallGemm4x4"

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) {
        port(row)(col).poke(values(row)(col).S)
      }
    }
  }

  private def expectMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) {
        port(row)(col).expect(values(row)(col).S)
      }
    }
  }

  it should "return the input matrix when multiplied by identity" in {
    test(new SmallGemm4x4()) { dut =>
      val a = Seq(
        Seq(1, 2, 3, 4),
        Seq(5, 6, 7, 8),
        Seq(-1, -2, -3, -4),
        Seq(2, 0, -2, 1)
      )
      val identity = Seq(
        Seq(1, 0, 0, 0),
        Seq(0, 1, 0, 0),
        Seq(0, 0, 1, 0),
        Seq(0, 0, 0, 1)
      )

      pokeMatrix(dut.io.a, a)
      pokeMatrix(dut.io.b, identity)
      expectMatrix(dut.io.c, a)
    }
  }

  it should "compute a hand-checked 4x4 matrix multiply" in {
    test(new SmallGemm4x4()) { dut =>
      val a = Seq(
        Seq(1, 2, 3, 4),
        Seq(0, 1, 0, 1),
        Seq(-1, 2, -3, 4),
        Seq(2, 2, 2, 2)
      )
      val b = Seq(
        Seq(1, 0, 2, 1),
        Seq(0, 1, 2, -1),
        Seq(1, 1, 0, 2),
        Seq(2, 0, -1, 1)
      )
      val expected = Seq(
        Seq(12, 5, 2, 9),
        Seq(2, 1, 1, 0),
        Seq(4, -1, -2, -5),
        Seq(8, 4, 6, 6)
      )

      pokeMatrix(dut.io.a, a)
      pokeMatrix(dut.io.b, b)
      expectMatrix(dut.io.c, expected)
    }
  }

  it should "output all zeros when both inputs are zero" in {
    test(new SmallGemm4x4()) { dut =>
      val zeros = Seq.fill(4)(Seq.fill(4)(0))

      pokeMatrix(dut.io.a, zeros)
      pokeMatrix(dut.io.b, zeros)
      expectMatrix(dut.io.c, zeros)
    }
  }
}
