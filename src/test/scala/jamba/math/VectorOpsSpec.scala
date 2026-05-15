package jamba.math

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class VectorOpsSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "VectorOps"

  private def pokeVector(dut: VectorOps, a: Seq[Int], b: Seq[Int]): Unit = {
    for (i <- a.indices) {
      dut.io.a(i).poke(a(i).S)
      dut.io.b(i).poke(b(i).S)
    }
  }

  it should "compute element-wise add, subtract, multiply, and ReLU" in {
    test(new VectorOps()) { dut =>
      pokeVector(dut, Seq(1, -2, 3, -4), Seq(5, 6, -7, -8))

      Seq(6, 4, -4, -12).zipWithIndex.foreach { case (value, i) => dut.io.add(i).expect(value.S) }
      Seq(-4, -8, 10, 4).zipWithIndex.foreach { case (value, i) => dut.io.sub(i).expect(value.S) }
      Seq(5, -12, -21, 32).zipWithIndex.foreach { case (value, i) => dut.io.mul(i).expect(value.S) }
      Seq(1, 0, 3, 0).zipWithIndex.foreach { case (value, i) => dut.io.relu(i).expect(value.S) }
    }
  }

  it should "output zeros for zero inputs" in {
    test(new VectorOps()) { dut =>
      pokeVector(dut, Seq(0, 0, 0, 0), Seq(0, 0, 0, 0))

      for (i <- 0 until 4) {
        dut.io.add(i).expect(0.S)
        dut.io.sub(i).expect(0.S)
        dut.io.mul(i).expect(0.S)
        dut.io.relu(i).expect(0.S)
      }
    }
  }
}
