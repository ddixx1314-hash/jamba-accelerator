package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SharedLinear4Spec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SharedLinear4"

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).poke(values(i).S)
    }
  }

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) {
        port(row)(col).poke(values(row)(col).S)
      }
    }
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).expect(values(i).S)
    }
  }

  it should "compute an identity projection with bias" in {
    test(new SharedLinear4()) { dut =>
      pokeVector(dut.io.x, Seq(1, -2, 3, -4))
      pokeMatrix(
        dut.io.weight,
        Seq(
          Seq(1, 0, 0, 0),
          Seq(0, 1, 0, 0),
          Seq(0, 0, 1, 0),
          Seq(0, 0, 0, 1)
        )
      )
      pokeVector(dut.io.bias, Seq(10, 20, 30, 40))
      expectVector(dut.io.y, Seq(11, 18, 33, 36))
    }
  }

  it should "compute a reverse identity projection" in {
    test(new SharedLinear4()) { dut =>
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeMatrix(
        dut.io.weight,
        Seq(
          Seq(0, 0, 0, 1),
          Seq(0, 0, 1, 0),
          Seq(0, 1, 0, 0),
          Seq(1, 0, 0, 0)
        )
      )
      pokeVector(dut.io.bias, Seq(0, 0, 0, 0))
      expectVector(dut.io.y, Seq(4, 3, 2, 1))
    }
  }

  it should "compute a hand-checked signed projection" in {
    test(new SharedLinear4()) { dut =>
      pokeVector(dut.io.x, Seq(1, -2, 3, -4))
      pokeMatrix(
        dut.io.weight,
        Seq(
          Seq(1, 0, 0, 0),
          Seq(0, 1, 0, 0),
          Seq(1, 1, 1, 1),
          Seq(2, 0, -1, 1)
        )
      )
      pokeVector(dut.io.bias, Seq(10, 20, 30, 40))
      expectVector(dut.io.y, Seq(11, 18, 28, 35))
    }
  }
}
