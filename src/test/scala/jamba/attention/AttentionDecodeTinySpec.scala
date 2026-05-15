package jamba.attention

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class AttentionDecodeTinySpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "AttentionDecodeTiny"

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

  it should "compute dot-product scores and weighted value sums" in {
    test(new AttentionDecodeTiny()) { dut =>
      pokeVector(dut.io.q, Seq(1, 2, 0, -1))
      pokeMatrix(
        dut.io.keys,
        Seq(
          Seq(1, 0, 0, 0),
          Seq(0, 1, 0, 0),
          Seq(1, 1, 0, 0),
          Seq(0, 0, 0, 1)
        )
      )
      pokeMatrix(
        dut.io.values,
        Seq(
          Seq(1, 0, 0, 0),
          Seq(0, 1, 0, 0),
          Seq(1, 1, 1, 1),
          Seq(2, 0, 0, 1)
        )
      )

      expectVector(dut.io.scores, Seq(1, 2, 3, -1))
      expectVector(dut.io.y, Seq(2, 5, 3, 2))
    }
  }

  it should "return zeros for zero query" in {
    test(new AttentionDecodeTiny()) { dut =>
      pokeVector(dut.io.q, Seq(0, 0, 0, 0))
      pokeMatrix(dut.io.keys, Seq.fill(4)(Seq(1, 2, 3, 4)))
      pokeMatrix(dut.io.values, Seq.fill(4)(Seq(1, -1, 2, -2)))

      expectVector(dut.io.scores, Seq(0, 0, 0, 0))
      expectVector(dut.io.y, Seq(0, 0, 0, 0))
    }
  }
}
