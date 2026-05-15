package jamba.core

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class DenseMLPMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "DenseMLPMini"

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

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) {
        port(row)(col).poke(values(row)(col).S)
      }
    }
  }

  private def pokeIdentity(port: Vec[Vec[SInt]]): Unit = {
    pokeMatrix(
      port,
      Seq(
        Seq(1, 0, 0, 0),
        Seq(0, 1, 0, 0),
        Seq(0, 0, 1, 0),
        Seq(0, 0, 0, 1)
      )
    )
  }

  private def pokeReverseIdentity(port: Vec[Vec[SInt]]): Unit = {
    pokeMatrix(
      port,
      Seq(
        Seq(0, 0, 0, 1),
        Seq(0, 0, 1, 0),
        Seq(0, 1, 0, 0),
        Seq(1, 0, 0, 0)
      )
    )
  }

  it should "match the deterministic Python dense_mlp_step fixture" in {
    test(new DenseMLPMini()) { dut =>
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeIdentity(dut.io.gateWeight)
      pokeVector(dut.io.gateBias, Seq(1, 1, 1, 1))
      pokeReverseIdentity(dut.io.upWeight)
      pokeVector(dut.io.upBias, Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.downWeight)
      pokeVector(dut.io.downBias, Seq(0, 0, 0, 0))

      expectVector(dut.io.gate, Seq(2, 3, 4, 5))
      expectVector(dut.io.up, Seq(4, 3, 2, 1))
      expectVector(dut.io.activatedGate, Seq(2, 3, 4, 5))
      expectVector(dut.io.hidden, Seq(8, 9, 8, 5))
      expectVector(dut.io.y, Seq(8, 9, 8, 5))
    }
  }
}
