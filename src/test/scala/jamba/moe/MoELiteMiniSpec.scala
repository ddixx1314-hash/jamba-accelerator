package jamba.moe

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class MoELiteMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MoELiteMini"

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

  private def pokeZeroMatrix(port: Vec[Vec[SInt]]): Unit = {
    pokeMatrix(port, Seq.fill(4)(Seq.fill(4)(0)))
  }

  private def pokeExpertWeights(dut: MoELiteMini): Unit = {
    for (expert <- 0 until 2) {
      pokeIdentity(dut.io.expertGateWeight(expert))
      pokeVector(dut.io.expertGateBias(expert), Seq(1, 1, 1, 1))
      pokeIdentity(dut.io.expertUpWeight(expert))
      pokeVector(dut.io.expertUpBias(expert), Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.expertDownWeight(expert))
      pokeVector(dut.io.expertDownBias(expert), Seq.fill(4)(expert))
    }
  }

  it should "route a token to expert 1 when its router score is larger" in {
    test(new MoELiteMini()) { dut =>
      pokeVector(dut.io.x, Seq(2, 1, 0, 0))
      pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 2, 0, 0)))
      pokeVector(dut.io.routerBias, Seq(0, 1))
      pokeExpertWeights(dut)
      dut.io.dispatchReady.poke(true.B)
      dut.io.combineReady.poke(true.B)

      dut.io.routerScores(0).expect(2.S)
      dut.io.routerScores(1).expect(3.S)
      dut.io.selectedExpert.expect(1.U)
      dut.io.dispatchValid.expect(true.B)
      dut.io.combineValid.expect(true.B)
      expectVector(dut.io.y, Seq(7, 3, 1, 1))
    }
  }

  it should "choose expert 0 on ties" in {
    test(new MoELiteMini()) { dut =>
      pokeVector(dut.io.x, Seq(1, 1, 0, 0))
      pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 1, 0, 0)))
      pokeVector(dut.io.routerBias, Seq(0, 0))
      pokeExpertWeights(dut)
      dut.io.dispatchReady.poke(true.B)
      dut.io.combineReady.poke(true.B)

      dut.io.routerScores(0).expect(1.S)
      dut.io.routerScores(1).expect(1.S)
      dut.io.selectedExpert.expect(0.U)
      expectVector(dut.io.y, Seq(2, 2, 0, 0))
    }
  }
}
