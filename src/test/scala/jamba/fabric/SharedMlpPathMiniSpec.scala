package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SharedMlpPathMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Shared MLP-path fabric"

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

  private def pokeExpertWeights(
      gateWeight: Vec[Vec[Vec[SInt]]],
      gateBias:   Vec[Vec[SInt]],
      upWeight:   Vec[Vec[Vec[SInt]]],
      upBias:     Vec[Vec[SInt]],
      downWeight: Vec[Vec[Vec[SInt]]],
      downBias:   Vec[Vec[SInt]]
  ): Unit = {
    for (expert <- 0 until 2) {
      pokeIdentity(gateWeight(expert))
      pokeVector(gateBias(expert), Seq(1, 1, 1, 1))
      pokeIdentity(upWeight(expert))
      pokeVector(upBias(expert), Seq(0, 0, 0, 0))
      pokeIdentity(downWeight(expert))
      pokeVector(downBias(expert), Seq.fill(4)(expert))
    }
  }

  private def pokeDensePathWeights(dut: SharedMlpPathMini): Unit = {
    pokeIdentity(dut.io.gateWeight)
    pokeVector(dut.io.gateBias, Seq(1, 1, 1, 1))
    pokeReverseIdentity(dut.io.upWeight)
    pokeVector(dut.io.upBias, Seq(0, 0, 0, 0))
    pokeIdentity(dut.io.downWeight)
    pokeVector(dut.io.downBias, Seq(0, 0, 0, 0))
  }

  it should "route to expert 1 when its shared dot-product score is larger" in {
    test(new SharedRouterMini()) { dut =>
      pokeVector(dut.io.x, Seq(2, 1, 0, 0))
      pokeMatrix(dut.io.weight, Seq(Seq(1, 0, 0, 0), Seq(0, 2, 0, 0)))
      pokeVector(dut.io.bias, Seq(0, 1))

      dut.io.scores(0).expect(2.S)
      dut.io.scores(1).expect(3.S)
      dut.io.selectedExpert.expect(1.U)
    }
  }

  it should "run one expert MLP on the shared dense fabric" in {
    test(new SharedExpertMLPMini()) { dut =>
      pokeVector(dut.io.x, Seq(2, 1, 0, 0))
      pokeIdentity(dut.io.gateWeight)
      pokeVector(dut.io.gateBias, Seq(1, 1, 1, 1))
      pokeIdentity(dut.io.upWeight)
      pokeVector(dut.io.upBias, Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.downWeight)
      pokeVector(dut.io.downBias, Seq(1, 1, 1, 1))

      expectVector(dut.io.y, Seq(7, 3, 1, 1))
    }
  }

  it should "select the routed shared expert output" in {
    test(new SharedMoELiteMini()) { dut =>
      pokeVector(dut.io.x, Seq(2, 1, 0, 0))
      pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 2, 0, 0)))
      pokeVector(dut.io.routerBias, Seq(0, 1))
      pokeExpertWeights(
        dut.io.expertGateWeight,
        dut.io.expertGateBias,
        dut.io.expertUpWeight,
        dut.io.expertUpBias,
        dut.io.expertDownWeight,
        dut.io.expertDownBias
      )
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

  it should "choose dense or MoE output with the same MLP-path contract" in {
    test(new SharedMlpPathMini()) { dut =>
      pokeVector(dut.io.x, Seq(2, 1, 0, 0))
      pokeDensePathWeights(dut)
      pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 2, 0, 0)))
      pokeVector(dut.io.routerBias, Seq(0, 1))
      pokeExpertWeights(
        dut.io.expertGateWeight,
        dut.io.expertGateBias,
        dut.io.expertUpWeight,
        dut.io.expertUpBias,
        dut.io.expertDownWeight,
        dut.io.expertDownBias
      )
      dut.io.dispatchReady.poke(true.B)
      dut.io.combineReady.poke(true.B)

      dut.io.enableMoE.poke(false.B)
      expectVector(dut.io.denseY, Seq(0, 0, 1, 2))
      expectVector(dut.io.y, Seq(0, 0, 1, 2))
      dut.io.dispatchValid.expect(false.B)
      dut.io.combineValid.expect(false.B)
      dut.io.selectedExpert.expect(0.U)

      dut.io.enableMoE.poke(true.B)
      expectVector(dut.io.moeY, Seq(7, 3, 1, 1))
      expectVector(dut.io.y, Seq(7, 3, 1, 1))
      dut.io.dispatchValid.expect(true.B)
      dut.io.combineValid.expect(true.B)
      dut.io.selectedExpert.expect(1.U)
    }
  }
}
