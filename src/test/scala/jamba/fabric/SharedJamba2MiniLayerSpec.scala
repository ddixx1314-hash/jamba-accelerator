package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SharedJamba2MiniLayerSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SharedJamba2MiniLayer"

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

  private def pokeZeroMatrix(port: Vec[Vec[SInt]]): Unit = {
    pokeMatrix(port, Seq.fill(4)(Seq.fill(4)(0)))
  }

  private def pokeKernel(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (tap <- values.indices) {
      for (lane <- values(tap).indices) {
        port(tap)(lane).poke(values(tap)(lane).S)
      }
    }
  }

  private def pokeExpertWeights(dut: SharedJamba2MiniLayer): Unit = {
    pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 1, 0, 0)))
    pokeVector(dut.io.routerBias, Seq(0, 0))
    for (expert <- 0 until 2) {
      pokeIdentity(dut.io.expertGateWeight(expert))
      pokeVector(dut.io.expertGateBias(expert), Seq(1, 1, 1, 1))
      pokeIdentity(dut.io.expertUpWeight(expert))
      pokeVector(dut.io.expertUpBias(expert), Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.expertDownWeight(expert))
      pokeVector(dut.io.expertDownBias(expert), Seq.fill(4)(expert))
    }
  }

  private def pokeDefaultWeights(dut: SharedJamba2MiniLayer): Unit = {
    pokeVector(dut.io.norm1Weight, Seq(1, 1, 1, 1))
    pokeVector(dut.io.norm2Weight, Seq(1, 1, 1, 1))

    pokeIdentity(dut.io.mambaInputWeight)
    pokeVector(dut.io.mambaInputBias, Seq(0, 0, 0, 0))
    pokeZeroMatrix(dut.io.mambaBWeight)
    pokeVector(dut.io.mambaBBias, Seq(2, 2, 2, 2))
    pokeZeroMatrix(dut.io.mambaCWeight)
    pokeVector(dut.io.mambaCBias, Seq(1, 1, 1, 1))
    pokeVector(dut.io.mambaA, Seq(1, 1, 1, 1))
    pokeKernel(dut.io.mambaKernel, Seq.fill(4)(Seq(1, 1, 1, 1)))

    pokeIdentity(dut.io.qWeight)
    pokeVector(dut.io.qBias, Seq(0, 0, 0, 0))
    pokeIdentity(dut.io.kWeight)
    pokeVector(dut.io.kBias, Seq(0, 0, 0, 0))
    pokeIdentity(dut.io.vWeight)
    pokeVector(dut.io.vBias, Seq(0, 0, 0, 0))
    pokeIdentity(dut.io.attentionOutWeight)
    pokeVector(dut.io.attentionOutBias, Seq(0, 0, 0, 0))

    pokeIdentity(dut.io.mlpGateWeight)
    pokeVector(dut.io.mlpGateBias, Seq(1, 1, 1, 1))
    pokeReverseIdentity(dut.io.mlpUpWeight)
    pokeVector(dut.io.mlpUpBias, Seq(0, 0, 0, 0))
    pokeIdentity(dut.io.mlpDownWeight)
    pokeVector(dut.io.mlpDownBias, Seq(0, 0, 0, 0))
    pokeExpertWeights(dut)
  }

  it should "run the shared MLP layer in Mamba mode" in {
    test(new SharedJamba2MiniLayer()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      pokeDefaultWeights(dut)

      expectVector(dut.io.mixerY, Seq(0, 0, 0, 0))
      expectVector(dut.io.firstResidual, Seq(1, 0, 0, 0))
      expectVector(dut.io.mlpY, Seq(0, 0, 0, 1))
      expectVector(dut.io.y, Seq(1, 0, 0, 1))
      dut.clock.step()

      expectVector(dut.io.stateOut, Seq(2, 0, 0, 0))
      expectVector(dut.io.mixerY, Seq(2, 0, 0, 0))
      expectVector(dut.io.firstResidual, Seq(3, 0, 0, 0))
      expectVector(dut.io.mlpY, Seq(0, 0, 0, 1))
      expectVector(dut.io.y, Seq(3, 0, 0, 1))
      dut.io.mixerType.expect(false.B)
      dut.io.dispatchValid.expect(false.B)
      dut.io.combineValid.expect(false.B)
      dut.io.selectedExpert.expect(0.U)
    }
  }

  it should "run the shared MLP layer in Attention mode" in {
    test(new SharedJamba2MiniLayer()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))
      pokeDefaultWeights(dut)

      expectVector(dut.io.mixerY, Seq(0, 0, 0, 0))
      expectVector(dut.io.firstResidual, Seq(4, 0, 0, 0))
      expectVector(dut.io.mlpY, Seq(0, 0, 0, 1))
      expectVector(dut.io.y, Seq(4, 0, 0, 1))
      dut.io.kvWriteIndex.expect(0.U)
      dut.io.kvValidCount.expect(0.U)
      dut.clock.step()

      dut.io.kvWriteIndex.expect(1.U)
      dut.io.kvValidCount.expect(1.U)
      dut.io.mixerType.expect(true.B)
      dut.io.dispatchValid.expect(false.B)
      dut.io.combineValid.expect(false.B)
      dut.io.selectedExpert.expect(0.U)
    }
  }

  it should "activate the shared MoE-lite MLP path" in {
    test(new SharedJamba2MiniLayer()) { dut =>
      dut.io.en.poke(false.B)
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(true.B)
      pokeVector(dut.io.x, Seq(1, 1, 0, 0))
      pokeDefaultWeights(dut)
      pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 2, 0, 0)))
      pokeVector(dut.io.routerBias, Seq(0, 1))

      dut.io.dispatchValid.expect(true.B)
      dut.io.combineValid.expect(true.B)
      dut.io.selectedExpert.expect(1.U)
    }
  }
}
