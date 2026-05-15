package jamba.core

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class Jamba2MiniHybridCoreSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Jamba2MiniHybridCore"

  private val debugConfig = Jamba2MiniConfig.debug

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

  private def pokeDefaultWeights(dut: Jamba2MiniHybridCore): Unit = {
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
  }

  it should "schedule sparse attention according to the debug period" in {
    test(new Jamba2MiniHybridCore(debugConfig)) { dut =>
      pokeDefaultWeights(dut)
      dut.io.en.poke(false.B)
      dut.io.clear.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))

      dut.io.layerUsesAttention(0).expect(false.B)
      dut.io.layerUsesAttention(1).expect(false.B)
      dut.io.layerUsesAttention(2).expect(false.B)
      dut.io.layerUsesAttention(3).expect(true.B)
    }
  }

  it should "run all layers and report valid after enable" in {
    test(new Jamba2MiniHybridCore(debugConfig)) { dut =>
      pokeDefaultWeights(dut)
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))

      dut.io.valid.expect(false.B)
      expectVector(dut.io.layerOutputs(0), Seq(1, 0, 0, 1))
      dut.clock.step()

      dut.io.valid.expect(true.B)
      dut.io.layerStateOut(0)(0).expect(2.S)
    }
  }

  it should "clear valid and child state" in {
    test(new Jamba2MiniHybridCore(debugConfig)) { dut =>
      pokeDefaultWeights(dut)
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.valid.expect(true.B)

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.valid.expect(false.B)
      dut.io.layerStateOut(0)(0).expect(0.S)
    }
  }
}
