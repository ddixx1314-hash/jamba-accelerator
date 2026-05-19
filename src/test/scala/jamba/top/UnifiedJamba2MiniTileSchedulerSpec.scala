package jamba.top

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class UnifiedJamba2MiniTileSchedulerSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "UnifiedJamba2MiniTileScheduler"

  private val config = Jamba2MiniConfig.debug.copy(
    numLayers = 2,
    attentionLayerPeriod = 2,
    attentionLayerOffset = 1,
    contextLength = 4
  )

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).poke(values(i).S)
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).expect(values(i).S)
  }

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) port(row)(col).poke(values(row)(col).S)
    }
  }

  private def pokeIdentity(port: Vec[Vec[SInt]]): Unit =
    pokeMatrix(port, Seq(Seq(1,0,0,0), Seq(0,1,0,0), Seq(0,0,1,0), Seq(0,0,0,1)))

  private def pokeReverseIdentity(port: Vec[Vec[SInt]]): Unit =
    pokeMatrix(port, Seq(Seq(0,0,0,1), Seq(0,0,1,0), Seq(0,1,0,0), Seq(1,0,0,0)))

  private def pokeZeroMatrix(port: Vec[Vec[SInt]]): Unit =
    pokeMatrix(port, Seq.fill(4)(Seq.fill(4)(0)))

  private def pokeKernel(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (tap <- values.indices) {
      for (lane <- values(tap).indices) port(tap)(lane).poke(values(tap)(lane).S)
    }
  }

  private def pokeExpertWeights(dut: UnifiedJamba2MiniTileScheduler): Unit = {
    pokeMatrix(dut.io.routerWeight, Seq(Seq(1,0,0,0), Seq(0,1,0,0)))
    pokeVector(dut.io.routerBias, Seq(0, 1))
    for (expert <- 0 until 2) {
      pokeIdentity(dut.io.expertGateWeight(expert))
      pokeVector(dut.io.expertGateBias(expert), Seq(1, 1, 1, 1))
      pokeIdentity(dut.io.expertUpWeight(expert))
      pokeVector(dut.io.expertUpBias(expert), Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.expertDownWeight(expert))
      pokeVector(dut.io.expertDownBias(expert), Seq.fill(4)(expert))
    }
  }

  private def pokeDefaultWeights(dut: UnifiedJamba2MiniTileScheduler): Unit = {
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

    pokeIdentity(dut.io.qWeight); pokeVector(dut.io.qBias, Seq(0,0,0,0))
    pokeIdentity(dut.io.kWeight); pokeVector(dut.io.kBias, Seq(0,0,0,0))
    pokeIdentity(dut.io.vWeight); pokeVector(dut.io.vBias, Seq(0,0,0,0))
    pokeIdentity(dut.io.attentionOutWeight); pokeVector(dut.io.attentionOutBias, Seq(0,0,0,0))

    pokeIdentity(dut.io.mlpGateWeight); pokeVector(dut.io.mlpGateBias, Seq(1,1,1,1))
    pokeReverseIdentity(dut.io.mlpUpWeight); pokeVector(dut.io.mlpUpBias, Seq(0,0,0,0))
    pokeIdentity(dut.io.mlpDownWeight); pokeVector(dut.io.mlpDownBias, Seq(0,0,0,0))
    pokeExpertWeights(dut)
  }

  private def runToDone(dut: UnifiedJamba2MiniTileScheduler, maxCycles: Int = 900): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"UnifiedJamba2MiniTileScheduler did not finish within $maxCycles cycles")
  }

  it should "schedule two unified layers with sparse attention placement" in {
    test(new UnifiedJamba2MiniTileScheduler(config)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      pokeDefaultWeights(dut)

      dut.io.layerUsesAttention(0).expect(false.B)
      dut.io.layerUsesAttention(1).expect(true.B)
      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.ready.expect(false.B)

      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.layerOutputs(0), Seq(3, 0, 0, 1))
      dut.io.layerKvWriteIndex(1).expect(1.U)
      dut.io.layerKvValidCount(1).expect(1.U)
      expectVector(dut.io.y, Seq(3, 0, 0, 2))
    }
  }

  it should "clear scheduler and child layer state" in {
    test(new UnifiedJamba2MiniTileScheduler(config)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      pokeDefaultWeights(dut)

      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.clear.poke(false.B)

      dut.io.ready.expect(true.B)
      dut.io.busy.expect(false.B)
      dut.io.done.expect(false.B)
      expectVector(dut.io.layerStateOut(0), Seq(0, 0, 0, 0))
      dut.io.layerKvWriteIndex(1).expect(0.U)
      dut.io.layerKvValidCount(1).expect(0.U)
    }
  }
}
