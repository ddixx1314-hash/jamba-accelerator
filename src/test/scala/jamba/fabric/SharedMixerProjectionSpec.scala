package jamba.fabric

import chisel3._
import chiseltest._
import jamba.attention.AttentionMixerMini
import jamba.mamba.Jamba2MambaMixerMini
import org.scalatest.flatspec.AnyFlatSpec

class MambaMixerProjectionComparisonHarness extends Module {
  val io = IO(new Bundle {
    val en          = Input(Bool())
    val clear       = Input(Bool())
    val x           = Input(Vec(4, SInt(8.W)))
    val inputWeight = Input(Vec(4, Vec(4, SInt(8.W))))
    val inputBias   = Input(Vec(4, SInt(32.W)))
    val bWeight     = Input(Vec(4, Vec(4, SInt(8.W))))
    val bBias       = Input(Vec(4, SInt(32.W)))
    val cWeight     = Input(Vec(4, Vec(4, SInt(8.W))))
    val cBias       = Input(Vec(4, SInt(32.W)))
    val a           = Input(Vec(4, SInt(8.W)))
    val kernel      = Input(Vec(4, Vec(4, SInt(8.W))))

    val baselineProjected = Output(Vec(4, SInt(8.W)))
    val sharedProjected   = Output(Vec(4, SInt(8.W)))
    val baselineConv      = Output(Vec(4, SInt(32.W)))
    val sharedConv        = Output(Vec(4, SInt(32.W)))
    val baselineB         = Output(Vec(4, SInt(8.W)))
    val sharedB           = Output(Vec(4, SInt(8.W)))
    val baselineC         = Output(Vec(4, SInt(8.W)))
    val sharedC           = Output(Vec(4, SInt(8.W)))
    val baselineStateOut  = Output(Vec(4, SInt(32.W)))
    val sharedStateOut    = Output(Vec(4, SInt(32.W)))
    val baselineY         = Output(Vec(4, SInt(32.W)))
    val sharedY           = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new Jamba2MambaMixerMini())
  val shared = Module(new SharedJamba2MambaMixerMini())

  baseline.io.en := io.en
  baseline.io.clear := io.clear
  baseline.io.x := io.x
  baseline.io.inputWeight := io.inputWeight
  baseline.io.inputBias := io.inputBias
  baseline.io.bWeight := io.bWeight
  baseline.io.bBias := io.bBias
  baseline.io.cWeight := io.cWeight
  baseline.io.cBias := io.cBias
  baseline.io.a := io.a
  baseline.io.kernel := io.kernel
  shared.io.en := io.en
  shared.io.clear := io.clear
  shared.io.x := io.x
  shared.io.inputWeight := io.inputWeight
  shared.io.inputBias := io.inputBias
  shared.io.bWeight := io.bWeight
  shared.io.bBias := io.bBias
  shared.io.cWeight := io.cWeight
  shared.io.cBias := io.cBias
  shared.io.a := io.a
  shared.io.kernel := io.kernel

  io.baselineProjected := baseline.io.projected
  io.sharedProjected := shared.io.projected
  io.baselineConv := baseline.io.conv
  io.sharedConv := shared.io.conv
  io.baselineB := baseline.io.b
  io.sharedB := shared.io.b
  io.baselineC := baseline.io.c
  io.sharedC := shared.io.c
  io.baselineStateOut := baseline.io.stateOut
  io.sharedStateOut := shared.io.stateOut
  io.baselineY := baseline.io.y
  io.sharedY := shared.io.y
}

class AttentionMixerProjectionComparisonHarness extends Module {
  val io = IO(new Bundle {
    val en        = Input(Bool())
    val clear     = Input(Bool())
    val x         = Input(Vec(4, SInt(8.W)))
    val qWeight   = Input(Vec(4, Vec(4, SInt(8.W))))
    val qBias     = Input(Vec(4, SInt(32.W)))
    val kWeight   = Input(Vec(4, Vec(4, SInt(8.W))))
    val kBias     = Input(Vec(4, SInt(32.W)))
    val vWeight   = Input(Vec(4, Vec(4, SInt(8.W))))
    val vBias     = Input(Vec(4, SInt(32.W)))
    val outWeight = Input(Vec(4, Vec(4, SInt(8.W))))
    val outBias   = Input(Vec(4, SInt(32.W)))

    val baselineQ            = Output(Vec(4, SInt(8.W)))
    val sharedQ              = Output(Vec(4, SInt(8.W)))
    val baselineK            = Output(Vec(4, SInt(8.W)))
    val sharedK              = Output(Vec(4, SInt(8.W)))
    val baselineV            = Output(Vec(4, SInt(8.W)))
    val sharedV              = Output(Vec(4, SInt(8.W)))
    val baselineScores       = Output(Vec(2, SInt(32.W)))
    val sharedScores         = Output(Vec(2, SInt(32.W)))
    val baselineWeights      = Output(Vec(2, SInt(32.W)))
    val sharedWeights        = Output(Vec(2, SInt(32.W)))
    val baselineRawY         = Output(Vec(4, SInt(32.W)))
    val sharedRawY           = Output(Vec(4, SInt(32.W)))
    val baselineY            = Output(Vec(4, SInt(32.W)))
    val sharedY              = Output(Vec(4, SInt(32.W)))
    val baselineKvWriteIndex = Output(UInt(1.W))
    val sharedKvWriteIndex   = Output(UInt(1.W))
    val baselineKvValidCount = Output(UInt(2.W))
    val sharedKvValidCount   = Output(UInt(2.W))
  })

  val baseline = Module(new AttentionMixerMini(contextLength = 2))
  val shared = Module(new SharedAttentionMixerMini(contextLength = 2))

  baseline.io.en := io.en
  baseline.io.clear := io.clear
  baseline.io.x := io.x
  baseline.io.qWeight := io.qWeight
  baseline.io.qBias := io.qBias
  baseline.io.kWeight := io.kWeight
  baseline.io.kBias := io.kBias
  baseline.io.vWeight := io.vWeight
  baseline.io.vBias := io.vBias
  baseline.io.outWeight := io.outWeight
  baseline.io.outBias := io.outBias
  shared.io.en := io.en
  shared.io.clear := io.clear
  shared.io.x := io.x
  shared.io.qWeight := io.qWeight
  shared.io.qBias := io.qBias
  shared.io.kWeight := io.kWeight
  shared.io.kBias := io.kBias
  shared.io.vWeight := io.vWeight
  shared.io.vBias := io.vBias
  shared.io.outWeight := io.outWeight
  shared.io.outBias := io.outBias

  io.baselineQ := baseline.io.q
  io.sharedQ := shared.io.q
  io.baselineK := baseline.io.k
  io.sharedK := shared.io.k
  io.baselineV := baseline.io.v
  io.sharedV := shared.io.v
  io.baselineScores := baseline.io.scores
  io.sharedScores := shared.io.scores
  io.baselineWeights := baseline.io.weights
  io.sharedWeights := shared.io.weights
  io.baselineRawY := baseline.io.rawY
  io.sharedRawY := shared.io.rawY
  io.baselineY := baseline.io.y
  io.sharedY := shared.io.y
  io.baselineKvWriteIndex := baseline.io.kvWriteIndex
  io.sharedKvWriteIndex := shared.io.kvWriteIndex
  io.baselineKvValidCount := baseline.io.kvValidCount
  io.sharedKvValidCount := shared.io.kvValidCount
}

class SharedMixerProjectionSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "shared mixer projection fabric"

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

  private def pokeKernel(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (tap <- values.indices) {
      for (lane <- values(tap).indices) {
        port(tap)(lane).poke(values(tap)(lane).S)
      }
    }
  }

  it should "match Jamba2MambaMixerMini over two tokens" in {
    test(new MambaMixerProjectionComparisonHarness) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeIdentity(dut.io.inputWeight)
      pokeVector(dut.io.inputBias, Seq(0, 0, 0, 0))
      pokeZeroMatrix(dut.io.bWeight)
      pokeVector(dut.io.bBias, Seq(2, 2, 2, 2))
      pokeZeroMatrix(dut.io.cWeight)
      pokeVector(dut.io.cBias, Seq(1, 1, 1, 1))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeKernel(dut.io.kernel, Seq.fill(4)(Seq(1, 1, 1, 1)))

      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      for (lane <- 0 until 4) {
        dut.io.sharedProjected(lane).expect(dut.io.baselineProjected(lane).peek())
        dut.io.sharedConv(lane).expect(dut.io.baselineConv(lane).peek())
        dut.io.sharedB(lane).expect(dut.io.baselineB(lane).peek())
        dut.io.sharedC(lane).expect(dut.io.baselineC(lane).peek())
        dut.io.sharedY(lane).expect(dut.io.baselineY(lane).peek())
      }
      dut.clock.step()

      pokeVector(dut.io.x, Seq(0, 1, 0, 0))
      for (lane <- 0 until 4) {
        dut.io.sharedProjected(lane).expect(dut.io.baselineProjected(lane).peek())
        dut.io.sharedConv(lane).expect(dut.io.baselineConv(lane).peek())
        dut.io.sharedB(lane).expect(dut.io.baselineB(lane).peek())
        dut.io.sharedC(lane).expect(dut.io.baselineC(lane).peek())
        dut.io.sharedStateOut(lane).expect(dut.io.baselineStateOut(lane).peek())
        dut.io.sharedY(lane).expect(dut.io.baselineY(lane).peek())
      }
    }
  }

  it should "match AttentionMixerMini projection, cache, and output behavior" in {
    test(new AttentionMixerProjectionComparisonHarness) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeIdentity(dut.io.qWeight)
      pokeVector(dut.io.qBias, Seq(0, 0, 0, 0))
      pokeMatrix(dut.io.kWeight, Seq(Seq(0, 1, 0, 0), Seq(1, 0, 0, 0), Seq(0, 0, 1, 0), Seq(0, 0, 0, 1)))
      pokeVector(dut.io.kBias, Seq(1, -1, 0, 0))
      pokeIdentity(dut.io.vWeight)
      pokeVector(dut.io.vBias, Seq(0, 0, -3, 0))
      pokeMatrix(dut.io.outWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 1, 0, 0), Seq(0, 0, 1, 0), Seq(1, 1, 1, 1)))
      pokeVector(dut.io.outBias, Seq(0, 0, 0, 0))

      pokeVector(dut.io.x, Seq(3, -2, 1, 0))
      for (lane <- 0 until 4) {
        dut.io.sharedQ(lane).expect(dut.io.baselineQ(lane).peek())
        dut.io.sharedK(lane).expect(dut.io.baselineK(lane).peek())
        dut.io.sharedV(lane).expect(dut.io.baselineV(lane).peek())
        dut.io.sharedRawY(lane).expect(dut.io.baselineRawY(lane).peek())
        dut.io.sharedY(lane).expect(dut.io.baselineY(lane).peek())
      }
      for (row <- 0 until 2) {
        dut.io.sharedScores(row).expect(dut.io.baselineScores(row).peek())
        dut.io.sharedWeights(row).expect(dut.io.baselineWeights(row).peek())
      }
      dut.clock.step()
      dut.io.sharedKvWriteIndex.expect(dut.io.baselineKvWriteIndex.peek())
      dut.io.sharedKvValidCount.expect(dut.io.baselineKvValidCount.peek())
    }
  }
}
