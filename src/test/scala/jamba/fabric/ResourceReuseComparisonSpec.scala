package jamba.fabric

import chisel3._
import chiseltest._
import jamba.attention.AttentionDecodeTiny
import jamba.core.{DenseMLPMini, MlpPathMini, TinyJambaBlock}
import jamba.mamba.{CausalConv1D, MambaStateUpdate, SelectiveScanTiny, TinyMambaBlock}
import jamba.math.{DotProduct, Linear4}
import org.scalatest.flatspec.AnyFlatSpec

class DotProductComparisonHarness extends Module {
  val io = IO(new Bundle {
    val a        = Input(Vec(4, SInt(8.W)))
    val b        = Input(Vec(4, SInt(8.W)))
    val baseline = Output(SInt(32.W))
    val shared   = Output(SInt(32.W))
  })

  val baseline = Module(new DotProduct())
  val shared = Module(new SharedDotProduct())

  baseline.io.a := io.a
  baseline.io.b := io.b
  shared.io.a := io.a
  shared.io.b := io.b

  io.baseline := baseline.io.y
  io.shared := shared.io.y
}

class Linear4ComparisonHarness extends Module {
  val io = IO(new Bundle {
    val x        = Input(Vec(4, SInt(8.W)))
    val weight   = Input(Vec(4, Vec(4, SInt(8.W))))
    val bias     = Input(Vec(4, SInt(32.W)))
    val baseline = Output(Vec(4, SInt(32.W)))
    val shared   = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new Linear4())
  val shared = Module(new SharedLinear4())

  baseline.io.x := io.x
  baseline.io.weight := io.weight
  baseline.io.bias := io.bias
  shared.io.x := io.x
  shared.io.weight := io.weight
  shared.io.bias := io.bias

  io.baseline := baseline.io.y
  io.shared := shared.io.y
}

class SerialLinear4ComparisonHarness extends Module {
  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(4, SInt(8.W)))
    val weight = Input(Vec(4, Vec(4, SInt(8.W))))
    val bias = Input(Vec(4, SInt(32.W)))
    val serialDone = Output(Bool())
    val serialBusy = Output(Bool())
    val baseline = Output(Vec(4, SInt(32.W)))
    val shared = Output(Vec(4, SInt(32.W)))
    val serial = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new Linear4())
  val shared = Module(new SharedLinear4())
  val serial = Module(new SerialSharedLinear4())

  baseline.io.x := io.x
  baseline.io.weight := io.weight
  baseline.io.bias := io.bias
  shared.io.x := io.x
  shared.io.weight := io.weight
  shared.io.bias := io.bias
  serial.io.start := io.start
  serial.io.clear := io.clear
  serial.io.x := io.x
  serial.io.weight := io.weight
  serial.io.bias := io.bias

  io.serialDone := serial.io.done
  io.serialBusy := serial.io.busy
  io.baseline := baseline.io.y
  io.shared := shared.io.y
  io.serial := serial.io.y
}

class AttentionDecodeComparisonHarness extends Module {
  val io = IO(new Bundle {
    val q              = Input(Vec(4, SInt(8.W)))
    val keys           = Input(Vec(4, Vec(4, SInt(8.W))))
    val values         = Input(Vec(4, Vec(4, SInt(8.W))))
    val baselineScores = Output(Vec(4, SInt(32.W)))
    val sharedScores   = Output(Vec(4, SInt(32.W)))
    val baselineY      = Output(Vec(4, SInt(32.W)))
    val sharedY        = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new AttentionDecodeTiny())
  val shared = Module(new SharedAttentionDecodeTiny())

  baseline.io.q := io.q
  baseline.io.keys := io.keys
  baseline.io.values := io.values
  shared.io.q := io.q
  shared.io.keys := io.keys
  shared.io.values := io.values

  io.baselineScores := baseline.io.scores
  io.sharedScores := shared.io.scores
  io.baselineY := baseline.io.y
  io.sharedY := shared.io.y
}

class CausalConv1DComparisonHarness extends Module {
  val io = IO(new Bundle {
    val en       = Input(Bool())
    val clear    = Input(Bool())
    val x        = Input(Vec(4, SInt(8.W)))
    val kernel   = Input(Vec(3, Vec(4, SInt(8.W))))
    val baseline = Output(Vec(4, SInt(32.W)))
    val shared   = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new CausalConv1D())
  val shared = Module(new SharedCausalConv1D())

  baseline.io.en := io.en
  baseline.io.clear := io.clear
  baseline.io.x := io.x
  baseline.io.kernel := io.kernel
  shared.io.en := io.en
  shared.io.clear := io.clear
  shared.io.x := io.x
  shared.io.kernel := io.kernel

  io.baseline := baseline.io.y
  io.shared := shared.io.y
}

class MambaStateUpdateComparisonHarness extends Module {
  val io = IO(new Bundle {
    val en       = Input(Bool())
    val clear    = Input(Bool())
    val x        = Input(Vec(4, SInt(8.W)))
    val a        = Input(Vec(4, SInt(8.W)))
    val b        = Input(Vec(4, SInt(8.W)))
    val baseline = Output(Vec(4, SInt(32.W)))
    val shared   = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new MambaStateUpdate())
  val shared = Module(new SharedMambaStateUpdate())

  baseline.io.en := io.en
  baseline.io.clear := io.clear
  baseline.io.x := io.x
  baseline.io.a := io.a
  baseline.io.b := io.b
  shared.io.en := io.en
  shared.io.clear := io.clear
  shared.io.x := io.x
  shared.io.a := io.a
  shared.io.b := io.b

  io.baseline := baseline.io.stateOut
  io.shared := shared.io.stateOut
}

class SelectiveScanTinyComparisonHarness extends Module {
  val io = IO(new Bundle {
    val en             = Input(Bool())
    val clear          = Input(Bool())
    val x              = Input(Vec(4, SInt(8.W)))
    val a              = Input(Vec(4, SInt(8.W)))
    val b              = Input(Vec(4, SInt(8.W)))
    val gate           = Input(Vec(4, SInt(8.W)))
    val baselineState  = Output(Vec(4, SInt(32.W)))
    val sharedState    = Output(Vec(4, SInt(32.W)))
    val baselineOutput = Output(Vec(4, SInt(32.W)))
    val sharedOutput   = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new SelectiveScanTiny())
  val shared = Module(new SharedSelectiveScanTiny())

  baseline.io.en := io.en
  baseline.io.clear := io.clear
  baseline.io.x := io.x
  baseline.io.a := io.a
  baseline.io.b := io.b
  baseline.io.gate := io.gate
  shared.io.en := io.en
  shared.io.clear := io.clear
  shared.io.x := io.x
  shared.io.a := io.a
  shared.io.b := io.b
  shared.io.gate := io.gate

  io.baselineState := baseline.io.stateOut
  io.sharedState := shared.io.stateOut
  io.baselineOutput := baseline.io.y
  io.sharedOutput := shared.io.y
}

class TinyMambaBlockComparisonHarness extends Module {
  val io = IO(new Bundle {
    val en            = Input(Bool())
    val clear         = Input(Bool())
    val x             = Input(Vec(4, SInt(8.W)))
    val kernel        = Input(Vec(3, Vec(4, SInt(8.W))))
    val a             = Input(Vec(4, SInt(8.W)))
    val b             = Input(Vec(4, SInt(8.W)))
    val c             = Input(Vec(4, SInt(8.W)))
    val gate          = Input(Vec(4, SInt(8.W)))
    val baselineState = Output(Vec(4, SInt(32.W)))
    val sharedState   = Output(Vec(4, SInt(32.W)))
    val baselineY     = Output(Vec(4, SInt(32.W)))
    val sharedY       = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new TinyMambaBlock())
  val shared = Module(new SharedTinyMambaBlock())

  baseline.io.en := io.en
  baseline.io.clear := io.clear
  baseline.io.x := io.x
  baseline.io.kernel := io.kernel
  baseline.io.a := io.a
  baseline.io.b := io.b
  baseline.io.c := io.c
  baseline.io.gate := io.gate
  shared.io.en := io.en
  shared.io.clear := io.clear
  shared.io.x := io.x
  shared.io.kernel := io.kernel
  shared.io.a := io.a
  shared.io.b := io.b
  shared.io.c := io.c
  shared.io.gate := io.gate

  io.baselineState := baseline.io.stateOut
  io.sharedState := shared.io.stateOut
  io.baselineY := baseline.io.y
  io.sharedY := shared.io.y
}

class TinyJambaBlockComparisonHarness extends Module {
  val io = IO(new Bundle {
    val en                     = Input(Bool())
    val clear                  = Input(Bool())
    val useAttention           = Input(Bool())
    val x                      = Input(Vec(4, SInt(8.W)))
    val kernel                 = Input(Vec(3, Vec(4, SInt(8.W))))
    val mambaA                 = Input(Vec(4, SInt(8.W)))
    val mambaB                 = Input(Vec(4, SInt(8.W)))
    val mambaC                 = Input(Vec(4, SInt(8.W)))
    val gate                   = Input(Vec(4, SInt(8.W)))
    val attentionKeys          = Input(Vec(4, Vec(4, SInt(8.W))))
    val attentionValues        = Input(Vec(4, Vec(4, SInt(8.W))))
    val baselineState          = Output(Vec(4, SInt(32.W)))
    val sharedState            = Output(Vec(4, SInt(32.W)))
    val baselineAttentionScore = Output(Vec(4, SInt(32.W)))
    val sharedAttentionScore   = Output(Vec(4, SInt(32.W)))
    val baselineY              = Output(Vec(4, SInt(32.W)))
    val sharedY                = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new TinyJambaBlock())
  val shared = Module(new SharedTinyJambaBlock())

  baseline.io.en := io.en
  baseline.io.clear := io.clear
  baseline.io.useAttention := io.useAttention
  baseline.io.x := io.x
  baseline.io.kernel := io.kernel
  baseline.io.mambaA := io.mambaA
  baseline.io.mambaB := io.mambaB
  baseline.io.mambaC := io.mambaC
  baseline.io.gate := io.gate
  baseline.io.attentionKeys := io.attentionKeys
  baseline.io.attentionValues := io.attentionValues
  shared.io.en := io.en
  shared.io.clear := io.clear
  shared.io.useAttention := io.useAttention
  shared.io.x := io.x
  shared.io.kernel := io.kernel
  shared.io.mambaA := io.mambaA
  shared.io.mambaB := io.mambaB
  shared.io.mambaC := io.mambaC
  shared.io.gate := io.gate
  shared.io.attentionKeys := io.attentionKeys
  shared.io.attentionValues := io.attentionValues

  io.baselineState := baseline.io.stateOut
  io.sharedState := shared.io.stateOut
  io.baselineAttentionScore := baseline.io.attentionScores
  io.sharedAttentionScore := shared.io.attentionScores
  io.baselineY := baseline.io.y
  io.sharedY := shared.io.y
}

class DenseMLPComparisonHarness extends Module {
  val io = IO(new Bundle {
    val x = Input(Vec(4, SInt(8.W)))
    val gateWeight = Input(Vec(4, Vec(4, SInt(8.W))))
    val gateBias = Input(Vec(4, SInt(32.W)))
    val upWeight = Input(Vec(4, Vec(4, SInt(8.W))))
    val upBias = Input(Vec(4, SInt(32.W)))
    val downWeight = Input(Vec(4, Vec(4, SInt(8.W))))
    val downBias = Input(Vec(4, SInt(32.W)))
    val baselineGate = Output(Vec(4, SInt(32.W)))
    val sharedGate = Output(Vec(4, SInt(32.W)))
    val baselineUp = Output(Vec(4, SInt(32.W)))
    val sharedUp = Output(Vec(4, SInt(32.W)))
    val baselineActivatedGate = Output(Vec(4, SInt(8.W)))
    val sharedActivatedGate = Output(Vec(4, SInt(8.W)))
    val baselineHidden = Output(Vec(4, SInt(8.W)))
    val sharedHidden = Output(Vec(4, SInt(8.W)))
    val baselineY = Output(Vec(4, SInt(32.W)))
    val sharedY = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new DenseMLPMini())
  val shared = Module(new SharedDenseMLPMini())

  baseline.io.x := io.x
  baseline.io.gateWeight := io.gateWeight
  baseline.io.gateBias := io.gateBias
  baseline.io.upWeight := io.upWeight
  baseline.io.upBias := io.upBias
  baseline.io.downWeight := io.downWeight
  baseline.io.downBias := io.downBias
  shared.io.x := io.x
  shared.io.gateWeight := io.gateWeight
  shared.io.gateBias := io.gateBias
  shared.io.upWeight := io.upWeight
  shared.io.upBias := io.upBias
  shared.io.downWeight := io.downWeight
  shared.io.downBias := io.downBias

  io.baselineGate := baseline.io.gate
  io.sharedGate := shared.io.gate
  io.baselineUp := baseline.io.up
  io.sharedUp := shared.io.up
  io.baselineActivatedGate := baseline.io.activatedGate
  io.sharedActivatedGate := shared.io.activatedGate
  io.baselineHidden := baseline.io.hidden
  io.sharedHidden := shared.io.hidden
  io.baselineY := baseline.io.y
  io.sharedY := shared.io.y
}

class MlpPathComparisonHarness extends Module {
  val io = IO(new Bundle {
    val enableMoE = Input(Bool())
    val x = Input(Vec(4, SInt(8.W)))
    val gateWeight = Input(Vec(4, Vec(4, SInt(8.W))))
    val gateBias = Input(Vec(4, SInt(32.W)))
    val upWeight = Input(Vec(4, Vec(4, SInt(8.W))))
    val upBias = Input(Vec(4, SInt(32.W)))
    val downWeight = Input(Vec(4, Vec(4, SInt(8.W))))
    val downBias = Input(Vec(4, SInt(32.W)))
    val routerWeight = Input(Vec(2, Vec(4, SInt(8.W))))
    val routerBias = Input(Vec(2, SInt(32.W)))
    val expertGateWeight = Input(Vec(2, Vec(4, Vec(4, SInt(8.W)))))
    val expertGateBias = Input(Vec(2, Vec(4, SInt(32.W))))
    val expertUpWeight = Input(Vec(2, Vec(4, Vec(4, SInt(8.W)))))
    val expertUpBias = Input(Vec(2, Vec(4, SInt(32.W))))
    val expertDownWeight = Input(Vec(2, Vec(4, Vec(4, SInt(8.W)))))
    val expertDownBias = Input(Vec(2, Vec(4, SInt(32.W))))
    val baselineDispatchValid = Output(Bool())
    val sharedDispatchValid = Output(Bool())
    val baselineCombineValid = Output(Bool())
    val sharedCombineValid = Output(Bool())
    val baselineSelectedExpert = Output(UInt(1.W))
    val sharedSelectedExpert = Output(UInt(1.W))
    val baselineDenseY = Output(Vec(4, SInt(32.W)))
    val sharedDenseY = Output(Vec(4, SInt(32.W)))
    val baselineMoeY = Output(Vec(4, SInt(32.W)))
    val sharedMoeY = Output(Vec(4, SInt(32.W)))
    val baselineY = Output(Vec(4, SInt(32.W)))
    val sharedY = Output(Vec(4, SInt(32.W)))
  })

  val baseline = Module(new MlpPathMini())
  val shared = Module(new SharedMlpPathMini())

  baseline.io.enableMoE := io.enableMoE
  baseline.io.x := io.x
  baseline.io.gateWeight := io.gateWeight
  baseline.io.gateBias := io.gateBias
  baseline.io.upWeight := io.upWeight
  baseline.io.upBias := io.upBias
  baseline.io.downWeight := io.downWeight
  baseline.io.downBias := io.downBias
  baseline.io.routerWeight := io.routerWeight
  baseline.io.routerBias := io.routerBias
  baseline.io.expertGateWeight := io.expertGateWeight
  baseline.io.expertGateBias := io.expertGateBias
  baseline.io.expertUpWeight := io.expertUpWeight
  baseline.io.expertUpBias := io.expertUpBias
  baseline.io.expertDownWeight := io.expertDownWeight
  baseline.io.expertDownBias := io.expertDownBias
  baseline.io.dispatchReady := true.B
  baseline.io.combineReady := true.B
  shared.io.enableMoE := io.enableMoE
  shared.io.x := io.x
  shared.io.gateWeight := io.gateWeight
  shared.io.gateBias := io.gateBias
  shared.io.upWeight := io.upWeight
  shared.io.upBias := io.upBias
  shared.io.downWeight := io.downWeight
  shared.io.downBias := io.downBias
  shared.io.routerWeight := io.routerWeight
  shared.io.routerBias := io.routerBias
  shared.io.expertGateWeight := io.expertGateWeight
  shared.io.expertGateBias := io.expertGateBias
  shared.io.expertUpWeight := io.expertUpWeight
  shared.io.expertUpBias := io.expertUpBias
  shared.io.expertDownWeight := io.expertDownWeight
  shared.io.expertDownBias := io.expertDownBias
  shared.io.dispatchReady := true.B
  shared.io.combineReady := true.B

  io.baselineDispatchValid := baseline.io.dispatchValid
  io.sharedDispatchValid := shared.io.dispatchValid
  io.baselineCombineValid := baseline.io.combineValid
  io.sharedCombineValid := shared.io.combineValid
  io.baselineSelectedExpert := baseline.io.selectedExpert
  io.sharedSelectedExpert := shared.io.selectedExpert
  io.baselineDenseY := baseline.io.denseY
  io.sharedDenseY := shared.io.denseY
  io.baselineMoeY := baseline.io.moeY
  io.sharedMoeY := shared.io.moeY
  io.baselineY := baseline.io.y
  io.sharedY := shared.io.y
}

class ResourceReuseComparisonSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "resource reuse comparison harnesses"

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

  private def pokeMlpPathWeights(dut: MlpPathComparisonHarness): Unit = {
    pokeIdentity(dut.io.gateWeight)
    pokeVector(dut.io.gateBias, Seq(1, 1, 1, 1))
    pokeReverseIdentity(dut.io.upWeight)
    pokeVector(dut.io.upBias, Seq(0, 0, 0, 0))
    pokeIdentity(dut.io.downWeight)
    pokeVector(dut.io.downBias, Seq(0, 0, 0, 0))
    pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 2, 0, 0)))
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

  it should "match DotProduct and SharedDotProduct" in {
    test(new DotProductComparisonHarness) { dut =>
      pokeVector(dut.io.a, Seq(1, -2, 3, -4))
      pokeVector(dut.io.b, Seq(5, 6, -7, -8))
      dut.io.shared.expect(dut.io.baseline.peek())
    }
  }

  it should "match Linear4 and SharedLinear4" in {
    test(new Linear4ComparisonHarness) { dut =>
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

      for (i <- 0 until 4) {
        dut.io.shared(i).expect(dut.io.baseline(i).peek())
      }
    }
  }

  it should "match Linear4, SharedLinear4, and SerialSharedLinear4 after serial execution" in {
    test(new SerialLinear4ComparisonHarness) { dut =>
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
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()

      dut.io.start.poke(false.B)
      var done = false
      for (_ <- 0 until 20) {
        if (!done) {
          dut.clock.step()
          done = dut.io.serialDone.peek().litToBoolean
        }
      }
      assert(done, "SerialSharedLinear4 did not finish in the comparison harness")

      for (i <- 0 until 4) {
        dut.io.shared(i).expect(dut.io.baseline(i).peek())
        dut.io.serial(i).expect(dut.io.baseline(i).peek())
      }
    }
  }

  it should "match AttentionDecodeTiny and SharedAttentionDecodeTiny" in {
    test(new AttentionDecodeComparisonHarness) { dut =>
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

      for (i <- 0 until 4) {
        dut.io.sharedScores(i).expect(dut.io.baselineScores(i).peek())
        dut.io.sharedY(i).expect(dut.io.baselineY(i).peek())
      }
    }
  }

  it should "match CausalConv1D and SharedCausalConv1D over time" in {
    test(new CausalConv1DComparisonHarness) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeMatrix(dut.io.kernel, Seq.fill(3)(Seq.fill(4)(1)))

      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      for (i <- 0 until 4) {
        dut.io.shared(i).expect(dut.io.baseline(i).peek())
      }
      dut.clock.step()

      pokeVector(dut.io.x, Seq(5, 6, 7, 8))
      for (i <- 0 until 4) {
        dut.io.shared(i).expect(dut.io.baseline(i).peek())
      }
      dut.clock.step()

      dut.io.en.poke(false.B)
      pokeVector(dut.io.x, Seq(2, 2, 2, 2))
      for (i <- 0 until 4) {
        dut.io.shared(i).expect(dut.io.baseline(i).peek())
      }
    }
  }

  it should "match MambaStateUpdate and SharedMambaStateUpdate over time" in {
    test(new MambaStateUpdateComparisonHarness) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, -2, 3, -4))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(3, 3, 3, 3))

      for (i <- 0 until 4) {
        dut.io.shared(i).expect(dut.io.baseline(i).peek())
      }
      dut.clock.step()

      for (i <- 0 until 4) {
        dut.io.shared(i).expect(dut.io.baseline(i).peek())
      }

      dut.io.en.poke(false.B)
      pokeVector(dut.io.x, Seq(7, 7, 7, 7))
      dut.clock.step()
      for (i <- 0 until 4) {
        dut.io.shared(i).expect(dut.io.baseline(i).peek())
      }
    }
  }

  it should "match SelectiveScanTiny and SharedSelectiveScanTiny over time" in {
    test(new SelectiveScanTinyComparisonHarness) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, -2, 3, -4))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(2, 2, 2, 2))
      pokeVector(dut.io.gate, Seq(3, 3, 3, 3))

      for (i <- 0 until 4) {
        dut.io.sharedState(i).expect(dut.io.baselineState(i).peek())
        dut.io.sharedOutput(i).expect(dut.io.baselineOutput(i).peek())
      }
      dut.clock.step()

      for (i <- 0 until 4) {
        dut.io.sharedState(i).expect(dut.io.baselineState(i).peek())
        dut.io.sharedOutput(i).expect(dut.io.baselineOutput(i).peek())
      }

      dut.io.en.poke(false.B)
      pokeVector(dut.io.x, Seq(7, 7, 7, 7))
      dut.clock.step()
      for (i <- 0 until 4) {
        dut.io.sharedState(i).expect(dut.io.baselineState(i).peek())
        dut.io.sharedOutput(i).expect(dut.io.baselineOutput(i).peek())
      }
    }
  }

  it should "match TinyMambaBlock and SharedTinyMambaBlock over time" in {
    test(new TinyMambaBlockComparisonHarness) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeMatrix(dut.io.kernel, Seq(Seq(1, 1, 1, 1), Seq(0, 0, 0, 0), Seq(0, 0, 0, 0)))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(2, 2, 2, 2))
      pokeVector(dut.io.c, Seq(3, 3, 3, 3))
      pokeVector(dut.io.gate, Seq(1, 1, 1, 1))

      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      for (i <- 0 until 4) {
        dut.io.sharedState(i).expect(dut.io.baselineState(i).peek())
        dut.io.sharedY(i).expect(dut.io.baselineY(i).peek())
      }
      dut.clock.step()

      for (i <- 0 until 4) {
        dut.io.sharedState(i).expect(dut.io.baselineState(i).peek())
        dut.io.sharedY(i).expect(dut.io.baselineY(i).peek())
      }
    }
  }

  it should "match TinyJambaBlock and SharedTinyJambaBlock in attention mode" in {
    test(new TinyJambaBlockComparisonHarness) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeMatrix(dut.io.kernel, Seq(Seq(1, 1, 1, 1), Seq(0, 0, 0, 0), Seq(0, 0, 0, 0)))
      pokeVector(dut.io.mambaA, Seq(1, 1, 1, 1))
      pokeVector(dut.io.mambaB, Seq(2, 2, 2, 2))
      pokeVector(dut.io.mambaC, Seq(3, 3, 3, 3))
      pokeVector(dut.io.gate, Seq(1, 1, 1, 1))
      pokeMatrix(dut.io.attentionKeys, Seq.fill(4)(Seq(1, 0, 0, 0)))
      pokeMatrix(dut.io.attentionValues, Seq.fill(4)(Seq(1, 1, 1, 1)))

      for (i <- 0 until 4) {
        dut.io.sharedState(i).expect(dut.io.baselineState(i).peek())
        dut.io.sharedAttentionScore(i).expect(dut.io.baselineAttentionScore(i).peek())
        dut.io.sharedY(i).expect(dut.io.baselineY(i).peek())
      }
      dut.clock.step()

      for (i <- 0 until 4) {
        dut.io.sharedState(i).expect(dut.io.baselineState(i).peek())
        dut.io.sharedAttentionScore(i).expect(dut.io.baselineAttentionScore(i).peek())
        dut.io.sharedY(i).expect(dut.io.baselineY(i).peek())
      }
    }
  }

  it should "match DenseMLPMini and SharedDenseMLPMini" in {
    test(new DenseMLPComparisonHarness) { dut =>
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeMatrix(
        dut.io.gateWeight,
        Seq(
          Seq(1, 0, 0, 0),
          Seq(0, 1, 0, 0),
          Seq(0, 0, 1, 0),
          Seq(0, 0, 0, 1)
        )
      )
      pokeVector(dut.io.gateBias, Seq(1, 1, 1, 1))
      pokeMatrix(
        dut.io.upWeight,
        Seq(
          Seq(0, 0, 0, 1),
          Seq(0, 0, 1, 0),
          Seq(0, 1, 0, 0),
          Seq(1, 0, 0, 0)
        )
      )
      pokeVector(dut.io.upBias, Seq(0, 0, 0, 0))
      pokeMatrix(
        dut.io.downWeight,
        Seq(
          Seq(1, 0, 0, 0),
          Seq(0, 1, 0, 0),
          Seq(0, 0, 1, 0),
          Seq(0, 0, 0, 1)
        )
      )
      pokeVector(dut.io.downBias, Seq(0, 0, 0, 0))

      for (i <- 0 until 4) {
        dut.io.sharedGate(i).expect(dut.io.baselineGate(i).peek())
        dut.io.sharedUp(i).expect(dut.io.baselineUp(i).peek())
        dut.io.sharedActivatedGate(i).expect(dut.io.baselineActivatedGate(i).peek())
        dut.io.sharedHidden(i).expect(dut.io.baselineHidden(i).peek())
        dut.io.sharedY(i).expect(dut.io.baselineY(i).peek())
      }
    }
  }

  it should "match MlpPathMini and SharedMlpPathMini in dense and MoE modes" in {
    test(new MlpPathComparisonHarness) { dut =>
      pokeVector(dut.io.x, Seq(2, 1, 0, 0))
      pokeMlpPathWeights(dut)

      dut.io.enableMoE.poke(false.B)
      dut.io.sharedDispatchValid.expect(dut.io.baselineDispatchValid.peek())
      dut.io.sharedCombineValid.expect(dut.io.baselineCombineValid.peek())
      dut.io.sharedSelectedExpert.expect(dut.io.baselineSelectedExpert.peek())
      for (i <- 0 until 4) {
        dut.io.sharedDenseY(i).expect(dut.io.baselineDenseY(i).peek())
        dut.io.sharedMoeY(i).expect(dut.io.baselineMoeY(i).peek())
        dut.io.sharedY(i).expect(dut.io.baselineY(i).peek())
      }

      dut.io.enableMoE.poke(true.B)
      dut.io.sharedDispatchValid.expect(dut.io.baselineDispatchValid.peek())
      dut.io.sharedCombineValid.expect(dut.io.baselineCombineValid.peek())
      dut.io.sharedSelectedExpert.expect(dut.io.baselineSelectedExpert.peek())
      for (i <- 0 until 4) {
        dut.io.sharedDenseY(i).expect(dut.io.baselineDenseY(i).peek())
        dut.io.sharedMoeY(i).expect(dut.io.baselineMoeY(i).peek())
        dut.io.sharedY(i).expect(dut.io.baselineY(i).peek())
      }
    }
  }
}
