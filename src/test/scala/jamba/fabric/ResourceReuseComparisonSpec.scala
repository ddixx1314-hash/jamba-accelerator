package jamba.fabric

import chisel3._
import chiseltest._
import jamba.attention.AttentionDecodeTiny
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
}
