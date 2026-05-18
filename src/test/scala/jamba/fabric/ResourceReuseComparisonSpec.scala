package jamba.fabric

import chisel3._
import chiseltest._
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
}
