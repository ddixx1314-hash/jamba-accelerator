package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SharedCausalConv1DSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SharedCausalConv1D"

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).poke(values(i).S)
    }
  }

  private def pokeKernel(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (tap <- values.indices) {
      for (lane <- values(tap).indices) {
        port(tap)(lane).poke(values(tap)(lane).S)
      }
    }
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).expect(values(i).S)
    }
  }

  it should "compute a three-tap causal convolution over time" in {
    test(new SharedCausalConv1D()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeKernel(dut.io.kernel, Seq.fill(3)(Seq.fill(4)(1)))

      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      expectVector(dut.io.y, Seq(1, 2, 3, 4))
      dut.clock.step()

      pokeVector(dut.io.x, Seq(5, 6, 7, 8))
      expectVector(dut.io.y, Seq(6, 8, 10, 12))
      dut.clock.step()

      pokeVector(dut.io.x, Seq(1, 1, 1, 1))
      expectVector(dut.io.y, Seq(7, 9, 11, 13))
    }
  }

  it should "hold delays when disabled and reset them on clear" in {
    test(new SharedCausalConv1D()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeKernel(dut.io.kernel, Seq.fill(3)(Seq.fill(4)(1)))
      pokeVector(dut.io.x, Seq(1, 1, 1, 1))
      dut.clock.step()

      dut.io.en.poke(false.B)
      pokeVector(dut.io.x, Seq(2, 2, 2, 2))
      expectVector(dut.io.y, Seq(3, 3, 3, 3))
      dut.clock.step()
      expectVector(dut.io.y, Seq(3, 3, 3, 3))

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.clear.poke(false.B)
      expectVector(dut.io.y, Seq(2, 2, 2, 2))
    }
  }
}
