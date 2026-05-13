package basic

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class TinyMambaBlockSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "TinyMambaBlock"

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

  it should "combine convolution, SSM state, and residual gate" in {
    test(new TinyMambaBlock()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeKernel(dut.io.kernel, Seq(Seq(1, 1, 1, 1), Seq(0, 0, 0, 0), Seq(0, 0, 0, 0)))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeVector(dut.io.b, Seq(2, 2, 2, 2))
      pokeVector(dut.io.c, Seq(3, 3, 3, 3))
      pokeVector(dut.io.gate, Seq(1, 1, 1, 1))

      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      expectVector(dut.io.y, Seq(1, 2, 3, 4))
      dut.clock.step()

      expectVector(dut.io.stateOut, Seq(2, 4, 6, 8))
      expectVector(dut.io.y, Seq(7, 14, 21, 28))
    }
  }
}
