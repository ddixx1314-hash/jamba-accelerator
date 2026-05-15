package jamba.mamba

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class Jamba2MambaMixerMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Jamba2MambaMixerMini"

  // Expected values in this spec are generated from python.golden.mamba_ops.mamba_mixer_step
  // using jamba2_mini_fixture().

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

  it should "match the deterministic Python Mamba mixer fixture for one token" in {
    test(new Jamba2MambaMixerMini()) { dut =>
      dut.io.en.poke(true.B)
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      pokeIdentity(dut.io.inputWeight)
      pokeVector(dut.io.inputBias, Seq(0, 0, 0, 0))
      pokeZeroMatrix(dut.io.bWeight)
      pokeVector(dut.io.bBias, Seq(2, 2, 2, 2))
      pokeZeroMatrix(dut.io.cWeight)
      pokeVector(dut.io.cBias, Seq(1, 1, 1, 1))
      pokeVector(dut.io.a, Seq(1, 1, 1, 1))
      pokeKernel(dut.io.kernel, Seq.fill(4)(Seq(1, 1, 1, 1)))

      expectVector(dut.io.projected, Seq(1, 0, 0, 0))
      expectVector(dut.io.conv, Seq(1, 0, 0, 0))
      expectVector(dut.io.b, Seq(2, 2, 2, 2))
      expectVector(dut.io.c, Seq(1, 1, 1, 1))
      expectVector(dut.io.stateOut, Seq(0, 0, 0, 0))
      expectVector(dut.io.y, Seq(0, 0, 0, 0))
      dut.clock.step()

      expectVector(dut.io.stateOut, Seq(2, 0, 0, 0))
      expectVector(dut.io.y, Seq(2, 0, 0, 0))
    }
  }

  it should "match Python golden values across two tokens with convolution history" in {
    test(new Jamba2MambaMixerMini()) { dut =>
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
      expectVector(dut.io.projected, Seq(1, 0, 0, 0))
      expectVector(dut.io.conv, Seq(1, 0, 0, 0))
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(2, 0, 0, 0))
      expectVector(dut.io.y, Seq(2, 0, 0, 0))

      pokeVector(dut.io.x, Seq(0, 1, 0, 0))
      expectVector(dut.io.projected, Seq(0, 1, 0, 0))
      expectVector(dut.io.conv, Seq(1, 1, 0, 0))
      expectVector(dut.io.b, Seq(2, 2, 2, 2))
      expectVector(dut.io.c, Seq(1, 1, 1, 1))
      dut.clock.step()
      expectVector(dut.io.stateOut, Seq(4, 2, 0, 0))
      expectVector(dut.io.y, Seq(4, 2, 0, 0))
    }
  }
}
