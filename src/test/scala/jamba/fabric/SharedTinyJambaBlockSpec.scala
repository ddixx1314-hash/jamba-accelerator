package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SharedTinyJambaBlockSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SharedTinyJambaBlock"

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

  private def pokeKernel(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = pokeMatrix(port, values)

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).expect(values(i).S)
    }
  }

  private def pokeCommon(dut: SharedTinyJambaBlock): Unit = {
    dut.io.en.poke(true.B)
    dut.io.clear.poke(false.B)
    pokeVector(dut.io.x, Seq(1, 2, 3, 4))
    pokeKernel(dut.io.kernel, Seq(Seq(1, 1, 1, 1), Seq(0, 0, 0, 0), Seq(0, 0, 0, 0)))
    pokeVector(dut.io.mambaA, Seq(1, 1, 1, 1))
    pokeVector(dut.io.mambaB, Seq(2, 2, 2, 2))
    pokeVector(dut.io.mambaC, Seq(3, 3, 3, 3))
    pokeVector(dut.io.gate, Seq(1, 1, 1, 1))
    pokeMatrix(dut.io.attentionKeys, Seq.fill(4)(Seq(1, 0, 0, 0)))
    pokeMatrix(dut.io.attentionValues, Seq.fill(4)(Seq(1, 1, 1, 1)))
  }

  it should "emit the shared Mamba path when attention is disabled" in {
    test(new SharedTinyJambaBlock()) { dut =>
      pokeCommon(dut)
      dut.io.useAttention.poke(false.B)

      expectVector(dut.io.y, Seq(1, 2, 3, 4))
      dut.clock.step()
      expectVector(dut.io.y, Seq(7, 14, 21, 28))
    }
  }

  it should "add shared attention output when attention is enabled" in {
    test(new SharedTinyJambaBlock()) { dut =>
      pokeCommon(dut)
      dut.io.useAttention.poke(true.B)

      expectVector(dut.io.attentionScores, Seq(1, 1, 1, 1))
      expectVector(dut.io.y, Seq(5, 6, 7, 8))
      dut.clock.step()
      expectVector(dut.io.y, Seq(11, 18, 25, 32))
    }
  }
}
