package basic

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class Jamba2MiniAcceleratorSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Jamba2MiniAccelerator"

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

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).expect(values(i).S)
    }
  }

  private def pokeCommon(dut: Jamba2MiniAccelerator): Unit = {
    dut.io.en.poke(true.B)
    dut.io.clear.poke(false.B)
    pokeVector(dut.io.x, Seq(1, 2, 3, 4))
    pokeMatrix(dut.io.kernel, Seq(Seq(1, 1, 1, 1), Seq(0, 0, 0, 0), Seq(0, 0, 0, 0)))
    pokeVector(dut.io.mambaA, Seq(1, 1, 1, 1))
    pokeVector(dut.io.mambaB, Seq(2, 2, 2, 2))
    pokeVector(dut.io.mambaC, Seq(3, 3, 3, 3))
    pokeVector(dut.io.gate, Seq(1, 1, 1, 1))
    pokeMatrix(dut.io.attentionKeys, Seq.fill(4)(Seq(1, 0, 0, 0)))
    pokeMatrix(dut.io.attentionValues, Seq.fill(4)(Seq(1, 1, 1, 1)))
  }

  it should "run the tiny Jamba-like datapath and report valid after enable" in {
    test(new Jamba2MiniAccelerator()) { dut =>
      pokeCommon(dut)
      dut.io.useAttention.poke(true.B)

      dut.io.valid.expect(false.B)
      expectVector(dut.io.attentionScores, Seq(1, 1, 1, 1))
      expectVector(dut.io.y, Seq(5, 6, 7, 8))

      dut.clock.step()
      dut.io.valid.expect(true.B)
      expectVector(dut.io.stateOut, Seq(2, 4, 6, 8))
      expectVector(dut.io.y, Seq(11, 18, 25, 32))
    }
  }

  it should "clear state and valid" in {
    test(new Jamba2MiniAccelerator()) { dut =>
      pokeCommon(dut)
      dut.io.useAttention.poke(false.B)
      dut.clock.step()
      dut.io.valid.expect(true.B)

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.valid.expect(false.B)
      expectVector(dut.io.stateOut, Seq(0, 0, 0, 0))
    }
  }
}
