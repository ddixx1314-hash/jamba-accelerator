package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SerialMambaMixerMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SerialMambaMixerMini"

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

  private def pokeDefaultWeights(dut: SerialMambaMixerMini): Unit = {
    pokeIdentity(dut.io.inputWeight)
    pokeVector(dut.io.inputBias, Seq(0, 0, 0, 0))
    pokeZeroMatrix(dut.io.bWeight)
    pokeVector(dut.io.bBias, Seq(2, 2, 2, 2))
    pokeZeroMatrix(dut.io.cWeight)
    pokeVector(dut.io.cBias, Seq(1, 1, 1, 1))
    pokeVector(dut.io.a, Seq(1, 1, 1, 1))
    pokeKernel(dut.io.kernel, Seq.fill(4)(Seq(1, 1, 1, 1)))
  }

  private def runToDone(dut: SerialMambaMixerMini, maxCycles: Int = 80): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"SerialMambaMixerMini did not finish within $maxCycles cycles")
  }

  it should "process one token and report post-token Mamba state" in {
    test(new SerialMambaMixerMini()) { dut =>
      dut.io.clear.poke(false.B)
      pokeDefaultWeights(dut)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))

      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.valid.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.projected, Seq(1, 0, 0, 0))
      expectVector(dut.io.conv, Seq(1, 0, 0, 0))
      expectVector(dut.io.b, Seq(2, 2, 2, 2))
      expectVector(dut.io.c, Seq(1, 1, 1, 1))
      expectVector(dut.io.stateOut, Seq(2, 0, 0, 0))
      expectVector(dut.io.y, Seq(2, 0, 0, 0))
    }
  }

  it should "preserve convolution history and scan state across two tokens" in {
    test(new SerialMambaMixerMini()) { dut =>
      dut.io.clear.poke(false.B)
      pokeDefaultWeights(dut)

      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      expectVector(dut.io.stateOut, Seq(2, 0, 0, 0))
      expectVector(dut.io.y, Seq(2, 0, 0, 0))

      pokeVector(dut.io.x, Seq(0, 1, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)

      expectVector(dut.io.projected, Seq(0, 1, 0, 0))
      expectVector(dut.io.conv, Seq(1, 1, 0, 0))
      expectVector(dut.io.stateOut, Seq(4, 2, 0, 0))
      expectVector(dut.io.y, Seq(4, 2, 0, 0))
    }
  }

  it should "clear projection, convolution, and scan state" in {
    test(new SerialMambaMixerMini()) { dut =>
      dut.io.clear.poke(false.B)
      pokeDefaultWeights(dut)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))

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
      expectVector(dut.io.projected, Seq(0, 0, 0, 0))
      expectVector(dut.io.conv, Seq(0, 0, 0, 0))
      expectVector(dut.io.stateOut, Seq(0, 0, 0, 0))
      expectVector(dut.io.y, Seq(0, 0, 0, 0))
    }
  }
}
