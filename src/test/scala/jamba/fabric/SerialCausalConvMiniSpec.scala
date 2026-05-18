package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class SerialCausalConvMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "SerialCausalConvMini"

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

  private def pokeKernel(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (tap <- values.indices) {
      for (lane <- values(tap).indices) {
        port(tap)(lane).poke(values(tap)(lane).S)
      }
    }
  }

  private def runToDone(dut: SerialCausalConvMini, maxCycles: Int = 24): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"SerialCausalConvMini did not finish within $maxCycles cycles")
  }

  it should "compute one four-tap causal convolution token" in {
    test(new SerialCausalConvMini()) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeKernel(dut.io.kernel, Seq.fill(4)(Seq(1, 1, 1, 1)))

      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.y, Seq(1, 2, 3, 4))
    }
  }

  it should "preserve causal history across tokens" in {
    test(new SerialCausalConvMini()) { dut =>
      dut.io.clear.poke(false.B)
      pokeKernel(dut.io.kernel, Seq.fill(4)(Seq(1, 1, 1, 1)))

      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      expectVector(dut.io.y, Seq(1, 0, 0, 0))

      pokeVector(dut.io.x, Seq(0, 1, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      expectVector(dut.io.y, Seq(1, 1, 0, 0))

      pokeVector(dut.io.x, Seq(0, 0, 1, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      expectVector(dut.io.y, Seq(1, 1, 1, 0))
    }
  }

  it should "clear active schedule and convolution history" in {
    test(new SerialCausalConvMini()) { dut =>
      dut.io.clear.poke(false.B)
      pokeKernel(dut.io.kernel, Seq.fill(4)(Seq(1, 1, 1, 1)))
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
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
      expectVector(dut.io.y, Seq(0, 0, 0, 0))

      pokeVector(dut.io.x, Seq(0, 1, 0, 0))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      expectVector(dut.io.y, Seq(0, 1, 0, 0))
    }
  }
}
