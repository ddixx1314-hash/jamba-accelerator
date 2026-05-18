package jamba.top

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class Jamba2MiniTileSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "Jamba2MiniTile"

  private val config = Jamba2MiniConfig.debug

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).poke(values(i).S)
    }
  }

  private def expectStableOutput(dut: Jamba2MiniTile, expected: Seq[BigInt]): Unit = {
    for (lane <- expected.indices) {
      dut.io.out(lane).expect(expected(lane).S)
    }
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) {
      port(i).expect(values(i).S)
    }
  }

  private def pokeIdle(dut: Jamba2MiniTile): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.start.poke(true.B)
    dut.io.enableMoE.poke(false.B)
    dut.io.useLoadedWeights.poke(false.B)
    dut.io.inValid.poke(false.B)
    dut.io.outReady.poke(false.B)
    pokeVector(dut.io.in, Seq(1, 0, 0, 0))
    dut.io.weightWriteValid.poke(false.B)
    dut.io.weightWriteAddr.poke(0.U)
    dut.io.weightWriteData.poke(0.S)
    dut.io.weightReadAddr.poke(0.U)
  }

  it should "clear output status and stop input acceptance while preserving loaded weights" in {
    test(new Jamba2MiniTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.weightWriteValid.poke(true.B)
      dut.io.weightWriteAddr.poke(7.U)
      dut.io.weightWriteData.poke(123.S)
      dut.io.weightWriteReady.expect(true.B)
      dut.clock.step()

      dut.io.weightWriteValid.poke(false.B)
      dut.io.inValid.poke(true.B)
      dut.clock.step()
      dut.io.outValid.expect(true.B)
      dut.io.busy.expect(true.B)
      dut.io.done.expect(true.B)

      dut.io.clear.poke(true.B)
      dut.io.inReady.expect(false.B)
      dut.clock.step()

      dut.io.outValid.expect(false.B)
      dut.io.busy.expect(false.B)
      dut.io.done.expect(false.B)
      dut.io.error.expect(false.B)
      dut.io.weightReadAddr.poke(7.U)
      dut.io.weightReadData.expect(123.S)
    }
  }

  it should "accept one token and hold output stable under backpressure" in {
    test(new Jamba2MiniTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.inReady.expect(true.B)
      dut.io.outValid.expect(false.B)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      dut.io.inReady.expect(false.B)
      val held = dut.io.out.map(_.peek().litValue)

      pokeVector(dut.io.in, Seq(4, 3, 2, 1))
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      dut.io.inReady.expect(false.B)
      expectStableOutput(dut, held)
    }
  }

  it should "consume and replace output in the same cycle" in {
    test(new Jamba2MiniTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()
      dut.io.outValid.expect(true.B)
      val first = dut.io.out.map(_.peek().litValue)

      dut.io.outReady.poke(true.B)
      pokeVector(dut.io.in, Seq(2, 0, 0, 0))
      dut.io.inReady.expect(true.B)
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      val second = dut.io.out.map(_.peek().litValue)
      assert(second != first, "replacement token should update the output buffer")
    }
  }

  it should "gate input acceptance with start" in {
    test(new Jamba2MiniTile(config)) { dut =>
      pokeIdle(dut)
      dut.io.start.poke(false.B)
      dut.io.inValid.poke(true.B)
      dut.io.inReady.expect(false.B)
      dut.clock.step()

      dut.io.outValid.expect(false.B)
      dut.io.busy.expect(false.B)
      dut.io.done.expect(false.B)
    }
  }

  it should "expose sparse attention schedule and MoE-lite debug outputs" in {
    test(new Jamba2MiniTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.debugLayerUsesAttention(0).expect(false.B)
      dut.io.debugLayerUsesAttention(1).expect(false.B)
      dut.io.debugLayerUsesAttention(2).expect(false.B)
      dut.io.debugLayerUsesAttention(3).expect(true.B)

      dut.io.enableMoE.poke(true.B)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(0, 3, 0, 0))
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      dut.io.debugLayerSelectedExpert(0).expect(1.U)
    }
  }

  it should "run a two-token end-to-end demo trace against Python golden values" in {
    test(new Jamba2MiniTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.weightWriteValid.poke(true.B)
      dut.io.weightWriteAddr.poke(3.U)
      dut.io.weightWriteData.poke(77.S)
      dut.clock.step()
      dut.io.weightWriteValid.poke(false.B)

      dut.io.clear.poke(true.B)
      dut.clock.step()
      dut.io.clear.poke(false.B)
      dut.io.weightReadAddr.poke(3.U)
      dut.io.weightReadData.expect(77.S)

      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      expectVector(dut.io.out, Seq(3, 0, 0, 3))
      expectVector(dut.io.debugLayerStateOut(0), Seq(2, 0, 0, 0))
      dut.io.debugLayerKvWriteIndex(3).expect(1.U)
      dut.io.debugLayerKvValidCount(3).expect(1.U)

      dut.io.inValid.poke(false.B)
      dut.io.outReady.poke(true.B)
      dut.clock.step()
      dut.io.outValid.expect(false.B)

      dut.io.outReady.poke(false.B)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(2, 0, 0, 0))
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      expectVector(dut.io.out, Seq(6, 0, 0, 3))
      expectVector(dut.io.debugLayerStateOut(0), Seq(8, 0, 0, 0))
      dut.io.debugLayerKvWriteIndex(3).expect(2.U)
      dut.io.debugLayerKvValidCount(3).expect(2.U)
    }
  }

  it should "drive the core from loaded weight-store values when selected" in {
    test(new Jamba2MiniTile(config)) { dut =>
      pokeIdle(dut)

      dut.io.weightWriteValid.poke(true.B)
      dut.io.weightWriteAddr.poke(232.U)
      dut.io.weightWriteData.poke(5.S)
      dut.clock.step()
      dut.io.weightWriteValid.poke(false.B)

      dut.io.useLoadedWeights.poke(true.B)
      dut.io.inValid.poke(true.B)
      pokeVector(dut.io.in, Seq(1, 0, 0, 0))
      dut.clock.step()

      dut.io.outValid.expect(true.B)
      expectVector(dut.io.out, Seq(21, 0, 0, 0))
      dut.io.weightReadAddr.poke(232.U)
      dut.io.weightReadData.expect(5.S)
    }
  }
}
