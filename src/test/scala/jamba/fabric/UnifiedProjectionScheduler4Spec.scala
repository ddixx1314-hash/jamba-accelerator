package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class UnifiedProjectionScheduler4Spec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "UnifiedProjectionScheduler4"

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

  private def pokeReverseIdentity(port: Vec[Vec[SInt]]): Unit = {
    pokeMatrix(
      port,
      Seq(
        Seq(0, 0, 0, 1),
        Seq(0, 0, 1, 0),
        Seq(0, 1, 0, 0),
        Seq(1, 0, 0, 0)
      )
    )
  }

  private def zeroAll(dut: UnifiedProjectionScheduler4): Unit = {
    for (slot <- 0 until UnifiedProjectionSlots.NumSlots) {
      dut.io.slotEnable(slot).poke(false.B)
      pokeVector(dut.io.x(slot), Seq(0, 0, 0, 0))
      pokeMatrix(dut.io.weight(slot), Seq.fill(4)(Seq(0, 0, 0, 0)))
      pokeVector(dut.io.bias(slot), Seq(0, 0, 0, 0))
    }
  }

  private def runToDone(dut: UnifiedProjectionScheduler4, maxCycles: Int = 250): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        if (dut.io.done.peek().litToBoolean) {
          seenDone = true
        }
      }
    }
    assert(seenDone, s"UnifiedProjectionScheduler4 did not finish within $maxCycles cycles")
  }

  it should "schedule selected Jamba projection slots with separate input vectors" in {
    test(new UnifiedProjectionScheduler4()) { dut =>
      zeroAll(dut)

      val mIn = UnifiedProjectionSlots.MambaInput
      val mC = UnifiedProjectionSlots.MambaC
      val aOut = UnifiedProjectionSlots.AttentionOut
      val gate = UnifiedProjectionSlots.MlpGate
      val down = UnifiedProjectionSlots.MlpDown

      dut.io.slotEnable(mIn).poke(true.B)
      pokeVector(dut.io.x(mIn), Seq(1, 2, 3, 4))
      pokeIdentity(dut.io.weight(mIn))

      dut.io.slotEnable(mC).poke(true.B)
      pokeVector(dut.io.x(mC), Seq(1, 2, 3, 4))
      pokeReverseIdentity(dut.io.weight(mC))
      pokeVector(dut.io.bias(mC), Seq(10, 10, 10, 10))

      dut.io.slotEnable(aOut).poke(true.B)
      pokeVector(dut.io.x(aOut), Seq(2, 0, -1, 3))
      pokeMatrix(dut.io.weight(aOut), Seq.fill(4)(Seq(0, 0, 0, 1)))
      pokeVector(dut.io.bias(aOut), Seq(1, 2, 3, 4))

      dut.io.slotEnable(gate).poke(true.B)
      pokeVector(dut.io.x(gate), Seq(-1, 2, -3, 4))
      pokeMatrix(dut.io.weight(gate), Seq.fill(4)(Seq(1, 1, 0, 0)))

      dut.io.slotEnable(down).poke(true.B)
      pokeVector(dut.io.x(down), Seq(5, 6, 7, 8))
      pokeIdentity(dut.io.weight(down))
      pokeVector(dut.io.bias(down), Seq(-1, -1, -1, -1))

      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()

      dut.io.start.poke(false.B)
      dut.io.busy.expect(true.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.y(mIn), Seq(1, 2, 3, 4))
      expectVector(dut.io.y(mC), Seq(14, 13, 12, 11))
      expectVector(dut.io.y(aOut), Seq(4, 5, 6, 7))
      expectVector(dut.io.y(gate), Seq(1, 1, 1, 1))
      expectVector(dut.io.y(down), Seq(4, 5, 6, 7))
      expectVector(dut.io.y(UnifiedProjectionSlots.MambaB), Seq(0, 0, 0, 0))
    }
  }

  it should "latch slot inputs and ignore changes while busy" in {
    test(new UnifiedProjectionScheduler4()) { dut =>
      zeroAll(dut)
      val q = UnifiedProjectionSlots.AttentionQ

      dut.io.slotEnable(q).poke(true.B)
      pokeVector(dut.io.x(q), Seq(2, -1, 0, 3))
      pokeIdentity(dut.io.weight(q))
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()

      pokeVector(dut.io.x(q), Seq(9, 9, 9, 9))
      dut.io.start.poke(true.B)
      dut.clock.step(2)
      dut.io.start.poke(false.B)
      runToDone(dut)

      expectVector(dut.io.y(q), Seq(2, -1, 0, 3))
    }
  }

  it should "finish immediately when no slots are enabled" in {
    test(new UnifiedProjectionScheduler4()) { dut =>
      zeroAll(dut)
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut, maxCycles = 4)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      for (slot <- 0 until UnifiedProjectionSlots.NumSlots) {
        expectVector(dut.io.y(slot), Seq(0, 0, 0, 0))
      }
    }
  }

  it should "clear an active unified schedule" in {
    test(new UnifiedProjectionScheduler4()) { dut =>
      zeroAll(dut)
      val up = UnifiedProjectionSlots.MlpUp
      dut.io.slotEnable(up).poke(true.B)
      pokeVector(dut.io.x(up), Seq(1, 2, 3, 4))
      pokeIdentity(dut.io.weight(up))

      dut.io.clear.poke(false.B)
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
      expectVector(dut.io.y(up), Seq(0, 0, 0, 0))
    }
  }
}
