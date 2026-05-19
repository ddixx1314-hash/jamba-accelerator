package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class UnifiedMoEPathMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "UnifiedMoEPathMini"

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).poke(values(i).S)
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit = {
    for (i <- values.indices) port(i).expect(values(i).S)
  }

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) port(row)(col).poke(values(row)(col).S)
    }
  }

  private def pokeIdentity(port: Vec[Vec[SInt]]): Unit =
    pokeMatrix(port, Seq(Seq(1,0,0,0), Seq(0,1,0,0), Seq(0,0,1,0), Seq(0,0,0,1)))

  private def pokeExpertWeights(dut: UnifiedMoEPathMini): Unit = {
    for (expert <- 0 until 2) {
      pokeIdentity(dut.io.expertGateWeight(expert))
      pokeVector(dut.io.expertGateBias(expert), Seq(1, 1, 1, 1))
      pokeIdentity(dut.io.expertUpWeight(expert))
      pokeVector(dut.io.expertUpBias(expert), Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.expertDownWeight(expert))
      pokeVector(dut.io.expertDownBias(expert), Seq.fill(4)(expert))
    }
  }

  private def runToDone(dut: UnifiedMoEPathMini, maxCycles: Int = 200): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"UnifiedMoEPathMini did not finish within $maxCycles cycles")
  }

  it should "route to expert 1 and run only the selected expert MLP" in {
    test(new UnifiedMoEPathMini()) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.dispatchReady.poke(true.B)
      dut.io.combineReady.poke(true.B)
      pokeVector(dut.io.x, Seq(2, 1, 0, 0))
      pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 2, 0, 0)))
      pokeVector(dut.io.routerBias, Seq(0, 1))
      pokeExpertWeights(dut)

      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      dut.io.routerScores(0).expect(2.S)
      dut.io.routerScores(1).expect(3.S)
      dut.io.selectedExpert.expect(1.U)
      dut.io.combineValid.expect(true.B)
      expectVector(dut.io.y, Seq(7, 3, 1, 1))
    }
  }

  it should "choose expert 0 on ties" in {
    test(new UnifiedMoEPathMini()) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.dispatchReady.poke(true.B)
      dut.io.combineReady.poke(true.B)
      pokeVector(dut.io.x, Seq(1, 1, 0, 0))
      pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(1, 0, 0, 0)))
      pokeVector(dut.io.routerBias, Seq(0, 0))
      pokeExpertWeights(dut)

      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)

      dut.io.routerScores(0).expect(1.S)
      dut.io.routerScores(1).expect(1.S)
      dut.io.selectedExpert.expect(0.U)
      expectVector(dut.io.y, Seq(2, 2, 0, 0))
    }
  }

  it should "clear an active MoE schedule" in {
    test(new UnifiedMoEPathMini()) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.dispatchReady.poke(true.B)
      dut.io.combineReady.poke(true.B)
      pokeVector(dut.io.x, Seq(2, 1, 0, 0))
      pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 2, 0, 0)))
      pokeVector(dut.io.routerBias, Seq(0, 1))
      pokeExpertWeights(dut)

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
      dut.io.selectedExpert.expect(0.U)
      expectVector(dut.io.y, Seq(0, 0, 0, 0))
    }
  }
}
