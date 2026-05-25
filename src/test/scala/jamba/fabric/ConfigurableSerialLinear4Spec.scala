package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class ConfigurableSerialLinear4Spec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "ConfigurableSerialLinear4"

  private def pokeVector(port: Vec[SInt], values: Seq[Int]): Unit =
    for (i <- values.indices) port(i).poke(values(i).S)

  private def pokeMatrix(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (row <- values.indices) {
      for (col <- values(row).indices) {
        port(row)(col).poke(values(row)(col).S)
      }
    }
  }

  private def expectVector(port: Vec[SInt], values: Seq[Int]): Unit =
    for (i <- values.indices) port(i).expect(values(i).S)

  private def runToDone(dut: ConfigurableSerialLinear4, maxCycles: Int = 30): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"ConfigurableSerialLinear4 did not finish within $maxCycles cycles")
  }

  private def runConfig(macLanes: Int, x: Seq[Int], weight: Seq[Seq[Int]], bias: Seq[Int], expected: Seq[Int]): Unit = {
    test(new ConfigurableSerialLinear4(macLanes = macLanes)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x)
      pokeMatrix(dut.io.weight, weight)
      pokeVector(dut.io.bias, bias)
      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.y, expected)
    }
  }

  it should "match identity, reverse, bias, zero, and signed cases for all MAC-lane counts" in {
    val cases = Seq(
      (
        "identity",
        Seq(1, 2, 3, 4),
        Seq(Seq(1, 0, 0, 0), Seq(0, 1, 0, 0), Seq(0, 0, 1, 0), Seq(0, 0, 0, 1)),
        Seq(0, 0, 0, 0),
        Seq(1, 2, 3, 4)
      ),
      (
        "reverse identity",
        Seq(1, 2, 3, 4),
        Seq(Seq(0, 0, 0, 1), Seq(0, 0, 1, 0), Seq(0, 1, 0, 0), Seq(1, 0, 0, 0)),
        Seq(0, 0, 0, 0),
        Seq(4, 3, 2, 1)
      ),
      (
        "bias",
        Seq(1, 2, 3, 4),
        Seq.fill(4)(Seq(1, 1, 1, 1)),
        Seq(1, 2, 3, 4),
        Seq(11, 12, 13, 14)
      ),
      (
        "zero",
        Seq(0, 0, 0, 0),
        Seq.fill(4)(Seq(7, -2, 5, 1)),
        Seq(0, 0, 0, 0),
        Seq(0, 0, 0, 0)
      ),
      (
        "mixed signed",
        Seq(1, -2, 3, -4),
        Seq(Seq(1, 0, 0, 0), Seq(0, 1, 0, 0), Seq(1, 1, 1, 1), Seq(2, 0, -1, 1)),
        Seq(10, 20, 30, 40),
        Seq(11, 18, 28, 35)
      )
    )

    for ((_, x, weight, bias, expected) <- cases; macLanes <- Seq(1, 2, 4)) {
      runConfig(macLanes, x, weight, bias, expected)
    }
  }

  it should "produce identical output to SerialSharedLinear4 when macLanes is 1" in {
    val x = Seq(2, -3, 4, -5)
    val weight = Seq(
      Seq(1, 2, 0, -1),
      Seq(0, -1, 3, 1),
      Seq(2, 0, -2, 1),
      Seq(-1, 1, 1, 0)
    )
    val bias = Seq(7, -4, 5, 2)
    var legacyOut = Seq.fill(4)(0L)
    var configurableOut = Seq.fill(4)(0L)

    test(new SerialSharedLinear4()) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x)
      pokeMatrix(dut.io.weight, weight)
      pokeVector(dut.io.bias, bias)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      var seenDone = false
      for (_ <- 0 until 30) {
        if (!seenDone) {
          dut.clock.step()
          seenDone = dut.io.done.peek().litToBoolean
        }
      }
      assert(seenDone, "SerialSharedLinear4 should finish")
      legacyOut = (0 until 4).map(i => dut.io.y(i).peek().litValue.toLong)
    }

    test(new ConfigurableSerialLinear4(macLanes = 1)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x)
      pokeMatrix(dut.io.weight, weight)
      pokeVector(dut.io.bias, bias)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      configurableOut = (0 until 4).map(i => dut.io.y(i).peek().litValue.toLong)
    }

    assert(configurableOut == legacyOut,
      s"macLanes=1 changed legacy behavior: configurable=$configurableOut legacy=$legacyOut")
  }

  it should "clear busy state and output registers" in {
    test(new ConfigurableSerialLinear4(macLanes = 2)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 2, 3, 4))
      pokeMatrix(dut.io.weight, Seq.fill(4)(Seq(1, 1, 1, 1)))
      pokeVector(dut.io.bias, Seq(0, 0, 0, 0))
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
    }
  }

  // ---- M12-A: Column-major sparse iteration (columnSkip=true, macLanes=1) ----

  /** Count the number of clock.steps() in runToDone (plus the 1 start step). */
  private def countCycles(
      x:      Seq[Int],
      weight: Seq[Seq[Int]],
      bias:   Seq[Int],
      columnSkip: Boolean): Int = {
    var cycles = 0
    test(new ConfigurableSerialLinear4(columnSkip = columnSkip)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, x)
      pokeMatrix(dut.io.weight, weight)
      pokeVector(dut.io.bias, bias)
      dut.io.start.poke(true.B)
      dut.clock.step(); cycles += 1
      dut.io.start.poke(false.B)
      var seenDone = false
      var limit = 40
      while (!seenDone && limit > 0) {
        dut.clock.step(); cycles += 1; limit -= 1
        seenDone = dut.io.done.peek().litToBoolean
      }
      assert(seenDone, "columnSkip test did not finish within 40 cycles")
    }
    cycles
  }

  it should "produce identical output with columnSkip=true vs standard for sparse inputs" in {
    // Three representative cases: dense, 2-nonzero, all-zero
    val cases = Seq(
      (
        "dense",
        Seq(1, 2, 3, 4),
        Seq(Seq(1,0,0,0), Seq(0,1,0,0), Seq(0,0,1,0), Seq(0,0,0,1)),
        Seq(0,0,0,0),
        Seq(1,2,3,4)
      ),
      (
        "k=2 sparse (col0 and col2 non-zero)",
        Seq(3, 0, -2, 0),
        Seq(Seq(1,-1,2,0), Seq(0,1,-1,1), Seq(2,0,1,-1), Seq(-1,1,0,2)),
        Seq(1,-1,0,2),
        Seq(3*1+(-2)*2+1, 3*0+(-2)*(-1)+(-1), 3*2+(-2)*1+0, 3*(-1)+(-2)*0+2)
        // y[0] = 3*1+0*(-1)+(-2)*2+0*0 + 1 = 3-4+1 = 0
        // y[1] = 3*0+0*1+(-2)*(-1)+0*1 + (-1) = 2-1 = 1
        // y[2] = 3*2+0*0+(-2)*1+0*(-1) + 0 = 6-2 = 4
        // y[3] = 3*(-1)+0*1+(-2)*0+0*2 + 2 = -3+2 = -1
      ),
      (
        "k=0 all-zero input",
        Seq(0, 0, 0, 0),
        Seq.fill(4)(Seq(7, -2, 5, 1)),
        Seq(3, -1, 2, 0),
        Seq(3, -1, 2, 0)   // y = 0 + bias = bias
      )
    )

    for ((label, x, weight, bias, expected) <- cases) {
      // Verify expected against golden standard first
      var stdOut = Seq.empty[Long]
      var sparseOut = Seq.empty[Long]

      test(new ConfigurableSerialLinear4(columnSkip = false)) { dut =>
        dut.io.clear.poke(false.B)
        pokeVector(dut.io.x, x)
        pokeMatrix(dut.io.weight, weight)
        pokeVector(dut.io.bias, bias)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        runToDone(dut)
        stdOut = dut.io.y.map(_.peek().litValue.toLong)
      }

      test(new ConfigurableSerialLinear4(columnSkip = true)) { dut =>
        dut.io.clear.poke(false.B)
        pokeVector(dut.io.x, x)
        pokeMatrix(dut.io.weight, weight)
        pokeVector(dut.io.bias, bias)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        runToDone(dut, maxCycles = 40)
        sparseOut = dut.io.y.map(_.peek().litValue.toLong)
      }

      assert(stdOut == sparseOut,
        s"[$label] columnSkip output mismatch: std=$stdOut sparse=$sparseOut")
      println(s"[M12-A] $label: std=$stdOut  sparse=$sparseOut ✓")
    }
  }

  it should "complete faster than standard for k=0 (all-zero input, 1 cycle vs lanes^2)" in {
    val x      = Seq(0, 0, 0, 0)
    val weight = Seq.fill(4)(Seq(1, 2, 3, 4))
    val bias   = Seq(5, -3, 0, 2)

    val cyclesStd    = countCycles(x, weight, bias, columnSkip = false)
    val cyclesSparse = countCycles(x, weight, bias, columnSkip = true)

    println(s"[M12-A] k=0 cycles: standard=$cyclesStd  sparse=$cyclesSparse  saved=${cyclesStd - cyclesSparse}")
    assert(cyclesSparse < cyclesStd,
      s"columnSkip k=0: sparse ($cyclesSparse) should be faster than standard ($cyclesStd)")
    // k=0: sparse should finish in 2 busyReg cycles (commit on first cycle + 1 latch) = 3 total
    // Standard: lanes^2 = 16 cycles + overhead
    assert(cyclesSparse <= cyclesStd / 4,
      s"k=0: sparse ($cyclesSparse) should be at least 4× faster than standard ($cyclesStd)")
  }

  it should "complete in k*lanes+overhead cycles for k=2 sparse input (fewer than standard)" in {
    // x has 2 non-zero columns: col0=3 and col2=-2, others zero
    val x      = Seq(3, 0, -2, 0)
    val weight = Seq(Seq(1,-1,2,0), Seq(0,1,-1,1), Seq(2,0,1,-1), Seq(-1,1,0,2))
    val bias   = Seq(1,-1,0,2)

    val cyclesStd    = countCycles(x, weight, bias, columnSkip = false)
    val cyclesSparse = countCycles(x, weight, bias, columnSkip = true)

    println(s"[M12-A] k=2 cycles: standard=$cyclesStd  sparse=$cyclesSparse  saved=${cyclesStd - cyclesSparse}")
    assert(cyclesSparse < cyclesStd,
      s"columnSkip k=2: sparse ($cyclesSparse) should be fewer cycles than standard ($cyclesStd)")
    // With lanes=4, k=2: sparse = 2*4 + commit(1) + start(1) + done_latch(1) = 11
    // Standard = lanes^2 + start + done_latch = 16 + 2 = 18
    assert(cyclesStd - cyclesSparse >= 4,
      s"k=2 should save at least lanes=4 cycles: std=$cyclesStd sparse=$cyclesSparse")
  }

  it should "produce correct output for k=0 (bias-only result)" in {
    val bias = Seq(5, -3, 0, 2)
    test(new ConfigurableSerialLinear4(columnSkip = true)) { dut =>
      dut.io.clear.poke(false.B)
      pokeVector(dut.io.x, Seq(0, 0, 0, 0))
      pokeMatrix(dut.io.weight, Seq.fill(4)(Seq(3, 7, -2, 1)))
      pokeVector(dut.io.bias, bias)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut, maxCycles = 10)
      dut.io.done.expect(true.B)
      // y = 0 * w + bias = bias
      expectVector(dut.io.y, bias)
    }
  }
}
