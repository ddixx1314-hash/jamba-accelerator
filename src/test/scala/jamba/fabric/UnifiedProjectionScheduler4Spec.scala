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

  private def runToDoneCycles(dut: UnifiedProjectionScheduler4, maxCycles: Int = 250): Int = {
    var seenDone = false
    var cycles = 0
    while (!seenDone && cycles < maxCycles) {
      dut.clock.step()
      cycles += 1
      seenDone = dut.io.done.peek().litToBoolean
    }
    assert(seenDone, s"UnifiedProjectionScheduler4 did not finish within $maxCycles cycles")
    cycles
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

  // ── M10-D: Dynamic vector bypass tests ──────────────────────────────────────

  it should "bypass all-zero-input slots when vectorBypass=true" in {
    // When all elements of x[slot] are zero, W·0+b = b, so output must equal bias.
    // The MAC should be skipped entirely; vectorBypassCount should equal the number
    // of enabled slots (all are bypassed here).
    val bias0 = Seq(3, -2, 7, 1)
    val bias1 = Seq(-1, 5, 0, 4)
    val m  = UnifiedProjectionSlots.MambaInput
    val mg = UnifiedProjectionSlots.MlpGate

    test(new UnifiedProjectionScheduler4(vectorBypass = true)) { dut =>
      zeroAll(dut)
      // Enable two slots, zero inputs, non-zero biases
      dut.io.slotEnable(m).poke(true.B)
      pokeVector(dut.io.bias(m), bias0)
      pokeIdentity(dut.io.weight(m))          // non-zero weight (should not matter)

      dut.io.slotEnable(mg).poke(true.B)
      pokeVector(dut.io.bias(mg), bias1)
      pokeIdentity(dut.io.weight(mg))

      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      runToDone(dut, maxCycles = 8)           // bypass: 1 findSlot cycle per bypassed slot

      dut.io.done.expect(true.B)
      // Output must equal bias (bypass wrote bias directly)
      expectVector(dut.io.y(m),  bias0)
      expectVector(dut.io.y(mg), bias1)
      dut.io.vectorBypassCount.expect(2.U)   // both slots bypassed
    }
  }

  it should "only bypass zero-input slots, MAC the rest" in {
    // Slot 0 (MambaInput): x = [0,0,0,0] → bypassed, y = bias
    // Slot 1 (MambaB):     x = [1,0,0,0], W = I → y = x + bias = x (bias=0)
    val m  = UnifiedProjectionSlots.MambaInput
    val mb = UnifiedProjectionSlots.MambaB
    val bias0 = Seq(9, -3, 2, -5)

    test(new UnifiedProjectionScheduler4(vectorBypass = true)) { dut =>
      zeroAll(dut)
      // Slot 0: zero input, non-zero bias → bypass path
      dut.io.slotEnable(m).poke(true.B)
      pokeVector(dut.io.bias(m), bias0)
      pokeIdentity(dut.io.weight(m))

      // Slot 1: non-zero input, identity weight, zero bias → MAC path
      dut.io.slotEnable(mb).poke(true.B)
      pokeVector(dut.io.x(mb), Seq(3, -1, 0, 7))
      pokeIdentity(dut.io.weight(mb))

      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      runToDone(dut, maxCycles = 64)

      dut.io.done.expect(true.B)
      // Slot 0: bypassed → output = bias
      expectVector(dut.io.y(m),  bias0)
      // Slot 1: MAC'd  → W·x = identity·[3,-1,0,7] = [3,-1,0,7], bias=0
      expectVector(dut.io.y(mb), Seq(3, -1, 0, 7))
      dut.io.vectorBypassCount.expect(1.U)   // only slot 0 bypassed
    }
  }

  it should "produce zero bypassCount when vectorBypass=false even with zero inputs" in {
    // Without the bypass feature, all slots go through the MAC even with zero x.
    val m = UnifiedProjectionSlots.MambaInput
    val bias0 = Seq(5, -1, 0, 3)

    test(new UnifiedProjectionScheduler4(vectorBypass = false)) { dut =>
      zeroAll(dut)
      dut.io.slotEnable(m).poke(true.B)
      pokeVector(dut.io.bias(m), bias0)
      pokeIdentity(dut.io.weight(m))

      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      runToDone(dut, maxCycles = 64)
      dut.io.done.expect(true.B)
      // MAC: W·0 + bias = bias
      expectVector(dut.io.y(m), bias0)
      // No bypass feature → always 0
      dut.io.vectorBypassCount.expect(0.U)
    }
  }

  it should "give identical outputs for non-zero inputs with and without vectorBypass" in {
    // Regardless of the bypass flag, non-zero inputs must produce the same result.
    val q = UnifiedProjectionSlots.AttentionQ
    val x      = Seq(2, -1, 3, 4)
    val weight = Seq(Seq(1,0,0,0), Seq(0,-1,0,0), Seq(1,1,1,1), Seq(2,0,-1,1))
    val bias   = Seq(0, 5, -2, 7)
    val expected = Seq(2, 6, 6, 12)

    for (bypass <- Seq(false, true)) {
      test(new UnifiedProjectionScheduler4(vectorBypass = bypass)) { dut =>
        zeroAll(dut)
        dut.io.slotEnable(q).poke(true.B)
        pokeVector(dut.io.x(q), x)
        pokeMatrix(dut.io.weight(q), weight)
        pokeVector(dut.io.bias(q), bias)
        dut.io.clear.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        runToDone(dut, maxCycles = 64)
        dut.io.done.expect(true.B)
        expectVector(dut.io.y(q), expected)
        dut.io.vectorBypassCount.expect(0.U)  // not bypassed (x != 0)
      }
    }
  }

  it should "be faster with bypass when all inputs are zero" in {
    // Bypassing zero-input slots should complete in fewer cycles than running the MAC.
    val m  = UnifiedProjectionSlots.MambaInput
    val mb = UnifiedProjectionSlots.MambaB
    val mc = UnifiedProjectionSlots.MambaC

    def measure(bypass: Boolean): Int = {
      var cycles = 0
      test(new UnifiedProjectionScheduler4(vectorBypass = bypass)) { dut =>
        zeroAll(dut)
        for (s <- Seq(m, mb, mc)) {
          dut.io.slotEnable(s).poke(true.B)
          pokeIdentity(dut.io.weight(s))
        }
        dut.io.clear.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        cycles = runToDoneCycles(dut)
      }
      cycles
    }

    val cyclesWithBypass    = measure(bypass = true)
    val cyclesWithoutBypass = measure(bypass = false)
    assert(
      cyclesWithBypass < cyclesWithoutBypass,
      s"Bypass ($cyclesWithBypass) should be faster than no-bypass ($cyclesWithoutBypass) for all-zero inputs"
    )
  }

  it should "reset bypassCount on each new start" in {
    // Run two consecutive scheduled rounds; bypassCount should reset to 0 each time.
    val m  = UnifiedProjectionSlots.MambaInput
    val mb = UnifiedProjectionSlots.MambaB

    test(new UnifiedProjectionScheduler4(vectorBypass = true)) { dut =>
      // Round 1: both slots zero → bypass both
      zeroAll(dut)
      dut.io.slotEnable(m).poke(true.B)
      dut.io.slotEnable(mb).poke(true.B)
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut, maxCycles = 8)
      dut.io.vectorBypassCount.expect(2.U)

      // Round 2: slot m has non-zero x → only mb bypassed
      zeroAll(dut)
      dut.io.slotEnable(m).poke(true.B)
      dut.io.slotEnable(mb).poke(true.B)
      pokeVector(dut.io.x(m), Seq(1, 1, 1, 1))
      pokeIdentity(dut.io.weight(m))
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut, maxCycles = 64)
      dut.io.vectorBypassCount.expect(1.U)   // reset: only mb bypassed this round
    }
  }

  // ── M13-S: Sparse-aware scheduler (columnSkip propagated to ConfigurableSerialLinear4) ─────

  it should "produce correct output with columnSkip=true for k=2 sparse slot input" in {
    // Use slot MambaInput with x=[3,0,-2,0] (k=2 non-zero columns).
    // columnSkip=true enables the sparse FSM in ConfigurableSerialLinear4.
    // Output must match the dense result computed by columnSkip=false.
    val m = UnifiedProjectionSlots.MambaInput
    val x      = Seq(3, 0, -2, 0)
    val weight = Seq(Seq(1,-1,2,0), Seq(0,1,-1,1), Seq(2,0,1,-1), Seq(-1,1,0,2))
    val bias   = Seq(1, -1, 0, 2)
    // Expected: y[r] = sum_c(weight[r][c]*x[c]) + bias[r]
    // y[0]=3*1+0*(-1)+(-2)*2+0*0 + 1 = 3-4+1 = 0
    // y[1]=3*0+0*1+(-2)*(-1)+0*1 + (-1) = 2-1 = 1
    // y[2]=3*2+0*0+(-2)*1+0*(-1) + 0 = 6-2 = 4
    // y[3]=3*(-1)+0*1+(-2)*0+0*2 + 2 = -3+2 = -1
    val expected = Seq(0, 1, 4, -1)

    var denseOut  = Seq.empty[Long]
    var sparseOut = Seq.empty[Long]

    // Dense run (columnSkip=false, default)
    test(new UnifiedProjectionScheduler4()) { dut =>
      zeroAll(dut)
      dut.io.slotEnable(m).poke(true.B)
      pokeVector(dut.io.x(m), x)
      pokeMatrix(dut.io.weight(m), weight)
      pokeVector(dut.io.bias(m), bias)
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      denseOut = (0 until 4).map(i => dut.io.y(m)(i).peek().litValue.toLong)
    }

    // Sparse run (columnSkip=true)
    test(new UnifiedProjectionScheduler4(columnSkip = true)) { dut =>
      zeroAll(dut)
      dut.io.slotEnable(m).poke(true.B)
      pokeVector(dut.io.x(m), x)
      pokeMatrix(dut.io.weight(m), weight)
      pokeVector(dut.io.bias(m), bias)
      dut.io.clear.poke(false.B)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      sparseOut = (0 until 4).map(i => dut.io.y(m)(i).peek().litValue.toLong)
    }

    println(s"[M13-S] k=2 sparse: dense=$denseOut  sparse=$sparseOut")
    assert(denseOut == expected.map(_.toLong),
      s"Dense output mismatch: got $denseOut expected $expected")
    assert(sparseOut == expected.map(_.toLong),
      s"Sparse (columnSkip=true) output mismatch: got $sparseOut expected $expected")
    assert(denseOut == sparseOut,
      s"columnSkip output must match dense: dense=$denseOut sparse=$sparseOut")
  }

  it should "finish faster with columnSkip=true than columnSkip=false for k=2 sparse input" in {
    // With k=2 non-zero columns and lanes=4:
    //   standard cycles ≈ lanes²/macLanes + overhead = 16 + overhead
    //   sparse cycles   ≈ k*lanes + overhead         = 8  + overhead
    val m      = UnifiedProjectionSlots.MambaInput
    val x      = Seq(3, 0, -2, 0)
    val weight = Seq(Seq(1,-1,2,0), Seq(0,1,-1,1), Seq(2,0,1,-1), Seq(-1,1,0,2))
    val bias   = Seq(1, -1, 0, 2)

    def measureCycles(colSkip: Boolean): Int = {
      var cycles = 0
      test(new UnifiedProjectionScheduler4(columnSkip = colSkip)) { dut =>
        zeroAll(dut)
        dut.io.slotEnable(m).poke(true.B)
        pokeVector(dut.io.x(m), x)
        pokeMatrix(dut.io.weight(m), weight)
        pokeVector(dut.io.bias(m), bias)
        dut.io.clear.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step(); cycles += 1
        dut.io.start.poke(false.B)
        var limit = 100
        while (!dut.io.done.peek().litToBoolean && limit > 0) {
          dut.clock.step(); cycles += 1; limit -= 1
        }
        assert(dut.io.done.peek().litToBoolean, s"columnSkip=$colSkip did not finish within 100 cycles")
      }
      cycles
    }

    val cyclesDense  = measureCycles(colSkip = false)
    val cyclesSparse = measureCycles(colSkip = true)
    println(s"[M13-S] k=2 scheduler cycles: dense=$cyclesDense  sparse(columnSkip)=$cyclesSparse  saved=${cyclesDense - cyclesSparse}")
    assert(cyclesSparse < cyclesDense,
      s"columnSkip=true should finish faster for k=2 sparse input: sparse=$cyclesSparse dense=$cyclesDense")
  }

  it should "produce identical results with columnSkip=true vs false for dense (k=4) input" in {
    // Dense input (all columns non-zero) should produce the same output either way.
    val q      = UnifiedProjectionSlots.AttentionQ
    val x      = Seq(1, -2, 3, -4)
    val weight = Seq(Seq(1,0,0,0), Seq(0,1,0,0), Seq(0,0,1,0), Seq(0,0,0,1))
    val bias   = Seq(10, 20, 30, 40)
    // y = identity * x + bias = [11, 18, 33, 36]
    val expected = Seq(11, 18, 33, 36)

    for (colSkip <- Seq(false, true)) {
      test(new UnifiedProjectionScheduler4(columnSkip = colSkip)) { dut =>
        zeroAll(dut)
        dut.io.slotEnable(q).poke(true.B)
        pokeVector(dut.io.x(q), x)
        pokeMatrix(dut.io.weight(q), weight)
        pokeVector(dut.io.bias(q), bias)
        dut.io.clear.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        runToDone(dut)
        dut.io.done.expect(true.B)
        expectVector(dut.io.y(q), expected)
        println(s"[M13-S] dense k=4 columnSkip=$colSkip: correct ✓")
      }
    }
  }

  it should "produce identical results and lower latency as projection MAC lanes increase" in {
    val q = UnifiedProjectionSlots.AttentionQ
    val x = Seq(2, -1, 3, 4)
    val weight = Seq(
      Seq(1, 0, 0, 0),
      Seq(0, -1, 0, 0),
      Seq(1, 1, 1, 1),
      Seq(2, 0, -1, 1)
    )
    val bias = Seq(0, 5, -2, 7)
    val expected = Seq(2, 6, 6, 12)
    var cyclesByMac = Map.empty[Int, Int]

    for (macLanes <- Seq(1, 2, 4)) {
      test(new UnifiedProjectionScheduler4(projectionMacLanes = macLanes)) { dut =>
        zeroAll(dut)
        dut.io.slotEnable(q).poke(true.B)
        pokeVector(dut.io.x(q), x)
        pokeMatrix(dut.io.weight(q), weight)
        pokeVector(dut.io.bias(q), bias)
        dut.io.clear.poke(false.B)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)

        val cycles = runToDoneCycles(dut)
        cyclesByMac += macLanes -> cycles
        expectVector(dut.io.y(q), expected)
      }
    }

    assert(cyclesByMac(2) < cyclesByMac(1), s"Mac2 should finish faster than Mac1: $cyclesByMac")
    assert(cyclesByMac(4) < cyclesByMac(2), s"Mac4 should finish faster than Mac2: $cyclesByMac")
  }
}
