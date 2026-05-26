package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class UnifiedJamba2MiniLayerSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "UnifiedJamba2MiniLayer"

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

  private def pokeReverseIdentity(port: Vec[Vec[SInt]]): Unit =
    pokeMatrix(port, Seq(Seq(0,0,0,1), Seq(0,0,1,0), Seq(0,1,0,0), Seq(1,0,0,0)))

  private def pokeZeroMatrix(port: Vec[Vec[SInt]]): Unit =
    pokeMatrix(port, Seq.fill(4)(Seq.fill(4)(0)))

  private def pokeKernel(port: Vec[Vec[SInt]], values: Seq[Seq[Int]]): Unit = {
    for (tap <- values.indices) {
      for (lane <- values(tap).indices) port(tap)(lane).poke(values(tap)(lane).S)
    }
  }

  private def pokeExpertWeights(dut: UnifiedJamba2MiniLayer): Unit = {
    pokeMatrix(dut.io.routerWeight, Seq(Seq(1,0,0,0), Seq(0,1,0,0)))
    pokeVector(dut.io.routerBias, Seq(0, 0))
    for (expert <- 0 until 2) {
      pokeIdentity(dut.io.expertGateWeight(expert))
      pokeVector(dut.io.expertGateBias(expert), Seq(1, 1, 1, 1))
      pokeIdentity(dut.io.expertUpWeight(expert))
      pokeVector(dut.io.expertUpBias(expert), Seq(0, 0, 0, 0))
      pokeIdentity(dut.io.expertDownWeight(expert))
      pokeVector(dut.io.expertDownBias(expert), Seq.fill(4)(expert))
    }
  }

  private def pokeDefaultWeights(dut: UnifiedJamba2MiniLayer): Unit = {
    pokeVector(dut.io.norm1Weight, Seq(1, 1, 1, 1))
    pokeVector(dut.io.norm2Weight, Seq(1, 1, 1, 1))

    pokeIdentity(dut.io.mambaInputWeight)
    pokeVector(dut.io.mambaInputBias, Seq(0, 0, 0, 0))
    pokeZeroMatrix(dut.io.mambaBWeight)
    pokeVector(dut.io.mambaBBias, Seq(2, 2, 2, 2))
    pokeZeroMatrix(dut.io.mambaCWeight)
    pokeVector(dut.io.mambaCBias, Seq(1, 1, 1, 1))
    pokeVector(dut.io.mambaA, Seq(1, 1, 1, 1))
    pokeKernel(dut.io.mambaKernel, Seq.fill(4)(Seq(1, 1, 1, 1)))

    pokeIdentity(dut.io.qWeight); pokeVector(dut.io.qBias, Seq(0,0,0,0))
    pokeIdentity(dut.io.kWeight); pokeVector(dut.io.kBias, Seq(0,0,0,0))
    pokeIdentity(dut.io.vWeight); pokeVector(dut.io.vBias, Seq(0,0,0,0))
    pokeIdentity(dut.io.attentionOutWeight); pokeVector(dut.io.attentionOutBias, Seq(0,0,0,0))

    pokeIdentity(dut.io.mlpGateWeight); pokeVector(dut.io.mlpGateBias, Seq(1,1,1,1))
    pokeReverseIdentity(dut.io.mlpUpWeight); pokeVector(dut.io.mlpUpBias, Seq(0,0,0,0))
    pokeIdentity(dut.io.mlpDownWeight); pokeVector(dut.io.mlpDownBias, Seq(0,0,0,0))
    pokeExpertWeights(dut)
  }

  private def runToDone(dut: UnifiedJamba2MiniLayer, maxCycles: Int = 260): Unit = {
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"UnifiedJamba2MiniLayer did not finish within $maxCycles cycles")
  }

  it should "run Mamba mode through one unified projection scheduler" in {
    test(new UnifiedJamba2MiniLayer()) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      pokeDefaultWeights(dut)

      dut.io.start.poke(true.B)
      dut.io.ready.expect(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      dut.io.ready.expect(true.B)
      expectVector(dut.io.mixerY,       Seq(2, 0, 0, 0))
      expectVector(dut.io.firstResidual, Seq(3, 0, 0, 0))
      expectVector(dut.io.mlpY,         Seq(0, 0, 0, 1))
      expectVector(dut.io.y,            Seq(3, 0, 0, 1))
      expectVector(dut.io.stateOut,     Seq(2, 0, 0, 0))
      dut.io.mixerType.expect(false.B)
      dut.io.dispatchValid.expect(false.B)
      dut.io.combineValid.expect(false.B)
    }
  }

  it should "run Attention mode and update KV cache" in {
    test(new UnifiedJamba2MiniLayer()) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))
      pokeDefaultWeights(dut)

      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)

      dut.io.done.expect(true.B)
      expectVector(dut.io.mixerY,       Seq(0, 0, 0, 0))
      expectVector(dut.io.firstResidual, Seq(4, 0, 0, 0))
      expectVector(dut.io.y,            Seq(4, 0, 0, 1))
      dut.io.kvWriteIndex.expect(1.U)
      dut.io.kvValidCount.expect(1.U)
      dut.io.mixerType.expect(true.B)
    }
  }

  it should "activate the unified MoE-lite MLP path" in {
    test(new UnifiedJamba2MiniLayer()) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(true.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      pokeDefaultWeights(dut)
      pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 1, 0, 0)))
      pokeVector(dut.io.routerBias, Seq(0, 2))

      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut, maxCycles = 320)

      dut.io.done.expect(true.B)
      expectVector(dut.io.mixerY, Seq(2, 0, 0, 0))
      expectVector(dut.io.firstResidual, Seq(3, 0, 0, 0))
      expectVector(dut.io.mlpY, Seq(3, 1, 1, 1))
      expectVector(dut.io.y, Seq(6, 1, 1, 1))
      dut.io.dispatchValid.expect(true.B)
      dut.io.combineValid.expect(true.B)
      dut.io.selectedExpert.expect(1.U)
    }
  }

  it should "clear resets persistent state and cache" in {
    test(new UnifiedJamba2MiniLayer()) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      pokeDefaultWeights(dut)

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
      expectVector(dut.io.stateOut, Seq(0, 0, 0, 0))
      dut.io.kvWriteIndex.expect(0.U)
      dut.io.kvValidCount.expect(0.U)
    }
  }

  it should "produce identical Mamba and Attention outputs across projection MAC-lane counts" in {
    for ((useAttention, expectedY) <- Seq(
      (false, Seq(3, 0, 0, 1)),
      (true,  Seq(4, 0, 0, 1))
    )) {
      var outputs = Map.empty[Int, Seq[Long]]
      for (macLanes <- Seq(1, 2, 4)) {
        test(new UnifiedJamba2MiniLayer(projectionMacLanes = macLanes)) { dut =>
          dut.io.clear.poke(false.B)
          dut.io.useAttention.poke(useAttention.B)
          dut.io.enableMoE.poke(false.B)
          pokeVector(dut.io.x, if (useAttention) Seq(4, 0, 0, 0) else Seq(1, 0, 0, 0))
          pokeDefaultWeights(dut)

          dut.io.start.poke(true.B)
          dut.clock.step()
          dut.io.start.poke(false.B)
          runToDone(dut)

          val y = (0 until 4).map(i => dut.io.y(i).peek().litValue.toLong)
          outputs += macLanes -> y
          expectVector(dut.io.y, expectedY)
        }
      }
      assert(outputs(1) == outputs(2) && outputs(2) == outputs(4),
        s"useAttention=$useAttention output changed across macLanes: $outputs")
    }
  }

  it should "produce identical MoE-lite output across projection MAC-lane counts" in {
    var outputs = Map.empty[Int, Seq[Long]]
    for (macLanes <- Seq(1, 2, 4)) {
      test(new UnifiedJamba2MiniLayer(projectionMacLanes = macLanes)) { dut =>
        dut.io.clear.poke(false.B)
        dut.io.useAttention.poke(false.B)
        dut.io.enableMoE.poke(true.B)
        pokeVector(dut.io.x, Seq(1, 0, 0, 0))
        pokeDefaultWeights(dut)
        pokeMatrix(dut.io.routerWeight, Seq(Seq(1, 0, 0, 0), Seq(0, 1, 0, 0)))
        pokeVector(dut.io.routerBias, Seq(0, 2))

        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        runToDone(dut, maxCycles = 320)

        val y = (0 until 4).map(i => dut.io.y(i).peek().litValue.toLong)
        outputs += macLanes -> y
        expectVector(dut.io.y, Seq(6, 1, 1, 1))
      }
    }
    assert(outputs(1) == outputs(2) && outputs(2) == outputs(4),
      s"MoE-lite output changed across macLanes: $outputs")
  }

  // ── M11-F: fusedOperators — fuse computeFirstResidual and computeHidden ───────

  /** Count cycles from start pulse until io.done pulses. */
  private def countCyclesToDone(dut: UnifiedJamba2MiniLayer, maxCycles: Int = 400): Int = {
    var cycles = 0
    var seenDone = false
    for (_ <- 0 until maxCycles) {
      if (!seenDone) {
        dut.clock.step()
        cycles += 1
        seenDone = dut.io.done.peek().litToBoolean
      }
    }
    assert(seenDone, s"UnifiedJamba2MiniLayer did not finish within $maxCycles cycles")
    cycles
  }

  it should "produce identical Mamba output with and without fusedOperators" in {
    val unfusedY = Array.ofDim[Long](4)
    val fusedY   = Array.ofDim[Long](4)

    test(new UnifiedJamba2MiniLayer(fusedOperators = false)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      pokeDefaultWeights(dut)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      for (i <- 0 until 4) unfusedY(i) = dut.io.y(i).peek().litValue.toLong
    }

    test(new UnifiedJamba2MiniLayer(fusedOperators = true)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(1, 0, 0, 0))
      pokeDefaultWeights(dut)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      for (i <- 0 until 4) fusedY(i) = dut.io.y(i).peek().litValue.toLong
    }

    assert(unfusedY.toSeq == fusedY.toSeq,
      s"Mamba output differs: unfused=${unfusedY.toSeq} fused=${fusedY.toSeq}")
  }

  it should "produce identical Attention output with and without fusedOperators" in {
    val unfusedY = Array.ofDim[Long](4)
    val fusedY   = Array.ofDim[Long](4)

    test(new UnifiedJamba2MiniLayer(fusedOperators = false)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))
      pokeDefaultWeights(dut)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      for (i <- 0 until 4) unfusedY(i) = dut.io.y(i).peek().litValue.toLong
    }

    test(new UnifiedJamba2MiniLayer(fusedOperators = true)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))
      pokeDefaultWeights(dut)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      runToDone(dut)
      for (i <- 0 until 4) fusedY(i) = dut.io.y(i).peek().litValue.toLong
    }

    assert(unfusedY.toSeq == fusedY.toSeq,
      s"Attention output differs: unfused=${unfusedY.toSeq} fused=${fusedY.toSeq}")
  }

  it should "run 2 fewer cycles per non-MoE token with fusedOperators" in {
    // Mamba non-MoE path: fusedOperators eliminates computeFirstResidual (1 cy)
    // and computeHidden (1 cy) = 2 cycles saved.
    for (useAttention <- Seq(false, true)) {
      val inputVec = if (useAttention) Seq(4, 0, 0, 0) else Seq(1, 0, 0, 0)

      var cyclesUnfused = 0
      var cyclesFused   = 0

      test(new UnifiedJamba2MiniLayer(fusedOperators = false)) { dut =>
        dut.io.clear.poke(false.B)
        dut.io.useAttention.poke(useAttention.B)
        dut.io.enableMoE.poke(false.B)
        pokeVector(dut.io.x, inputVec)
        pokeDefaultWeights(dut)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        cyclesUnfused = countCyclesToDone(dut)
      }

      test(new UnifiedJamba2MiniLayer(fusedOperators = true)) { dut =>
        dut.io.clear.poke(false.B)
        dut.io.useAttention.poke(useAttention.B)
        dut.io.enableMoE.poke(false.B)
        pokeVector(dut.io.x, inputVec)
        pokeDefaultWeights(dut)
        dut.io.start.poke(true.B)
        dut.clock.step()
        dut.io.start.poke(false.B)
        cyclesFused = countCyclesToDone(dut)
      }

      println(s"[M11-F] useAttention=$useAttention: unfused=$cyclesUnfused fused=$cyclesFused saved=${cyclesUnfused - cyclesFused}")
      assert(cyclesUnfused - cyclesFused == 2,
        s"Expected 2 fewer cycles with fusion (useAttention=$useAttention): " +
        s"unfused=$cyclesUnfused fused=$cyclesFused diff=${cyclesUnfused - cyclesFused}")
    }
  }

  // ---- M12-P: Power-of-Two A matrix ----

  it should "produce identical Mamba output for first token: useShiftA a=0 ↔ standard a=1" in {
    // state=0 initially → (state >> 0) = 0 = state*1.  Outputs must be bit-exact.
    val inputVec = Seq(2, 3, 1, 4)
    var outStd   = Seq.empty[Long]
    var outShift = Seq.empty[Long]

    test(new UnifiedJamba2MiniLayer(useShiftA = false)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, inputVec)
      pokeDefaultWeights(dut)
      // mambaA = [1,1,1,1] (standard: multiply state by 1)
      for (i <- 0 until 4) dut.io.mambaA(i).poke(1.S)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      while (!dut.io.done.peek().litToBoolean) dut.clock.step()
      outStd = dut.io.y.map(_.peek().litValue.toLong)
    }

    test(new UnifiedJamba2MiniLayer(useShiftA = true)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, inputVec)
      pokeDefaultWeights(dut)
      // mambaA = [0,0,0,0] in shift mode = shift by 0 = multiply by 1
      for (i <- 0 until 4) dut.io.mambaA(i).poke(0.S)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      while (!dut.io.done.peek().litToBoolean) dut.clock.step()
      outShift = dut.io.y.map(_.peek().litValue.toLong)
    }

    assert(outStd == outShift,
      s"First-token Mamba output must match: standard=$outStd vs useShiftA=$outShift")
  }

  it should "run lanes fewer cycles per Mamba token with useShiftA=true" in {
    // useShiftA eliminates 1 MAC op per lane in the scan → saves lanes=4 FSM cycles
    val inputVec = Seq(1, 2, 3, 4)
    var cyclesStd   = 0
    var cyclesShift = 0

    test(new UnifiedJamba2MiniLayer(useShiftA = false)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, inputVec)
      pokeDefaultWeights(dut)
      for (i <- 0 until 4) dut.io.mambaA(i).poke(1.S)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      cyclesStd = countCyclesToDone(dut)
    }

    test(new UnifiedJamba2MiniLayer(useShiftA = true)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(false.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, inputVec)
      pokeDefaultWeights(dut)
      for (i <- 0 until 4) dut.io.mambaA(i).poke(0.S)  // shift 0 = ×1
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      cyclesShift = countCyclesToDone(dut)
    }

    println(s"[M12-P] Mamba token cycles: standard=$cyclesStd useShiftA=$cyclesShift saved=${cyclesStd - cyclesShift}")
    assert(cyclesStd - cyclesShift == 4,
      s"useShiftA should save exactly lanes=4 cycles per Mamba token: " +
      s"std=$cyclesStd shift=$cyclesShift diff=${cyclesStd - cyclesShift}")
  }

  // ---- M12-K: Sliding Window Attention (Samba-style) ----

  it should "produce identical output for token 1 with any attentionWindowSize (cache cold)" in {
    // Token 1 always sees validCount=1 ≤ windowSize, so window never masks anything.
    val inputVec = Seq(4, 0, 0, 0)
    var outFull   = Seq.empty[Long]
    var outWindow = Seq.empty[Long]

    test(new UnifiedJamba2MiniLayer(attentionWindowSize = 0)) { dut =>   // full context
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, inputVec)
      pokeDefaultWeights(dut)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      while (!dut.io.done.peek().litToBoolean) dut.clock.step()
      outFull = dut.io.y.map(_.peek().litValue.toLong)
    }

    test(new UnifiedJamba2MiniLayer(attentionWindowSize = 1)) { dut =>   // window = 1
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, inputVec)
      pokeDefaultWeights(dut)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      while (!dut.io.done.peek().litToBoolean) dut.clock.step()
      outWindow = dut.io.y.map(_.peek().litValue.toLong)
    }

    assert(outFull == outWindow,
      s"Token-1 output must be identical regardless of window size: full=$outFull window=$outWindow")
  }

  it should "differ on token 2 between full-context and window=1 when token 1 contributed non-zero KV" in {
    // With normShift=2 the default K·Q scores are too small (score>>2=0).
    // Use normShift=0 so attention weights are not zeroed out.
    // Strategy: preload KV cache with K_1=[4,0,0,0], V_1=[4,0,0,0] before token 2.
    // Token 2 with input [4,0,0,0]: Q≈[1,0,0,0] (after norm+proj), K_2=[1,0,0,0].
    //   full context: scores for K_1 and K_2 are both 4; weights = 4; rawY uses both V's
    //   window=1:     only K_2/V_2 is attended to → different output
    //
    // Implementation: directly preload the KV cache using loadKvState, then run 1 token.
    // We preload K_1=[4,0,0,0], V_1=[4,0,0,0] (large enough to survive normShift=0).
    def runWithPreloadedKV(dut: UnifiedJamba2MiniLayer): Seq[Long] = {
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(false.B)
      pokeDefaultWeights(dut)

      // Preload token-1's K/V into slot 0 of the cache
      dut.io.loadKvState.poke(true.B)
      // K_1 = [4,0,0,0], V_1 = [4,0,0,0] in slot 0
      dut.io.keyCacheIn(0)(0).poke(4.S)
      for (l <- 1 until 4) dut.io.keyCacheIn(0)(l).poke(0.S)
      dut.io.valueCacheIn(0)(0).poke(4.S)
      for (l <- 1 until 4) dut.io.valueCacheIn(0)(l).poke(0.S)
      // Remaining slots zero
      for (s <- 1 until 4; l <- 0 until 4) {
        dut.io.keyCacheIn(s)(l).poke(0.S)
        dut.io.valueCacheIn(s)(l).poke(0.S)
      }
      dut.io.kvWriteIndexIn.poke(1.U)  // next write goes to slot 1
      dut.io.kvValidCountIn.poke(1.U)  // 1 entry valid
      dut.clock.step()
      dut.io.loadKvState.poke(false.B)

      // Now run token 2 (input x=[4,0,0,0])
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      while (!dut.io.done.peek().litToBoolean) dut.clock.step()
      dut.io.y.map(_.peek().litValue.toLong)
    }

    var outFull   = Seq.empty[Long]
    var outWindow = Seq.empty[Long]

    // normShift=0: attention weights = scores (not right-shifted to zero)
    test(new UnifiedJamba2MiniLayer(contextLength = 4, attentionWindowSize = 0, normShift = 0)) { dut =>
      outFull = runWithPreloadedKV(dut)
    }
    test(new UnifiedJamba2MiniLayer(contextLength = 4, attentionWindowSize = 1, normShift = 0)) { dut =>
      outWindow = runWithPreloadedKV(dut)
    }

    println(s"[M12-K] Token-2 with preloaded KV: fullCtx=$outFull  window1=$outWindow")
    assert(outFull != outWindow,
      s"Sliding window (size=1) should exclude preloaded KV_1 from token-2 attention: " +
      s"full=$outFull vs window=$outWindow must differ")
  }

  it should "apply attentionWindowSize masking in Attention+MoE mode" in {
    // When useAttention=true AND enableMoE=true, the attention mixer output (mixerY)
    // feeds into the MoE MLP.  attentionWindowSize must still mask KV entries correctly
    // even when the MLP path is MoE instead of dense.
    //
    // Same preload strategy as "differ on token 2":
    //   K_1=[4,0,0,0], V_1=[4,0,0,0] at cache slot 0, validCount=1.
    // Token-2 input x=[4,0,0,0], normShift=0 so attention scores are not zeroed.
    // window=1 excludes K_1/V_1 from the score → mixerY differs → y differs.
    def runAttnMoEWithPreloadedKV(dut: UnifiedJamba2MiniLayer): (Seq[Long], Seq[Long]) = {
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(true.B)          // MoE path enabled
      pokeDefaultWeights(dut)

      // Preload slot 0: K_1=[4,0,0,0], V_1=[4,0,0,0]
      dut.io.loadKvState.poke(true.B)
      dut.io.keyCacheIn(0)(0).poke(4.S)
      for (l <- 1 until 4) dut.io.keyCacheIn(0)(l).poke(0.S)
      dut.io.valueCacheIn(0)(0).poke(4.S)
      for (l <- 1 until 4) dut.io.valueCacheIn(0)(l).poke(0.S)
      for (s <- 1 until 4; l <- 0 until 4) {
        dut.io.keyCacheIn(s)(l).poke(0.S)
        dut.io.valueCacheIn(s)(l).poke(0.S)
      }
      dut.io.kvWriteIndexIn.poke(1.U)
      dut.io.kvValidCountIn.poke(1.U)
      dut.clock.step()
      dut.io.loadKvState.poke(false.B)

      // Run token 2
      pokeVector(dut.io.x, Seq(4, 0, 0, 0))
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      while (!dut.io.done.peek().litToBoolean) dut.clock.step()
      val mixerY = dut.io.mixerY.map(_.peek().litValue.toLong)
      val y      = dut.io.y.map(_.peek().litValue.toLong)
      (mixerY, y)
    }

    var (mixerYFull, yFull)     = (Seq.empty[Long], Seq.empty[Long])
    var (mixerYWindow, yWindow) = (Seq.empty[Long], Seq.empty[Long])

    test(new UnifiedJamba2MiniLayer(contextLength = 4, attentionWindowSize = 0, normShift = 0)) { dut =>
      val (m, y) = runAttnMoEWithPreloadedKV(dut)
      mixerYFull = m; yFull = y
    }
    test(new UnifiedJamba2MiniLayer(contextLength = 4, attentionWindowSize = 1, normShift = 0)) { dut =>
      val (m, y) = runAttnMoEWithPreloadedKV(dut)
      mixerYWindow = m; yWindow = y
    }

    println(s"[M12-K+MoE] Attn+MoE token-2: fullCtx_mixerY=$mixerYFull window1_mixerY=$mixerYWindow")
    println(s"[M12-K+MoE] Attn+MoE token-2: fullCtx_y=$yFull window1_y=$yWindow")
    assert(mixerYFull != mixerYWindow,
      s"attentionWindowSize must affect attention output (mixerY) even in MoE mode: " +
      s"full=$mixerYFull window=$mixerYWindow must differ")
    assert(yFull != yWindow,
      s"attentionWindowSize must affect final output even in MoE mode: " +
      s"full=$yFull window=$yWindow must differ")
  }

  it should "match full-context output when attentionWindowSize=contextLength" in {
    // window = contextLength is the edge case that equals full context — outputs must be identical.
    val inputVec = Seq(4, 0, 0, 0)
    var outFull   = Seq.empty[Long]
    var outEqual  = Seq.empty[Long]

    test(new UnifiedJamba2MiniLayer(contextLength = 4, attentionWindowSize = 0)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, inputVec)
      pokeDefaultWeights(dut)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      while (!dut.io.done.peek().litToBoolean) dut.clock.step()
      outFull = dut.io.y.map(_.peek().litValue.toLong)
    }

    // attentionWindowSize=contextLength should behave identically to attentionWindowSize=0
    test(new UnifiedJamba2MiniLayer(contextLength = 4, attentionWindowSize = 4)) { dut =>
      dut.io.clear.poke(false.B)
      dut.io.useAttention.poke(true.B)
      dut.io.enableMoE.poke(false.B)
      pokeVector(dut.io.x, inputVec)
      pokeDefaultWeights(dut)
      dut.io.start.poke(true.B); dut.clock.step(); dut.io.start.poke(false.B)
      while (!dut.io.done.peek().litToBoolean) dut.clock.step()
      outEqual = dut.io.y.map(_.peek().litValue.toLong)
    }

    assert(outFull == outEqual,
      s"attentionWindowSize=contextLength must equal full-context: full=$outFull eq=$outEqual")
  }
}
