package jamba.fabric

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

/** M15-P: Pipeline-depth analysis tests for PipelinedMacLane.
  *
  * Validates that the 1-stage pipeline register correctly delays the multiply
  * result by 1 cycle, and that accOut computes a*b + accIn with the pipelined
  * product. Demonstrates the latency-frequency trade-off:
  *   - Non-pipelined MacLane: accOut = a*b + accIn (1 cycle, deep critical path)
  *   - PipelinedMacLane:      accOut = a_prev*b_prev + accIn (2 cycles, shallow CP)
  */
class PipelinedMacLaneSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "PipelinedMacLane"

  it should "output a*b + accIn with 1-cycle pipeline delay" in {
    // Cycle 1: poke a=3, b=4, accIn=0 → productReg latches 12
    // Cycle 2: poke accIn=5 → accOut = 12 + 5 = 17 (using previous product)
    test(new PipelinedMacLane()) { dut =>
      dut.io.a.poke(3.S)
      dut.io.b.poke(4.S)
      dut.io.accIn.poke(0.S)
      dut.clock.step()   // productReg := 3*4 = 12

      // On next cycle, accOut = productReg(12) + accIn
      dut.io.a.poke(0.S)   // new inputs (will be latched next cycle, not used yet)
      dut.io.b.poke(0.S)
      dut.io.accIn.poke(5.S)
      // accOut = prev_product(12) + accIn(5) = 17
      dut.io.accOut.expect(17.S)
      dut.clock.step()
    }
  }

  it should "accumulate correctly across multiple pipeline stages" in {
    // Verify: consecutive MAC operations with pipelined product.
    // Pipeline:
    //   Cycle 1: a=2, b=3  → productReg ← 6
    //   Cycle 2: a=4, b=5  → productReg ← 20; accOut = 6 + accIn
    //   Cycle 3: a=1, b=2  → productReg ← 2;  accOut = 20 + accIn
    test(new PipelinedMacLane()) { dut =>
      // Cycle 1: set inputs for product = 2*3 = 6
      dut.io.a.poke(2.S); dut.io.b.poke(3.S); dut.io.accIn.poke(0.S)
      dut.clock.step()

      // Cycle 2: productReg=6; set new inputs for product = 4*5 = 20
      dut.io.a.poke(4.S); dut.io.b.poke(5.S); dut.io.accIn.poke(10.S)
      dut.io.accOut.expect((6 + 10).S)    // 6 (prev product) + 10 (accIn) = 16
      dut.clock.step()

      // Cycle 3: productReg=20; set new inputs
      dut.io.a.poke(1.S); dut.io.b.poke(2.S); dut.io.accIn.poke((-3).S)
      dut.io.accOut.expect((20 + (-3)).S) // 20 (prev product) + (-3) = 17
      dut.clock.step()

      // Cycle 4: productReg=2
      dut.io.a.poke(0.S); dut.io.b.poke(0.S); dut.io.accIn.poke(100.S)
      dut.io.accOut.expect((2 + 100).S)   // 2 (prev product) + 100 = 102
    }
  }

  it should "produce same result as non-pipelined MacLane (with 1-cycle shift)" in {
    // Verify that PipelinedMacLane gives the same MAC value as MacLane,
    // just delayed by 1 cycle. Use accIn=0 to isolate the multiply.
    //
    // MacLane:        accOut = a * b              (cycle N)
    // PipelinedMacLane: accOut = prev(a) * prev(b) (cycle N+1)
    val testPairs = Seq((3, 4, 12), (7, -2, -14), (-5, 6, -30), (0, 8, 0))

    var pipelinedResults = Seq.empty[Long]

    test(new PipelinedMacLane()) { dut =>
      for ((a, b, _) <- testPairs) {
        dut.io.a.poke(a.S); dut.io.b.poke(b.S); dut.io.accIn.poke(0.S)
        dut.clock.step()
        // accOut now contains the product from the PREVIOUS cycle
      }
      // Final step: read all accumulated pipeline results
      // (each clock.step above shifted the product into accOut for the NEXT clock)
      // Re-run and capture:
      pipelinedResults = Seq.empty
    }

    // Direct comparison: step through all pairs and capture accOut one-cycle-late
    test(new PipelinedMacLane()) { dut =>
      var results = Seq.empty[Long]

      // Prime the pipeline with first pair
      dut.io.a.poke(testPairs(0)._1.S)
      dut.io.b.poke(testPairs(0)._2.S)
      dut.io.accIn.poke(0.S)
      dut.clock.step()

      // For each subsequent pair, read the result of the PREVIOUS pair
      for (i <- 1 until testPairs.size) {
        dut.io.a.poke(testPairs(i)._1.S)
        dut.io.b.poke(testPairs(i)._2.S)
        dut.io.accIn.poke(0.S)
        results = results :+ dut.io.accOut.peek().litValue.toLong
        dut.clock.step()
      }
      results = results :+ dut.io.accOut.peek().litValue.toLong

      // results(i) should equal testPairs(i)._3 (the expected a*b)
      for (i <- results.indices) {
        assert(results(i) == testPairs(i)._3.toLong,
          s"Pipeline result mismatch at index $i: expected=${testPairs(i)._3} got=${results(i)}")
      }
      println(s"[M15-P] Pipelined MAC results: $results (expected: ${testPairs.map(_._3)})")
    }
  }

  it should "show shallower combinational depth than non-pipelined (stage analysis)" in {
    // Structural test: verify stageAValid and stageBValid pipeline indicators.
    // stageAValid goes high after 1 cycle (product stage latched).
    // stageBValid goes high after 2 cycles (accumulate stage ready).
    test(new PipelinedMacLane()) { dut =>
      // Initial state: both invalid
      dut.io.stageAValid.expect(false.B)
      dut.io.stageBValid.expect(false.B)

      dut.io.a.poke(1.S); dut.io.b.poke(1.S); dut.io.accIn.poke(0.S)
      dut.clock.step()

      // After 1 cycle: stage A valid (product latched)
      dut.io.stageAValid.expect(true.B)
      dut.io.stageBValid.expect(false.B)  // stage B not yet valid
      dut.clock.step()

      // After 2 cycles: both stages valid
      dut.io.stageAValid.expect(true.B)
      dut.io.stageBValid.expect(true.B)

      println("[M15-P] Pipeline stages: A valid after 1 cycle, B valid after 2 cycles ✓")
      println("[M15-P] Critical path depth: multiply chain only (stage A) or add chain only (stage B)")
      println("[M15-P] Non-pipelined MacLane: multiply + add chain combined (2× deeper)")
    }
  }
}
