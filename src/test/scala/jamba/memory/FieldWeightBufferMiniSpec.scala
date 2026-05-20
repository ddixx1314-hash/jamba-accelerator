package jamba.memory

import chisel3._
import chiseltest._
import jamba.common.Jamba2MiniConfig
import org.scalatest.flatspec.AnyFlatSpec

class FieldWeightBufferMiniSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "FieldWeightBufferMini"

  private val config = Jamba2MiniConfig.debug.copy(numLayers = 1, convTaps = 4)

  private def pokeIdle(dut: FieldWeightBufferMini): Unit = {
    dut.io.clear.poke(false.B)
    dut.io.start.poke(false.B)
    dut.io.fieldId.poke(0.U)
    dut.io.inValid.poke(false.B)
    dut.io.inData.poke(0.S)
    dut.io.inIsAcc.poke(false.B)
    dut.io.inElementIndex.poke(0.U)
    dut.io.inNumElements.poke(0.U)
    dut.io.inRow.poke(0.U)
    dut.io.inCol.poke(0.U)
    dut.io.inLane.poke(0.U)
    dut.io.inTap.poke(0.U)
    dut.io.inExpert.poke(0.U)
  }

  private def feedElement(
      dut: FieldWeightBufferMini,
      data: Int,
      isAcc: Boolean,
      elemIdx: Int,
      numElems: Int,
      row: Int = 0, col: Int = 0, lane: Int = 0, tap: Int = 0, expert: Int = 0
  ): Unit = {
    dut.io.inValid.poke(true.B)
    dut.io.inData.poke(data.S)
    dut.io.inIsAcc.poke(isAcc.B)
    dut.io.inElementIndex.poke(elemIdx.U)
    dut.io.inNumElements.poke(numElems.U)
    dut.io.inRow.poke(row.U)
    dut.io.inCol.poke(col.U)
    dut.io.inLane.poke(lane.U)
    dut.io.inTap.poke(tap.U)
    dut.io.inExpert.poke(expert.U)
    dut.clock.step()
    dut.io.inValid.poke(false.B)
  }

  it should "fill a data vector field and assert done" in {
    test(new FieldWeightBufferMini(config)) { dut =>
      pokeIdle(dut)
      dut.io.fieldId.poke(WeightAddressGenMini.Norm1Weight.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      for (lane <- 0 until 4) {
        feedElement(dut, lane * 3 + 1, isAcc = false, lane, 4, lane = lane)
      }
      dut.clock.step()
      dut.io.done.expect(true.B)
      for (lane <- 0 until 4) {
        dut.io.dataVec(lane).expect((lane * 3 + 1).S)
      }
    }
  }

  it should "fill an accumulator bias vector field" in {
    test(new FieldWeightBufferMini(config)) { dut =>
      pokeIdle(dut)
      dut.io.fieldId.poke(WeightAddressGenMini.MambaInputBias.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      for (lane <- 0 until 4) {
        feedElement(dut, lane * 10, isAcc = true, lane, 4, lane = lane)
      }
      dut.clock.step()
      dut.io.done.expect(true.B)
      for (lane <- 0 until 4) {
        dut.io.accVec(lane).expect((lane * 10).S)
      }
    }
  }

  it should "fill a data matrix field row-major" in {
    test(new FieldWeightBufferMini(config)) { dut =>
      pokeIdle(dut)
      dut.io.fieldId.poke(WeightAddressGenMini.QWeight.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var idx = 0
      for (row <- 0 until 4; col <- 0 until 4) {
        feedElement(dut, row * 4 + col + 1, isAcc = false, idx, 16, row = row, col = col)
        idx += 1
      }
      dut.clock.step()
      dut.io.done.expect(true.B)
      for (row <- 0 until 4; col <- 0 until 4) {
        dut.io.dataMatrix(row)(col).expect((row * 4 + col + 1).S)
      }
    }
  }

  it should "fill expert matrix and bias fields" in {
    test(new FieldWeightBufferMini(config)) { dut =>
      pokeIdle(dut)
      dut.io.fieldId.poke(WeightAddressGenMini.ExpertGateWeight.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)

      var idx = 0
      for (exp <- 0 until 2; row <- 0 until 4; col <- 0 until 4) {
        feedElement(dut, exp * 100 + row * 4 + col, isAcc = false, idx, 32, row = row, col = col, expert = exp)
        idx += 1
      }
      dut.clock.step()
      dut.io.done.expect(true.B)
      for (exp <- 0 until 2; row <- 0 until 4; col <- 0 until 4) {
        dut.io.expertMatrix(exp)(row)(col).expect((exp * 100 + row * 4 + col).S)
      }

      // Now test expert bias
      dut.io.fieldId.poke(WeightAddressGenMini.ExpertDownBias.U)
      dut.io.start.poke(true.B)
      dut.clock.step()
      dut.io.start.poke(false.B)
      idx = 0
      for (exp <- 0 until 2; lane <- 0 until 4) {
        feedElement(dut, exp * 10 + lane, isAcc = true, idx, 8, lane = lane, expert = exp)
        idx += 1
      }
      dut.clock.step()
      dut.io.done.expect(true.B)
      for (exp <- 0 until 2; lane <- 0 until 4) {
        dut.io.expertAccVec(exp)(lane).expect((exp * 10 + lane).S)
      }
    }
  }
}
