package jamba.common

import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class FixedPointSaturateHarness(inWidth: Int, outWidth: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(SInt(inWidth.W))
    val out = Output(SInt(outWidth.W))
  })

  io.out := FixedPointMath.saturate(io.in, outWidth)
}

class FixedPointShiftHarness(width: Int, shift: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(SInt(width.W))
    val out = Output(SInt(width.W))
  })

  io.out := FixedPointMath.roundedShiftRight(io.in, shift)
}

class FixedPointMultiplyRescaleHarness(aWidth: Int, bWidth: Int, outWidth: Int, shift: Int) extends Module {
  val io = IO(new Bundle {
    val a = Input(SInt(aWidth.W))
    val b = Input(SInt(bWidth.W))
    val out = Output(SInt(outWidth.W))
  })

  io.out := FixedPointMath.multiplyRescale(io.a, io.b, outWidth, shift)
}

class FixedPointSaturatingAddHarness(width: Int, outWidth: Int) extends Module {
  val io = IO(new Bundle {
    val a = Input(SInt(width.W))
    val b = Input(SInt(width.W))
    val out = Output(SInt(outWidth.W))
  })

  io.out := FixedPointMath.saturatingAdd(io.a, io.b, outWidth)
}

class FixedPointMathSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "FixedPointMath"

  it should "saturate signed values into the target width" in {
    test(new FixedPointSaturateHarness(16, 8)) { dut =>
      dut.io.in.poke(200.S)
      dut.io.out.expect(127.S)

      dut.io.in.poke((-200).S)
      dut.io.out.expect((-128).S)

      dut.io.in.poke(42.S)
      dut.io.out.expect(42.S)
    }
  }

  it should "round arithmetic right shifts away from zero" in {
    test(new FixedPointShiftHarness(16, 2)) { dut =>
      dut.io.in.poke(7.S)
      dut.io.out.expect(2.S)

      dut.io.in.poke((-7).S)
      dut.io.out.expect((-3).S)
    }
  }

  it should "multiply, rescale, and saturate" in {
    test(new FixedPointMultiplyRescaleHarness(8, 8, 8, 2)) { dut =>
      dut.io.a.poke(7.S)
      dut.io.b.poke(3.S)
      dut.io.out.expect(5.S)

      dut.io.a.poke(100.S)
      dut.io.b.poke(8.S)
      dut.io.out.expect(127.S)

      dut.io.a.poke((-7).S)
      dut.io.b.poke(3.S)
      dut.io.out.expect((-6).S)
    }
  }

  it should "saturate additions" in {
    test(new FixedPointSaturatingAddHarness(8, 8)) { dut =>
      dut.io.a.poke(100.S)
      dut.io.b.poke(50.S)
      dut.io.out.expect(127.S)

      dut.io.a.poke((-100).S)
      dut.io.b.poke((-50).S)
      dut.io.out.expect((-128).S)

      dut.io.a.poke(10.S)
      dut.io.b.poke((-3).S)
      dut.io.out.expect(7.S)
    }
  }
}
