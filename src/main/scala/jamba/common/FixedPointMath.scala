package jamba.common

import chisel3._

object FixedPointMath {
  def maxSigned(width: Int): SInt = ((BigInt(1) << (width - 1)) - 1).S(width.W)

  def minSigned(width: Int): SInt = (-(BigInt(1) << (width - 1))).S(width.W)

  def saturate(value: SInt, outWidth: Int): SInt = {
    val out = Wire(SInt(outWidth.W))
    val maxValue = maxSigned(outWidth)
    val minValue = minSigned(outWidth)

    when(value > maxValue) {
      out := maxValue
    }.elsewhen(value < minValue) {
      out := minValue
    }.otherwise {
      out := value.asSInt
    }

    out
  }

  def roundedShiftRight(value: SInt, shift: Int): SInt = {
    require(shift >= 0, "shift must be non-negative")

    if (shift == 0) {
      value
    } else {
      val rounded = Wire(SInt((value.getWidth + 1).W))
      val half = (BigInt(1) << (shift - 1)).S((value.getWidth + 1).W)
      val widened = value.asSInt.pad(value.getWidth + 1)

      when(value >= 0.S) {
        rounded := widened + half
      }.otherwise {
        rounded := widened - half
      }

      (rounded >> shift).asSInt
    }
  }

  def multiplyRescale(a: SInt, b: SInt, outWidth: Int, shift: Int): SInt = {
    val product = a * b
    val shifted = roundedShiftRight(product.asSInt, shift)
    saturate(shifted, outWidth)
  }

  def saturatingAdd(a: SInt, b: SInt, outWidth: Int): SInt = {
    val sum = a +& b
    saturate(sum.asSInt, outWidth)
  }
}
