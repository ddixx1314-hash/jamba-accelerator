package jamba.common

import chisel3._

object SignedMath {
  def resize(value: SInt, width: Int): SInt = {
    val resized = Wire(SInt(width.W))
    resized := value
    resized
  }
}
