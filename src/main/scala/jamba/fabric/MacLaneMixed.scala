package jamba.fabric

import chisel3._
import jamba.common.SignedMath

/** Shared signed MAC lane for operands with different widths.
  *
  * When zeroSkip=true, the multiply-accumulate is bypassed and accIn is passed
  * through unchanged whenever either operand is zero. This mirrors MacLane's
  * zero-skip behaviour and enables sparsification analysis for the scan path.
  */
class MacLaneMixed(aWidth: Int = 32, bWidth: Int = 8, accWidth: Int = 32, zeroSkip: Boolean = false)
    extends Module {
  require(aWidth > 0, "MacLaneMixed aWidth must be positive")
  require(bWidth > 0, "MacLaneMixed bWidth must be positive")
  require(accWidth > 0, "MacLaneMixed accWidth must be positive")

  val io = IO(new Bundle {
    val a      = Input(SInt(aWidth.W))
    val b      = Input(SInt(bWidth.W))
    val accIn  = Input(SInt(accWidth.W))
    val accOut = Output(SInt(accWidth.W))
  })

  val product = SignedMath.resize(io.a * io.b, accWidth)
  val mac = SignedMath.resize(product +& io.accIn, accWidth)
  if (zeroSkip) {
    io.accOut := Mux(io.a === 0.S || io.b === 0.S, io.accIn, mac)
  } else {
    io.accOut := mac
  }
}
