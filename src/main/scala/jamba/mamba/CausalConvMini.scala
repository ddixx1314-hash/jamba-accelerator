package jamba.mamba

import chisel3._
import jamba.common.SignedMath

/** Parameterized per-lane causal convolution for the Jamba2 mini Mamba mixer. */
class CausalConvMini(lanes: Int = 4, taps: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(lanes > 0, "CausalConvMini lanes must be positive")
  require(taps > 0, "CausalConvMini taps must be positive")
  require(dataWidth > 0, "CausalConvMini dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "CausalConvMini accWidth should hold products")

  val io = IO(new Bundle {
    val en = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val kernel = Input(Vec(taps, Vec(lanes, SInt(dataWidth.W))))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
  })

  val history = RegInit(VecInit(Seq.fill(taps - 1)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W))))))

  for (lane <- 0 until lanes) {
    val terms = Wire(Vec(taps, SInt(accWidth.W)))
    terms(0) := SignedMath.resize(io.x(lane) * io.kernel(0)(lane), accWidth)

    for (tap <- 1 until taps) {
      terms(tap) := SignedMath.resize(history(tap - 1)(lane) * io.kernel(tap)(lane), accWidth)
    }

    io.y(lane) := terms.reduce((a, b) => SignedMath.resize(a +& b, accWidth))
  }

  when(io.clear) {
    history := VecInit(Seq.fill(taps - 1)(VecInit(Seq.fill(lanes)(0.S(dataWidth.W)))))
  }.elsewhen(io.en) {
    if (taps > 1) {
      for (tap <- (taps - 2) to 1 by -1) {
        history(tap) := history(tap - 1)
      }
      history(0) := io.x
    }
  }
}
