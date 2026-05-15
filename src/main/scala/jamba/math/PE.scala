package jamba.math

import chisel3._

/** Processing Element — combinational multiply-accumulate unit for SSM/Attention */
class PE(dataWidth: Int = 16, accWidth: Int = 32) extends Module {
  require(dataWidth > 0, "PE dataWidth must be positive")
  require(accWidth >= 2 * dataWidth, "PE accWidth should hold the full product")

  val io = IO(new Bundle {
    val a       = Input(SInt(dataWidth.W))
    val b       = Input(SInt(dataWidth.W))
    val acc_in  = Input(SInt(accWidth.W))
    val acc_out = Output(SInt(accWidth.W))
    val valid   = Output(Bool())
  })

  val mulResult = Wire(SInt(accWidth.W))
  mulResult := io.a * io.b

  io.acc_out := mulResult + io.acc_in
  io.valid   := true.B
}
