package basic

import chisel3._

/** Processing Element — multiply-accumulate unit for SSM/Attention */
class PE(width: Int = 16) extends Module {
  val io = IO(new Bundle {
    val a      = Input(SInt(width.W))
    val b      = Input(SInt(width.W))
    val acc_in = Input(SInt(width.W))
    val acc_out = Output(SInt(width.W))
    val valid   = Output(Bool())
  })

  private val mulResult = io.a * io.b
  io.acc_out := mulResult + io.acc_in
  io.valid   := true.B
}
