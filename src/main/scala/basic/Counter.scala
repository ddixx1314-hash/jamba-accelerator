package basic

import chisel3._

class Counter(width: Int = 8) extends Module {
  val io = IO(new Bundle {
    val en  = Input(Bool())
    val out = Output(UInt(width.W))
  })

  val cnt = RegInit(0.U(width.W))

  when(io.en) {
    cnt := cnt + 1.U
  }

  io.out := cnt
}
