package jamba.memory

import chisel3._
import chisel3.util.log2Ceil

/** Small register-file-backed weight storage with valid/ready write and combinational read. */
class WeightStoreMini(depth: Int = 64, dataWidth: Int = 32) extends Module {
  require(depth > 0, "WeightStoreMini depth must be positive")
  require(dataWidth > 0, "WeightStoreMini dataWidth must be positive")

  private val addrWidth = math.max(1, log2Ceil(depth))

  val io = IO(new Bundle {
    val clear = Input(Bool())
    val writeValid = Input(Bool())
    val writeReady = Output(Bool())
    val writeAddr = Input(UInt(addrWidth.W))
    val writeData = Input(SInt(dataWidth.W))
    val readAddr = Input(UInt(addrWidth.W))
    val readData = Output(SInt(dataWidth.W))
  })

  val mem = RegInit(VecInit(Seq.fill(depth)(0.S(dataWidth.W))))

  io.writeReady := true.B

  when(io.writeValid && io.writeReady) {
    mem(io.writeAddr) := io.writeData
  }

  // Clear is intentionally ignored by the weight store. Weights survive tile clears.
  io.readData := mem(io.readAddr)
}
