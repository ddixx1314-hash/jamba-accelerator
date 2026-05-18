package jamba.fabric

import chisel3._

/** Semantic wrapper for serial Mamba input/B/C projections.
  *
  * All three projections consume the same token input, so this wrapper can use
  * the generic serial projection scheduler directly.
  */
class SerialMambaProjectionGroup(dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  val lanes = 4

  val io = IO(new Bundle {
    val start = Input(Bool())
    val clear = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))

    val inputWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val inputBias = Input(Vec(lanes, SInt(accWidth.W)))
    val bWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val bBias = Input(Vec(lanes, SInt(accWidth.W)))
    val cWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val cBias = Input(Vec(lanes, SInt(accWidth.W)))

    val ready = Output(Bool())
    val busy = Output(Bool())
    val done = Output(Bool())
    val projectedRaw = Output(Vec(lanes, SInt(accWidth.W)))
    val bRaw = Output(Vec(lanes, SInt(accWidth.W)))
    val cRaw = Output(Vec(lanes, SInt(accWidth.W)))
    val projected = Output(Vec(lanes, SInt(dataWidth.W)))
    val b = Output(Vec(lanes, SInt(dataWidth.W)))
    val c = Output(Vec(lanes, SInt(dataWidth.W)))
  })

  private def narrowToData(value: SInt): SInt = value(dataWidth - 1, 0).asSInt

  val scheduler = Module(new SerialProjectionScheduler4(numProjections = 3, dataWidth, accWidth))
  scheduler.io.start := io.start
  scheduler.io.clear := io.clear
  scheduler.io.x := io.x
  scheduler.io.weight(0) := io.inputWeight
  scheduler.io.weight(1) := io.bWeight
  scheduler.io.weight(2) := io.cWeight
  scheduler.io.bias(0) := io.inputBias
  scheduler.io.bias(1) := io.bBias
  scheduler.io.bias(2) := io.cBias

  io.ready := scheduler.io.ready
  io.busy := scheduler.io.busy
  io.done := scheduler.io.done
  io.projectedRaw := scheduler.io.y(0)
  io.bRaw := scheduler.io.y(1)
  io.cRaw := scheduler.io.y(2)

  for (lane <- 0 until lanes) {
    io.projected(lane) := narrowToData(io.projectedRaw(lane))
    io.b(lane) := narrowToData(io.bRaw(lane))
    io.c(lane) := narrowToData(io.cRaw(lane))
  }
}
