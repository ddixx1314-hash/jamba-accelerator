package jamba.fabric

import chisel3._

/** One MoE expert MLP whose dense projections use the shared fabric. */
class SharedExpertMLPMini(lanes: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  val io = IO(new Bundle {
    val x          = Input(Vec(lanes, SInt(dataWidth.W)))
    val gateWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val gateBias   = Input(Vec(lanes, SInt(accWidth.W)))
    val upWeight   = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val upBias     = Input(Vec(lanes, SInt(accWidth.W)))
    val downWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val downBias   = Input(Vec(lanes, SInt(accWidth.W)))
    val y          = Output(Vec(lanes, SInt(accWidth.W)))
  })

  val dense = Module(new SharedDenseMLPMini(lanes, dataWidth, accWidth))
  dense.io.x := io.x
  dense.io.gateWeight := io.gateWeight
  dense.io.gateBias := io.gateBias
  dense.io.upWeight := io.upWeight
  dense.io.upBias := io.upBias
  dense.io.downWeight := io.downWeight
  dense.io.downBias := io.downBias

  io.y := dense.io.y
}
