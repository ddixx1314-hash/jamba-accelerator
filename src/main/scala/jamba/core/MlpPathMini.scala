package jamba.core

import chisel3._

/** MLP wrapper that currently runs DenseMLP and reserves a future MoE dispatch/combine boundary. */
class MlpPathMini(lanes: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(lanes == 4, "MlpPathMini currently uses DenseMLPMini and requires lanes == 4")

  val io = IO(new Bundle {
    val enableMoE = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val gateWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val gateBias = Input(Vec(lanes, SInt(accWidth.W)))
    val upWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val upBias = Input(Vec(lanes, SInt(accWidth.W)))
    val downWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val downBias = Input(Vec(lanes, SInt(accWidth.W)))

    val dispatchValid = Output(Bool())
    val dispatchReady = Input(Bool())
    val combineValid = Output(Bool())
    val combineReady = Input(Bool())
    val selectedExpert = Output(UInt(2.W))

    val denseY = Output(Vec(lanes, SInt(accWidth.W)))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
  })

  val dense = Module(new DenseMLPMini(lanes, dataWidth, accWidth))
  dense.io.x := io.x
  dense.io.gateWeight := io.gateWeight
  dense.io.gateBias := io.gateBias
  dense.io.upWeight := io.upWeight
  dense.io.upBias := io.upBias
  dense.io.downWeight := io.downWeight
  dense.io.downBias := io.downBias

  io.dispatchValid := false.B
  io.combineValid := false.B
  io.selectedExpert := 0.U
  io.denseY := dense.io.y
  io.y := dense.io.y
}
