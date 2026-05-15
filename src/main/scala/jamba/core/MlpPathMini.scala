package jamba.core

import chisel3._
import jamba.moe.MoELiteMini

/** MLP wrapper that currently runs DenseMLP and reserves a future MoE dispatch/combine boundary. */
class MlpPathMini(lanes: Int = 4, numExperts: Int = 2, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(lanes == 4, "MlpPathMini currently uses DenseMLPMini and requires lanes == 4")
  require(numExperts == 2, "MlpPathMini first MoE-lite implementation supports exactly 2 experts")

  val io = IO(new Bundle {
    val enableMoE = Input(Bool())
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val gateWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val gateBias = Input(Vec(lanes, SInt(accWidth.W)))
    val upWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val upBias = Input(Vec(lanes, SInt(accWidth.W)))
    val downWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val downBias = Input(Vec(lanes, SInt(accWidth.W)))
    val routerWeight = Input(Vec(numExperts, Vec(lanes, SInt(dataWidth.W))))
    val routerBias = Input(Vec(numExperts, SInt(accWidth.W)))
    val expertGateWeight = Input(Vec(numExperts, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertGateBias = Input(Vec(numExperts, Vec(lanes, SInt(accWidth.W))))
    val expertUpWeight = Input(Vec(numExperts, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertUpBias = Input(Vec(numExperts, Vec(lanes, SInt(accWidth.W))))
    val expertDownWeight = Input(Vec(numExperts, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertDownBias = Input(Vec(numExperts, Vec(lanes, SInt(accWidth.W))))

    val dispatchValid = Output(Bool())
    val dispatchReady = Input(Bool())
    val combineValid = Output(Bool())
    val combineReady = Input(Bool())
    val selectedExpert = Output(UInt(1.W))

    val denseY = Output(Vec(lanes, SInt(accWidth.W)))
    val moeY = Output(Vec(lanes, SInt(accWidth.W)))
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

  val moe = Module(new MoELiteMini(lanes, numExperts, dataWidth, accWidth))
  moe.io.x := io.x
  moe.io.routerWeight := io.routerWeight
  moe.io.routerBias := io.routerBias
  moe.io.expertGateWeight := io.expertGateWeight
  moe.io.expertGateBias := io.expertGateBias
  moe.io.expertUpWeight := io.expertUpWeight
  moe.io.expertUpBias := io.expertUpBias
  moe.io.expertDownWeight := io.expertDownWeight
  moe.io.expertDownBias := io.expertDownBias
  moe.io.dispatchReady := io.dispatchReady
  moe.io.combineReady := io.combineReady

  io.dispatchValid := io.enableMoE && moe.io.dispatchValid
  io.combineValid := io.enableMoE && moe.io.combineValid
  io.selectedExpert := Mux(io.enableMoE, moe.io.selectedExpert, 0.U)
  io.denseY := dense.io.y
  io.moeY := moe.io.y
  io.y := Mux(io.enableMoE, moe.io.y, dense.io.y)
}
