package jamba.fabric

import chisel3._

/** Token-serial top-1 MoE-lite whose router and expert MLPs use shared fabric blocks. */
class SharedMoELiteMini(lanes: Int = 4, numExperts: Int = 2, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(numExperts == 2, "SharedMoELiteMini first implementation supports exactly 2 experts")

  val io = IO(new Bundle {
    val x                = Input(Vec(lanes, SInt(dataWidth.W)))
    val routerWeight     = Input(Vec(numExperts, Vec(lanes, SInt(dataWidth.W))))
    val routerBias       = Input(Vec(numExperts, SInt(accWidth.W)))
    val expertGateWeight = Input(Vec(numExperts, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertGateBias   = Input(Vec(numExperts, Vec(lanes, SInt(accWidth.W))))
    val expertUpWeight   = Input(Vec(numExperts, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertUpBias     = Input(Vec(numExperts, Vec(lanes, SInt(accWidth.W))))
    val expertDownWeight = Input(Vec(numExperts, Vec(lanes, Vec(lanes, SInt(dataWidth.W)))))
    val expertDownBias   = Input(Vec(numExperts, Vec(lanes, SInt(accWidth.W))))

    val dispatchValid  = Output(Bool())
    val dispatchReady  = Input(Bool())
    val combineValid   = Output(Bool())
    val combineReady   = Input(Bool())
    val selectedExpert = Output(UInt(1.W))
    val routerScores   = Output(Vec(numExperts, SInt(accWidth.W)))
    val y              = Output(Vec(lanes, SInt(accWidth.W)))
  })

  val router = Module(new SharedRouterMini(lanes, numExperts, dataWidth, accWidth))
  router.io.x := io.x
  router.io.weight := io.routerWeight
  router.io.bias := io.routerBias

  val experts = Seq.fill(numExperts)(Module(new SharedExpertMLPMini(lanes, dataWidth, accWidth)))
  for (expert <- 0 until numExperts) {
    experts(expert).io.x := io.x
    experts(expert).io.gateWeight := io.expertGateWeight(expert)
    experts(expert).io.gateBias := io.expertGateBias(expert)
    experts(expert).io.upWeight := io.expertUpWeight(expert)
    experts(expert).io.upBias := io.expertUpBias(expert)
    experts(expert).io.downWeight := io.expertDownWeight(expert)
    experts(expert).io.downBias := io.expertDownBias(expert)
  }

  io.dispatchValid := true.B
  io.combineValid := true.B
  io.selectedExpert := router.io.selectedExpert
  io.routerScores := router.io.scores

  for (lane <- 0 until lanes) {
    io.y(lane) := Mux(router.io.selectedExpert === 1.U, experts(1).io.y(lane), experts(0).io.y(lane))
  }
}
