package jamba.fabric

import chisel3._

/** Dense MLP path built from shared linear projections and MAC lanes. */
class SharedDenseMLPMini(lanes: Int = 4, dataWidth: Int = 8, accWidth: Int = 32) extends Module {
  require(lanes == 4, "SharedDenseMLPMini currently uses SharedLinear4 and requires lanes == 4")

  val io = IO(new Bundle {
    val x = Input(Vec(lanes, SInt(dataWidth.W)))
    val gateWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val gateBias = Input(Vec(lanes, SInt(accWidth.W)))
    val upWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val upBias = Input(Vec(lanes, SInt(accWidth.W)))
    val downWeight = Input(Vec(lanes, Vec(lanes, SInt(dataWidth.W))))
    val downBias = Input(Vec(lanes, SInt(accWidth.W)))

    val gate = Output(Vec(lanes, SInt(accWidth.W)))
    val up = Output(Vec(lanes, SInt(accWidth.W)))
    val activatedGate = Output(Vec(lanes, SInt(dataWidth.W)))
    val hidden = Output(Vec(lanes, SInt(dataWidth.W)))
    val y = Output(Vec(lanes, SInt(accWidth.W)))
  })

  private def narrowToData(value: SInt): SInt = value(dataWidth - 1, 0).asSInt

  val gateProjection = Module(new SharedLinear4(dataWidth, accWidth))
  gateProjection.io.x := io.x
  gateProjection.io.weight := io.gateWeight
  gateProjection.io.bias := io.gateBias

  val upProjection = Module(new SharedLinear4(dataWidth, accWidth))
  upProjection.io.x := io.x
  upProjection.io.weight := io.upWeight
  upProjection.io.bias := io.upBias

  val hiddenMacs = Seq.fill(lanes)(Module(new MacLane(dataWidth, accWidth)))

  for (lane <- 0 until lanes) {
    io.gate(lane) := gateProjection.io.y(lane)
    io.up(lane) := upProjection.io.y(lane)
    io.activatedGate(lane) := Mux(gateProjection.io.y(lane) < 0.S, 0.S, narrowToData(gateProjection.io.y(lane)))

    hiddenMacs(lane).io.a := io.activatedGate(lane)
    hiddenMacs(lane).io.b := narrowToData(upProjection.io.y(lane))
    hiddenMacs(lane).io.accIn := 0.S(accWidth.W)
    io.hidden(lane) := narrowToData(hiddenMacs(lane).io.accOut)
  }

  val downProjection = Module(new SharedLinear4(dataWidth, accWidth))
  downProjection.io.x := io.hidden
  downProjection.io.weight := io.downWeight
  downProjection.io.bias := io.downBias

  io.y := downProjection.io.y
}
