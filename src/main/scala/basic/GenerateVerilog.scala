package basic

import circt.stage.ChiselStage

object GenerateVerilog extends App {
  ChiselStage.emitSystemVerilogFile(
    new Jamba2MiniAccelerator(),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--lowering-options=disallowLocalVariables"
    ),
    args = Array("--target-dir", "generated/verilog")
  )

  ChiselStage.emitSystemVerilogFile(
    new Jamba2MiniCore(),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--lowering-options=disallowLocalVariables"
    ),
    args = Array("--target-dir", "generated/verilog")
  )
}
