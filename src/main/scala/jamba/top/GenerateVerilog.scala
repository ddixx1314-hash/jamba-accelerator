package jamba.top

import circt.stage.ChiselStage
import jamba.core.{Jamba2MiniAccelerator, Jamba2MiniCore}
import jamba.stream.Jamba2MiniStream

object GenerateVerilog extends App {
  ChiselStage.emitSystemVerilogFile(
    new Jamba2MiniTile(),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--lowering-options=disallowLocalVariables"
    ),
    args = Array("--target-dir", "generated/verilog")
  )

  ChiselStage.emitSystemVerilogFile(
    new JambaMiniTile(),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--lowering-options=disallowLocalVariables"
    ),
    args = Array("--target-dir", "generated/verilog")
  )

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

  ChiselStage.emitSystemVerilogFile(
    new Jamba2MiniStream(),
    firtoolOpts = Array(
      "-disable-all-randomization",
      "-strip-debug-info",
      "--lowering-options=disallowLocalVariables"
    ),
    args = Array("--target-dir", "generated/verilog")
  )
}
