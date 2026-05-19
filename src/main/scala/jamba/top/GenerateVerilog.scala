package jamba.top

import circt.stage.ChiselStage
import jamba.common.Jamba2MiniConfig
import jamba.core.{Jamba2MiniAccelerator, Jamba2MiniCore}
import jamba.stream.Jamba2MiniStream

object GenerateVerilog extends App {
  private val firtoolOptions = Array(
    "-disable-all-randomization",
    "-strip-debug-info",
    "--lowering-options=disallowLocalVariables"
  )

  ChiselStage.emitSystemVerilogFile(
    new UnifiedJamba2MiniAcceleratorTile(),
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", "generated/verilog")
  )

  ChiselStage.emitSystemVerilogFile(
    new UnifiedJamba2MiniTileScheduler(),
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", "generated/verilog")
  )

  ChiselStage.emitSystemVerilogFile(
    new UnifiedJamba2MiniFullTile(
      Jamba2MiniConfig.debug.copy(numLayers = 2, attentionLayerPeriod = 2, attentionLayerOffset = 1, contextLength = 4),
      weightDepth = 64
    ),
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", "generated/verilog")
  )

  ChiselStage.emitSystemVerilogFile(
    new Jamba2MiniTile(),
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", "generated/verilog")
  )

  ChiselStage.emitSystemVerilogFile(
    new JambaMiniTile(),
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", "generated/verilog")
  )

  ChiselStage.emitSystemVerilogFile(
    new Jamba2MiniAccelerator(),
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", "generated/verilog")
  )

  ChiselStage.emitSystemVerilogFile(
    new Jamba2MiniCore(),
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", "generated/verilog")
  )

  ChiselStage.emitSystemVerilogFile(
    new Jamba2MiniStream(),
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", "generated/verilog")
  )
}
