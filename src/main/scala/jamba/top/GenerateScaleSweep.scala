package jamba.top

import circt.stage.ChiselStage
import jamba.common.Jamba2MiniConfig

/** Generate several Jamba2MiniTile variants for lightweight scale analysis. */
object GenerateScaleSweep extends App {
  private val targetDir = if (args.nonEmpty) args(0) else "generated/scale"
  private val firtoolOptions = Array(
    "-disable-all-randomization",
    "-strip-debug-info",
    "--lowering-options=disallowLocalVariables"
  )

  private case class SweepCase(name: String, config: Jamba2MiniConfig, weightDepth: Int = 256)

  private val cases = Seq(
    SweepCase("Jamba2MiniTile_Debug4L_Context8", Jamba2MiniConfig.debug),
    SweepCase(
      "Jamba2MiniTile_Formal8L_Context16",
      Jamba2MiniConfig()
    ),
    SweepCase(
      "Jamba2MiniTile_Debug4L_Context16",
      Jamba2MiniConfig.debug.copy(contextLength = 16)
    ),
    SweepCase(
      "Jamba2MiniTile_Debug4L_AttnPeriod2",
      Jamba2MiniConfig.debug.copy(attentionLayerPeriod = 2, attentionLayerOffset = 1)
    ),
    SweepCase(
      "Jamba2MiniTile_Formal8L_Context32",
      Jamba2MiniConfig(contextLength = 32)
    )
  )

  cases.foreach { sweepCase =>
    ChiselStage.emitSystemVerilogFile(
      new Jamba2MiniTile(sweepCase.config, sweepCase.weightDepth) {
        override def desiredName: String = sweepCase.name
      },
      firtoolOpts = firtoolOptions,
      args = Array("--target-dir", targetDir)
    )
  }

  ChiselStage.emitSystemVerilogFile(
    new UnifiedJamba2MiniAcceleratorTile(Jamba2MiniConfig.debug) {
      override def desiredName: String = "UnifiedJamba2MiniAcceleratorTile_Debug_Context8"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new UnifiedJamba2MiniTileScheduler(
      Jamba2MiniConfig.debug.copy(numLayers = 2, attentionLayerPeriod = 2, attentionLayerOffset = 1, contextLength = 4)
    ) {
      override def desiredName: String = "UnifiedJamba2MiniTileScheduler_2L_Context4"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new UnifiedJamba2MiniFullTile(
      Jamba2MiniConfig.debug.copy(numLayers = 2, attentionLayerPeriod = 2, attentionLayerOffset = 1, contextLength = 4),
      weightDepth = 64
    ) {
      override def desiredName: String = "UnifiedJamba2MiniFullTile_2L_Context4"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )
}
