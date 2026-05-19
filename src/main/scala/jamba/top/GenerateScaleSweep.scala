package jamba.top

import circt.stage.ChiselStage
import jamba.common.Jamba2MiniConfig
import jamba.fabric.UnifiedJamba2MiniLayer

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
      weightDepth = 512
    ) {
      override def desiredName: String = "UnifiedJamba2MiniFullTile_2L_Context4"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  // === Unified Layer Scale Sweep (contextLength) ===
  private val layerSweepCases = Seq(
    (4,  "UnifiedLayer_Context4"),
    (8,  "UnifiedLayer_Context8"),
    (16, "UnifiedLayer_Context16")
  )

  for ((ctx, layerName) <- layerSweepCases) {
    ChiselStage.emitSystemVerilogFile(
      new UnifiedJamba2MiniLayer(contextLength = ctx) {
        override def desiredName: String = layerName
      },
      firtoolOpts = firtoolOptions,
      args = Array("--target-dir", targetDir)
    )
  }

  // === Unified FullTile Scale Sweep (numLayers + contextLength) ===
  private val tileSweepCases = Seq(
    ("UnifiedFullTile_2L_Context8",
      Jamba2MiniConfig.debug.copy(numLayers = 2, attentionLayerPeriod = 2, attentionLayerOffset = 1), 256),
    ("UnifiedFullTile_4L_Context8",
      Jamba2MiniConfig.debug, 256),
    ("UnifiedFullTile_4L_Context16",
      Jamba2MiniConfig.debug.copy(contextLength = 16), 256),
    ("UnifiedFullTile_8L_Context16",
      Jamba2MiniConfig(numLayers = 8, attentionLayerPeriod = 8, attentionLayerOffset = 7, contextLength = 16), 512)
  )

  for ((tileName, cfg, wd) <- tileSweepCases) {
    ChiselStage.emitSystemVerilogFile(
      new UnifiedJamba2MiniFullTile(cfg, wd) {
        override def desiredName: String = tileName
      },
      firtoolOpts = firtoolOptions,
      args = Array("--target-dir", targetDir)
    )
  }
}
