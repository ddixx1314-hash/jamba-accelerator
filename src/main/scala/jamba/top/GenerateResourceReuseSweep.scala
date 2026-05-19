package jamba.top

import circt.stage.ChiselStage
import jamba.attention.{AttentionDecodeTiny, AttentionMixerMini}
import jamba.core.{DenseMLPMini, Jamba2MiniLayer, MlpPathMini, TinyJambaBlock}
import jamba.fabric.{
  MacLane,
  MacLaneMixed,
  SharedDenseMLPMini,
  SharedAttentionDecodeTiny,
  SharedAttentionMixerMini,
  SharedCausalConv1D,
  SharedDotProduct,
  SharedJamba2MiniLayer,
  SharedJamba2MambaMixerMini,
  SharedLinear4,
  SharedMambaStateUpdate,
  SharedMlpPathMini,
  SharedMoELiteMini,
  SharedSelectiveScanTiny,
  SharedTinyJambaBlock,
  SharedTinyMambaBlock,
  SharedReduction,
  SerialAttentionMixerMini,
  SerialAttentionProjectionGroup,
  SerialJamba2MiniLayer,
  SerialCausalConvMini,
  SerialMambaProjectionGroup,
  SerialMambaMixerMini,
  SerialProjectionScheduler4,
  SerialSelectiveScanMini,
  SerialSharedLinear4,
  UnifiedJamba2MiniLayer,
  UnifiedMoEPathMini,
  UnifiedProjectionScheduler4
}
import jamba.mamba.{CausalConv1D, Jamba2MambaMixerMini, MambaStateUpdate, SelectiveScanTiny, TinyMambaBlock}
import jamba.math.{DotProduct, Linear4}
import jamba.moe.MoELiteMini

/** Generate baseline and shared-fabric operator variants for resource-reuse analysis. */
object GenerateResourceReuseSweep extends App {
  private val targetDir = if (args.nonEmpty) args(0) else "generated/resource_reuse"
  private val firtoolOptions = Array(
    "-disable-all-randomization",
    "-strip-debug-info",
    "--lowering-options=disallowLocalVariables"
  )

  ChiselStage.emitSystemVerilogFile(
    new MacLane() {
      override def desiredName: String = "MacLane_ResourceReuse"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new MacLaneMixed() {
      override def desiredName: String = "MacLaneMixed_ResourceReuse"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedReduction() {
      override def desiredName: String = "SharedReduction4_ResourceReuse"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new DotProduct() {
      override def desiredName: String = "DotProduct_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedDotProduct() {
      override def desiredName: String = "DotProduct_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new Linear4() {
      override def desiredName: String = "Linear4_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedLinear4() {
      override def desiredName: String = "Linear4_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialSharedLinear4() {
      override def desiredName: String = "Linear4_SerialSharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialProjectionScheduler4(numProjections = 3) {
      override def desiredName: String = "MambaProjectionGroup_SerialSharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialProjectionScheduler4(numProjections = 4) {
      override def desiredName: String = "AttentionProjectionGroup_SerialSharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialMambaProjectionGroup() {
      override def desiredName: String = "MambaProjectionGroup_SemanticSerial"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialAttentionProjectionGroup() {
      override def desiredName: String = "AttentionProjectionGroup_SemanticSerial"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new UnifiedProjectionScheduler4() {
      override def desiredName: String = "Jamba2LayerProjectionGroup_UnifiedSerial"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new AttentionDecodeTiny() {
      override def desiredName: String = "AttentionDecodeTiny_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedAttentionDecodeTiny() {
      override def desiredName: String = "AttentionDecodeTiny_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new CausalConv1D() {
      override def desiredName: String = "CausalConv1D_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedCausalConv1D() {
      override def desiredName: String = "CausalConv1D_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialCausalConvMini() {
      override def desiredName: String = "CausalConvMini_SerialSharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialSelectiveScanMini() {
      override def desiredName: String = "SelectiveScanMini_SerialSharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new MambaStateUpdate() {
      override def desiredName: String = "MambaStateUpdate_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedMambaStateUpdate() {
      override def desiredName: String = "MambaStateUpdate_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SelectiveScanTiny() {
      override def desiredName: String = "SelectiveScanTiny_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedSelectiveScanTiny() {
      override def desiredName: String = "SelectiveScanTiny_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new TinyMambaBlock() {
      override def desiredName: String = "TinyMambaBlock_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedTinyMambaBlock() {
      override def desiredName: String = "TinyMambaBlock_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new TinyJambaBlock() {
      override def desiredName: String = "TinyJambaBlock_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedTinyJambaBlock() {
      override def desiredName: String = "TinyJambaBlock_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new DenseMLPMini() {
      override def desiredName: String = "DenseMLPMini_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedDenseMLPMini() {
      override def desiredName: String = "DenseMLPMini_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new MoELiteMini() {
      override def desiredName: String = "MoELiteMini_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedMoELiteMini() {
      override def desiredName: String = "MoELiteMini_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new UnifiedMoEPathMini() {
      override def desiredName: String = "MoELiteMini_UnifiedSerial"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new MlpPathMini() {
      override def desiredName: String = "MlpPathMini_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedMlpPathMini() {
      override def desiredName: String = "MlpPathMini_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new Jamba2MiniLayer() {
      override def desiredName: String = "Jamba2MiniLayer_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedJamba2MiniLayer() {
      override def desiredName: String = "Jamba2MiniLayer_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new Jamba2MambaMixerMini() {
      override def desiredName: String = "Jamba2MambaMixerMini_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedJamba2MambaMixerMini() {
      override def desiredName: String = "Jamba2MambaMixerMini_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialMambaMixerMini() {
      override def desiredName: String = "Jamba2MambaMixerMini_SemanticSerial"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new AttentionMixerMini() {
      override def desiredName: String = "AttentionMixerMini_Baseline"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SharedAttentionMixerMini() {
      override def desiredName: String = "AttentionMixerMini_SharedFabric"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialAttentionMixerMini() {
      override def desiredName: String = "AttentionMixerMini_SemanticSerial"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialJamba2MiniLayer() {
      override def desiredName: String = "Jamba2MiniLayer_SemanticSerial"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new UnifiedJamba2MiniLayer() {
      override def desiredName: String = "Jamba2MiniLayer_UnifiedSerial"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  // === Quantization Precision Sweep ===
  private val quantConfigs = Seq(
    (4, 16, "INT4"),
    (6, 24, "INT6"),
    (8, 32, "INT8")
  )

  for ((dw, aw, precName) <- quantConfigs) {
    ChiselStage.emitSystemVerilogFile(
      new UnifiedJamba2MiniLayer(
        lanes         = 4,
        taps          = 4,
        contextLength = 8,
        dataWidth     = dw,
        stateWidth    = aw,
        accWidth      = aw
      ) {
        override def desiredName: String = s"Jamba2MiniLayer_UnifiedSerial_${precName}"
      },
      firtoolOpts = firtoolOptions,
      args = Array("--target-dir", targetDir)
    )
  }

  // === Zero-Skip Sparsification Variants ===
  ChiselStage.emitSystemVerilogFile(
    new MacLane(zeroSkip = true) {
      override def desiredName: String = "MacLane_ZeroSkip"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )

  ChiselStage.emitSystemVerilogFile(
    new SerialSharedLinear4(zeroSkip = true) {
      override def desiredName: String = "Linear4_SerialSharedFabric_ZeroSkip"
    },
    firtoolOpts = firtoolOptions,
    args = Array("--target-dir", targetDir)
  )
}
