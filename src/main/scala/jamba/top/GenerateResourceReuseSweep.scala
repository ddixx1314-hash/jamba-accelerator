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
  SerialSharedLinear4
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
}
