package jamba.top

import circt.stage.ChiselStage
import jamba.attention.AttentionDecodeTiny
import jamba.fabric.{
  MacLane,
  MacLaneMixed,
  SharedAttentionDecodeTiny,
  SharedCausalConv1D,
  SharedDotProduct,
  SharedLinear4,
  SharedReduction
}
import jamba.mamba.CausalConv1D
import jamba.math.{DotProduct, Linear4}

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
}
