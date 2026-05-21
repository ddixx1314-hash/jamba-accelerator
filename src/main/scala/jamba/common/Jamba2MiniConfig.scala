package jamba.common

case class Jamba2MiniConfig(
  hiddenSize: Int = 8,
  lanes: Int = 4,
  numLayers: Int = 8,
  attentionLayerPeriod: Int = 8,
  attentionLayerOffset: Int = 7,
  mambaStateSize: Int = 8,
  convTaps: Int = 4,
  numAttentionHeads: Int = 2,
  numKvHeads: Int = 1,
  contextLength: Int = 16,
  mlpExpansion: Int = 2,
  enableMoE: Boolean = false,
  numExperts: Int = 0,
  dataWidth: Int = 8,
  weightWidth: Int = 8,
  accWidth: Int = 32,
  activationBits: Int = 8,
  weightBits: Int = 8,
  accumulatorBits: Int = 32,
  ssmStateBits: Int = 32,
  kvCacheBits: Int = 8,
  projectionMacLanes: Int = 1
) {
  require(hiddenSize > 0, "hiddenSize must be positive")
  require(lanes > 0, "lanes must be positive")
  require(numLayers > 0, "numLayers must be positive")
  require(attentionLayerPeriod > 0, "attentionLayerPeriod must be positive")
  require(attentionLayerOffset >= 0, "attentionLayerOffset must be non-negative")
  require(attentionLayerOffset < attentionLayerPeriod, "attentionLayerOffset must be inside the attention period")
  require(mambaStateSize > 0, "mambaStateSize must be positive")
  require(convTaps > 0, "convTaps must be positive")
  require(numAttentionHeads > 0, "numAttentionHeads must be positive")
  require(numKvHeads > 0, "numKvHeads must be positive")
  require(numAttentionHeads % numKvHeads == 0, "numAttentionHeads must be divisible by numKvHeads")
  require(contextLength > 0, "contextLength must be positive")
  require(mlpExpansion > 0, "mlpExpansion must be positive")
  require(numExperts >= 0, "numExperts must be non-negative")
  require(enableMoE || numExperts == 0, "numExperts should be zero when MoE is disabled")
  require(dataWidth > 0, "dataWidth must be positive")
  require(weightWidth > 0, "weightWidth must be positive")
  require(accWidth > 0, "accWidth must be positive")
  require(activationBits > 0, "activationBits must be positive")
  require(weightBits > 0, "weightBits must be positive")
  require(accumulatorBits > 0, "accumulatorBits must be positive")
  require(ssmStateBits >= accumulatorBits, "ssmStateBits should be at least accumulatorBits")
  require(kvCacheBits > 0, "kvCacheBits must be positive")
  require(projectionMacLanes >= 1, "projectionMacLanes must be positive")
  require(projectionMacLanes <= lanes, "projectionMacLanes must be <= lanes")
  require(lanes % projectionMacLanes == 0, "lanes must be divisible by projectionMacLanes")

  def isAttentionLayer(layerIndex: Int): Boolean =
    layerIndex % attentionLayerPeriod == attentionLayerOffset
}

object Jamba2MiniConfig {
  val DebugAttentionPeriod: Int = 4

  def debug: Jamba2MiniConfig =
    Jamba2MiniConfig(
      numLayers = 4,
      attentionLayerPeriod = DebugAttentionPeriod,
      attentionLayerOffset = DebugAttentionPeriod - 1,
      contextLength = 8
    )
}
