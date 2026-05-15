package jamba.common

case class FixedPointConfig(
  activationBits: Int = 8,
  weightBits: Int = 8,
  accumulatorBits: Int = 32,
  ssmStateBits: Int = 32,
  kvCacheBits: Int = 8,
  activationFracBits: Int = 4,
  weightFracBits: Int = 6,
  outputFracBits: Int = 4
) {
  require(activationBits > 1, "activationBits must include sign and magnitude bits")
  require(weightBits > 1, "weightBits must include sign and magnitude bits")
  require(accumulatorBits > 1, "accumulatorBits must include sign and magnitude bits")
  require(ssmStateBits >= accumulatorBits, "ssmStateBits should be at least accumulatorBits")
  require(kvCacheBits > 1, "kvCacheBits must include sign and magnitude bits")
  require(activationFracBits >= 0, "activationFracBits must be non-negative")
  require(weightFracBits >= 0, "weightFracBits must be non-negative")
  require(outputFracBits >= 0, "outputFracBits must be non-negative")

  val productFracBits: Int = activationFracBits + weightFracBits
  val productToOutputShift: Int = productFracBits - outputFracBits

  require(productToOutputShift >= 0, "outputFracBits cannot exceed product fractional bits")
}
