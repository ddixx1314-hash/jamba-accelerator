package jamba.common

case class JambaMiniConfig(
  lanes: Int = 4,
  dataWidth: Int = 8,
  accWidth: Int = 32,
  convTaps: Int = 3
)
