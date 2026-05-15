# Fixed-Point Policy

This document defines the fixed-point arithmetic policy for the Jamba2 Mini accelerator prototype.

The current hardware still uses mostly integer teaching modules. This policy is the shared contract for the next hardware stages.

## Numeric Domains

`FixedPointConfig` separates the important numeric domains:

| Domain | Default Bits | Purpose |
| --- | ---: | --- |
| Activation | 8 | Token activations and layer outputs |
| Weight | 8 | Projection weights, conv kernels, and MLP weights |
| Accumulator | 32 | Dot products and reductions |
| SSM State | 32 | Recurrent Mamba/SSM state |
| KV Cache | 8 | Stored attention keys and values |

The SSM state domain is intentionally wider than activations because recurrent updates can accumulate over time.

## Fractional Bits

The first policy uses:

| Field | Default |
| --- | ---: |
| `activationFracBits` | 4 |
| `weightFracBits` | 6 |
| `outputFracBits` | 4 |

For activation-weight multiplication:

```text
productFracBits = activationFracBits + weightFracBits
productToOutputShift = productFracBits - outputFracBits
```

The product is rounded, shifted, and saturated back into the target output domain.

## Saturation

All narrowing operations use signed saturation.

For an `N`-bit signed value:

```text
max =  2^(N-1) - 1
min = -2^(N-1)
```

Values above `max` clamp to `max`. Values below `min` clamp to `min`.

## Rounding

Right shifts use arithmetic shift with away-from-zero rounding before the shift:

```text
positive: (x + 2^(shift-1)) >> shift
negative: (x - 2^(shift-1)) >> shift
```

This rule is simple, deterministic, and mirrored exactly in the Python golden model.

## Shared Helpers

Chisel implementation:

```text
jamba.common.FixedPointMath
```

Python golden implementation:

```text
fixed_saturate
fixed_round_shift_right
fixed_multiply_rescale
fixed_saturating_add
```

All future fixed-point datapath modules should use these rules unless a later document explicitly changes the policy.
