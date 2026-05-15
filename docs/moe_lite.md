# MoE-Lite

This document describes the first MoE-lite path for the Jamba2 Mini accelerator prototype.

The goal is to exercise the hardware structure of routing, dispatch, expert execution, and combine without implementing production-scale Jamba2 Mini MoE.

## Scope

The first implementation is intentionally small:

- token-serial
- top-1 routing
- two experts
- one token processed at a time
- deterministic integer arithmetic

This is not a full production MoE system. It is a hardware integration path that can later be expanded to vectorized dispatch/combine.

## Modules

- `RouterMini`: computes one score per expert and chooses the highest score.
- `ExpertMLPMini`: wraps one dense expert MLP.
- `MoELiteMini`: routes the token to one of two experts and returns the selected expert output.
- `MlpPathMini`: selects Dense MLP or MoE-lite based on `enableMoE`.

## Dispatch and Combine

`MlpPathMini` exposes the reserved boundary:

```text
dispatchValid
dispatchReady
combineValid
combineReady
selectedExpert
```

In the current token-serial implementation, `dispatchValid` and `combineValid` are asserted only when `enableMoE` is true.

## Tie Breaking

If expert scores tie, expert 0 is selected.

This keeps tests deterministic.

## Future Extension

The interface is intended to grow toward:

- more experts
- top-2 routing
- vectorized token dispatch
- expert output accumulation in combine logic
- load-balanced routing statistics
