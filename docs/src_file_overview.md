# src 文件职责总览

这份文档按目录说明 `src/` 下每个 Scala 文件的作用。可以把它当成项目地图。

## 三层架构导读

项目分三个层次，从底向上：

```
【基础层】common / math / norm / mamba / attention / moe / core
  → 组合逻辑或简单时序，每个模块只做一件事，容易阅读和测试

【资源复用层】fabric
  → 把上面的算子映射到共享 MAC lane、串行 scheduler 或统一 projection fabric
  → 核心贡献所在：SerialSharedLinear4、ConfigurableSerialLinear4、
    UnifiedProjectionScheduler4、UnifiedJamba2MiniLayer

【Tile 层】top
  → 多层调度、权重存储、状态虚拟化
  → 核心贡献：SinglePhysicalLayerTile（单物理层服务 L 逻辑层）
```

**新读者建议路径**：先读 `MacLane → SerialSharedLinear4 → UnifiedProjectionScheduler4 → UnifiedJamba2MiniLayer → SinglePhysicalLayerTile`，就能看懂整个资源复用链路。

---

## src/main/scala

### jamba/common

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/common/Counter.scala` | 最基础的带 `en` 使能计数器，用来练习 `RegInit`、`when` 和时序逻辑。 |
| `src/main/scala/jamba/common/FixedPointConfig.scala` | 固定点格式参数定义，描述数据位宽、累加位宽、小数位等配置。 |
| `src/main/scala/jamba/common/FixedPointMath.scala` | 固定点辅助函数集合，包含饱和、舍入右移、乘法重标定、饱和加法等。 |
| `src/main/scala/jamba/common/JambaMiniConfig.scala` | 第一阶段 mini tile 的配置参数，给早期 Jamba 小模块提供统一参数入口。 |
| `src/main/scala/jamba/common/Jamba2MiniConfig.scala` | Jamba2 mini 架构配置，包含层数、lane 数、卷积 tap、注意力层周期、context length、MoE 开关等，并提供 debug 配置。 |
| `src/main/scala/jamba/common/SignedMath.scala` | 有符号数位宽调整工具，主要用于把宽累加结果安全裁剪或扩展到目标位宽。 |

### jamba/math

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/math/PE.scala` | 组合乘加处理单元，计算 `a * b + accIn`，是后续 dot product、GEMM 和共享 MAC lane 的概念起点。 |
| `src/main/scala/jamba/math/DotProduct.scala` | 组合有符号点积，计算两个向量的乘积求和。 |
| `src/main/scala/jamba/math/SmallGemm4x4.scala` | 组合 4x4 矩阵乘法，输出 4x4 结果矩阵。 |
| `src/main/scala/jamba/math/VectorOps.scala` | 组合向量逐元素运算，提供 add、mul、ReLU/gate 等后续 block 会用到的基础向量算子。 |
| `src/main/scala/jamba/math/Linear4.scala` | 4 lane 线性投影，计算 `y(row) = bias(row) + sum(weight(row)(col) * x(col))`，对应神经网络里的全连接层。 |

### jamba/norm

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/norm/RmsNormStats.scala` | 计算 RMSNorm 需要的整数统计量：平方和与 mean square。 |
| `src/main/scala/jamba/norm/RmsNormApprox.scala` | 整数友好的 RMSNorm 近似实现，用 mean square 做缩放并处理全零输入。 |

### jamba/mamba

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/mamba/MambaStateUpdate.scala` | 最小 SSM 状态更新单元，计算 `state := a * state + b * x`，带 `en` 和 `clear`。 |
| `src/main/scala/jamba/mamba/SelectiveScanTiny.scala` | 小型 selective scan：先更新 SSM state，再用 gate/C 系数得到可见输出。 |
| `src/main/scala/jamba/mamba/CausalConv1D.scala` | 三 tap、逐通道 causal convolution，用移位寄存器保存历史 token。 |
| `src/main/scala/jamba/mamba/TinyMambaBlock.scala` | 最小 Mamba-like block，把 causal conv、selective scan 和 residual/gate 组合起来。 |
| `src/main/scala/jamba/mamba/CausalConvMini.scala` | 参数化 per-lane causal convolution，支持多 lane、多 tap，是 Jamba2 mini mixer 的卷积部分。 |
| `src/main/scala/jamba/mamba/SelectiveScanMini.scala` | token-serial SSM scan，计算 `state := a * state + b * x`，再输出 `state * c`。 |
| `src/main/scala/jamba/mamba/Jamba2MambaMixerMini.scala` | Jamba2 mini 的 Mamba mixer：输入投影、B/C 投影、causal conv、selective scan 和 C gate。 |

### jamba/attention

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/attention/AttentionDecodeTiny.scala` | 极简 attention decode，不做 softmax：先算 `q dot k` 分数，再用分数加权 value。 |
| `src/main/scala/jamba/attention/AttentionMixerMini.scala` | Jamba2 mini attention mixer，包含 Q/K/V/out 投影、环形 KV cache、decode 和输出投影。 |

### jamba/moe

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/moe/RouterMini.scala` | token-serial top-1 MoE router，计算每个 expert 的分数并选择最大者，平分时选 expert 0。 |
| `src/main/scala/jamba/moe/ExpertMLPMini.scala` | 单个 MoE expert 的 MLP 包装，内部复用 `DenseMLPMini`。 |
| `src/main/scala/jamba/moe/MoELiteMini.scala` | 两 expert 的简化 MoE-lite 路径，包含 router、expert dispatch/combine 和输出选择。 |

### jamba/core

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/core/DenseMLPMini.scala` | Dense MLP 路径：`ReLU(gate projection) * up projection` 后接 down projection。 |
| `src/main/scala/jamba/core/MlpPathMini.scala` | MLP 路径包装器，在 dense MLP 和预留 MoE 边界之间统一接口。 |
| `src/main/scala/jamba/core/TinyJambaBlock.scala` | Tiny Jamba-like block，在 Mamba path 上可选叠加 tiny attention path。 |
| `src/main/scala/jamba/core/Jamba2MiniCore.scala` | 更完整的 tiny Jamba2 风格 core：RMSNorm、投影、sequence block 和输出投影。 |
| `src/main/scala/jamba/core/Jamba2MiniLayer.scala` | 正式 mini layer：`RMSNorm -> Mixer -> residual -> RMSNorm -> MLP -> residual`。 |
| `src/main/scala/jamba/core/Jamba2MiniHybridCore.scala` | 多层 hybrid core，根据配置让部分层走 Attention，其余主要走 Mamba。 |
| `src/main/scala/jamba/core/Jamba2MiniAccelerator.scala` | 顶层固定点原型 accelerator block，给早期端到端实验提供一个简单外壳。 |

### jamba/fabric

这一目录主要是“资源复用版”实现：把原来并行的 dot product、linear、conv、mixer 等映射到共享 MAC lane、串行 scheduler 或统一 projection fabric 上，用来研究面积/延迟权衡。

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/fabric/MacLane.scala` | 共享有符号 MAC lane，计算 `a * b + accIn`，支持可选 zero-skip。 |
| `src/main/scala/jamba/fabric/MacLaneMixed.scala` | 混合位宽 MAC lane，支持宽 state 与窄系数相乘，也支持 zero-skip。 |
| `src/main/scala/jamba/fabric/SharedReduction.scala` | 组合有符号归约器，把多个累加宽度输入求和。 |
| `src/main/scala/jamba/fabric/SharedDotProduct.scala` | 用可复用 MAC lane 组织出的 dot product。 |
| `src/main/scala/jamba/fabric/SharedLinear4.scala` | 用 shared-style dot product 组成的 4 lane 线性投影。 |
| `src/main/scala/jamba/fabric/SerialSharedLinear4.scala` | 用一个可复用 MAC lane 串行完成 4x4 linear，暴露 `start/ready/busy/done` 控制。 |
| `src/main/scala/jamba/fabric/ConfigurableSerialLinear4.scala` | 可配置 MAC lane 并行度的串行 linear，用于比较 1/2/4 lane 的延迟和资源；`columnSkip=true` 时激活稀疏 FSM，通过优先级编码器跳过零列，周期数 k×lanes+2（k 为非零输入列数）（M12-A）。 |
| `src/main/scala/jamba/fabric/SerialProjectionScheduler4.scala` | 把多个 4x4 projection 排队映射到一个串行 `Linear4` fabric。 |
| `src/main/scala/jamba/fabric/UnifiedProjectionScheduler4.scala` | 给 layer 内命名投影槽位统一调度，包括 Mamba、Attention 和 MLP 相关投影；`vectorBypass=true` 支持运行时零输入跳过（M10-D）；`columnSkip=true` 将稀疏列跳过传播到 ConfigurableSerialLinear4（M13-S）。 |
| `src/main/scala/jamba/fabric/SerialMambaProjectionGroup.scala` | 语义包装：把 Mamba input/B/C projection 放到串行 projection group 里。 |
| `src/main/scala/jamba/fabric/SerialAttentionProjectionGroup.scala` | 语义包装：把 Attention Q/K/V/out projection 放到串行 projection group 里。 |
| `src/main/scala/jamba/fabric/SharedCausalConv1D.scala` | shared-fabric 版三 tap causal conv。 |
| `src/main/scala/jamba/fabric/SerialCausalConvMini.scala` | 用一个可复用 MAC lane 串行计算多 lane、多 tap causal conv。 |
| `src/main/scala/jamba/fabric/SharedMambaStateUpdate.scala` | shared mixed-width MAC 版 SSM state update。 |
| `src/main/scala/jamba/fabric/SharedSelectiveScanTiny.scala` | shared-fabric 版 tiny selective scan。 |
| `src/main/scala/jamba/fabric/SerialSelectiveScanMini.scala` | 串行 selective scan，用单个混合位宽 MAC 分三步完成 recurrent、input、output gate；`useShiftA=true` 时将 state×A MAC 替换为算术右移，2-op 调度节省 lanes 个周期/token（M12-P）；参数化 lanes，lanes=8 已验证：standard 25 cy，useShiftA 17 cy（M14-C）。 |
| `src/main/scala/jamba/fabric/SharedTinyMambaBlock.scala` | shared-fabric 版 tiny Mamba block，由共享 conv、scan、MAC 资源组成。 |
| `src/main/scala/jamba/fabric/SharedJamba2MambaMixerMini.scala` | shared-fabric 版 Jamba2 Mamba mixer，投影映射到共享 linear fabric。 |
| `src/main/scala/jamba/fabric/SerialMambaMixerMini.scala` | token-level 串行 Mamba mixer shell，按阶段复用 projection、conv 和 scan 资源。 |
| `src/main/scala/jamba/fabric/SharedAttentionDecodeTiny.scala` | shared-fabric 版 tiny attention decode。 |
| `src/main/scala/jamba/fabric/SharedAttentionMixerMini.scala` | shared linear fabric 版 Attention mixer，复用 Q/K/V/out 投影资源。 |
| `src/main/scala/jamba/fabric/SerialAttentionMixerMini.scala` | token-level 串行 Attention mixer，用一个 serial linear 复用四个 projection。 |
| `src/main/scala/jamba/fabric/SharedRouterMini.scala` | shared dot-product lane 版 MoE top-1 router。 |
| `src/main/scala/jamba/fabric/SharedExpertMLPMini.scala` | shared-fabric 版单个 expert MLP。 |
| `src/main/scala/jamba/fabric/SharedDenseMLPMini.scala` | shared-fabric 版 Dense MLP。 |
| `src/main/scala/jamba/fabric/SharedMoELiteMini.scala` | router 和 expert MLP 都映射到 shared fabric 的 MoE-lite。 |
| `src/main/scala/jamba/fabric/SharedMlpPathMini.scala` | Dense/MoE 二选一路径的 shared-fabric 统一包装。 |
| `src/main/scala/jamba/fabric/UnifiedMoEPathMini.scala` | 通过统一 projection scheduler 实现的 token-serial top-1 MoE-lite path。 |
| `src/main/scala/jamba/fabric/SharedTinyJambaBlock.scala` | shared-fabric 版 Tiny Jamba block，组合 shared Mamba 和 tiny attention。 |
| `src/main/scala/jamba/fabric/SharedJamba2MiniLayer.scala` | MLP 侧映射到 shared fabric 的正式 Jamba2 mini layer。 |
| `src/main/scala/jamba/fabric/SerialJamba2MiniLayer.scala` | 全 layer 串行/时间复用版本，Mamba 和 Attention mixer 都按多周期调度。 |
| `src/main/scala/jamba/fabric/UnifiedJamba2MiniLayer.scala` | 使用一个统一 projection scheduler 的 Jamba2 mini layer，是后续 tile 复用架构的核心 layer；`useShiftA` 传递 PoT A 矩阵（M12-P）；`attentionWindowSize` 控制 Samba 风格滑动窗口注意力（M12-K）；`fusedOperators` 消除 FSM 中间状态（M11-F，Mamba -2 cy/Attention -2 cy）；`fuseInnerLaunch` 将 launchConv/launchScan/launchAttentionOut 融合进前序等待态（M14-F，Mamba -2 cy/Attention -1 cy）；`fuseMlpLaunch` 将 launchMlpDown 融合进 computeHidden/waitMlpGateUp（M15-N，各节省 1 cy，Mamba 153/Attention 145）。 |

### jamba/memory

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/memory/WeightStoreMini.scala` | 小型寄存器文件式 weight store，支持 valid/ready 写入和组合读出，clear 不清权重。 |
| `src/main/scala/jamba/memory/WeightAddressGenMini.scala` | 权重地址生成器，把 layer、field、row/col/lane/tap/expert 映射成扁平地址。 |
| `src/main/scala/jamba/memory/SequentialWeightLoaderMini.scala` | 顺序 field-element 地址遍历器，用于按字段逐元素加载 mini Jamba 权重。 |
| `src/main/scala/jamba/memory/SequentialWeightCaptureMini.scala` | 连接 loader 和外部 memory read port，把 readData 作为元素流向后级输出。 |
| `src/main/scala/jamba/memory/FieldWeightBufferMini.scala` | 从元素流填充完整权重字段，写入 typed registers，例如向量、矩阵、kernel、expert 参数。 |
| `src/main/scala/jamba/memory/SequentialWeightLoadPathMini.scala` | 单个 field 的端到端顺序加载路径：地址生成、读数、缓冲到 typed outputs。 |
| `src/main/scala/jamba/memory/LayeredWeightStoreMini.scala` | 多层 field-banked weight store，把扁平地址空间解码成每层各类 typed weight 输出。 |

### jamba/stream

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/stream/Jamba2MiniStream.scala` | `Jamba2MiniCore` 的流式包装，提供输入 ready、输出 valid/ready 和一项输出缓冲。 |

### jamba/top

| 文件 | 作用 |
| --- | --- |
| `src/main/scala/jamba/top/JambaMiniTile.scala` | 第一阶段 mini accelerator tile 顶层，带 token stream、控制信号和 tiny Jamba datapath。 |
| `src/main/scala/jamba/top/Jamba2MiniTile.scala` | 正式 Jamba2 mini tile，包含 token stream、command/status 和 weight-load shell。 |
| `src/main/scala/jamba/top/UnifiedJamba2MiniAcceleratorTile.scala` | 单层 unified Jamba2 mini accelerator shell，支持模式选择、权重写读、输出缓冲。 |
| `src/main/scala/jamba/top/UnifiedJamba2MiniTileScheduler.scala` | 多层 unified layer 的顺序 scheduler，按配置安排 sparse Attention layer。 |
| `src/main/scala/jamba/top/UnifiedJamba2MiniFullTile.scala` | 多层 unified scheduler 外壳，带 weight store、流式输入输出和调试状态。 |
| `src/main/scala/jamba/top/SinglePhysicalLayerTile.scala` | 单物理 layer、多逻辑 layer 的 tile，用 per-layer state virtualization 复用一个物理层。 |
| `src/main/scala/jamba/top/GenerateVerilog.scala` | Verilog 生成入口，用 ChiselStage 生成选定模块的 Verilog。 |
| `src/main/scala/jamba/top/GenerateScaleSweep.scala` | 生成多组 tile/layer 配置的 Verilog，用于规模变化分析。 |
| `src/main/scala/jamba/top/GenerateResourceReuseSweep.scala` | 生成 baseline、shared-fabric、serial 等变体 Verilog，用于资源复用实验。 |

## src/test/scala

### jamba/common

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/common/CounterTest.scala` | 验证计数器 enable 计数、disable 保持和位宽溢出回绕。 |
| `src/test/scala/jamba/common/FixedPointMathSpec.scala` | 验证固定点饱和、舍入右移、乘法重标定和饱和加法。 |

### jamba/math

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/math/PESpec.scala` | 验证 PE 正数、负数和零乘积的乘加行为，并生成过 VCD。 |
| `src/test/scala/jamba/math/DotProductSpec.scala` | 验证 dot product 的正常、负数和全零输入。 |
| `src/test/scala/jamba/math/SmallGemm4x4Spec.scala` | 验证 4x4 GEMM 的普通矩阵、单位矩阵/零矩阵等情况。 |
| `src/test/scala/jamba/math/VectorOpsSpec.scala` | 验证逐元素向量运算，包括加法、乘法和门控/激活相关输出。 |
| `src/test/scala/jamba/math/Linear4Spec.scala` | 验证 4 lane linear projection 的矩阵-向量加 bias 结果。 |

### jamba/norm

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/norm/RmsNormStatsSpec.scala` | 验证平方和、整数 mean square、负数输入和全零输入。 |
| `src/test/scala/jamba/norm/RmsNormApproxSpec.scala` | 验证整数 RMSNorm 近似缩放和全零保护。 |

### jamba/mamba

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/mamba/MambaStateUpdateSpec.scala` | 验证 SSM state enabled 更新、disabled 保持和 clear 清零。 |
| `src/test/scala/jamba/mamba/SelectiveScanTinySpec.scala` | 验证 tiny selective scan 对可见 SSM state 的 gate 行为。 |
| `src/test/scala/jamba/mamba/CausalConv1DSpec.scala` | 验证三 tap causal conv 的跨 token 历史、disabled 保持和 clear。 |
| `src/test/scala/jamba/mamba/TinyMambaBlockSpec.scala` | 验证 tiny Mamba block 组合 conv、SSM state 和 residual gate。 |
| `src/test/scala/jamba/mamba/CausalConvMiniSpec.scala` | 验证四 tap causal conv over time、history 保持和 clear。 |
| `src/test/scala/jamba/mamba/SelectiveScanMiniSpec.scala` | 验证 token-serial scan 的状态累加、输出 gate、disabled 保持和 clear。 |
| `src/test/scala/jamba/mamba/Jamba2MambaMixerMiniSpec.scala` | 用 Python golden fixture 验证单 token 和两 token convolution history 的 Mamba mixer。 |

### jamba/attention

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/attention/AttentionDecodeTinySpec.scala` | 验证 dot-product scores、weighted value sum 和零 query。 |
| `src/test/scala/jamba/attention/AttentionMixerMiniSpec.scala` | 验证 Q/K/V cache 写入、环形 cache wrap、clear、disabled 行为、负数/非单位权重和饱和。 |

### jamba/moe

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/moe/MoELiteMiniSpec.scala` | 验证 MoE-lite 根据 router score 选择 expert，平分时选 expert 0。 |

### jamba/core

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/core/DenseMLPMiniSpec.scala` | 对照 Python deterministic fixture 验证 Dense MLP step。 |
| `src/test/scala/jamba/core/TinyJambaBlockSpec.scala` | 验证 attention disabled 时只输出 Mamba path，enabled 时叠加 tiny attention。 |
| `src/test/scala/jamba/core/Jamba2MiniCoreSpec.scala` | 验证 tiny Jamba2 core 的 Mamba/Attention 路径、valid 和关键中间输出。 |
| `src/test/scala/jamba/core/Jamba2MiniLayerSpec.scala` | 验证正式 layer 的 Mamba mode、Attention mode 和 MoE-lite MLP path。 |
| `src/test/scala/jamba/core/Jamba2MiniHybridCoreSpec.scala` | 验证多层 hybrid core 的 sparse Attention placement、层输出和 clear/valid 行为。 |
| `src/test/scala/jamba/core/Jamba2MiniAcceleratorSpec.scala` | 验证顶层 accelerator 原型的 enable 后 valid、状态输出和 clear。 |

### jamba/fabric

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/fabric/MacLaneSpec.scala` | 验证 shared MAC lane 的正数、负数和零乘积行为。 |
| `src/test/scala/jamba/fabric/MacLaneZeroSkipSpec.scala` | 验证 `MacLane(zeroSkip = true)` 对零 operand 跳过乘法但保留累加器语义。 |
| `src/test/scala/jamba/fabric/MacLaneMixedZeroSkipSpec.scala` | 验证混合位宽 MAC 的 zero-skip、非 zero-skip 和符号扩展相关行为。 |
| `src/test/scala/jamba/fabric/SharedDotProductSpec.scala` | 验证 shared-fabric dot product 的普通、负数和全零输入。 |
| `src/test/scala/jamba/fabric/SharedLinear4Spec.scala` | 验证 shared linear 的矩阵-向量、bias 和多行输出。 |
| `src/test/scala/jamba/fabric/SerialSharedLinear4Spec.scala` | 验证串行 linear 的 start/done 协议、输出结果、busy 和 clear。 |
| `src/test/scala/jamba/fabric/ConfigurableSerialLinear4Spec.scala` | 验证不同 MAC lane 并行度下结果一致、串行版本兼容性；M12-A：验证 columnSkip 输出正确性、k=0/k=2 稀疏加速倍率和 k=0 bias-only 输出；M13-L：lanes=8 macLanes=1/2/4 正确性和 Pareto 延迟排序；M15-W：lanes=16 macLanes=1/2/4 正确性（257/129/65 cy）和 L²/M+1 公式确认。 |
| `src/test/scala/jamba/fabric/SerialProjectionScheduler4Spec.scala` | 验证多个 projection 通过一个 serial linear 的调度、slot 跳过和清除。 |
| `src/test/scala/jamba/fabric/UnifiedProjectionScheduler4Spec.scala` | 验证统一投影调度器的命名 slot、结果保存、busy/done、clear 和 MAC lane 延迟趋势；M10-D：bypass 零输入、bypassCount 计数；M13-S：columnSkip k=2 稀疏 slot 正确性和延迟节省（14 vs 21 cycles），k=4 dense 正确性。 |
| `src/test/scala/jamba/fabric/SerialSemanticProjectionGroupSpec.scala` | 验证 Mamba/Attention 语义 projection group 的输出和控制协议。 |
| `src/test/scala/jamba/fabric/SharedCausalConv1DSpec.scala` | 验证 shared 版三 tap causal conv 的历史和 clear。 |
| `src/test/scala/jamba/fabric/SerialCausalConvMiniSpec.scala` | 验证串行 causal conv 的单 token 结果、跨 token history 和 clear。 |
| `src/test/scala/jamba/fabric/SharedMambaStateUpdateSpec.scala` | 验证 shared 版 SSM state update 的 enabled、disabled 和 clear。 |
| `src/test/scala/jamba/fabric/SharedSelectiveScanTinySpec.scala` | 验证 shared 版 tiny selective scan 的 gate 输出。 |
| `src/test/scala/jamba/fabric/SerialSelectiveScanMiniSpec.scala` | 验证串行 selective scan 的单 token、两 token 状态累加和 clear；M12-P：验证 useShiftA 输出正确性、两 token 累加、lanes 周期节省和首 token a=0↔a=1 等价性；M14-C：lanes=8 identity 输出正确性、useShiftA 2-op 正确性和 saved=8=lanes Pareto 验证；M15-W：lanes=16 identity 正确性（49/33 cy）和 saved=16=lanes N×lanes 公式确认。 |
| `src/test/scala/jamba/fabric/SharedTinyMambaBlockSpec.scala` | 验证 shared 版 tiny Mamba block 的 conv、SSM 和 residual gate 组合。 |
| `src/test/scala/jamba/fabric/SharedAttentionDecodeTinySpec.scala` | 验证 shared 版 attention decode 的 scores、weighted values 和零 query。 |
| `src/test/scala/jamba/fabric/SharedDenseMLPMiniSpec.scala` | 验证 shared 版 Dense MLP 与 deterministic fixture 结果一致。 |
| `src/test/scala/jamba/fabric/SharedMlpPathMiniSpec.scala` | 验证 shared router、shared expert、shared MoE 和 dense/MoE 统一 MLP path。 |
| `src/test/scala/jamba/fabric/UnifiedMoEPathMiniSpec.scala` | 验证通过 unified projection scheduler 的 MoE path、expert 选择和 dense/MoE 相关输出。 |
| `src/test/scala/jamba/fabric/SharedTinyJambaBlockSpec.scala` | 验证 shared 版 Tiny Jamba block 的 Mamba-only 和 Attention 混合输出。 |
| `src/test/scala/jamba/fabric/SharedJamba2MiniLayerSpec.scala` | 验证 shared MLP layer 的 Mamba mode、Attention mode 和 MoE-lite MLP path。 |
| `src/test/scala/jamba/fabric/SerialMambaMixerMiniSpec.scala` | 验证串行 Mamba mixer 的单 token、跨 token history 和 clear。 |
| `src/test/scala/jamba/fabric/SerialAttentionMixerMiniSpec.scala` | 验证串行 Attention mixer 的 cache 写入、输出、back-to-back token 和 clear。 |
| `src/test/scala/jamba/fabric/SharedMixerProjectionSpec.scala` | 对比 shared mixer projection 与 baseline Mamba/Attention mixer 的行为一致性。 |
| `src/test/scala/jamba/fabric/SerialJamba2MiniLayerSpec.scala` | 验证串行 full layer 的 Mamba/Attention 模式切换、KV cache 和 clear。 |
| `src/test/scala/jamba/fabric/UnifiedJamba2MiniLayerSpec.scala` | 验证 unified layer 的 Mamba/Attention/MoE 模式、状态保存恢复和 MAC lane 并行度；M11-F：fusedOperators 输出一致性和 2 周期节省；M12-P：useShiftA 输出一致性和 4 周期节省；M12-K：滑动窗口注意力输出差异验证（fullCtx vs window=1）；M12-K+MoE：Attention+MoE 联合模式 attentionWindowSize 掩码正确性；M14-F：fuseInnerLaunch 输出一致性（Mamba/Attention 各一用例）和 Mamba -2 cy / Attention -1 cy 周期节省验证；M15-N：fuseMlpLaunch 输出一致性（Mamba/Attention 各一用例）和 1 cy/token 节省验证（Mamba 153/Attention 145）。 |
| `src/test/scala/jamba/fabric/LatencyBudgetSpec.scala` | 测量/约束 serial linear、configurable linear、serial conv、serial scan 的周期预算。 |
| `src/test/scala/jamba/fabric/ResourceReuseComparisonSpec.scala` | 多个 comparison harness，对比 baseline 与 shared/serial 实现的输出一致性。 |

### jamba/memory

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/memory/WeightStoreMiniSpec.scala` | 验证权重写读、覆盖写和 clear 后权重保留。 |
| `src/test/scala/jamba/memory/WeightAddressGenMiniSpec.scala` | 验证 vector、matrix、kernel、router、expert 字段地址和非法 field 报错。 |
| `src/test/scala/jamba/memory/SequentialWeightLoaderMiniSpec.scala` | 验证顺序 loader 的 field 遍历、backpressure、层 stride 和错误处理。 |
| `src/test/scala/jamba/memory/SequentialWeightCaptureMiniSpec.scala` | 验证地址输出、readData 捕获、backpressure、layer stride 和非法 field。 |
| `src/test/scala/jamba/memory/FieldWeightBufferMiniSpec.scala` | 验证元素流填充向量、矩阵、bias、expert matrix/bias 字段。 |
| `src/test/scala/jamba/memory/SequentialWeightLoadPathMiniSpec.scala` | 验证一个 field 从地址生成到 readData 注入再到 buffer 输出的端到端路径。 |
| `src/test/scala/jamba/memory/LayeredWeightStoreMiniSpec.scala` | 验证多层 weight store 的扁平地址读写、字段解码和跨层隔离。 |

### jamba/stream

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/stream/Jamba2MiniStreamSpec.scala` | 验证 streaming wrapper 的输入 ready、输出 backpressure、同周期消费替换和 clear。 |

### jamba/top

| 文件 | 覆盖重点 |
| --- | --- |
| `src/test/scala/jamba/top/JambaMiniTileSpec.scala` | 验证第一阶段 tile 的 clear、输入输出握手、backpressure 和 attention mix。 |
| `src/test/scala/jamba/top/Jamba2MiniTileSpec.scala` | 验证 Jamba2 mini tile 的 start gate、输出 buffer、debug 信息、Python golden demo trace 和 loaded weights。 |
| `src/test/scala/jamba/top/UnifiedJamba2MiniAcceleratorTileSpec.scala` | 验证 unified accelerator shell 的 Mamba/Attention/MoE 模式、loaded weights、backpressure 和 clear。 |
| `src/test/scala/jamba/top/UnifiedJamba2MiniTileSchedulerSpec.scala` | 验证多层 unified scheduler 的 sparse Attention 调度和 clear。 |
| `src/test/scala/jamba/top/UnifiedJamba2MiniFullTileSpec.scala` | 验证 full tile scheduler shell 的 token 处理、输出 backpressure 和 clear 后权重保留。 |
| `src/test/scala/jamba/top/UnifiedJamba2MiniFullTileWeightLoadSpec.scala` | 验证 full tile 通过写端口加载权重后输出一致，并验证 weight readback。 |
| `src/test/scala/jamba/top/SinglePhysicalLayerTileSpec.scala` | 验证单物理层复用多个逻辑层、跨 token state virtualization（2-token + 3-layer 等价性）、backpressure、clear 和不同 MAC lane 配置一致性。 |

---

## python/

### python/golden

| 文件 | 作用 |
| --- | --- |
| `python/golden/mamba_ops.py` | 整个项目的 Python 参考实现，包含固定点乘法/右移/饱和、Mamba state update、causal conv、SSM scan、attention decode、dense MLP、MoE-lite routing、quantized step 等函数，供 ChiselTest 和独立测试用作 golden model。 |

### python/tests

| 文件 | 覆盖重点 |
| --- | --- |
| `python/tests/test_mamba_ops.py` | 对 `mamba_ops.py` 中各个 golden 函数做独立验证：固定点运算、单 token Mamba/Attention/MoE 步骤、两 token 历史、saturation/zero 边界情况，共 28 个测试。 |
| `python/tests/test_quantization_sweep.py` | 验证 `quantized_mamba_step` / `quantized_attention_step` 在 INT4/INT6/INT8 精度下行为正确，确认 mul-proxy 精度不变性的数学基础。 |

---

## scripts/

| 文件 | 作用 |
| --- | --- |
| `scripts/resource_reuse_analysis.sh` | 生成四层硬件映射资源对比报告（Baseline/SharedFabric/SemanticSerial/UnifiedSerial），含 instance-weighted mul-proxy 计算。 |
| `scripts/scale_analysis.sh` | 生成 context length / layer count 规模分析报告。 |
| `scripts/optimization_sweep.sh` | 生成 M8-O projectionMacLanes=1/2/4 资源-延迟 Pareto 分析报告。 |
| `scripts/sparsification_analysis.sh` | 生成零跳过稀疏化（zeroSkip）结构代理对比报告。 |
| `scripts/latency_model.py` | **M12-C 解析延迟模型**：实现 T_scheduler/T_mamba/T_attention 闭式公式；对 5 个实测数据点（SerialSharedLinear4、ConfigurableSerialLinear4 M1/M2/M4、Mamba token、Attention token）做验证；输出延迟投影表和各里程碑节省量汇总。 |

