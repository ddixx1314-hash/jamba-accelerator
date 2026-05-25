# 关键模块实现说明

> 本文档对项目中各关键算子和模块的实现方式、设计选择和近似方法作详细说明。
> 对应代码位于 `src/main/scala/jamba/`。

---

## 一、基础计算单元

### MacLane（共享 MAC 单元）

**作用**：整个项目的原子计算单元，计算 `accOut = a * b + accIn`。

**实现**：纯组合逻辑，无寄存器，由上层 FSM 控制何时采样结果。

**参数**：
- `dataWidth`：操作数位宽（默认 INT8）
- `accWidth`：累加器位宽（默认 INT32，防止溢出）
- `zeroSkip`：为 `true` 时，若 `a == 0` 或 `b == 0` 则直接透传 `accIn`，跳过乘法，用于减少动态功耗

**MacLaneMixed**：支持宽窄混合位宽（如 INT32 状态 × INT8 系数），专用于 SSM 状态更新。

---

## 二、Linear 投影

### SerialSharedLinear4（串行矩阵向量乘）

**作用**：用 1 个 MacLane 串行完成 4×4 矩阵向量乘，计算 `y(row) = bias(row) + Σ weight(row)(col) * x(col)`。

**实现**：
- 输入 `x` 和权重矩阵在 `start` 时锁存到寄存器
- FSM 按行列顺序遍历，每拍完成 1 次 MAC
- 4×4 矩阵共 16 次 MAC，总延迟 **16 个时钟周期**
- `done` 信号在最后一次 MAC 完成后拉高

**接口**：`start / busy / done` 握手协议，支持背压。

### ConfigurableSerialLinear4（可配置 MAC 并行度）

**作用**：在 `SerialSharedLinear4` 基础上，支持 `macLanes=1/2/4` 个 MAC lane 并行，用于资源-延迟 Pareto 分析。

**实现**：
- `macLanes` 个 MacLane 每拍同时计算相邻列，结果由规约加法树合并
- 延迟随并行度线性缩短：macLanes=1 → 16 周期，macLanes=2 → 8 周期，macLanes=4 → 4 周期
- bias 只在每行最后一次规约后加一次，避免重复累加

**Pareto 结果（Context=8，4 层）**：

| macLanes | 实例加权 Mul-proxy | Token 估算周期 |
|:---:|:---:|:---:|
| 1 | 92 | 564 |
| 2 | 94（+2.2%） | 364（−35%） |
| 4 | 98（+6.5%） | 264（−53%） |

---

## 三、统一投影调度器

### UnifiedProjectionScheduler4

**作用**：用一个槽表 FSM 统一调度 Jamba2 层内全部 10 次 Linear 投影（Mamba×3 + Attention×4 + MLP/MoE×3），共享同一个 `ConfigurableSerialLinear4` 实例。

**实现**：
- 槽表（`Seq`）定义每个投影槽位的名称、权重来源和结果寄存器
- FSM 按槽位顺序依次触发投影，等待 `done` 后推进到下一槽
- 每个投影结果保存在对应的输出寄存器，供后续专用单元使用
- 支持按层类型（Mamba / Attention / MoE）跳过不需要的槽位

**层级 Mul-proxy 对比**：

| 方案 | 层级 Mul-proxy |
|------|:---:|
| Baseline（各算子独立） | 96 |
| SharedFabric | 69 |
| SemanticSerial | 42 |
| UnifiedSerial（本方案） | 50 |

> UnifiedSerial 略高于 SemanticSerial（50 > 42）是因为 Mamba 和 Attention 的投影槽位在结构上同时展开，FIRRTL 不会根据运行时层类型消除未使用的槽位。

---

## 四、因果卷积

### SerialCausalConvMini（串行因果卷积）

**作用**：多 lane、多 tap 的逐通道因果卷积，计算 `y(lane) = Σ kernel(tap) * history(tap)(lane)`。

**实现**：
- 历史 token 保存在移位寄存器（`historyReg`），每次新 token 到来时移位
- 用 1 个 MacLane 串行遍历所有 `lane × tap` 组合
- 支持 `loadHistory` / `historyIn` 端口，由上层 Tile 在层切换时注入恢复的历史状态
- 总延迟：`lanes × taps` 个时钟周期

**状态持久性**：`historyReg` 跨 token 保持（记录前 `taps-1` 个 token 的输入），是 M7-B 状态虚拟化保存的内容之一。

---

## 五、选择性扫描（SSM）

### SerialSelectiveScanMini（串行 SSM）

**作用**：实现 Mamba 的递归状态更新，每个 token 三步完成：
1. `state := A * state + B * x`（recurrent）
2. `y := C * state`（output gate）

**实现**：
- 用 1 个 MacLaneMixed（宽 state × 窄系数）串行处理所有 lane
- 每个 lane 独立维护一个 `stateReg`（INT32），跨 token 保持
- 支持 `loadState` / `stateIn` 端口，用于 M7-B 状态虚拟化的 save/restore

**MacLaneMixed 的必要性**：SSM 状态累加后位宽会增长（INT32），但 A/B/C 系数是 INT8，若用统一 INT8 MAC 会截断状态精度，因此需要混合位宽 MAC。

---

## 六、注意力

### AttentionDecodeTiny（注意力解码）

**作用**：decode 阶段单 query token 的注意力计算（不含 softmax）。

**实现**：
- `score(t) = Q · K[t]`：对 KV cache 中每个历史 token 做点积
- `y = Σ score(t) * V[t]`：加权求和 value
- 近似：**用移位代替 softmax**（`score >> normShift`），避免指数运算和除法

**KV Cache**：环形寄存器阵列，大小为 `contextLength × lanes`，跨 token 保持，是 M7-B 状态虚拟化保存的内容之一。

---

## 七、RMSNorm 近似

### RmsNormApprox

**标准 RMSNorm**：
$$y_i = \frac{x_i \cdot w_i}{\sqrt{\frac{1}{N}\sum_{j} x_j^2}}$$

分母需要计算平方根，硬件实现成本高。

**本实现的近似**：
$$y_i = \frac{x_i \cdot w_i}{\frac{1}{N}\sum_{j} x_j^2}$$

用**均方值**（mean square）直接代替均方根，省去开根号电路。

**实现步骤**（`RmsNormStats` + `RmsNormApprox`）：
1. 计算 `sumSquares = Σ x_j²`（并行组合逻辑）
2. 整除得 `meanSquare = sumSquares / N`
3. 输出 `y_i = (x_i * weight_i) / meanSquare`，结果截断到 `dataWidth`
4. 零值保护：`meanSquare == 0` 时分母替换为 1，避免除以零

**近似误差说明**：均方值与均方根之间差一个开根号，输出的绝对数值尺度会发生变化，但在整数量化域内归一化的相对效果仍然保留。后续如有精度要求可替换为查表法或迭代开方。

---

## 八、MoE-lite

### UnifiedMoEPathMini（统一调度 MoE 路径）

**作用**：top-1 专家路由 + 单专家 MLP，通过统一投影调度器复用投影 MAC。

**实现**：
- `RouterMini`：对每个 expert 计算 `score = gate_weight · x` 的点积，选分数最大的 expert（平分时选 expert 0）
- 路由结果确定后，对选中 expert 的 gate/up/down 投影通过 `UnifiedProjectionScheduler4` 串行完成
- 未选中的 expert 投影槽位被跳过

---

## 九、多层状态虚拟化

### SinglePhysicalLayerTile（M7-A/B）

**作用**：用 1 个物理 `UnifiedJamba2MiniLayer` 服务 L 个逻辑层，实例加权 Mul-proxy 从 O(L) 降至接近 O(1)。

**需要虚拟化的状态**：

| 状态 | 模块 | 大小 |
|------|------|------|
| SSM 隐层状态 | `SerialSelectiveScanMini` | `lanes × stateWidth` |
| 因果卷积历史 | `SerialCausalConvMini` | `(taps-1) × lanes × dataWidth` |
| KV Cache | `AttentionDecodeTiny` | `contextLength × lanes × dataWidth` |

**save/restore FSM**：
- 每个逻辑层开始前（`restoreState` 阶段）：从 per-layer 状态寄存器文件读出对应层的状态，通过 `loadState` / `loadHistory` 端口注入物理层
- 当前逻辑层处理完成后（`saveState` 阶段）：从物理层读出状态，写回对应层的状态寄存器文件

**资源对比**（实例加权 Mul-proxy）：

| 配置 | UnifiedFullTile（L 实例） | SinglePhysicalTile（1 实例） |
|------|:---:|:---:|
| 2 层 Context=8 | 184 | 92 |
| 4 层 Context=8 | 368 | 92 |
| 8 层 Context=16 | 1248 | 156 |

---

## 十、权重存储

### LayeredWeightStoreMini + SequentialWeightLoaderMini

**作用**：片上寄存器文件式权重存储，支持按层、按字段读写。

**地址空间**：`layer_idx × field_type × element_offset`，由 `WeightAddressGenMini` 将 `(layer, field, row/col/tap/expert)` 映射为扁平地址，layer stride = 512。

**加载方式**：`SequentialWeightLoaderMini` 提供 BRAM 风格的顺序地址遍历接口，逐字段逐元素写入，方便后续替换为真实 BRAM 读接口（M8-BRAM）。

**当前局限**：权重存储在寄存器文件（而非 BRAM），每个权重元素占用一个寄存器，IO 端口数量随层数和字段数线性增长。替换为 `SyncReadMem`（BRAM 推断）是下一阶段工作（M8-BRAM）。

---

*最后更新：2026-05-25*
