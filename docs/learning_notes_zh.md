# Jamba 加速器中文学习笔记

这个仓库是一个用 Chisel 学习 Mamba/Jamba-like 硬件加速器的小型项目。它不是生产级 Jamba 2，也不能直接跑真实大模型权重。它的目标是把核心硬件概念拆成小模块：每个模块都能读懂、能测试、能生成 SystemVerilog。

当前采用的简化约定：

- 输入数据主要是 `SInt(8.W)`。
- 累加结果主要是 `SInt(32.W)`。
- 向量长度固定为 4，方便手算和写测试。
- 权重通过 IO (Input/Output，输入/输出) 端口输入，不实现 SRAM (Static Random-Access Memory，静态随机存取存储器)、DDR (Double Data Rate，双倍数据率内存) 或 AXI (Advanced eXtensible Interface，高级可扩展接口)。
- attention (注意力机制) 是 tiny decode datapath，不包含真实 softmax (归一化指数函数)。

## 背景概念：Mamba、SSM、Jamba 是什么

Mamba 是一种序列模型结构。它的目标之一是降低长序列处理成本。普通 Transformer 的 attention 往往需要让 token 之间两两交互，序列越长，计算和显存压力越大。Mamba 走的是另一条路：用 SSM (State Space Model，状态空间模型) 来维护一个会随时间更新的内部状态。

一个非常简化的 SSM 状态更新可以写成：

$$
h_{t+1} = A h_t + B x_t
$$

输出可以写成：

$$
y_t = C h_t
$$

这里 $x_t$ 是当前输入 token，$h_t$ 是当前内部状态，$h_{t+1}$ 是下一拍状态，$y_t$ 是当前输出。你可以把它想成“带记忆的状态机”：每来一个 token，就更新一次内部状态，再根据状态产生输出。

Selective Scan (选择性扫描) 是 Mamba 里很关键的思想。它不是使用一组完全固定的参数处理所有 token，而是让某些参数可以跟输入相关，从而更灵活地选择哪些信息进入状态、哪些信息被保留或遗忘。本项目里的 `SelectiveScanTiny` 是一个非常简化的版本：它只保留“状态更新 + gate”这个核心形状。

Jamba 是一种混合思路：把 Mamba/SSM 路径和 attention 路径放在同一个模型结构里。Mamba 路径擅长高效处理长序列状态，attention 路径擅长直接查找和组合上下文信息。本项目里的 `TinyJambaBlock` 就是这种思想的 mini 原型：它有一条 Mamba-like path，也有一条可选的 tiny attention path。

本项目里的对应关系：

- `MambaStateUpdate`：简化版状态更新。
- `SelectiveScanTiny`：简化版选择性扫描。
- `TinyMambaBlock`：小型 Mamba-like block。
- `AttentionDecodeTiny`：小型 attention decode path。
- `TinyJambaBlock`：Mamba path 和 attention path 的混合。
- `Jamba2MiniCore`：把 norm、projection、Mamba/attention block 和 output projection 串起来的 mini core。

## 整体框架：为什么需要这些算子

如果只看单个模块，会觉得它们像一堆分散的小算子：点积、矩阵乘、RMSNorm、卷积、状态更新、attention。其实它们在一个 mini Jamba block 里各自有位置，可以按“一个 token 怎么被处理”来理解。

第一步是输入 token。这里的 token 不是文字本身，而是一个向量，例如本项目里简化成 4 个 `SInt(8.W)` 数字。硬件看到的是：

$$
x = [x_0, x_1, x_2, x_3]
$$

第二步是归一化，也就是 `RmsNormApprox`。模型里不同 token 的数值幅度可能差很多，归一化的作用是先把输入缩放到比较稳定的范围，后面的投影和状态更新更容易工作。本项目用整数近似版 RMSNorm，只是为了先搭出硬件结构。

第三步是线性投影，也就是 `Linear4`。一个输入向量通常不会直接送进 Mamba 或 attention，而是先通过几组权重矩阵变成不同用途的向量：

- input projection：产生送入主计算路径的 token 表示。
- gate projection：产生 gate，用来控制哪些信息通过。
- B projection：产生或近似产生 SSM 输入相关参数。
- C projection：产生或近似产生 SSM 输出相关参数。
- output projection：把 block 内部结果变回输出向量。

所以 `DotProduct`、`SmallGemm4x4`、`Linear4` 不是孤立的数学练习，它们是在为“神经网络里的线性层/投影层”服务。

第四步是 Mamba path。它的目标是用状态来记住历史 token 信息：

```text
当前 token -> CausalConv1D -> MambaStateUpdate / SelectiveScanTiny -> Mamba 输出
```

`CausalConv1D` 负责看当前 token 和前几个 token 的局部关系。它像一个很小的滑动窗口，只看过去，不看未来，所以叫 causal。`MambaStateUpdate` 负责更新内部状态。`SelectiveScanTiny` 在状态更新后加 gate，模拟 Mamba 里“选择性保留/写入信息”的感觉。

第五步是 attention path。Mamba path 擅长用状态压缩历史，但有些场景下模型也需要直接从上下文里取信息。attention path 的作用就是用 query 和 keys 算相似度，再用这个相似度加权 values。本项目的 `AttentionDecodeTiny` 只保留最小形状：

$$
s_r = q \cdot k_r
$$

$$
y_c = \sum_r s_r v_{r,c}
$$

它没有 softmax、mask 或 scale，只是为了先理解 attention decode 的硬件数据流。

第六步是混合，也就是 `TinyJambaBlock`。如果 `useAttention = 0`，输出主要来自 Mamba path；如果 `useAttention = 1`，就把 Mamba path 和 attention path 加在一起。这样就形成了一个 tiny Jamba-like block。

第七步是顶层封装。`Jamba2MiniCore` 把前面的 norm、projection、Mamba/attention block、output projection 串起来，形成完整 mini datapath。`Jamba2MiniStream` 再给它加 `valid/ready` 握手，让它更像真实硬件系统里的一个可连接模块。

可以把整个项目的层次记成这样：

```text
基础硬件积木:
  Counter, PE, DotProduct, VectorOps, SmallGemm4x4

神经网络常用算子:
  RmsNormStats, RmsNormApprox, Linear4

Mamba/SSM 路径:
  CausalConv1D, MambaStateUpdate, SelectiveScanTiny, TinyMambaBlock

Attention 路径:
  AttentionDecodeTiny

Jamba-like 混合模块:
  TinyJambaBlock, Jamba2MiniCore

系统接口包装:
  Jamba2MiniAccelerator, Jamba2MiniStream
```

## 当前整体结构

当前最完整的数据通路顶层是 `Jamba2MiniCore`。

```text
x
 -> RmsNormApprox
 -> Linear4 input/gate/B/C projections
 -> TinyJambaBlock
      -> TinyMambaBlock
           -> CausalConv1D
           -> SelectiveScanTiny
      -> AttentionDecodeTiny optional path
 -> Linear4 output projection
 -> y
```

`Jamba2MiniStream` 在 `Jamba2MiniCore` 外面加了一层 `valid/ready` 流式接口。这样它更像一个真实硬件 block：上游送 token，下游准备好之后接收输出。

常用命令：

```bash
sbt test
./scripts/run_test.sh
./scripts/generate_verilog.sh
```

`./scripts/run_test.sh` 会依次运行 Chisel 测试、Python golden model 测试、SystemVerilog 生成和 Verilator lint。

## Counter

含义：Counter，计数器。

### 功能
`Counter` 是一个带 `en` 使能信号的无符号计数器。`en` 为真时，每个时钟周期加 1；`en` 为假时保持原值。

### 为什么需要
计数器是硬件控制逻辑的基础。后续加速器中会用计数器记录 token 位置、循环次数、pipeline 拍数、地址偏移等。

### Chisel 写法
`RegInit(0.U(width.W))` 表示一个复位值为 0 的寄存器。`when(io.en)` 表示只有使能有效时才更新寄存器。

### Verilog 对应
它类似：

```verilog
always_ff @(posedge clock) begin
  if (reset) cnt <= 0;
  else if (en) cnt <= cnt + 1;
end
assign out = cnt;
```

### 易错点
- `cnt` 是寄存器，只有时钟沿之后才变化。
- 固定位宽计数器溢出后会回绕。
- `width` 不能是 0。

## PE

含义：PE (Processing Element，处理单元)。

### 功能
`PE` 是一个组合逻辑乘加单元：

$$
y = a \times b + c
$$

其中 $c$ 对应代码里的 `acc_in`，$y$ 对应 `acc_out`。

### 为什么需要
乘加是神经网络加速器中最常见的操作。点积、矩阵乘、线性层、attention score 都可以拆成很多乘加。

### Chisel 写法
`a` 和 `b` 是 `SInt(dataWidth.W)`，`acc_in` 和 `acc_out` 是 `SInt(accWidth.W)`。这样输入位宽和累加位宽分开，避免乘法结果太早被截断。

### Verilog 对应
它相当于一个 signed multiplier 后面接 signed adder：

```verilog
assign acc_out = a * b + acc_in;
assign valid = 1'b1;
```

### 易错点
- `PE` 没有寄存器，所以 `valid` 只是说明组合输出有效，不是 pipeline done。
- `accWidth` 必须足够大，否则乘法结果会被截断。
- signed 和 unsigned 不要混用。

## DotProduct

含义：Dot Product，点积。

### 功能
`DotProduct` 计算两个向量的有符号点积：

$$
y = \sum_i a_i b_i
$$

### 为什么需要
点积是矩阵乘、attention score、线性投影的基础。

### Chisel 写法
`Vec(length, SInt(...))` 表示多 lane 输入。Scala `for` 循环在这里不是运行时循环，而是生成多个硬件乘法器。`reduce(_ +& _)` 生成加法归约网络。

### Verilog 对应
它会变成多个乘法器加一个加法树。每个 `a(i) * b(i)` 都是并行硬件。

### 易错点
- Chisel 的 `for` 是 elaboration-time 生成硬件，不是电路运行时循环。
- 乘法结果位宽通常比输入大。
- 组合逻辑测试不一定需要 `clock.step()`。

## SmallGemm4x4

含义：Small GEMM 4x4，其中 GEMM (General Matrix Multiply，通用矩阵乘法)，这里指小型 4x4 矩阵乘法。

### 功能
`SmallGemm4x4` 计算 4x4 矩阵乘法：

$$
c_{row,col} = \sum_k a_{row,k} b_{k,col}
$$

### 为什么需要
矩阵乘是线性层和投影层的核心。这个模块是从点积走向 GEMM (General Matrix Multiply，通用矩阵乘法) 的小练习。

### Chisel 写法
使用嵌套 `Vec` 表示二维矩阵，使用嵌套 `for` 生成 16 个输出元素的乘加逻辑。

### Verilog 对应
等价于手写 16 个 dot-product datapath。

### 易错点
- `b` 的索引是 `b(k)(col)`，不是 `b(row)(col)`。
- 四个乘积相加需要更宽的累加器。
- 测试里的 expected value 要按 row-column 手算。

## VectorOps

含义：Vector Operations，向量操作。

### 功能
`VectorOps` 做逐元素向量操作：加、减、乘、ReLU。

### 为什么需要
神经网络 block 里经常需要 residual、gate、activation 等逐元素操作。

### Chisel 写法
每个 lane 独立赋值。ReLU 使用 `Mux(io.a(i) < 0.S, 0.S, io.a(i))`。

### Verilog 对应
每个 lane 都会生成自己的加法器、减法器、乘法器和 mux。

### 易错点
- 逐元素乘法不是点积。
- ReLU 是比较器加 mux。
- 乘法输出不要用太窄的位宽。

## SignedMath

含义：Signed Math，有符号数学工具。

### 功能
`SignedMath.resize` 用来统一处理 signed 数值的位宽转换。

### 为什么需要
项目里经常要把乘法或累加结果缩回某个固定宽度。集中成 helper 后，代码更容易看出哪里发生了位宽变化。

### Chisel 写法
创建一个目标位宽的 `Wire(SInt(width.W))`，然后把原值赋进去。

### Verilog 对应
根据目标位宽不同，综合后可能是符号扩展，也可能是截断。

### 易错点
- resize 不是饱和运算。
- 缩窄会丢高位。
- 不要把数值 resize 和 bit reinterpret 混为一谈。

## RmsNormStats

含义：RMSNorm Statistics，其中 RMSNorm (Root Mean Square Normalization，均方根归一化)，这里指 RMSNorm 统计量。

### 功能
计算 RMSNorm 的整数统计量：

$$
S = \sum_i x_i^2
$$

$$
M = \left\lfloor \frac{S}{N} \right\rfloor
$$

其中 $S$ 对应 `sumSquares`，$M$ 对应 `meanSquare`，$N$ 是向量长度。

### 为什么需要
RMSNorm 的第一步就是计算平方和或均方值。

### Chisel 写法
先为每个 lane 计算平方，再用 `reduce` 相加。

### Verilog 对应
多个平方器接加法树，再接一个常数除法器。

### 易错点
- 这不是完整 RMSNorm，只是统计部分。
- 负数平方后是正贡献。
- 平方和需要更宽的累加位宽。

## RmsNormApprox

含义：RMSNorm Approximation，其中 RMSNorm (Root Mean Square Normalization，均方根归一化)，这里指 RMSNorm 近似实现。

### 功能
用整数除法近似 RMSNorm：

$$
y_i = \frac{x_i \gamma_i}{M}
$$

其中 $\gamma_i$ 对应 `weight(i)`，$M$ 对应 `meanSquare`。

如果 `meanSquare` 是 0，则用 1 作为 denominator，避免除零。

### 为什么需要
真实 RMSNorm 需要平方根和倒数，这对早期学习项目太重。这里用整数近似先搭出 normalization stage。

### Chisel 写法
复用 `RmsNormStats`，然后每个 lane 做乘法和除法。

### Verilog 对应
统计模块输出 denominator，每个 lane 有乘法器和除法器。

### 易错点
- 它不是浮点 RMSNorm。
- 整数除法会截断。
- 全 0 输入必须避免除 0。

## Linear4

含义：Linear 4，4 路线性投影。

### 功能
4 输入、4 输出的线性投影：

$$
y_r = \beta_r + \sum_c W_{r,c} x_c
$$

其中 $W$ 是 `weight` 矩阵，$\beta$ 是 `bias` 向量。

### 为什么需要
Jamba/Mamba block 需要很多投影：输入投影、gate 投影、B/C 参数投影、输出投影等。

### Chisel 写法
每一行权重对应一个 dot product，再加 bias。

### Verilog 对应
每个输出 row 是 4 个乘法器、一个加法树、一个 bias adder。

### 易错点
- weight 的行列不要反。
- bias 应该使用 `accWidth`。
- 当前权重是 IO 输入，不是内部存储。

## MambaStateUpdate

含义：Mamba State Update，Mamba 状态更新。

### 功能
更新 SSM 状态：

$$
h_{t+1} = a h_t + b x_t
$$

其中 $h_t$ 是当前 state，$h_{t+1}$ 是下一拍 state。

### 为什么需要
这是 Mamba selective scan 的核心思想之一：用状态递推处理序列。

### Chisel 写法
`state` 是 `RegInit(VecInit(...))`。`clear` 优先清零，`en` 有效时更新。

### Verilog 对应
`state` 是一组寄存器，`clear` 是同步清零，`en` 是寄存器写使能。

### 易错点
- 新 state 要到时钟沿之后才可见。
- `clear` 优先级要高于 `en`。
- 递推乘法结果可能很宽。

## CausalConv1D

含义：1D Causal Convolution，其中 1D (One-Dimensional，一维)，这里指一维因果卷积。

### 功能
三 tap causal convolution：

$$
y_t = k_0 x_t + k_1 x_{t-1} + k_2 x_{t-2}
$$

每个 lane 独立计算。

### 为什么需要
Mamba block 通常会在 selective scan 前做局部 causal convolution。

### Chisel 写法
`delay1` 和 `delay2` 是保存历史 token 的寄存器。

### Verilog 对应
两个 delay register 形成一个小 shift register，三个 tap 进入乘加逻辑。

### 易错点
- tap 顺序不要反。
- delay register 在时钟沿更新。
- `en` 为假时 delay 应保持不变。

## SelectiveScanTiny

含义：Tiny Selective Scan，小型选择性扫描。

### 功能
封装状态更新，并对输出 state 做 gate：

$$
y_i = h_i g_i
$$

其中 $h_i$ 对应 `stateOut(i)`，$g_i$ 对应 `gate(i)`。

### 为什么需要
它是 tiny Mamba scan path 的简化版本。

### Chisel 写法
实例化 `MambaStateUpdate`，再对每个 lane 做 gate 乘法。

### Verilog 对应
一个子模块实例加上一组并行乘法器。

### 易错点
- `stateOut` 是寄存器输出，更新有一拍延迟。
- 不要混淆 raw state 和 gated output。
- gate 乘法也需要考虑位宽。

## AttentionDecodeTiny

含义：Tiny Attention Decode，小型注意力解码。

### 功能
简化 attention decode：

$$
s_r = q \cdot k_r
$$

$$
y_c = \sum_r s_r v_{r,c}
$$

其中 $k_r$ 是第 $r$ 行 key，$v_{r,c}$ 是第 $r$ 行 value 的第 $c$ 个元素。

不包含 softmax。

### 为什么需要
Jamba 的特点之一是混合 Mamba/SSM 和 attention。这个模块提供 tiny attention path。

### Chisel 写法
先用 dot product 计算每个 key 的 score，再用 score 对 value 加权求和。

### Verilog 对应
生成多组乘法器和加法器：一部分算 score，一部分算 weighted values。

### 易错点
- 它不是完整 attention。
- 没有 softmax、scale、mask。
- score 乘 value 时位宽会继续变大。

## TinyMambaBlock

含义：Tiny Mamba Block，小型 Mamba 模块。

### 功能
组合 causal conv、selective scan、state projection 和 residual gate。

### 为什么需要
这是项目里的 tiny Mamba-like block。

### Chisel 写法
实例化 `CausalConv1D` 和 `SelectiveScanTiny`，并用 `Wire(Vec(...))` 连接中间结果。

### Verilog 对应
多个子模块通过总线连接。

### 易错点
- conv 和 scan 都有状态，时序要按 clock edge 理解。
- 这不是生产级 Mamba 数值实现。
- 中间结果缩窄到 8-bit 会损失范围。

## TinyJambaBlock

含义：Tiny Jamba Block，小型 Jamba 模块。

### 功能
把 Mamba path 和可选 attention path 组合起来：

$$
y =
\begin{cases}
y_m + y_a, & u = 1 \\
y_m, & u = 0
\end{cases}
$$

其中 $y_m$ 是 Mamba path 输出，$y_a$ 是 attention path 输出，$u$ 对应 `useAttention`。

### 为什么需要
它体现了 Jamba-like 混合结构：SSM path 加 attention path。

### Chisel 写法
并行实例化 `TinyMambaBlock` 和 `AttentionDecodeTiny`，最后用 `Mux` 选择/相加。

### Verilog 对应
两个 path 都会存在于硬件里，`useAttention` 只是控制输出逻辑。

### 易错点
- `useAttention=false` 不代表 attention 硬件不存在。
- 两条路径都会被 elaborated。
- 测试时需要观察 state 和 attentionScores。

## Jamba2MiniAccelerator

含义：Jamba 2 Mini Accelerator，Jamba 2 小型加速器。

### 功能
早期顶层 wrapper，直接包住 `TinyJambaBlock`，提供 `en`、`clear`、`valid` 等信号。

### 为什么需要
这是第一个可运行的顶层模块，适合在项目早期验证 Mamba/attention 组合。

### Chisel 写法
主要是 IO 转接和一个 `validReg`。

### Verilog 对应
一个 top module，内部实例化 tiny block，并注册 valid。

### 易错点
- `valid` 是寄存器，`y` 是当前 datapath 输出。
- 它不是最终最完整顶层。
- 没有权重存储或流式接口。

## Jamba2MiniCore

含义：Jamba 2 Mini Core，Jamba 2 小型核心计算模块。

### 功能
更完整的 mini core：

$$
x \rightarrow \mathrm{Norm} \rightarrow \mathrm{Proj} \rightarrow \mathrm{JambaBlock} \rightarrow \mathrm{OutProj}
$$

### 为什么需要
它把 normalization、projection、Mamba/attention block、output projection 串成一个较完整的 mini accelerator datapath。

### Chisel 写法
实例化多个子模块，并通过 `Vec` IO 连接它们。保留 debug 输出，例如 `normMeanSquare`、`projectedX`、`blockY`、`stateOut`。

### Verilog 对应
生成一个较大的顶层 datapath，里面有多个子模块实例和很多向量端口。

### 易错点
- 权重仍然是 IO 输入，不是 SRAM。
- recurrent state 只有接受 token 后才更新。
- accumulator 输出缩回 8-bit 时可能截断。

## Jamba2MiniStream

含义：Jamba 2 Mini Stream，带流式握手接口的 Jamba 2 小型模块。

### 功能
给 `Jamba2MiniCore` 加一层 token-level valid/ready wrapper。

输入握手：

$$
f = v_{in} \land r_{in}
$$

其中 $f$ 对应 `fire`，$v_{in}$ 对应 `inValid`，$r_{in}$ 对应 `inReady`。

输出消费：

$$
c = v_{out} \land r_{out}
$$

其中 $c$ 对应 `willConsume`，$v_{out}$ 对应 `outValid`，$r_{out}$ 对应 `outReady`。

### 为什么需要
真实硬件模块通常不会只有一个 `en`，而是要能和上下游模块握手。这个 wrapper 是向系统级加速器迈出的一步。

### Chisel 写法
`outputValid` 是输出 buffer 是否有效。`outputReg` 保存输出数据。下游没准备好时，输出必须保持稳定。

### Verilog 对应
它会生成一个一 entry 的 skid-buffer 风格结构：valid 寄存器加 data 寄存器。

### 易错点
- `outValid` 高且 `outReady` 低时不能覆盖输出。
- 同周期 consume 和 accept 时可以替换 buffer。
- `clear` 时 `inReady` 应该拉低，避免误接收 token。

## GenerateVerilog 和脚本

含义：Generate Verilog，生成 Verilog/SystemVerilog。

### 功能
`GenerateVerilog` 生成四个顶层的 SystemVerilog：

- `JambaMiniTile`
- `Jamba2MiniAccelerator`
- `Jamba2MiniCore`
- `Jamba2MiniStream`

### 为什么需要
Chisel 只是硬件生成语言，最终还需要生成 Verilog/SystemVerilog 给后端工具使用。

### Chisel 写法
使用 `ChiselStage.emitSystemVerilogFile`。

### Verilog 对应
生成的 `.sv` 文件可以被 Verilator lint，也可以作为后续综合/仿真的输入。

### 易错点
- 不要手改 generated Verilog，要改 Chisel 源码后重新生成。
- `generated/verilog` 被 git 忽略，属于可再生文件。
- Verilator lint 通过不等于功能完全正确，还需要测试。

## 推荐学习顺序

如果你是从 Verilog 背景开始学 Chisel，建议按这个顺序读：

1. `Counter`
2. `PE`
3. `DotProduct`
4. `SmallGemm4x4`
5. `VectorOps`
6. `RmsNormStats`
7. `RmsNormApprox`
8. `Linear4`
9. `MambaStateUpdate`
10. `CausalConv1D`
11. `SelectiveScanTiny`
12. `AttentionDecodeTiny`
13. `TinyMambaBlock`
14. `TinyJambaBlock`
15. `Jamba2MiniCore`
16. `Jamba2MiniStream`

## 当前项目能做什么

当前项目可以：

- 运行所有 Chisel 单元测试。
- 运行 Python golden model 测试。
- 生成 SystemVerilog。
- 用 Verilator lint 检查生成的顶层。
- 演示一个 integer tiny Jamba-like accelerator datapath。
- 演示一个简单 token valid/ready wrapper。

当前项目还不能：

- 跑真实 Jamba 2 权重。
- 支持 BF16/FP16。
- 支持 AXI/DDR。
- 实现真实 softmax attention。
- 实现 MoE 或大上下文 KV cache。
- 直接作为完整 FPGA/ASIC 系统部署。
