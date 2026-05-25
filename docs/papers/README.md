# Paper Library

This directory keeps the local paper/reference copies used by the Jamba2 Mini accelerator project.

## Core Papers

| Topic | Local file | Source |
| --- | --- | --- |
| Transformer attention | [attention_is_all_you_need_1706.03762.pdf](attention_is_all_you_need_1706.03762.pdf) | <https://arxiv.org/abs/1706.03762> |
| Mamba / selective SSM | [mamba_selective_ssm_2312.00752.pdf](mamba_selective_ssm_2312.00752.pdf) | <https://arxiv.org/abs/2312.00752> |
| Jamba hybrid Transformer-Mamba-MoE | [jamba_hybrid_transformer_mamba_2403.19887.pdf](jamba_hybrid_transformer_mamba_2403.19887.pdf) | <https://arxiv.org/abs/2403.19887> |
| Jamba 2 (MoE + GQA) | [jamba2_moe_hybrid_2408.12570.pdf](jamba2_moe_hybrid_2408.12570.pdf) | <https://arxiv.org/abs/2408.12570> |
| Samba hybrid Mamba + sliding-window attention | [samba_hybrid_ssm_swa_2406.07522.pdf](samba_hybrid_ssm_swa_2406.07522.pdf) | <https://arxiv.org/abs/2406.07522> |
| S4 structured state spaces | [s4_structured_state_spaces_2111.00396.pdf](s4_structured_state_spaces_2111.00396.pdf) | <https://arxiv.org/abs/2111.00396> |
| FlashAttention | [flashattention_2205.14135.pdf](flashattention_2205.14135.pdf) | <https://arxiv.org/abs/2205.14135> |
| FlashAttention-2 | [flashattention2_2307.08691.pdf](flashattention2_2307.08691.pdf) | <https://arxiv.org/abs/2307.08691> |

## FPGA 加速器论文（Baseline 候选）

| Topic | Local file | Source |
| --- | --- | --- |
| LightMamba：Mamba FPGA 加速器 + 量化协同设计 | [lightmamba_fpga_2502.15260.pdf](lightmamba_fpga_2502.15260.pdf) | <https://arxiv.org/abs/2502.15260> |
| FastMamba：Mamba2 FPGA 加速器（IEEE FPGA 2025） | [fastmamba_fpga_2505.18975.pdf](fastmamba_fpga_2505.18975.pdf) | <https://arxiv.org/abs/2505.18975> |

## Official Model References

| Topic | Local file | Source |
| --- | --- | --- |
| Jamba2 Mini model card | [official/ai21_jamba2_mini_model_card.md](official/ai21_jamba2_mini_model_card.md) | <https://huggingface.co/ai21labs/AI21-Jamba2-Mini> |

## Project Use

- **模型理解**：Jamba/Jamba2 定义完整的 mini 模型流程；Mamba、S4、Samba 支撑 SSM/Attention 混合路径和选择性扫描硬件设计
- **Baseline 对比**：LightMamba 和 FastMamba 作为 Mamba FPGA 加速器的外部 baseline，对比资源（LUT/DSP/BRAM）和延迟指标
- **技术参考**：FlashAttention 用于讨论注意力打分/值累加和内存感知调度
- 本目录作为本地阅读缓存；论文和报告中引用原始 arxiv URL
