# Paper Library

This directory keeps the local paper/reference copies used by the Jamba2 Mini accelerator project.

## Core Papers

| Topic | Local file | Source |
| --- | --- | --- |
| Transformer attention | [attention_is_all_you_need_1706.03762.pdf](attention_is_all_you_need_1706.03762.pdf) | <https://arxiv.org/abs/1706.03762> |
| Mamba / selective SSM | [mamba_selective_ssm_2312.00752.pdf](mamba_selective_ssm_2312.00752.pdf) | <https://arxiv.org/abs/2312.00752> |
| Jamba hybrid Transformer-Mamba-MoE | [jamba_hybrid_transformer_mamba_2403.19887.pdf](jamba_hybrid_transformer_mamba_2403.19887.pdf) | <https://arxiv.org/abs/2403.19887> |
| Samba hybrid Mamba + sliding-window attention | [samba_hybrid_ssm_swa_2406.07522.pdf](samba_hybrid_ssm_swa_2406.07522.pdf) | <https://arxiv.org/abs/2406.07522> |
| S4 structured state spaces | [s4_structured_state_spaces_2111.00396.pdf](s4_structured_state_spaces_2111.00396.pdf) | <https://arxiv.org/abs/2111.00396> |
| FlashAttention | [flashattention_2205.14135.pdf](flashattention_2205.14135.pdf) | <https://arxiv.org/abs/2205.14135> |
| FlashAttention-2 | [flashattention2_2307.08691.pdf](flashattention2_2307.08691.pdf) | <https://arxiv.org/abs/2307.08691> |

## Official Model References

| Topic | Local file | Source |
| --- | --- | --- |
| Jamba2 Mini model card | [official/ai21_jamba2_mini_model_card.md](official/ai21_jamba2_mini_model_card.md) | <https://huggingface.co/ai21labs/AI21-Jamba2-Mini> |

## Project Use

- Use the Jamba/Jamba2 references to define the complete mini model flow.
- Use Mamba, S4, and Samba to justify the SSM/attention hybrid path and selective-scan hardware.
- Use Transformer and FlashAttention to discuss attention score/value accumulation and memory-aware attention scheduling.
- Use this directory as a local reading cache; cite the original URLs in papers and reports.
