---
license: apache-2.0
pipeline_tag: text-generation
library_name: transformers
---

# Introduction

Jamba2 Mini is an open source small language model built for enterprise reliability. With 12B active parameters (52B total), it delivers precise question answering without the computational overhead of reasoning models. The model's SSM-Transformer architecture provides a memory-efficient solution for production agent stacks where consistent, grounded outputs are critical.

Released under Apache 2.0 License with a 256K context window, Jamba2 Mini is designed for enterprise workflows that demand accuracy and steerability. For more details, read the [full release blog post](https://www.ai21.com/blog/introducing-jamba2/).

# Key Advantages
* **Superior reliability-to-throughput ratio:** Maintains high performance at 100K+ token contexts
* **Category-leading benchmarks:** Excels on IFBench, IFEval, Collie, and FACTS
* **Statistically significant quality wins:** Outperforms comparable models on real-world enterprise tasks
* **256K context window:** Processes technical manuals, research papers, and knowledge bases
* **Apache 2.0 License:** Fully open source for commercial use
* **Production-optimized:** Lean memory footprint for scalable deployments

# Evaluation Results
Jamba2 Mini leads on instruction following and grounding metrics, demonstrating exceptional steerability and context faithfulness. In blind side-by-side evaluations on 100 real-world enterprise prompts, the model achieved statistically significant wins on output quality and factuality compared to Ministral3 14B.

<img src="https://huggingface.co/ai21labs/AI21-Jamba2-Mini/resolve/main/assets/Enterprise%20Reliability%20Benchmarks%20for%20Mini%20Models.png" width="900"/>

# Training and Evaluation Details
Jamba2 models were developed using a comprehensive post-training pipeline starting from Jamba 1.5 pre-training. The models underwent mid-training on 500B carefully curated tokens with increased representation of math, code, high-quality web data, and long documents. A state passing phase optimized the Mamba layers for effective context length generalization. Training continued with cold start supervised fine-tuning to establish instruction-following and reasoning capabilities, followed by DPO optimization.

The final training stages involved multiple on-policy reinforcement learning phases, progressively moving from short-context verifiable rewards to longer context training with mixed verifiable and model-based rewards. Evaluation focused on two key enterprise reliability signals: instruction-following benchmarks measuring steerability, and grounding benchmarks testing context faithfulness. Human evaluators assessed performance on real-world enterprise tasks using blind, counterbalanced side-by-side comparisons, rating outputs on factuality, style, constraint-adherence, instruction-following, and helpfulness.


# Quickstart
## Run with vLLM
Best results require vLLM version **0.12.0** or higher.

```
vllm serve "ai21labs/AI21-Jamba2-Mini" --mamba-ssm-cache-dtype float32 --enable-auto-tool-choice --tool-call-parser hermes --enable-prefix-caching --quantization experts_int8
```


## Run with Transformers

```
pip install transformers>=4.54.0
pip install flash-attn --no-build-isolation
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("ai21labs/AI21-Jamba2-Mini",
                                  dtype=torch.bfloat16,
attn_implementation="flash_attention_2", device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("ai21labs/AI21-Jamba2-Mini")

messages = [
    {"role": "system",
     "content": "You are an HR Policy Assistant.
                 Answer employee questions using only the provided policy documents.
                 If the answer isn't in the documents, say so clearly.
                 Be concise and cite the specific policy section when possible."
},
    {"role": "user",
     "content": "Context documents: {retrieved_chunks}.
                 Employee question: {user_question}.
                 Answer:"
},
]

prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

outputs = model.generate(**tokenizer(prompts, return_tensors="pt").to(model.device), do_sample=True, temperature=0.6)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

For more deployment guides and resources, visit our [official documentation](https://docs.ai21.com/home).
