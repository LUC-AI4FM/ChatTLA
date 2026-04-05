---
library_name: transformers
model_name: checkpoints_kto
tags:
- generated_from_trainer
- kto
- trl
licence: license
---

# Model Card for checkpoints_kto

This model is a fine-tuned version of [None](https://huggingface.co/None).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 



This model was trained with KTO, a method introduced in [KTO: Model Alignment as Prospect Theoretic Optimization](https://huggingface.co/papers/2402.01306).

### Framework versions

- TRL: 0.29.1
- Transformers: 5.3.0
- Pytorch: 2.10.0
- Datasets: 4.8.3
- Tokenizers: 0.22.2

## Citations

Cite KTO as:

```bibtex
@article{ethayarajh2024kto,
    title        = {{KTO: Model Alignment as Prospect Theoretic Optimization}},
    author       = {Kawin Ethayarajh and Winnie Xu and Niklas Muennighoff and Dan Jurafsky and Douwe Kiela},
    year         = 2024,
    eprint       = {arXiv:2402.01306},
}
```

Cite TRL as:
    
```bibtex
@software{vonwerra2020trl,
  title   = {{TRL: Transformers Reinforcement Learning}},
  author  = {von Werra, Leandro and Belkada, Younes and Tunstall, Lewis and Beeching, Edward and Thrush, Tristan and Lambert, Nathan and Huang, Shengyi and Rasul, Kashif and Gallouédec, Quentin},
  license = {Apache-2.0},
  url     = {https://github.com/huggingface/trl},
  year    = {2020}
}
```