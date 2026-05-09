# Swift-SVD

Swift-SVD is an activation-aware, training-free low-rank compression framework for large language models that combines theoretical optimality with practical efficiency, jointly reducing the memory footprint of static model weights and the KV cache.

## ✨ Introduction

[Swift-SVD: Theoretical Optimality Meets Practical Efficiency in Low-Rank LLM Compression](https://arxiv.org/abs/2604.01609v1)

International Conference on Machine Learning (ICML) 2026

---

## 🚀 Quick Start

### Full Pipeline

layer-importance -> SVD training -> uniform/adaptive rank allocation -> compression -> evaluation.

### Run

To simplify `ppl` evaluation, we use the reconstructed low-rank weight $W_k = A_k B_k$ to replace the original weight matrix. This is equivalent to applying the compressed weight, but avoids modifying the model-specific `forward` implementation during evaluation.

For deployment with explicit low-rank factors $A_k$ and $B_k$, users can adapt the `forward` pass for different model architectures accordingly.

```bash
# prepare environment
pip install -r requirements.txt

# calculate layer importance
cd llm-layer-importance
bash run_calculate_importance.sh
cd ..

# uniform compression
bash run_uniform.sh
# dynamic compression
bash run_dynamic.sh
```

---


## 📝 License

MIT License

---

## 📚 Citation

If you use this project, please cite:

```bibtex
@misc{qi2026swiftsvdtheoreticaloptimalitymeets,
      title={Swift-SVD: Theoretical Optimality Meets Practical Efficiency in Low-Rank LLM Compression},
      author={Ruoling Qi and Yirui Liu and Xuaner Wu and Xiangyu Wang and Ming Li and Chen Chen and Jian Chen and Yin Chen and Qizhen Weng},
      year={2026},
      eprint={2604.01609},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.01609},
}
```

---

## ✍️ Acknowledgement

This repository builds upon several excellent open-source projects. We sincerely thank the authors of [SVD-LLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM), [ShortGPT](https://github.com/sramshetty/ShortGPT/tree/hf-models)&[llm-layer-importance](https://github.com/Hik289/llm-layer-importance), and [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for their foundational contributions.

Specifically, our model-specific `forward` implementation is adapted from **SVD-LLM**, the layer-importance computation is based on **ShortGPT**, and the accuracy evaluation is conducted using **lm-evaluation-harness**.

We are grateful to these wonderful and inspiring open-source projects.

---