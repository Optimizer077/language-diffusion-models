<h1 align="center">🌀 Language Diffusion Models — from scratch</h1>

<p align="center">
  <b>A hands-on, build-it-yourself course on diffusion models for text.</b><br>
  Train a real (tiny) language diffusion model in seconds — every line of math tied to the paper it comes from.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CPU%20or%20CUDA-ee4c2c">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/Lessons%201--5-complete-success">
</p>

---

## Why this course?

Most explanations of language diffusion are either a wall of equations or a black-box library call. This course sits in between: **small, runnable notebooks that build each idea from scratch**, so you can *see* the forward process corrupt text, *watch* the model denoise it back, and *read* the exact equation each block of code implements.

> **Autoregressive (GPT-style) models** write left-to-right, one token at a time.
> **Diffusion models** start from noise and refine the *whole sequence in parallel* over a chosen number of steps — giving native infilling/editing, controllable compute, and a different scaling story. Recent work ([LLaDA, 2025](https://arxiv.org/abs/2502.09992)) shows the approach is competitive with autoregressive LLMs.

You'll build the masking models behind **MDLM** and **LLaDA**, the general **D3PM** framework, embedding-space diffusion (**Diffusion-LM**), a scaled-up conditional diffusion LLM, and score-entropy diffusion (**SEDD**) — all from scratch.

## Quickstart

```bash
# 1. Install (CPU build shown; for GPU use the matching CUDA wheel from pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 2. Open the first lesson
jupyter notebook 01_foundations_masked_diffusion.ipynb
```

The models are intentionally tiny — everything runs on a **laptop CPU in a minute
or two**. The notebooks auto-detect a GPU (`device = "cuda" if torch.cuda.is_available() else "cpu"`) and use it when present.

## The lessons

| # | Notebook | What you build | Papers |
|---|----------|----------------|--------|
| **1** | [Foundations: Masked Diffusion](01_foundations_masked_diffusion.ipynb) | A working masked diffusion LM — forward masking, bidirectional denoiser, ELBO→weighted-CE, sampling, **infilling** | MDLM, LLaDA |
| **2** | [D3PM: General Transition Kernels](02_d3pm_transition_kernels.ipynb) | **Uniform vs. absorbing** kernels with validated closed forms, the term-by-term ELBO, the hybrid loss, a kernel bake-off | D3PM |
| **3** | [Diffusion-LM: Continuous Diffusion](03_diffusion_lm_embedding_space.ipynb) | Gaussian diffusion in **embedding space** + rounding; DDPM sampling; **gradient-guided controllable generation** | Diffusion-LM, DDPM |
| **4** | [Making Masked Diffusion Practical](04_scaling_diffusion_llm.ipynb) | LLaDA-style: **generalization** (grammar corpus), **classifier-free guidance**, **confidence-based sampling** + the steps↔quality trade-off | LLaDA, MaskGIT |
| **5** | [SEDD: Score-Entropy Diffusion](05_sedd_score_entropy.ipynb) | Learn the distribution's **ratios** (concrete score) via score-entropy; the learned score is **verified** to match the true ratio's *shape* (corr ≈ 0.99) | SEDD |

📚 **See [`CONTENT.md`](CONTENT.md)** for the full syllabus, learning path, and concept map.
🧮 **See [`reference/D3PM_MATH.md`](reference/D3PM_MATH.md)** for the worked, numerically-validated D3PM derivations.

## What makes these notebooks different

- **Verified, not vibes.** The D3PM closed forms are checked against brute-force
  Bayes *inside the notebook*; a standalone oracle ([`tools/d3pm_reference_check.py`](tools/d3pm_reference_check.py)) cross-checks the matrix math. If the math were wrong, the notebook would fail.
- **Honest experiments.** Lesson 2's kernel comparison reports what the code
  *actually* shows (uniform can beat absorbing on tiny data!) and explains the
  real reason — *revisable vs. irreversible* sampling — instead of parroting a
  headline result.
- **Every claim is runnable.** Plots, corruption demos, denoising traces, and
  loss curves are generated live from the code beside them.

## Repository layout

```
.
├── README.md                              ← you are here
├── CONTENT.md                             ← full syllabus & learning path
├── 01_foundations_masked_diffusion.ipynb  ← Lesson 1
├── 02_d3pm_transition_kernels.ipynb       ← Lesson 2
├── 03_diffusion_lm_embedding_space.ipynb  ← Lesson 3
├── 04_scaling_diffusion_llm.ipynb         ← Lesson 4
├── 05_sedd_score_entropy.ipynb            ← Lesson 5
├── papers/                                ← the cited papers (PDFs) + a per-lesson index
├── reference/
│   └── D3PM_MATH.md                       ← worked, validated D3PM math
├── tools/
│   ├── build_lesson1.py … build_lesson5.py  ← regenerate each lesson's notebook
│   ├── d3pm_reference_check.py            ← numerical oracle for the D3PM math (Lesson 2)
│   └── sedd_reference_check.py            ← numerical oracle for the SEDD math (Lesson 5)
├── requirements.txt
└── LICENSE
```

> The notebooks are authored from the `tools/build_lesson*.py` scripts (cell
> sources live as plain Python, which is easier to review and diff than notebook
> JSON). Run a build script to regenerate its `.ipynb`; you never *need* to — the
> committed notebooks are ready to open.

## Contributing & feedback

This is a learning resource — issues, corrections, and suggestions are very
welcome. Found a clearer explanation, a bug, or a typo in the math? Open an issue
or a PR.

## License

[MIT](LICENSE) — free to use, learn from, and build on.

## References & Acknowledgments

This is a from-scratch **teaching re-implementation**. All credit for the ideas,
methods, and results belongs to the authors below — please cite the original
papers, not this course. PDFs are collected in [`papers/`](papers/) (see
[`papers/README.md`](papers/README.md) for the per-lesson mapping and a note on
redistribution).

**Foundations**
- Ho, Jain, Abbeel (2020) — *Denoising Diffusion Probabilistic Models* (DDPM) — [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
- Nichol, Dhariwal (2021) — *Improved Denoising Diffusion Probabilistic Models* (cosine schedule) — [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)

**Discrete / masked diffusion**
- Austin, Johnson, Ho, Tarlow, van den Berg (2021) — *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM) — [arXiv:2107.03006](https://arxiv.org/abs/2107.03006)
- Lou, Meng, Ermon (2023) — *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (SEDD) — [arXiv:2310.16834](https://arxiv.org/abs/2310.16834)
- Shi, Han, Wang, Doucet, Titsias (2024) — *Simplified and Generalized Masked Diffusion for Discrete Data* — [arXiv:2406.04329](https://arxiv.org/abs/2406.04329)
- Sahoo et al. (2024) — *Simple and Effective Masked Diffusion Language Models* (MDLM) — [arXiv:2406.07524](https://arxiv.org/abs/2406.07524)
- Nie et al. (2025) — *Large Language Diffusion Models* (LLaDA) — [arXiv:2502.09992](https://arxiv.org/abs/2502.09992)

**Continuous / embedding-space diffusion**
- Li, Thickstun, Gulrajani, Liang, Hashimoto (2022) — *Diffusion-LM Improves Controllable Text Generation* — [arXiv:2205.14217](https://arxiv.org/abs/2205.14217)
- Gulrajani, Hashimoto (2023) — *Likelihood-Based Diffusion Language Models* (Plaid) — [arXiv:2305.18619](https://arxiv.org/abs/2305.18619)

**Sampling / generation**
- Chang, Zhang, Jiang, Liu, Freeman (2022) — *MaskGIT: Masked Generative Image Transformer* — [arXiv:2202.04200](https://arxiv.org/abs/2202.04200)
