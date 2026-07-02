# Course Content & Learning Path

A from-scratch, build-it-yourself course on **diffusion models for language**.
Every lesson is a single self-contained, runnable Jupyter notebook: you train a
real (tiny) model on a toy corpus in seconds, with the math tied line-by-line to
the papers that introduced each idea. No prior diffusion knowledge required —
just PyTorch and the transformer.

---

## How to use this course

- **Work the notebooks in order.** Each builds on the last; later lessons reuse
  the forward-process / denoiser / sampler scaffolding from earlier ones.
- **Run every cell.** The notebooks are designed to be *executed*, not just read —
  the plots, corruption demos, and numerical checks are where the intuition lands.
- **Do the exercises.** Each lesson ends with exercises that extend the code; they
  are the difference between "I read it" and "I get it."
- **Read the margins.** Markdown cells connect each block of code to the exact
  paper and equation it implements.

**Prerequisites:** comfort with PyTorch and the transformer architecture; basic
probability (categorical distributions, KL divergence, Bayes' rule). Calculus and
linear algebra at the level of "I can read $\prod$, $\sum$, and a matrix product."

**Time:** ~45–90 minutes per lesson including exercises.

---

## The arc of the course

```
  DISCRETE tokens                                     CONTINUOUS embeddings
  ───────────────                                     ─────────────────────
  L1  masking (absorbing-state)                       L3  Diffusion-LM
  L2  D3PM — any transition kernel                        (Gaussian noise in R^d,
  L4  scaling up — LLaDA-style:                            + gradient guidance)
      generalization, guidance, fast sampling
  L5  SEDD — score/ratio matching  (advanced)

  Suggested order: L1 → L2 → L3 → L4 → L5
```

**All five lessons are complete and verified end-to-end.** 🎓

---

## Lesson 1 — Foundations: Masked (Absorbing-State) Diffusion ✅

**Notebook:** [`01_foundations_masked_diffusion.ipynb`](01_foundations_masked_diffusion.ipynb)

The simplest language diffusion model, end to end. The forward process replaces
tokens with `[MASK]`; a bidirectional transformer learns to fill them back in.

- **Concepts:** autoregressive vs. diffusion generation; the absorbing/masking
  forward process; the noise schedule; the bidirectional denoiser with time
  conditioning; the $1/t$-weighted masked cross-entropy from the continuous-time
  ELBO; ancestral sampling; zero-shot infilling.
- **You build:** a working masked diffusion LM that trains in seconds and
  generates and infills toy text; a step-by-step view of the denoising trajectory.
- **Papers:** MDLM (Sahoo et al. 2024), Shi et al. 2024, LLaDA (Nie et al. 2025).
- **Takeaway:** generation = iterative denoising from all-`[MASK]`; the mask token
  is a fixed "predict me" signal, not random noise.

## Lesson 2 — General Transition Kernels: the D3PM Framework ✅

**Notebook:** [`02_d3pm_transition_kernels.ipynb`](02_d3pm_transition_kernels.ipynb)
**Math reference:** [`reference/D3PM_MATH.md`](reference/D3PM_MATH.md)

Generalize Lesson 1's single kernel to *any* categorical transition matrix, and
compare the **uniform** and **absorbing** kernels head-to-head.

- **Concepts:** discrete diffusion as a Markov chain with transition matrices
  $Q_t$; closed-form $t$-step marginal and forward posterior; the
  x0-parameterized reverse; the **term-by-term ELBO** ($L_0 + \sum L_{t-1} + L_T$)
  and the **hybrid loss**; why the absorbing ELBO **collapses to Lesson 1's
  weighted cross-entropy**; revisable vs. irreversible sampling.
- **You build:** both kernels with numerically-validated closed forms (checked
  against brute-force Bayes), trained and sampled side by side; visualizations of
  $Q_t$, the schedule, and the two kinds of "noise."
- **Papers:** D3PM (Austin et al. 2021).
- **Takeaway:** choosing $Q_t$ chooses your diffusion; absorbing is a special case
  of a much larger family, and its simplicity is *why* it scales.

## Lesson 3 — Continuous / Embedding-Space Diffusion (Diffusion-LM) ✅

**Notebook:** [`03_diffusion_lm_embedding_space.ipynb`](03_diffusion_lm_embedding_space.ipynb)

A completely different way to diffuse language: instead of corrupting discrete
tokens, embed them into $\mathbb R^d$ and run **Gaussian** diffusion in that
continuous space (the exact DDPM image-diffusion math), then *round* back to tokens.

- **Concepts:** learned word embeddings + a cosine **rounding** readout; the
  Gaussian forward process and **predict-$z_0$** parameterization; a denoising MSE
  + rounding cross-entropy that learns the embeddings end-to-end; **DDPM sampling**
  with the **clamping trick**; and **gradient-guided controllable generation**.
- **You build:** a small Diffusion-LM that denoises in embedding space and decodes
  to text, plus a working demo that **steers samples toward a target prefix** using
  gradients — no retraining.
- **Papers:** Diffusion-LM (Li et al. 2022); DDPM (Ho et al. 2020); Plaid
  (Gulrajani & Hashimoto 2023).
- **Takeaway:** continuous latents make **gradient guidance** natural (the big win)
  but add embeddings + lossy rounding and awkward likelihoods — which is why recent
  work leans back toward the discrete/masking side.

## Lesson 4 — Making Masked Diffusion Practical (LLaDA-style) ✅

**Notebook:** [`04_scaling_diffusion_llm.ipynb`](04_scaling_diffusion_llm.ipynb)

Turn Lesson 1's masking model into a practical, LLaDA-style one — and measure real
generalization instead of memorization.

- **Concepts:** training on a **grammar-generated** corpus and measuring
  **generalization** (novel, grammar-valid samples, not copies); **conditional
  generation** via a label embedding + label-dropout; **classifier-free guidance
  (CFG)** and the guidance-weight dial; **confidence-based (MaskGIT/LLaDA)
  sampling** and the **steps↔quality (NFE) trade-off**.
- **You build:** a conditional masked diffusion LM that invents new valid
  sentences; a CFG demo where cranking the guidance weight makes samples obey a
  requested category harder; and an NFE plot showing confidence sampling beats
  random unmasking at low step counts.
- **Papers:** LLaDA (Nie et al. 2025); MaskGIT (Chang et al. 2022).
- **Takeaway:** "scaling" is mostly *technique* — generalization, guidance, and
  confidence sampling are what make masked diffusion usable; raw size is the rest.

## Lesson 5 — Score-Entropy Discrete Diffusion (SEDD) ✅

**Notebook:** [`05_sedd_score_entropy.ipynb`](05_sedd_score_entropy.ipynb)
**Math oracle:** [`tools/sedd_reference_check.py`](tools/sedd_reference_check.py)

The advanced capstone. Stop predicting the posterior/token; instead learn the
**ratios** $p_t(y)/p_t(x)$ of the data distribution — the **concrete score** (the
discrete analogue of a "score").

- **Concepts:** the concrete score / ratio target; the **denoising score-entropy**
  loss $s - r\log s$ (minimised at $s=r$); the **SEDD identity** (posterior-average
  of per-$x_0$ ratios = the true marginal ratio); continuous-time uniform process;
  **reverse tau-leaping** sampling.
- **You build:** a SEDD model on the toy corpus, with a **numerically-verified**
  headline result — the learned score matches the brute-force **true marginal
  ratio** (correlation ≈ 0.99) — plus a working reverse sampler.
- **Papers:** SEDD (Lou, Meng, Ermon 2023).
- **Takeaway:** learning ratios (not predictions) is a genuinely different lens on
  discrete diffusion — the one that first beat strong baselines on likelihood.

---

## Part II — Advanced / Frontier track (planned)

Part I (Lessons 1–5) covers the *foundations*. This planned second track digs into
the two things that make diffusion LMs practically exciting — **how you reveal
tokens** and **how you generate in parallel** — plus the **2025–2026 frontier**.

### Lesson 6 — Masking & unmasking *strategies* (the decoding zoo) 🔜
Lesson 1 revealed tokens at random; Lesson 4 revealed the most-confident first.
That's just the start. This lesson builds and compares the full menu:
- **Confidence / entropy / margin**-based unmasking orders, and cosine vs. linear
  reveal schedules.
- **Remasking & self-correction** — let the model *un-commit* a token it now
  regrets (directly fixes the "irreversible commitment" weakness we saw in Lessons
  2 and 4), via predictor–corrector / resample steps.
- **Any-order** generation and planned/adaptive decoding.
- Papers: MaskGIT, LLaDA, and remasking/self-correction samplers.

### Lesson 7 — Parallelization & fast generation 🔜
Diffusion's headline advantage over autoregressive LLMs: **generate many tokens at
once**. This lesson makes it real and measures it:
- **Parallel decoding** and the speed/quality frontier (tokens-per-step vs.
  accuracy) — building on Lesson 4's NFE plot.
- **Block / semi-autoregressive diffusion** for arbitrary-length text: generate a
  block with diffusion, freeze it, condition the next block on it (Block Diffusion).
- **Few-step generation** via distillation / consistency-style samplers.
- Throughput vs. an autoregressive baseline, honestly benchmarked.

### Lesson 8 — The 2025–2026 frontier 🔜
The research edge, tied back to what you built:
- **Discrete flow matching** (a flow-matching alternative to the diffusion ELBO).
- **Adapting pretrained autoregressive LLMs into diffusion LLMs** (cheaply turning
  a GPT into a LLaDA-style model) and scaling laws for masked diffusion.
- **Guidance & reward** for discrete diffusion (steering/aligning the samples).
- **Production diffusion LLMs** (e.g. commercial diffusion code/text models) and
  where diffusion stands vs. autoregressive at the frontier.

> ⚠️ **Honesty note on "2026 methods."** This course's author knowledge runs to
> **early 2026**. Lessons 6–7 rest on established, verified methods. Lesson 8's
> newest, post-cutoff work will get a **fresh, cited literature pass** before it's
> written — no invented papers. If you want the very latest, that literature review
> is the first step.

*(Part II is a roadmap, not yet built. Each lesson will follow Part I's recipe:
tiny, runnable, verified, beginner-friendly, one paper-family at a time.)*

---

## Concept map (where each idea is introduced)

| Concept | Lesson |
|---|---|
| Forward (corruption) process | 1 |
| Noise schedule $\bar\alpha_t$ | 1, 2 |
| Bidirectional denoiser + time conditioning | 1 |
| ELBO → weighted cross-entropy | 1 |
| Ancestral sampling, infilling | 1 |
| Transition matrices $Q_t$, the kernel family | 2 |
| Closed-form marginal & posterior, Bayes | 2 |
| x0-parameterized reverse, term-by-term ELBO, hybrid loss | 2 |
| Continuous/embedding diffusion, rounding | 3 |
| Gradient guidance (continuous) / classifier-free guidance | 3, 4 |
| Generalization, conditional generation | 4 |
| Confidence-based sampling, steps↔quality (NFE) | 4 |
| Score / ratio matching | 5 |
| Continuous-time discrete diffusion | 5 |

---

## Papers referenced across the course

- Ho, Jain, Abbeel 2020 — *Denoising Diffusion Probabilistic Models* (DDPM) — https://arxiv.org/abs/2006.11239
- Nichol, Dhariwal 2021 — *Improved Denoising Diffusion Probabilistic Models* (cosine schedule) — https://arxiv.org/abs/2102.09672
- Austin et al. 2021 — *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM) — https://arxiv.org/abs/2107.03006
- Li et al. 2022 — *Diffusion-LM Improves Controllable Text Generation* — https://arxiv.org/abs/2205.14217
- Chang et al. 2022 — *MaskGIT: Masked Generative Image Transformer* — https://arxiv.org/abs/2202.04200
- Gulrajani & Hashimoto 2023 — *Likelihood-Based Diffusion Language Models* (Plaid) — https://arxiv.org/abs/2305.18619
- Lou, Meng, Ermon 2023 — *Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution* (SEDD) — https://arxiv.org/abs/2310.16834
- Shi et al. 2024 — *Simplified and Generalized Masked Diffusion for Discrete Data* — https://arxiv.org/abs/2406.04329
- Sahoo et al. 2024 — *Simple and Effective Masked Diffusion Language Models* (MDLM) — https://arxiv.org/abs/2406.07524
- Nie et al. 2025 — *Large Language Diffusion Models* (LLaDA) — https://arxiv.org/abs/2502.09992

---

*All five lessons are done and run end-to-end. The course is complete: a
from-scratch, verified tour of language diffusion — discrete and continuous,
prediction-based and score-based.*
