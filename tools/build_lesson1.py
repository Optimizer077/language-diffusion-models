"""Builds 01_foundations_masked_diffusion.ipynb using nbformat.

Keeping the notebook authored from a script means the cell sources live in
plain Python here (easy to lint/diff) instead of hand-edited notebook JSON.
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
def md(src):  cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

# ----------------------------------------------------------------------------
md(r"""# Lesson 1 — Foundations of Language Diffusion Models
### Masked (absorbing-state) discrete diffusion, from scratch

Welcome — this is the first lesson, and the front door to the whole course. By the
end you'll have **built and trained a working language diffusion model** from
scratch, and you'll understand the three pieces every diffusion language model is
made of:

1. **A forward (corruption) process** — gradually turn clean text into noise.
2. **A denoiser** — a network that learns to undo the corruption.
3. **A sampler** — run the denoiser in reverse to *generate* new text.

The whole idea in one sentence: **learn to fill in the blanks, then generate by
starting from all-blanks and filling them in.** That's it. Everything else is
detail.

> 🟢 **New to this? Don't worry about the math symbols.** There's a plain-English
> glossary in the next cell, and every formula is explained in words right beside
> it. If you can follow "blank out some letters and train a model to guess them
> back," you can follow this entire lesson.

We use the **masking** (a.k.a. *absorbing-state*) flavor of diffusion because it's
the simplest to picture, and it's exactly what powers modern diffusion LLMs like
**LLaDA**.

---
**Prerequisites:** comfort with PyTorch and the transformer. You do **not** need any
prior diffusion knowledge — that's what we're building.

**Key papers (we connect the code to these):**
- Austin et al. 2021 — *Structured Denoising Diffusion Models in Discrete State-Spaces* (**D3PM**) — Lesson 2
- Li et al. 2022 — *Diffusion-LM* (continuous/embedding diffusion) — Lesson 3
- Sahoo et al. 2024 (**MDLM**) / Shi et al. 2024 (**MD4**) — masked diffusion LMs — this lesson's objective
- Nie et al. 2025 — *Large Language Diffusion Models* (**LLaDA**) — Lesson 4
""")

# ----------------------------------------------------------------------------
md(r"""## 📖 A 60-second glossary (skim now, refer back later)

You don't need to memorize these — just know they're here.

| Term | In plain words |
|---|---|
| **token / letter** | one character of text (we work letter-by-letter). |
| **vocabulary** | the set of all possible characters. |
| **`[MASK]`** | a special "blanked-out" symbol — this is our "noise." |
| **absorbing state** | once a letter becomes `[MASK]` it *stays* `[MASK]`, and the blank tells you nothing about what was there. |
| **$x_0$** | the **clean** original text. (The subscript is the *noise level*.) |
| **$x_t$** | the text after corrupting it to noise level $t$. |
| **forward process** | adding noise: blank out more and more letters. |
| **reverse process / sampling** | removing noise: fill blanks back in. This is generation. |
| **noise level $t$** | a number in $(0,1]$: $t\to0$ is clean, $t\to1$ is all-blanks. |
| **schedule $\alpha_t$** | the probability a letter has *survived* (not been blanked) at level $t$. |
| **denoiser** | the neural net that guesses the blanked letters. |
| **bidirectional** | the denoiser sees the *whole* sequence (left **and** right) — unlike GPT. |
| **cross-entropy** | the standard "how wrong was the guess?" loss for picking a class (here, a letter). |
| **ELBO** | the principled training objective for diffusion models; here it gives a $1/t$ weighting on the loss. |
| **ancestral sampling** | generate by undoing the noise one step at a time. |
| **temperature** | a randomness dial when sampling (higher = more varied). |
| **infilling** | filling blanks while *keeping* some known letters fixed. |
""")

# ----------------------------------------------------------------------------
md(r"""## 0. Autoregressive vs. diffusion: the core idea

A standard LLM (GPT-style) is **autoregressive (AR)**: it writes text strictly
left-to-right, one token at a time, each conditioned on all the previous ones:

$$p_\theta(x) = \prod_{i=1}^{L} p_\theta(x_i \mid x_{<i}).$$

A **diffusion LM** throws out the left-to-right rule. Instead it learns to
*denoise*: given a partly-corrupted sequence, restore it. To generate, it starts
from **pure noise** (here: an all-`[MASK]` sequence) and **refines the whole
sequence in parallel** over a number of steps — a bit like a sculptor revealing a
figure from marble, rather than a writer typing left to right.

| | Autoregressive (GPT) | Diffusion (this lesson) |
|---|---|---|
| Generation order | strictly left→right | any order / all at once |
| Steps to make L tokens | L | a chosen number T (can be ≪ L) |
| Sees future context? | no (causal mask) | yes (full, bidirectional attention) |
| Infilling / editing the middle | hard | easy (just leave known tokens unmasked) |

That last row is the practical payoff: with diffusion, conditioning on the
*middle* of a sequence is as natural as conditioning on the start.
""")

# ----------------------------------------------------------------------------
code(r"""import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("torch", torch.__version__, "| device:", device)""")

# ----------------------------------------------------------------------------
md(r"""## 1. A toy corpus

To keep training fast (seconds on a CPU) and the results *readable*, we use a
small, structured char-level corpus. Each line follows a simple pattern, so a tiny
model can learn the regularities and we can clearly see whether generated text
looks like the training data.

We work at the **character level**: the vocabulary is just the characters that
appear, plus two special tokens:
- `[PAD]` — pads short lines to a fixed length `L`.
- `[MASK]` — the *absorbing state*; the "noise" token the forward process puts in
  place of real characters.

> **Heads-up (consistency across the course):** here we use a dedicated `[PAD]`
> token. Lessons 2–4 simplify by just padding with spaces (treating space as a
> normal character), so you won't see `[PAD]` there. Same idea, one less special
> token to track.
""")

code(r"""raw_lines = [
    "the cat sat on the mat",
    "the dog ran in the fog",
    "the sun is in the sky",
    "the fish swims in the sea",
    "the bird flew over the hill",
    "the fox hid in the den",
    "a red car drove down the road",
    "a blue boat sailed on the lake",
    "the old man read a good book",
    "the girl drew a big green tree",
    "we ran fast to the bus stop",
    "they sang songs all night long",
    "the rain fell soft on the roof",
    "the moon rose high in the dark",
    "my cat likes to nap in the sun",
    "the kids ran home for the meal",
]

# Build the char vocabulary from the corpus.
chars = sorted(set("".join(raw_lines)))
PAD, MASK = "_", "#"          # symbols chosen to not collide with corpus chars
assert PAD not in chars and MASK not in chars
vocab = chars + [PAD, MASK]
stoi = {c: i for i, c in enumerate(vocab)}
itos = {i: c for c, i in stoi.items()}
PAD_ID, MASK_ID = stoi[PAD], stoi[MASK]
V = len(vocab)

L = max(len(s) for s in raw_lines)   # fixed block length
print(f"vocab size V = {V}  | block length L = {L}")
print("vocab:", "".join(vocab))

def encode(s):
    s = s[:L] + PAD * (L - len(s))           # truncate / pad to length L
    return torch.tensor([stoi[c] for c in s], dtype=torch.long)

def decode(t):
    return "".join(itos[int(i)] for i in t)

data = torch.stack([encode(s) for s in raw_lines])   # (N, L)
print("data tensor:", tuple(data.shape))
print("example encode/decode:", repr(decode(data[0])))""")

# ----------------------------------------------------------------------------
md(r"""## 2. The forward process: masking as "adding noise"

In the **absorbing-state** formulation, corruption is dead simple: **blank out
letters at random.** We pick a **noise level** $t \in (0, 1]$ (bigger = noisier),
and define a **schedule** $\alpha_t$ = the probability a letter *survives* (is
kept, not blanked). We use the simplest schedule, $\alpha_t = 1 - t$:

- $t \to 0$: $\alpha_t \to 1$ — almost nothing blanked → nearly clean text.
- $t \to 1$: $\alpha_t \to 0$ — everything blanked → pure noise.

Given clean text $x_0$, the corrupted $x_t$ replaces **each letter independently**
with `[MASK]` with probability $1 - \alpha_t = t$:

$$q(x_t^i \mid x_0^i) = \begin{cases} \alpha_t & x_t^i = x_0^i \quad(\text{survived}) \\ 1 - \alpha_t & x_t^i = \texttt{[MASK]} \quad(\text{blanked}) \end{cases}$$

> 🧠 **`[MASK]` is not random noise — it's a *blank*.** In image diffusion, "noise"
> means adding random numbers to pixels. Here, "noise" means **replacing a letter
> with one fixed placeholder** (`[MASK]`) that carries *zero* information about what
> was there. That "no information left" property is what *absorbing* means: once
> blanked, always blanked, and unrecoverable except by the model's guess.

> 📐 **Notation bridge (read this once).** Because $t$ here is a single continuous
> "total noise" knob, our $\alpha_t = 1-t$ is the probability a letter survives *all
> the way* to level $t$ — a **cumulative** quantity. In Lessons 2–4 we instead take
> many small discrete steps and write the *per-step* survival as $\alpha_t$ and the
> *cumulative* one as $\bar\alpha_t$ ("alpha-bar"). **So this lesson's $\alpha_t$ is
> the same idea as $\bar\alpha_t$ in the later lessons.** Don't let the reused
> letter trip you up.

Two nice properties: masking is **independent per position**, and a blank is
information-free. We never blank `[PAD]` — padding isn't real signal.
""")

code(r"""def alpha(t):
    # Linear schedule: probability of KEEPING a token at time t.
    return 1.0 - t

def forward_mask(x0, t):
    '''Corrupt a clean batch x0 (B, L) at time t (B,) -> (x_t, mask_bool).

    Each non-pad position is replaced by [MASK] with probability (1 - alpha_t).
    Returns the corrupted sequence and a boolean mask of which positions were
    turned into [MASK] (these are the ones the model must predict).
    '''
    keep_prob = alpha(t).unsqueeze(1)                 # (B, 1)
    rand = torch.rand_like(x0, dtype=torch.float)     # (B, L)
    is_pad = (x0 == PAD_ID)
    masked = (rand > keep_prob) & (~is_pad)           # True -> replace w/ [MASK]
    x_t = torch.where(masked, torch.full_like(x0, MASK_ID), x0)
    return x_t, masked

# Demo: corrupt one line at a few noise levels.
demo = data[0:1]
for tval in [0.1, 0.4, 0.8]:
    xt, m = forward_mask(demo, torch.tensor([tval]))
    print(f"t={tval:.1f}  ->  {decode(xt[0])!r}")
print(f"\noriginal     ->  {decode(demo[0])!r}")""")

# ----------------------------------------------------------------------------
md(r"""## 3. The denoiser network

The denoiser $f_\theta(x_t, t)$ is a **fill-in-the-blanks network**: it reads the
corrupted sequence and, for every position, predicts a probability distribution
over the vocabulary — most importantly, it must guess the real character behind
each `[MASK]`.

The key difference from a GPT: the denoiser uses **full (bidirectional)
attention** — when guessing a blank it looks both **left and right**. We also tell
it the current noise level by feeding $t$ through a small "time embedding," so it
can behave differently when the input is barely corrupted vs. almost all blanks.

Concretely it's just a small **encoder-style transformer** with a linear head that
outputs one score per vocabulary letter, per position.
""")

code(r"""class TimeEmbedding(nn.Module):
    '''Sinusoidal embedding of the scalar time t in (0,1], then an MLP.'''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t):                          # t: (B,)
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t[:, None] * freqs[None, :] * 1000.0
        emb = torch.cat([ang.sin(), ang.cos()], dim=-1)
        return self.mlp(emb)                       # (B, dim)


class DenoiserTransformer(nn.Module):
    def __init__(self, V, L, d_model=128, nhead=4, nlayers=3, ff=256, p=0.1):
        super().__init__()
        self.tok = nn.Embedding(V, d_model)
        self.pos = nn.Embedding(L, d_model)
        self.time = TimeEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=ff, dropout=p,
            batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, V)
        self.register_buffer("pos_ids", torch.arange(L))

    def forward(self, x_t, t):                     # x_t: (B, L), t: (B,)
        h = self.tok(x_t) + self.pos(self.pos_ids)[None]
        h = h + self.time(t)[:, None, :]           # broadcast time over positions
        h = self.encoder(h)                        # full bidirectional attention
        return self.head(self.norm(h))             # (B, L, V) logits

model = DenoiserTransformer(V, L).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"denoiser parameters: {n_params:,}")""")

# ----------------------------------------------------------------------------
md(r"""## 4. The training objective (built up gently)

**Goal:** a single number (the *loss*) that, made small, turns the network into a
good blank-filler. Let's build it up.

**Step 1 — the obvious loss.** Corrupt some text, ask the model to guess the
blanked letters, and measure how wrong it is with **cross-entropy** (the standard
loss for "pick the right class," here the right letter). Score it **only on the
blanked positions** — the un-blanked letters are given, nothing to learn there.

**Step 2 — the principled tweak: weight by $1/t$.** Diffusion models have a proper
training objective called the **ELBO** (evidence lower bound). You don't need its
derivation; the punchline is that it turns the plain cross-entropy into a
**time-weighted** one. For the masking process, the continuous-time negative ELBO
is (MDLM, Sahoo et al. 2024 / Shi et al. 2024):

$$\mathcal{L} = \mathbb{E}_{t \sim U(0,1)}\, \mathbb{E}_{x_0,\,x_t}
\left[ \frac{\alpha'_t}{1 - \alpha_t} \sum_{i\,:\,x_t^i = \texttt{[MASK]}}
\log p_\theta(x_0^i \mid x_t) \right].$$

With our schedule $\alpha_t = 1 - t$ (so $\alpha'_t = -1$ and $1 - \alpha_t = t$),
that weight is just $\tfrac{1}{t}$:

$$\mathcal{L} = \mathbb{E}_{t,\,x_0,\,x_t}
\left[ \frac{1}{t} \sum_{i\,:\,x_t^i=\texttt{[MASK]}} -\log p_\theta(x_0^i \mid x_t) \right].$$

**In plain words:** cross-entropy on the blanked positions, weighted by $1/t$ —
low-noise examples (small $t$) count more.

> **One honest footnote on the code.** The bound above *sums* over the blanked
> positions; our `diffusion_loss` (next cell) instead *averages* the cross-entropy
> over them (divides by how many were blanked) before multiplying by $1/t$. That
> per-token normalization is a common, more stable choice — it keeps the loss on a
> fixed scale — though it does re-weight the noise levels slightly versus the exact
> sum-form bound. The takeaway is unchanged: recover the blanks, and weight
> low-noise examples more.

So the training loop is:
1. Sample clean lines $x_0$ and a random noise level $t \sim U(0,1]$.
2. Corrupt → $x_t$ (Section 2).
3. Predict logits; take cross-entropy **only on blanked positions**.
4. Weight each example by $1/t$ and take a gradient step.

*(Lesson 2 shows where this weighting comes from — the ELBO written out term by
term — and generalizes it beyond masking.)*
""")

code(r"""def diffusion_loss(model, x0):
    B = x0.size(0)
    t = torch.rand(B, device=x0.device).clamp(min=1e-3)   # avoid 1/t blow-up
    x_t, masked = forward_mask(x0, t)
    logits = model(x_t, t)                                # (B, L, V)

    # Per-token cross-entropy, kept only where we actually masked.
    ce = F.cross_entropy(logits.reshape(-1, V), x0.reshape(-1), reduction="none")
    ce = ce.reshape(B, L) * masked.float()
    per_example = ce.sum(dim=1) / masked.float().sum(dim=1).clamp(min=1)
    weight = 1.0 / t
    return (weight * per_example).mean()""")

# ----------------------------------------------------------------------------
code(r"""opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
STEPS, BATCH = 2000, 32
data_dev = data.to(device)
losses = []

model.train()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, data_dev.size(0), (BATCH,), device=device)
    loss = diffusion_loss(model, data_dev[idx])
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    losses.append(loss.item())
    if step % 250 == 0 or step == 1:
        print(f"step {step:4d} | loss {np.mean(losses[-50:]):.4f}")

plt.figure(figsize=(6, 3))
plt.plot(np.convolve(losses, np.ones(25)/25, mode="valid"))
plt.xlabel("step"); plt.ylabel("loss (smoothed)"); plt.title("Diffusion training loss")
plt.tight_layout(); plt.show()""")

# ----------------------------------------------------------------------------
md(r"""## 5. Sampling: running the process in reverse

To generate, we **denoise from pure noise**. Start at $t = 1$ with an all-`[MASK]`
sequence and walk down to $t = 0$ over $T$ steps. At each step we move from noise
level $t$ to an earlier $s < t$ and decide which blanks to **reveal** (fill in).
Once a position is revealed, it stays fixed — this is called **ancestral
sampling**.

For the absorbing process, the principled reverse step reveals each still-blanked
position independently with probability

$$\frac{\alpha_s - \alpha_t}{1 - \alpha_t} \;=\; \frac{t - s}{t}
\quad (\text{linear schedule}),$$

and when it reveals one, it draws the letter from the model's prediction
$p_\theta(\cdot \mid x_t)$. (Intuition: $\tfrac{t-s}{t}$ is the fraction of the
remaining blanks that "come due" to be filled this step.) This is the **ancestral
sampler** of MDLM.

Two knobs worth knowing:
- **More steps $T$** → finer-grained, usually higher quality (but slower).
- **Temperature** on the logits trades diversity vs. fidelity.

> A popular alternative (MaskGIT / LLaDA) is *confidence-based* unmasking: predict
> all blanks, then commit only the most-confident ones first. You'll meet it in the
> exercises here, and build it for real in **Lesson 4**.
""")

code(r"""@torch.no_grad()
def sample(model, n=8, T=64, temperature=1.0, seed=None):
    if seed is not None: torch.manual_seed(seed)
    model.eval()
    x = torch.full((n, L), MASK_ID, device=device)        # start: all masked (t=1)
    ts = torch.linspace(1.0, 0.0, T + 1, device=device)   # times t -> s

    for i in range(T):
        t, s = ts[i], ts[i + 1]
        still_masked = (x == MASK_ID)
        if not still_masked.any():
            break
        logits = model(x, t.expand(n)) / temperature
        probs = F.softmax(logits, dim=-1)
        pred = torch.multinomial(probs.reshape(-1, V), 1).reshape(n, L)

        reveal_prob = ((t - s) / t).clamp(0, 1)            # per-step reveal prob
        do_reveal = (torch.rand_like(x, dtype=torch.float) < reveal_prob) & still_masked
        x = torch.where(do_reveal, pred, x)

    # Any positions left masked at the end: fill with the final greedy prediction.
    if (x == MASK_ID).any():
        logits = model(x, ts[-1].expand(n))
        x = torch.where(x == MASK_ID, logits.argmax(-1), x)
    return x

print("Samples from the trained diffusion LM:\n")
for row in sample(model, n=10, T=64, seed=1):
    print("  ", repr(decode(row).replace(PAD, "")))""")

# ----------------------------------------------------------------------------
md(r"""## 6. Watching it denoise

The fun part of diffusion is that generation is a *trajectory*. Let's print the
sequence at a few checkpoints as it goes from all-`[MASK]` to finished text, so you
can literally watch the model commit letters over time.
""")

code(r"""@torch.no_grad()
def sample_with_trace(model, T=64, temperature=1.0, seed=2, checkpoints=6):
    torch.manual_seed(seed); model.eval()
    x = torch.full((1, L), MASK_ID, device=device)
    ts = torch.linspace(1.0, 0.0, T + 1, device=device)
    show = {int(k) for k in torch.linspace(0, T - 1, checkpoints).round().tolist()}
    for i in range(T):
        t, s = ts[i], ts[i + 1]
        sm = (x == MASK_ID)
        logits = model(x, t.expand(1)) / temperature
        pred = torch.multinomial(F.softmax(logits, -1).reshape(-1, V), 1).reshape(1, L)
        reveal = (torch.rand_like(x, dtype=torch.float) < ((t - s)/t).clamp(0,1)) & sm
        x = torch.where(reveal, pred, x)
        if i in show:
            frac = (x == MASK_ID).float().mean().item()
            print(f"step {i:3d} (t={t:.2f}, {frac*100:4.0f}% masked): {decode(x[0])!r}")
    print("final"+ " "*22 + f": {decode(x[0]).replace(PAD,'')!r}")

sample_with_trace(model)""")

# ----------------------------------------------------------------------------
md(r"""## 7. Bonus: diffusion does infilling for free

Because the denoiser conditions on *whatever is unmasked*, conditional generation
needs **no retraining** — just clamp the known letters in place and let the model
fill the rest. This is the "edit the middle" ability that autoregressive models
lack (they can only extend from the left).
""")

code(r"""@torch.no_grad()
def infill(model, template, T=64, temperature=0.8, seed=3):
    '''template: string using '#' for positions to fill in, real chars to keep.'''
    torch.manual_seed(seed); model.eval()
    base = encode(template)
    known = (base != MASK_ID) & (base != PAD_ID)            # positions to clamp
    x = base.clone().unsqueeze(0).to(device)
    known = known.unsqueeze(0).to(device)
    ts = torch.linspace(1.0, 0.0, T + 1, device=device)
    for i in range(T):
        t, s = ts[i], ts[i + 1]
        sm = (x == MASK_ID)
        if not sm.any(): break
        logits = model(x, t.expand(1)) / temperature
        pred = torch.multinomial(F.softmax(logits, -1).reshape(-1, V), 1).reshape(1, L)
        reveal = (torch.rand_like(x, dtype=torch.float) < ((t - s)/t).clamp(0,1)) & sm
        x = torch.where(reveal, pred, x)
        x = torch.where(known, base.to(device), x)          # re-clamp known tokens
    if (x == MASK_ID).any():
        logits = model(x, ts[-1].expand(1))
        x = torch.where(x == MASK_ID, logits.argmax(-1), x)
    return decode(x[0]).replace(PAD, "")

# Keep "the " ... "sat on the " and let the model fill the gaps:
print(infill(model, "the ### sat on the ###"))
print(infill(model, "the ### ran ### the ###"))""")

# ----------------------------------------------------------------------------
md(r"""## 8. What you built, and where it goes next

You implemented a complete **masked discrete diffusion language model**:
- **Forward process** — independent per-token masking with a noise schedule $\alpha_t$.
- **Denoiser** — a bidirectional transformer that fills in blanks.
- **Objective** — the $1/t$-weighted masked cross-entropy from the ELBO.
- **Sampler** — the ancestral reverse process, plus free infilling.

**How this maps to the literature**
- This *is* the **MDLM / LLaDA** recipe in miniature. LLaDA scales exactly this
  (masking + bidirectional denoiser + per-token CE) to billions of parameters and
  competes with autoregressive LLMs.
- **D3PM** generalizes the forward process to *any* token-transition rule (mask,
  random-replace, and more). Absorbing/masking is one choice — Lesson 2.
- **Diffusion-LM** takes a different route entirely: diffuse in *continuous
  embedding* space and round back to tokens — Lesson 3.

**Where the course goes from here**
- **Lesson 2 — D3PM & general transition kernels.** Any kind of noise as a
  probability table; the true ELBO term by term; why masking is special.
- **Lesson 3 — Continuous / embedding-space diffusion (Diffusion-LM).** Gaussian
  noise on word vectors + gradient-guided controllable generation.
- **Lesson 4 — Making masked diffusion practical (LLaDA-style).** Generalization,
  classifier-free guidance, and confidence-based fast sampling.
- **Lesson 5 — SEDD (score-entropy discrete diffusion).** The advanced capstone:
  learn the *ratios* of the distribution instead of the posterior.

---
## Exercises

1. **Confidence sampling.** Replace the random reveal in `sample` with
   MaskGIT/LLaDA-style unmasking: predict all blanks, then commit the top-`k` by
   predicted probability, with `k` on a cosine schedule. Compare quality at small
   `T`. *(You'll build this for real in Lesson 4.)*
2. **Schedule.** Swap the linear $\alpha_t = 1-t$ for a cosine schedule. Re-derive
   the loss weight $\alpha'_t/(1-\alpha_t)$ and update `diffusion_loss`. Does it
   help at this scale?
3. **Steps vs. quality.** Plot a quality proxy (e.g., fraction of samples that
   exactly match a training line) as a function of `T` ∈ {4, 8, 16, 64, 256}.
4. **Uniform kernel (preview of Lesson 2).** Change the forward process to replace
   letters with a *random* letter instead of `[MASK]`. What breaks, and why does
   the absorbing (masking) kernel make the objective simpler?
5. **Bigger data.** Point the corpus at a real text file (still char-level) and see
   how far this tiny model gets.
""")

# ----------------------------------------------------------------------------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
# Write the notebook to the repository root (one level up from tools/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(ROOT, "01_foundations_masked_diffusion.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out_path, "with", len(cells), "cells")
