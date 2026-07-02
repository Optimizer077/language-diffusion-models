"""Builds 03_diffusion_lm_embedding_space.ipynb (Lesson 3) using nbformat.

Lesson 3 moves from DISCRETE diffusion (Lessons 1-2) to CONTINUOUS diffusion:
embed tokens into vectors, add Gaussian noise (like image diffusion), train a
denoiser to predict the clean embeddings, and "round" back to tokens. This is the
Diffusion-LM recipe (Li et al. 2022), and it unlocks gradient-guided controllable
generation, which we demo.

Authoring from a script keeps cell sources as plain Python (easy to diff/lint).
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
def md(src):   cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

# ===========================================================================
md(r"""# Lesson 3 — Diffusion in *Continuous* Space (Diffusion-LM)
### Stop corrupting letters; start corrupting *vectors*

In Lessons 1 and 2 the noise acted directly on **letters**: blank one out, or swap
it for a random one. Everything lived in the discrete world of the vocabulary.

This lesson does something that sounds almost illegal for text: it turns each
letter into a **list of numbers** (a vector) and then adds ordinary **bell-curve
(Gaussian) noise** to those numbers — exactly the way image diffusion blurs a
photo into static. Then a network learns to remove the noise, and we **round** the
cleaned-up vectors back into letters.

This is **Diffusion-LM** (Li et al., 2022). Why bother, when discrete diffusion
already works?

> **Because a continuous space has *directions you can push in*.** Once text lives
> in a smooth vector space, you can **steer** generation with gradients — "make the
> output start with `my`", "make it more positive", "hit this length" — *without
> retraining the model*. That kind of whole-sequence, plug-and-play control is
> Diffusion-LM's headline trick — much more natural here than for autoregressive
> models (which can only nudge one step at a time). We build it at the end.

**What you'll build & learn**
1. **Embeddings**: turning letters into vectors, and a "rounding" step to get back.
2. The **Gaussian forward process** — the same math as image diffusion (DDPM),
   now applied to embedding vectors.
3. A denoiser that **predicts the clean vectors**, trained with a mean-squared
   error + a rounding cross-entropy.
4. **Sampling** by running Gaussian denoising backwards (with the "clamping" trick
   that keeps it honest about being text).
5. **Controllable generation**: nudging the sampler with gradients to satisfy a
   constraint — the payoff of going continuous.

> **Prereqs:** Lessons 1–2 (you know the forward/denoise/sample loop and the
> schedule $\bar\alpha_t$). Everything continuous-specific is explained here.
""")

# ===========================================================================
md(r"""## 📖 New words in this lesson (plain English)

| Term | In plain words |
|---|---|
| **embedding** | a letter turned into a short list of numbers (a vector). Similar letters can sit near each other. |
| **embedding space** | the continuous space those vectors live in ($\mathbb R^d$, here $d$ numbers per letter). |
| **Gaussian noise** | "bell-curve" random numbers — the classic static/fuzz you add to blur something. |
| **DDPM** | the standard image-diffusion recipe (Ho et al. 2020) for adding/removing Gaussian noise. We reuse its formulas. |
| **rounding / readout** | turning a (denoised) vector back into an actual letter — "which letter's embedding is this closest to?" |
| **$z_0$** | the clean **embedding vectors** of a sentence (continuous), vs. $x_0$ = the letters (discrete). |
| **$z_t$** | the embeddings after $t$ steps of Gaussian noise. |
| **predict-$x_0$** | we train the network to output the *clean* vectors $z_0$ directly (not the noise). |
| **guidance** | nudging the sampler each step, using the gradient of some "I want *this*" objective. |

The schedule $\bar\alpha_t$ is the same idea as before: how much of the original
**signal survives** after $t$ steps (now: how much of the clean vector remains vs.
how much is noise).
""")

# ===========================================================================
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

# ===========================================================================
md(r"""## 1. Data (same toy corpus)

Same tiny char-level corpus as before, padded with spaces. No `[MASK]` token this
time — continuous diffusion has no "blank" state; the "noise" is just fuzz on the
vectors. So the vocabulary is exactly the $K$ real characters.
""")

code(r"""raw_lines = [
    "the cat sat on the mat",   "the dog ran in the fog",
    "the sun is in the sky",    "the fish swims in the sea",
    "the bird flew over the hill", "the fox hid in the den",
    "a red car drove down the road", "a blue boat sailed on the lake",
    "the old man read a good book", "the girl drew a big green tree",
    "we ran fast to the bus stop",  "they sang songs all night long",
    "the rain fell soft on the roof", "the moon rose high in the dark",
    "my cat likes to nap in the sun", "the kids ran home for the meal",
]

chars = sorted(set("".join(raw_lines)))
K = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
L = max(len(s) for s in raw_lines)
def encode(s): return torch.tensor([stoi[c] for c in (s + " " * (L - len(s)))[:L]])
def decode(t): return "".join(itos[int(i)] for i in t)

data = torch.stack([encode(s) for s in raw_lines]).to(device)
print(f"K = {K} letters | block length L = {L}")
print("example:", repr(decode(data[0])))""")

# ===========================================================================
md(r"""## 2. Letters → vectors → letters (embeddings & rounding)

Two pieces bridge the discrete and continuous worlds:

- **Embed** (letters → vectors): a learnable table $E$ with one $d$-dimensional
  vector per letter. We use the *direction* of each vector and give it a fixed
  length $\sqrt d$, so every letter-vector has the same magnitude — this keeps the
  "signal" and the Gaussian "noise" on the same scale (important for the noise to
  actually compete with the signal).
- **Round / read out** (vectors → letters): given any vector $v$, score each letter
  by how *aligned* $v$ is with that letter's embedding (cosine similarity), and
  softmax into letter probabilities. Rounding = pick the most-aligned letter.

Both the embeddings and this readout are **learned** — the model gets to design a
vector space in which its own denoising is easy. (Diffusion-LM's key idea: learn
the embeddings end-to-end with the diffusion.)
""")

code(r"""d = 32                      # embedding dimension
LOGIT_SCALE = 10.0          # sharpness of the cosine readout

emb = nn.Embedding(K, d).to(device)
nn.init.normal_(emb.weight, std=1.0)

def to_z0(x):
    '''letters (..,) -> clean embedding vectors (..,d), each of length sqrt(d).'''
    return F.normalize(emb(x), dim=-1) * math.sqrt(d)

def readout(v):
    '''vectors (..,d) -> letter logits (..,K) via cosine similarity to each embedding.'''
    vn = F.normalize(v, dim=-1)
    en = F.normalize(emb.weight, dim=-1)      # (K, d)
    return LOGIT_SCALE * vn @ en.t()

# sanity: a clean embedding rounds back to its own letter
z = to_z0(data[0])
print("round-trip letters == original:", bool((readout(z).argmax(-1) == data[0]).all()))
print("z0 shape:", tuple(z.shape), "| per-vector length ~", round(float(z[0].norm()), 2))""")

# ===========================================================================
md(r"""## 3. The Gaussian forward process (image-diffusion math, on vectors)

Now the noise. This is **exactly DDPM** (the standard image-diffusion recipe),
applied to our embedding vectors. With the same survival schedule $\bar\alpha_t$
(1 at the start, ~0 at the end), the noised vectors at step $t$ are

$$z_t = \sqrt{\bar\alpha_t}\; z_0 \;+\; \sqrt{1-\bar\alpha_t}\;\epsilon,\qquad \epsilon\sim\mathcal N(0, I).$$

In words: **keep a $\sqrt{\bar\alpha_t}$ fraction of the clean vector, and add
Gaussian fuzz for the rest.** Early on ($\bar\alpha_t\approx 1$) it's basically the
clean vector; by the end ($\bar\alpha_t\approx 0$) it's pure $\mathcal N(0,I)$
static with no trace of the original. Because our clean vectors already have
per-coordinate variance ~1, the fully-noised $z_T$ matches standard normal noise —
which is where generation will start.

Let's watch a real sentence dissolve into static (we *round* the noisy vectors
back to letters at each level, just to visualize what the fuzz does to the text).
""")

code(r"""def make_schedule(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0]).clamp(1e-6, 1.0)
    return abar.to(device).float()

T = 200
abar = make_schedule(T)      # abar[t] = fraction of signal that survives to step t

def q_sample(z0, t, eps):
    '''Add t steps of Gaussian noise to clean vectors z0. t:(B,), z0:(B,L,d).'''
    ab = abar[t].view(-1, 1, 1)
    return ab.sqrt() * z0 + (1 - ab).sqrt() * eps

# Visualize: round the noised vectors back to letters at increasing noise.
z0 = to_z0(data[0:1])
print("original:", repr(decode(data[0])), "\n")
for frac in [0.1, 0.3, 0.5, 0.8]:
    t = torch.tensor([int(frac * T)], device=device)
    zt = q_sample(z0, t, torch.randn_like(z0))
    print(f"t={int(frac*T):3d} (signal {float(abar[t]):.2f}) -> {decode(readout(zt)[0].argmax(-1))!r}")""")

# ===========================================================================
md(r"""## 4. The denoiser: predict the *clean vectors*

The network reads the noisy vectors $z_t$ (and the time $t$) and outputs its guess
of the **clean vectors** $z_0$. It's the same transformer as before, but now its
input and output are continuous $d$-dimensional vectors instead of token IDs:

- an input `Linear` maps each $d$-vector up to the model width,
- the transformer mixes information across positions (bidirectional, as always),
- an output `Linear` maps back down to a $d$-vector — the predicted clean embedding.

Predicting the clean signal $z_0$ (rather than the noise $\epsilon$) is the
**"predict-$x_0$" parameterization** — the same choice we made in Lessons 1–2, and
it pairs naturally with the rounding step.
""")

code(r"""class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t[:, None] * freqs[None, :] * 1000.0
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))

class GaussDenoiser(nn.Module):
    def __init__(self, d, L, d_model=128, nhead=4, nlayers=3, ff=256, p=0.1):
        super().__init__()
        self.inp = nn.Linear(d, d_model)
        self.pos = nn.Embedding(L, d_model)
        self.time = TimeEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, p, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model); self.out = nn.Linear(d_model, d)
        self.register_buffer("pos_ids", torch.arange(L))
    def forward(self, z_t, t):                       # z_t:(B,L,d), t:(B,) in (0,1]
        h = self.inp(z_t) + self.pos(self.pos_ids)[None] + self.time(t)[:, None, :]
        return self.out(self.norm(self.encoder(h)))  # predicted clean z0  (B,L,d)

print("denoiser:", sum(p.numel() for p in GaussDenoiser(d, L).parameters()), "params")""")

# ===========================================================================
md(r"""## 5. The loss: match the vectors **and** stay decodable

Two jobs, two loss terms:

1. **Denoise (MSE).** Make the predicted clean vectors match the true ones:
   $\; \lVert \hat z_0 - z_0 \rVert^2$. This is the standard diffusion loss, just
   on embeddings. *(We stop gradients through the target $z_0$ so the network
   chases the embeddings, rather than the embeddings collapsing to meet the network
   — a known failure mode.)*
2. **Round (cross-entropy).** Make sure vectors actually decode to the right
   letters: cross-entropy of `readout(·)` against the true letters — applied to
   *both* the clean embeddings (so the embedding space is decodable at all) and the
   network's prediction (so its output lands on the right letter). This term is
   what **learns the embeddings**.

$$\mathcal L = \underbrace{\lVert \hat z_0 - z_0 \rVert^2}_{\text{denoise}} \;+\; \underbrace{\mathrm{CE}(\mathrm{readout}(\hat z_0), x_0) + \mathrm{CE}(\mathrm{readout}(z_0), x_0)}_{\text{stay decodable / learn embeddings}}.$$

As before, we Monte-Carlo over $t$: each step, pick a random noise level per
example.
""")

code(r"""def diffusion_lm_loss(model, x0):
    B = x0.size(0)
    z0 = to_z0(x0)                                  # (B,L,d) clean vectors
    t = torch.randint(1, T + 1, (B,), device=x0.device)
    eps = torch.randn_like(z0)
    zt = q_sample(z0, t, eps)
    z0_hat = model(zt, t.float() / T)               # predict clean vectors
    L_mse = F.mse_loss(z0_hat, z0.detach())
    ce = lambda v: F.cross_entropy(readout(v).reshape(-1, K), x0.reshape(-1))
    L_round = ce(z0_hat) + ce(z0)
    return L_mse + L_round, L_mse.item(), L_round.item()""")

# ===========================================================================
md(r"""## 6. Train

We train the denoiser **and** the embeddings together (the optimizer sees both).
Watch the rounding cross-entropy fall — that's the model carving out a vector
space where each letter has its own well-separated region.
""")

code(r"""model = GaussDenoiser(d, L).to(device)
opt = torch.optim.AdamW(list(model.parameters()) + list(emb.parameters()),
                        lr=2e-3, weight_decay=1e-4)
STEPS, BATCH = 4000, 64
hist = []
model.train()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, data.size(0), (BATCH,), device=device)
    loss, mse, rnd = diffusion_lm_loss(model, data[idx])
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(list(model.parameters()) + list(emb.parameters()), 1.0)
    opt.step(); hist.append((mse, rnd))
    if step % 500 == 0 or step == 1:
        print(f"step {step:4d} | mse {np.mean([h[0] for h in hist[-50:]]):.3f}"
              f" | round-CE {np.mean([h[1] for h in hist[-50:]]):.3f}")

h = np.array(hist)
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].plot(np.convolve(h[:,0], np.ones(25)/25, "valid")); ax[0].set_title("MSE (denoise)")
ax[1].plot(np.convolve(h[:,1], np.ones(25)/25, "valid")); ax[1].set_title("rounding CE")
for a in ax: a.set_xlabel("step")
plt.tight_layout(); plt.show()""")

# ===========================================================================
md(r"""## 7. Sampling: run the Gaussian denoising backwards

Generation is the **DDPM reverse process**: start from pure static
$z_T\sim\mathcal N(0,I)$, and step down $t = T\to 1$. At each step the model
predicts the clean vectors $\hat z_0$, and DDPM's formula tells us how to take one
small denoising step toward it (a weighted blend of "where we are" and "where the
model thinks clean is", plus a little noise). Finally we **round** to letters.

We add the Diffusion-LM **clamping trick**: at each step, snap the predicted clean
vector to the *nearest real letter's* embedding before stepping. This constantly
reminds the continuous process that it's ultimately producing discrete text, and
dramatically sharpens the samples.
""")

code(r"""def ddpm_mean(z_t, z0_hat, t):
    '''DDPM posterior mean of z_{t-1} given z_t and predicted z0 (t: python int).'''
    ab_t, ab_tm1 = abar[t], abar[t - 1]
    beta_t = 1 - ab_t / ab_tm1
    coef_z0 = ab_tm1.sqrt() * beta_t / (1 - ab_t)
    coef_zt = (ab_t / ab_tm1).sqrt() * (1 - ab_tm1) / (1 - ab_t)
    return coef_z0 * z0_hat + coef_zt * z_t, beta_t * (1 - ab_tm1) / (1 - ab_t)

@torch.no_grad()
def sample(model, n=10, clamp=True, seed=1):
    torch.manual_seed(seed); model.eval()
    z = torch.randn(n, L, d, device=device)
    for t in range(T, 0, -1):
        tt = torch.full((n,), t, device=device)
        z0_hat = model(z, tt.float() / T)
        if clamp:                                   # snap to nearest real letter
            z0_hat = to_z0(readout(z0_hat).argmax(-1))
        mean, var = ddpm_mean(z, z0_hat, t)
        z = mean + (var.sqrt() * torch.randn_like(z) if t > 1 else 0.0)
    return readout(z).argmax(-1)

print("=== samples (with clamping) ===")
for r in sample(model, n=10): print("  ", repr(decode(r).rstrip()))""")

# ===========================================================================
md(r"""## 8. The payoff: **controllable** generation with gradients

Here's what the continuous space buys us. Suppose we want samples that **start
with a chosen prefix** (say `"my "`). We don't retrain anything. Instead, at each
sampling step we:

1. ask the model for its clean-vector guess $\hat z_0$,
2. measure a **control loss** — how badly the first few letters miss the target
   prefix (cross-entropy of the readout vs. the target letters),
3. take the **gradient of that control loss with respect to $z_t$**, and nudge
   $z_t$ in the direction that lowers it.

That's **guidance**: the smooth vector space lets us *push* a sample toward any
differentiable objective. (Swap the objective for a sentiment classifier, a length
target, a "contains word X" score, … — same mechanism.) This is *far more awkward*
for autoregressive models — methods like PPLM (Dathathri et al. 2020) do steer a
*frozen* GPT, but by nudging per-step hidden states; here we take gradients over the
**whole sequence at once** in a smooth space, in a few lines.

The `gscale` knob sets *how hard* we push: turn it up and more samples satisfy the
constraint, but push too hard and fluency suffers (Exercise 4 sweeps it). We use a
moderate value that clearly steers while keeping the text valid.

Below: how often do unguided vs. guided samples start with `"my "`?
""")

code(r"""def sample_guided(model, prefix, n=10, gscale=6.0, clamp=True, seed=2):
    torch.manual_seed(seed); model.eval()
    P = len(prefix)
    target = encode(prefix)[:P].to(device).unsqueeze(0).expand(n, P)   # (n,P)
    z = torch.randn(n, L, d, device=device)
    for t in range(T, 0, -1):
        tt = torch.full((n,), t, device=device)
        z = z.detach().requires_grad_(True)
        z0_hat = model(z, tt.float() / T)
        ctrl = F.cross_entropy(readout(z0_hat)[:, :P].reshape(-1, K), target.reshape(-1))
        grad, = torch.autograd.grad(ctrl, z)
        with torch.no_grad():
            zc = to_z0(readout(z0_hat).argmax(-1)) if clamp else z0_hat
            mean, var = ddpm_mean(z.detach(), zc, t)
            z = mean - gscale * grad                                    # <-- the nudge
            if t > 1: z = z + var.sqrt() * torch.randn_like(z)
    return readout(z).argmax(-1)

def frac_startswith(toks, prefix):
    return np.mean([decode(r).startswith(prefix) for r in toks])

pre = "my "
uncond = sample(model, n=40, seed=5)
guided = sample_guided(model, pre, n=40, gscale=6.0, seed=5)
print(f"start with {pre!r}:  unguided {frac_startswith(uncond, pre):.0%}"
      f"   guided {frac_startswith(guided, pre):.0%}  (gscale=6)\n")
print("guided samples:")
for r in guided[:10]: print("  ", repr(decode(r).rstrip()))""")

# ===========================================================================
md(r"""## 9. Recap & what's next

**You built a continuous (embedding-space) language diffusion model — Diffusion-LM:**

- **Letters ↔ vectors.** A learned embedding turns letters into vectors; a cosine
  **readout** rounds vectors back to letters.
- **Gaussian forward process.** The exact **DDPM** image-diffusion math, applied to
  embeddings: keep $\sqrt{\bar\alpha_t}$ of the signal, add bell-curve noise.
- **Predict-$z_0$ denoiser** trained with a **denoising MSE + a rounding
  cross-entropy** that learns the embeddings end-to-end.
- **DDPM sampling** from $\mathcal N(0,I)$ with the **clamping trick** for sharp text.
- **Gradient guidance** — the big payoff: steer generation toward any
  differentiable objective *without retraining*.

**Discrete vs. continuous — the trade-off.**

| | Discrete (L1–2) | Continuous (this lesson) |
|---|---|---|
| Noise | mask / random token | Gaussian fuzz on vectors |
| Extra machinery | none | embeddings + rounding |
| Likelihoods | clean (it's the ELBO) | awkward (rounding breaks the bound) |
| Control | harder | **gradient guidance is natural** |

Continuous diffusion is finicky (embeddings can collapse; rounding is lossy), which
is why most *recent* text-diffusion work went back to the discrete/masking side
(MDLM, LLaDA). But the continuous route remains the cleanest way to get
**gradient-based control**, and it connects language diffusion to the whole image-
diffusion toolbox.

**Coming in Lesson 4 — Making masked diffusion practical (LLaDA-style):** back to
the discrete/masking side, scaled up with the tricks that make it usable —
generalization, classifier-free guidance, and fast confidence-based sampling. Then
**Lesson 5 (SEDD)** closes the course by learning the **ratios** of the data
distribution (a discrete "score") instead of the posterior.

---
### Exercises
1. **Predict-$\epsilon$ instead.** Re-parameterize the model to output the noise
   $\epsilon$ (as in vanilla DDPM) and recover $\hat z_0$ from it. Does quality change?
2. **Turn off clamping.** Set `clamp=False` in `sample`. How much worse are the
   samples, and why does snapping-to-nearest-letter help so much?
3. **A different control.** Swap the prefix objective for "the sample should
   *contain* the word `cat`" (cross-entropy at a sliding window). Tune `gscale`.
4. **Guidance strength.** Sweep `gscale` ∈ {0, 0.5, 1, 2, 4, 8}. Plot "fraction
   matching the constraint" vs. "fraction still valid English". Find the sweet spot
   (too strong → gibberish that technically satisfies the constraint).
5. **Fewer steps (DDIM).** Replace the DDPM sampler with a deterministic DDIM one
   and generate with 20–50 steps instead of 200. How much quality do you lose?
""")

# ===========================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(ROOT, "03_diffusion_lm_embedding_space.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out_path, "with", len(cells), "cells")
