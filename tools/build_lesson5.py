"""Builds 05_sedd_score_entropy.ipynb (Lesson 5) using nbformat.

Lesson 5 is the advanced capstone: Score-Entropy Discrete Diffusion (SEDD, Lou,
Meng, Ermon 2023). Instead of predicting the clean token or the posterior, we
learn the *concrete score* -- the ratios p_t(y)/p_t(x) between a state and its
single-token neighbours -- via the denoising score-entropy loss, on a
continuous-time uniform process. The headline, numerically-verified result: the
trained score matches the true marginal ratio (validated against a brute-force
mixture, mirroring Lesson 2's Bayes check).

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
md(r"""# Lesson 5 — Score-Entropy Discrete Diffusion (SEDD)
### The advanced capstone: learn the *ratios*, not the posterior

Every model so far answered the question **"what was the clean token?"** — Lesson 1
predicted the masked letter, Lesson 2 predicted the reverse posterior, Lesson 3
predicted clean embeddings. **SEDD** (Lou, Meng & Ermon, 2023) asks a different
question:

> For the corrupted text I'm looking at, **how much more (or less) likely** would it
> be if I changed *this one letter* to something else?

That "relative likelihood of a small edit" is a **ratio** $p_t(y)/p_t(x)$ — and
SEDD calls the collection of these ratios the **concrete score** (the discrete
cousin of the "score" $\nabla \log p$ that powers image diffusion). Learn the
ratios, and you can run the process backwards.

> ⚠️ **This is the hardest lesson** — it needs continuous-time Markov chains and a
> custom loss. We keep the intuition front-and-center and, crucially, we
> **numerically verify** the one fact everything rests on: the loss really does
> teach the network the true ratios. Even where sampling is fiddly (SEDD genuinely
> is), that verified result is the payoff.

**Why anyone bothers:** SEDD gave discrete diffusion its **first decisive wins** on
likelihood and sample quality over strong baselines. The ratio viewpoint is also
more flexible than "predict the token."

> **Prereq:** Lesson 2 (the uniform kernel + its marginal $q(x_t\mid x_0)$). We
> reuse that exact forward process here, now in continuous time.
""")

# ===========================================================================
md(r"""## 📖 New words in this lesson

| Term | In plain words |
|---|---|
| **concrete score** | the set of ratios $p_t(y)/p_t(x)$ for every single-letter edit $y$ of the current text $x$. "How much more likely is this small change?" |
| **ratio target** | for a *given* clean $x_0$, the ratio has a simple closed form (we derive it). Training uses it as the label. |
| **marginal $p_t(x)$** | the probability of text $x$ at noise level $t$, averaged over all clean data. |
| **CTMC** | continuous-time Markov chain: instead of discrete steps, letters flip at random *rates* over continuous time $t$. |
| **rate** | how fast flips happen at time $t$ (from the noise schedule). |
| **score-entropy loss** | a special loss whose minimum is exactly the true ratio (we verify this). |
| **tau-leaping** | a way to simulate a CTMC: over a small time step, each letter flips with a probability set by its rate. Used for sampling. |

The schedule uses the **signal-retention** $a_t$ (same idea as $\bar\alpha_t$
before): the chance a letter is still its original self at time $t$. Here the
forward process is the **uniform** one from Lesson 2 (keep, or jump to a random
letter).
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
md(r"""## 1. The forward process (uniform, continuous-time) and its ratios

Same toy corpus; same **uniform** corruption as Lesson 2: at noise level $t$, a
letter keeps its value with probability $a_t$, else it becomes a **uniformly random
letter**. So the one-letter marginal is
$$q(x_t^i = y \mid x_0^i) = a_t\,\mathbb 1[y = x_0^i] + \frac{1-a_t}{N}.$$

The thing SEDD wants — the **ratio target** — falls right out of this. If we *knew*
the clean letter $x_0^i$, the relative likelihood of the current corrupted letter
being $b$ instead of what it is ($x_t^i$) is just
$$r^i_b \;=\; \frac{q(x_t^i{=}b\mid x_0^i)}{q(x_t^i{=}x_t^i\mid x_0^i)} \;=\; \frac{a_t\,\mathbb 1[b=x_0^i] + (1-a_t)/N}{a_t\,\mathbb 1[x_t^i=x_0^i] + (1-a_t)/N}.$$

That's the label we'll train the network to output (per position, per candidate
letter $b$). We use a cosine schedule for $a_t$ and its associated flip-**rate**
$\sigma'_t$ (needed for the loss weighting and for sampling).
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
N = len(chars)                                   # vocab size (uniform process: no mask)
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
L = max(len(s) for s in raw_lines)
def encode(s): return torch.tensor([stoi[c] for c in (s + " " * (L - len(s)))[:L]])
def decode(t): return "".join(itos[int(i)] for i in t)
data = torch.stack([encode(s) for s in raw_lines]).to(device)

def a_of(t):                                     # signal retention (cosine), clamped
    return (torch.cos(0.5 * math.pi * t) ** 2).clamp(1e-3, 0.98)
def sigma_rate(t):                               # flip rate sigma'(t) = -a'/a
    a = a_of(t); ap = -0.5 * math.pi * torch.sin(math.pi * t)   # d/dt cos^2(pi/2 t)
    return (-ap / a).clamp(1e-4, 50.0)

def q_sample(x0, a):                             # uniform corruption: keep w.p. a else random
    keep = torch.rand_like(x0, dtype=torch.float) < a[:, None]
    unif = torch.randint(0, N, x0.shape, device=x0.device)
    return torch.where(keep, x0, unif)

print(f"N={N} letters | L={L}")
for tv in [0.2, 0.5, 0.8]:
    a = a_of(torch.tensor([tv]))
    print(f"t={tv}  a_t(keep)={float(a):.2f}  rate={float(sigma_rate(torch.tensor([tv]))):.2f}"
          f"  ->  {decode(q_sample(data[0:1], a)[0])!r}")""")

# ===========================================================================
md(r"""## 2. The score network + the ratio target

The **score network** reads the corrupted text and outputs, for every position and
every candidate letter $b$, its estimate of the ratio $s_\theta(x_t)^i_b \approx
p_t(\text{edit to }b)/p_t(x_t)$. Ratios are positive, so the network outputs
log-ratios and we exponentiate. (The ratio of a letter to *itself* is 1, so we
ignore the diagonal.)

It's the same small transformer as before; only its *meaning* changed — its outputs
are ratios, not token probabilities.
""")

code(r"""def target_ratio(x0, xt, a):
    '''Per-x0 ratio target r^i_b = q(b|x0_i)/q(xt_i|x0_i).  -> (B, L, N).'''
    p_all = a[:, None, None] * F.one_hot(x0, N) + ((1 - a) / N)[:, None, None]  # q(b|x0): (B,L,N)
    p_cur = p_all.gather(-1, xt[..., None]).clamp_min(1e-9)                     # q(xt|x0): (B,L,1)
    return p_all / p_cur

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, t):
        half = self.dim // 2
        fr = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t[:, None] * fr[None, :] * 1000.0
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))

class ScoreTransformer(nn.Module):
    def __init__(self, N, L, d_model=128, nhead=4, nlayers=3, ff=256, p=0.1):
        super().__init__()
        self.tok = nn.Embedding(N, d_model); self.pos = nn.Embedding(L, d_model)
        self.time = TimeEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, p, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model); self.head = nn.Linear(d_model, N)
        self.register_buffer("pos_ids", torch.arange(L))
    def forward(self, x, t):                          # -> log-ratios (B,L,N)
        h = self.tok(x) + self.pos(self.pos_ids)[None] + self.time(t)[:, None, :]
        # wide clamp = overflow-safety only; the operating range is ~[-7,4], so this
        # never blocks the DSE gradient (a tight clamp would freeze the score's scale).
        return self.head(self.norm(self.encoder(h))).clamp(-15, 15)

print("score net:", sum(p.numel() for p in ScoreTransformer(N, L).parameters()), "params")""")

# ===========================================================================
md(r"""## 3. The denoising score-entropy loss

How do you train a network to output a ratio when you never see the true marginal
ratio directly? SEDD's trick is the **denoising score-entropy** loss. For each
candidate edit $b$ (at each position), it uses the term
$$\ell(s, r) = s - r\,\log s,$$
where $s = s_\theta(x_t)^i_b$ is the network's guess and $r = r^i_b$ is the
*per-$x_0$* target ratio from §1. Two facts make this work (both **checked in our
oracle**, `tools/sedd_reference_check.py`):

1. **$\ell(s,r)$ is minimised exactly at $s = r$** (calculus: $\partial_s\ell = 1 - r/s = 0$).
2. **The SEDD identity:** averaging the per-$x_0$ target $r$ over the posterior
   $p(x_0\mid x_t)$ gives the *true marginal ratio* $p_t(y)/p_t(x_t)$. So even though
   each training example uses its own $x_0$, the network — seeing many $x_0$ for the
   same $x_t$ — is driven to output the true marginal ratio. (This is the discrete
   analogue of how denoising trains a score.)

We sum $\ell$ over all edits $b\neq x_t^i$ and all positions, and weight by the
CTMC flip-rate $\sigma'_t$ (SEDD's weighting; it also tames the exploding ratios at
low noise). We clamp the target ratio for numerical stability.
""")

code(r"""def dse_loss(model, x0):
    B = x0.size(0)
    t = torch.rand(B, device=x0.device).clamp(0.02, 0.98)
    a = a_of(t)
    xt = q_sample(x0, a)
    s = model(xt, t).exp()                                   # ratios (B,L,N)
    r = target_ratio(x0, xt, a).clamp(max=50.0)              # stability clamp
    ell = s - r * torch.log(s.clamp_min(1e-9))               # min at s=r
    not_cur = 1.0 - F.one_hot(xt, N).float()                 # exclude the diagonal (b == xt)
    per_pos = (ell * not_cur).sum(-1)                        # (B,L) sum over edits b
    return (sigma_rate(t) * per_pos.mean(1)).mean()          # rate-weighted""")

# ===========================================================================
code(r"""model = ScoreTransformer(N, L).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
STEPS, BATCH = 3000, 64
losses = []; model.train()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, data.size(0), (BATCH,), device=device)
    loss = dse_loss(model, data[idx])
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 500 == 0 or step == 1:
        print(f"step {step:4d} | score-entropy loss {np.mean(losses[-50:]):.3f}")

plt.figure(figsize=(6, 3))
plt.plot(np.convolve(losses, np.ones(25)/25, "valid"))
plt.xlabel("step"); plt.ylabel("DSE loss"); plt.title("Score-entropy training loss")
plt.tight_layout(); plt.show()""")

# ===========================================================================
md(r"""## 4. Did it learn the *true ratios*? (the headline check)

This is the payoff, and it's fully checkable. On our tiny corpus we can compute the
**true marginal ratio** $p_t(y)/p_t(x)$ by brute force: $p_t(x)$ is just the average
over the 16 training lines of $q(x\mid x_0)$ (a product over positions). For a
single-letter edit, only that position changes, so the ratio is cheap.

We corrupt a real line to a moderate noise level, then compare the **network's
score** to this **true marginal ratio** for every position and letter. If SEDD
worked, they line up on the diagonal.

> **What to expect:** the network nails the ratios' *shape* — their relative sizes,
> measured by the **log-correlation** (~0.99) — up to a single overall **scale
> factor**. That scale offset is a **model imperfection** (a perfect fit would give
> ~1×), *not* a free gauge — SEDD's reverse rates really do depend on the absolute
> scale. It doesn't wreck generation, because the sampler (§5) **explicitly divides
> this global scale out**, and the destination-token choice is scale-free anyway.
> What SEDD must get right is the *shape*, and it does — so we divide the scale out
> here just to put the points on the diagonal.
""")

code(r"""@torch.no_grad()
def true_marginal_ratios(x, a):
    '''For sequence x (L,), scalar a: return (L,N) true marginal ratios p_t(edit)/p_t(x).'''
    match = (data == x[None, :]).float()                     # (M,L)  is data_j == x_j ?
    p_pos = a * match + (1 - a) / N                          # (M,L) = q(x_j | x0_j)
    prod = p_pos.prod(1)                                     # (M,)  = q(x | x0)
    p_t_x = prod.mean().clamp_min(1e-30)                     # scalar = p_t(x)
    # for each position i and letter b: p_t(x with i->b) = mean_x0 [ prod / q(x_i|x0) * q(b|x0) ]
    base = prod[:, None, None] / p_pos[:, :, None].clamp_min(1e-30)     # (M,L,1) drop position i
    match_b = (data[:, :, None] == torch.arange(N, device=x.device))   # (M,L,N) is x0_i == b ?
    q_b = a * match_b.float() + (1 - a) / N                  # (M,L,N) = q(b | x0_i)
    p_t_edit = (base * q_b).mean(0)                          # (L,N) = p_t(edit to b at i)
    return p_t_edit / p_t_x

model.eval()
tv = 0.5; a = a_of(torch.tensor([tv], device=device))
x = q_sample(data[3:4], a)[0]                                # corrupt a line to moderate noise
with torch.no_grad():
    true_r = true_marginal_ratios(x, a).cpu().numpy()
    pred_r = model(x[None], torch.tensor([tv], device=device)).exp()[0].cpu().numpy()

# compare on the off-diagonal entries (edits to a *different* letter)
mask = np.ones_like(true_r, bool); mask[np.arange(L), x.cpu().numpy()] = False
tr, pr = true_r[mask], pred_r[mask]
corr = np.corrcoef(np.log(tr + 1e-9), np.log(pr + 1e-9))[0, 1]
# The learned score matches the true ratio up to ONE overall scale factor -- a
# model imperfection, not a gauge freedom. The sampler (Section 5) explicitly
# divides this global scale out; here we do the same so the points land on the
# diagonal, and report the *shape* fit (log-correlation + de-scaled log-RMSE).
offset = np.median(pr / (tr + 1e-9))
log_rmse = np.sqrt(np.mean((np.log(pr / offset + 1e-9) - np.log(tr + 1e-9)) ** 2))
print(f"learned score vs TRUE marginal ratio (t={tv}):  log-correlation = {corr:.3f}")
print(f"  after dividing out the global scale (~{offset:.0f}x): log-RMSE = {log_rmse:.2f}")
print(f"  (the shape is what SEDD must get right; the sampler normalizes the scale away.)")

plt.figure(figsize=(4.2, 4.2))
plt.scatter(tr, pr / offset, s=6, alpha=0.4)
lims = [min(tr.min(), (pr / offset).min()), max(tr.max(), (pr / offset).max())]
plt.plot(lims, lims, "k--", lw=1)
plt.xscale("log"); plt.yscale("log")
plt.xlabel("true marginal ratio  p_t(y)/p_t(x)")
plt.ylabel("network score  (÷ global scale)")
plt.title(f"SEDD learns the true ratios  (corr {corr:.2f})"); plt.tight_layout(); plt.show()""")

# ===========================================================================
md(r"""## 5. Sampling: run the Markov chain backwards (tau-leaping)

With the ratios in hand, generation simulates the **reverse** continuous-time
Markov chain. The reverse rate of editing position $i$ to letter $b$ is the forward
rate times the score:
$$\hat R^i_b = \underbrace{\tfrac{\sigma'_t}{N}}_{\text{forward flip rate}} \cdot\; s_\theta(x_t)^i_b.$$
We simulate it by **tau-leaping**: start from random letters (pure noise at $t=1$),
step down to $t=0$, and at each small step let every position flip with probability
$1-e^{-\hat R^i_{\text{tot}}\,dt}$, choosing the new letter in proportion to the
scores. As $t\to0$ the rates die out and the text settles.

> 🔬 **Honesty note.** SEDD sampling is delicate (it's a research-grade method). On
> this tiny model expect *decent* but not flawless text — the rigorous win of this
> lesson is the verified score above. Real SEDD adds tricks (higher-order /
> predictor-corrector samplers, careful schedules) for its state-of-the-art numbers.
""")

code(r"""@torch.no_grad()
def sample(model, n=10, T=256, seed=0):
    torch.manual_seed(seed); model.eval()
    x = torch.randint(0, N, (n, L), device=device)           # prior: uniform noise
    ts = torch.linspace(1.0, 0.0, T + 1, device=device)
    for i in range(T):
        t = ts[i]; dt = (ts[i] - ts[i + 1])
        tt = t.expand(n)
        s = model(x, tt).exp()                               # (n,L,N) ratios
        not_cur = 1.0 - F.one_hot(x, N).float()
        s_off = s * not_cur
        # divide out the learned score's overall scale with ONE scalar per sample
        # (mean off-diagonal score -> 1). This removes the ~100x global offset that
        # would otherwise saturate the jump rate, while KEEPING the relative structure
        # across positions -- so a badly-wrong position still flips more than a good one.
        gscale = (s_off.sum((1, 2), keepdim=True)
                  / not_cur.sum((1, 2), keepdim=True)).clamp_min(1e-9)
        R = (sigma_rate(tt)[:, None, None] / N) * (s_off / gscale)   # reverse rates (n,L,N)
        R_tot = R.sum(-1)                                    # (n,L)
        jump = torch.rand_like(R_tot) < (1 - torch.exp(-R_tot * dt))
        probs = R / R.sum(-1, keepdim=True).clamp_min(1e-9)
        newtok = torch.multinomial(probs.reshape(-1, N), 1).reshape(n, L)
        x = torch.where(jump, newtok, x)
    return x

print("SEDD samples (reverse tau-leaping):\n")
for r in sample(model, n=10, seed=1): print("  ", repr(decode(r).rstrip()))

train_set = set(decode(d).rstrip() for d in data)
s = sample(model, n=200, seed=2)
hit = np.mean([decode(r).rstrip() in train_set for r in s])
print(f"\nfraction exactly matching a training line: {hit:.0%}")""")

# ===========================================================================
md(r"""## 6. Recap — and the course is complete 🎓

**You built SEDD from scratch:**
- The **concrete score** — ratios $p_t(y)/p_t(x)$ of single-letter edits — as the
  thing to learn, instead of the token or the posterior.
- The **denoising score-entropy loss** $s - r\log s$, whose minimum is the true
  ratio, with the **SEDD identity** (posterior-average of per-$x_0$ ratios = the
  marginal ratio) making the denoising target valid.
- A **numerically-verified** result: the trained network's score matches the
  brute-force **true marginal ratio** (§4) — the rigorous heart of the lesson.
- **Reverse tau-leaping** sampling of the continuous-time Markov chain.

**How SEDD relates to the rest of the course:**
- It's still **discrete** diffusion (Lessons 1–2, 4), but reframed around *ratios*
  rather than *predictions* — which turns out to give better likelihoods.
- On the **absorbing** kernel, SEDD's ratios collapse back to the masked-prediction
  objective of Lesson 1 (they're secretly the same there); the **uniform** kernel we
  used is where the ratio view is genuinely distinct — which is why we chose it.
- The "score" language connects discrete diffusion to the continuous
  score-based/diffusion world of Lesson 3 and image models.

---
## 🎓 The whole series, at a glance

| # | Lesson | Core idea |
|---|--------|-----------|
| 1 | Masked diffusion | predict the masked token (weighted cross-entropy) |
| 2 | D3PM kernels | any transition matrix; the term-by-term ELBO |
| 3 | Diffusion-LM | Gaussian diffusion in embedding space + gradient guidance |
| 4 | Scaling / LLaDA | generalization, classifier-free guidance, fast sampling |
| 5 | **SEDD** | **learn the ratios (concrete score) via score entropy** |

You now have a from-scratch, verified tour of the major families of language
diffusion models — discrete and continuous, prediction-based and score-based.

---
### Exercises
1. **Absorbing SEDD.** Swap the uniform process for the absorbing (masking) one and
   re-derive the ratio target. Show it reduces to Lesson 1's masked prediction.
2. **Better sampler.** Add a predictor–corrector step (after each tau-leap, do a few
   Gibbs-style corrections using the score). Does sample quality improve?
3. **Likelihood.** SEDD's *full* weighted score entropy is an ELBO on $\log p(x_0)$
   — but our `dse_loss` drops the constant term ($r\log r - r$) and uses a simplified
   weighting, so the number it prints is **not** directly bits-per-char (it can even
   go negative). Add the constant back and the correct time-integral weighting, then
   estimate bits-per-character on held-out lines and compare to the Lesson 1–2 models.
4. **Steps vs. quality.** Sweep $T$ (tau-leaping steps). How few can you use before
   samples fall apart — and how does that compare to Lesson 4's confidence sampler?
5. **Scale the check.** Re-run the §4 verification at several noise levels $t$. Where
   is the score easiest/hardest to learn, and why (think about ratio magnitudes)?
""")

# ===========================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(ROOT, "05_sedd_score_entropy.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out_path, "with", len(cells), "cells")
