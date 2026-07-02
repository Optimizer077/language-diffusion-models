"""Builds 04_scaling_diffusion_llm.ipynb (Lesson 4) using nbformat.

Lesson 4 makes the masked diffusion LM of Lesson 1 *practical*, the way LLaDA /
MaskGIT do: train on a bigger (grammar-generated) corpus and measure real
generalization; add a class label + classifier-free guidance for conditional
generation; and replace random unmasking with confidence-based unmasking, then
show the steps-vs-quality (NFE) trade-off.

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
md(r"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Optimizer077/language-diffusion-models/blob/main/04_scaling_diffusion_llm.ipynb)

# Lesson 4 — Making Masked Diffusion *Practical* (LLaDA-style)
### Generalization, guidance, and fast sampling

Lessons 1–3 built the *ideas*. This lesson is about the *engineering* that turns a
masked diffusion model (Lesson 1) into something you'd actually want to use — the
tricks behind **LLaDA** and **MaskGIT**.

> **A note on "scaling."** We can't train a billion-parameter model on a laptop, so
> "scaling up" here means the **techniques that make masked diffusion work in
> practice** — not raw size. The tiny model stays tiny; what changes is *how* we
> train, condition, and sample it.

We reuse Lesson 1's masking model and add three practical pillars:

1. **Beyond memorization → generalization.** Lessons 1–3 used 16 fixed lines, so
   the model just memorized. Here we generate **hundreds of sentences from a
   grammar**, train on *some* of them, and measure whether the model invents *new,
   grammatically-valid* sentences it never saw. That's real generalization.
2. **Conditional generation with Classifier-Free Guidance (CFG).** We give each
   sentence a **category label** and teach one model to generate both with and
   without it — then *dial up* how strongly it obeys the category at sampling time.
   This is how real diffusion models do controllable generation.
3. **Confidence-based sampling & the speed/quality trade-off.** We swap Lesson 1's
   *random* unmasking for **"unmask what you're most sure of first"** (MaskGIT /
   LLaDA), and show it stays good with **far fewer steps** — the key to fast
   generation.

> **Prereq:** Lesson 1 (masking forward process, the denoiser, ancestral
> sampling). We reuse all of it.
""")

# ===========================================================================
md(r"""## 📖 New words in this lesson

| Term | In plain words |
|---|---|
| **generalization** | producing valid *new* outputs, not just copies of the training data. The real test of learning. |
| **conditioning** | telling the model *what kind* of thing to generate (here: a category label). |
| **classifier-free guidance (CFG)** | a dial that controls *how strongly* the model obeys the condition, by comparing its "with label" and "without label" predictions. |
| **guidance weight $w$** | the CFG dial. $w=0$ = plain conditional; higher $w$ = obey the label harder (until it starts breaking). |
| **confidence-based sampling** | when unmasking, fill in the tokens the model is **most sure about first** (instead of random ones). From MaskGIT. |
| **NFE (number of function evaluations)** | how many times we run the network to generate one sample = the number of sampling **steps** $T$ (classifier-free guidance runs it twice per step, so NFE = $2T$ there). Fewer = faster. |
| **steps↔quality trade-off** | fewer steps = faster but usually worse; a good sampler stays good with fewer steps. |

We keep Lesson 1's masking: corruption replaces letters with `[MASK]`; the noise
level is a continuous $t\in(0,1]$ with keep-probability $\alpha_t = 1-t$; training
is the $1/t$-weighted cross-entropy on masked positions.
""")

# ===========================================================================
code(r"""import math, random, itertools
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
md(r"""## 1. A bigger corpus from a *grammar*

To test generalization we need more than 16 lines and a way to check whether a
*new* sentence is "correct." So we define a tiny **grammar**: every sentence has
the frame **`the <subject> <verb> the <object>`**, and there are three
**categories**, each with its own words:

- **animal** — subjects like `cat/dog`, objects like `mouse/bird`
- **machine** — subjects like `car/bus`, objects like `truck/train`
- **sky** — subjects like `sun/moon`, objects like `hill/lake`

A sentence is **valid** if its subject, verb, and object all come from the *same*
category (the subject and object lists are distinct per category, so mixing them —
`"the cat chased the truck"` — is invalid). With 4 subjects × 4 verbs × 4 objects
per category × 3 categories, that's **192 valid sentences**. We **train on ~70%**
and hold out the rest, so we can later ask: *does the model generate valid
sentences it never saw?*
""")

code(r"""CATS = {
    "animal":  (["cat", "dog", "fox", "owl"],  ["chased", "liked", "watched", "found"], ["mouse", "bird", "bone", "fish"]),
    "machine": (["car", "bus", "van", "cab"],  ["passed", "towed", "chased", "met"],    ["truck", "train", "boat", "bike"]),
    "sky":     (["sun", "moon", "star", "cloud"], ["lit", "hid", "warmed", "chased"],   ["hill", "lake", "town", "field"]),
}
CAT_NAMES = list(CATS)
NULL = len(CAT_NAMES)                 # the "no label" id used for unconditional / CFG

# Enumerate every valid sentence and its category label.
all_sents, all_labels = [], []
for ci, (S, V, O) in enumerate(CATS.values()):
    for s, v, o in itertools.product(S, V, O):
        all_sents.append(f"the {s} {v} the {o}"); all_labels.append(ci)

# Deterministic 70/30 split: hold out every sentence whose index mod 10 is < 3.
train_sents = [s for i, s in enumerate(all_sents) if i % 10 >= 3]
train_labels = [l for i, l in enumerate(all_labels) if i % 10 >= 3]
valid_set = set(all_sents); train_set = set(train_sents)
held_out = valid_set - train_set

def is_valid(sent):
    '''Return the category id if `sent` fits the grammar, else None.'''
    tok = sent.split()
    if len(tok) != 5 or tok[0] != "the" or tok[3] != "the": return None
    s, v, o = tok[1], tok[2], tok[4]
    for ci, (S, V, O) in enumerate(CATS.values()):
        if s in S and v in V and o in O: return ci
    return None

# Char-level vocab (+ a MASK state). Pad with spaces (space is a normal char).
chars = sorted(set("".join(all_sents)))
K = len(chars)                        # data classes the model predicts over
MASK_ID = K; V_in = K + 1             # embedding vocab includes [MASK]
stoi = {c: i for i, c in enumerate(chars)}; itos = {i: c for c, i in stoi.items()}
itos[MASK_ID] = "#"
L = max(len(s) for s in all_sents)
def encode(s): return torch.tensor([stoi[c] for c in (s + " " * (L - len(s)))[:L]])
def decode(t): return "".join(itos[int(i)] for i in t)

data = torch.stack([encode(s) for s in train_sents]).to(device)
labels = torch.tensor(train_labels, device=device)
print(f"{len(all_sents)} valid sentences | train {len(train_sents)} | held-out {len(held_out)}")
print(f"K={K} chars | L={L} | categories={CAT_NAMES}")
print("examples:", [f"({CAT_NAMES[l]}) {s}" for s, l in list(zip(train_sents, train_labels))[:3]])""")

# ===========================================================================
md(r"""## 2. Masking + training (recap from Lesson 1)

Nothing new here — this is Lesson 1's masking forward process and $1/t$-weighted
loss. The **only** addition: during training we randomly **drop the label** (20%
of the time, replace it with the `NULL` "no label" id). That single trick lets
*one* model act both **conditionally** (label given) and **unconditionally** (no
label) — which is exactly what classifier-free guidance needs later.
""")

code(r"""def forward_mask(x0, t):
    '''Replace each letter with [MASK] with probability (1 - alpha_t) = t.'''
    keep = torch.rand_like(x0, dtype=torch.float) < (1 - t).unsqueeze(1)
    return torch.where(keep, x0, torch.full_like(x0, MASK_ID)), ~keep

def diffusion_loss(model, x0, lab, p_drop=0.2):
    B = x0.size(0)
    t = torch.rand(B, device=x0.device).clamp(min=1e-3)
    lab = lab.clone()
    lab[torch.rand(B, device=x0.device) < p_drop] = NULL      # label dropout for CFG
    x_t, masked = forward_mask(x0, t)
    logits = model(x_t, t, lab)                               # (B,L,K)
    ce = F.cross_entropy(logits.reshape(-1, K), x0.reshape(-1), reduction="none").reshape(B, L)
    per_ex = (ce * masked.float()).sum(1) / masked.float().sum(1).clamp(min=1)
    return ((1.0 / t) * per_ex).mean()""")

# ===========================================================================
md(r"""## 3. The conditional denoiser

Same transformer as Lesson 1, with **one extra input**: a **label embedding**
added alongside the token, position, and time embeddings. The label is the
sentence's category (or `NULL`). That's all it takes to make the model
controllable.
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

class CondDenoiser(nn.Module):
    def __init__(self, V_in, K_out, n_labels, L, d_model=128, nhead=4, nlayers=3, ff=256, p=0.1):
        super().__init__()
        self.tok = nn.Embedding(V_in, d_model)
        self.pos = nn.Embedding(L, d_model)
        self.time = TimeEmbedding(d_model)
        self.lab = nn.Embedding(n_labels + 1, d_model)      # +1 for the NULL label
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, p, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model); self.head = nn.Linear(d_model, K_out)
        self.register_buffer("pos_ids", torch.arange(L))
    def forward(self, x_t, t, label):                        # x_t:(B,L) t:(B,) label:(B,)
        h = (self.tok(x_t) + self.pos(self.pos_ids)[None]
             + self.time(t)[:, None, :] + self.lab(label)[:, None, :])
        return self.head(self.norm(self.encoder(h)))         # (B,L,K)

print("denoiser:", sum(p.numel() for p in
      CondDenoiser(V_in, K, len(CAT_NAMES), L).parameters()), "params")""")

# ===========================================================================
code(r"""model = CondDenoiser(V_in, K, len(CAT_NAMES), L).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
STEPS, BATCH = 3000, 64
losses = []; model.train()
for step in range(1, STEPS + 1):
    idx = torch.randint(0, data.size(0), (BATCH,), device=device)
    loss = diffusion_loss(model, data[idx], labels[idx])
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    losses.append(loss.item())
    if step % 500 == 0 or step == 1:
        print(f"step {step:4d} | loss {np.mean(losses[-50:]):.4f}")

plt.figure(figsize=(6, 3))
plt.plot(np.convolve(losses, np.ones(25)/25, "valid"))
plt.xlabel("step"); plt.ylabel("loss"); plt.title("training loss"); plt.tight_layout(); plt.show()""")

# ===========================================================================
md(r"""## 4. Two ways to unmask: random vs. confidence

At generation time we start from all-`[MASK]` and reveal letters over $T$ steps.
**How** we choose which to reveal matters a lot:

- **Random (Lesson 1):** each step, reveal a random subset of masked positions.
- **Confidence (MaskGIT / LLaDA):** each step, predict *all* masked positions, then
  **commit only the ones the model is most confident about**, and re-mask the rest
  to reconsider next step. Confident letters (which pin down the sentence) get
  locked in first, so the rest fall into place — much better when steps are scarce.

Both take a `label` (a category, or `None` for unconditional) and an optional CFG
weight (next section). The `_logits` helper handles conditioning + guidance in one
place.
""")

code(r"""def _logits(model, x, t_vec, label, cfg_w):
    '''Logits with optional classifier-free guidance.
       label=None -> unconditional; label=(n,) LongTensor -> conditional;
       cfg_w>0 amplifies the label (cond + w*(cond - uncond)).'''
    n = x.size(0)
    null = torch.full((n,), NULL, device=x.device)
    if label is None:
        return model(x, t_vec, null)
    cond = model(x, t_vec, label)
    if cfg_w == 0: return cond
    return cond + cfg_w * (cond - model(x, t_vec, null))

@torch.no_grad()
def sample_random(model, T, n=64, label=None, cfg_w=0.0, temperature=1.0, seed=0):
    torch.manual_seed(seed); model.eval()
    x = torch.full((n, L), MASK_ID, device=device)
    ts = torch.linspace(1.0, 0.0, T + 1, device=device)
    for i in range(T):
        t, s = ts[i], ts[i + 1]; sm = x == MASK_ID
        if not sm.any(): break
        logits = _logits(model, x, t.expand(n), label, cfg_w) / temperature
        pred = torch.multinomial(logits.softmax(-1).reshape(-1, K), 1).reshape(n, L)
        reveal = (torch.rand_like(x, dtype=torch.float) < ((t - s) / t).clamp(0, 1)) & sm
        x = torch.where(reveal, pred, x)
    if (x == MASK_ID).any():
        logits = _logits(model, x, torch.full((n,), 1e-3, device=device), label, cfg_w)
        x = torch.where(x == MASK_ID, logits.argmax(-1), x)
    return x

@torch.no_grad()
def sample_confidence(model, T, n=64, label=None, cfg_w=0.0, temperature=1.0, seed=0):
    '''MaskGIT/LLaDA-style: commit the most-confident letters first, re-mask the
       rest to reconsider. Gumbel-noised reveal order (annealed) gives diversity.'''
    torch.manual_seed(seed); model.eval()
    x = torch.full((n, L), MASK_ID, device=device)
    for i in range(T):
        masked = x == MASK_ID
        t_vec = masked.float().mean(1)                       # current noise level per sample
        probs = (_logits(model, x, t_vec, label, cfg_w) / temperature).softmax(-1)
        pred = torch.multinomial(probs.reshape(-1, K), 1).reshape(n, L)   # sample (not argmax)
        conf = probs.gather(-1, pred.unsqueeze(-1)).squeeze(-1).clamp_min(1e-9).log()
        anneal = temperature * (1 - (i + 1) / T)             # explore early, exploit late
        gumbel = -torch.log(-torch.log(torch.rand_like(conf).clamp_min(1e-9)).clamp_min(1e-9))
        score = (conf + anneal * gumbel).masked_fill(~masked, float("inf"))
        newx = torch.where(masked, pred, x)                  # fill masked; keep committed
        n_mask = int(math.floor(L * math.cos(math.pi / 2 * (i + 1) / T)))  # keep this many masked
        if n_mask > 0:                                       # re-mask least-confident
            idx = score.topk(n_mask, dim=1, largest=False).indices
            newx.scatter_(1, idx, MASK_ID)
        x = newx
    return x

# quick look: conditional generation with random categories
_lab = torch.randint(0, len(CAT_NAMES), (6,), device=device)
print("conditional samples (confidence, T=24):")
for r, l in zip(sample_confidence(model, T=24, n=6, label=_lab, seed=1), _lab):
    print(f"  ({CAT_NAMES[l]:7s})", repr(decode(r).rstrip()))""")

# ===========================================================================
md(r"""## 5. Did it *generalize*? (new valid sentences, not copies)

The real question: does the model produce **valid** sentences (grammar-correct,
categories not mixed), and among those, **novel** ones it never saw in training?
We generate a batch and score it against the grammar.
""")

code(r"""def score(samples):
    dec = [decode(r).rstrip() for r in samples]
    valid = [s for s in dec if is_valid(s) is not None]
    novel = [s for s in valid if s in held_out]              # valid AND never trained on
    return len(valid) / len(dec), len(novel) / len(dec), dec

gen = sample_confidence(model, T=24, n=300,
                        label=torch.randint(0, len(CAT_NAMES), (300,), device=device), seed=3)
v, nvl, dec = score(gen)
print(f"valid: {v:.0%}   |   novel-and-valid (never seen in training): {nvl:.0%}")
print("\nsome NOVEL valid sentences the model invented (held-out combos):")
seen = set()
for s in dec:
    if s in held_out and s not in seen:
        seen.add(s); print("  ", repr(s))
    if len(seen) >= 8: break""")

# ===========================================================================
md(r"""## 6. Conditioning + classifier-free guidance (CFG)

We trained one model to work **with** the label and **without** it (that was the
label-dropout). Two things to see:

**(a) Does conditioning even work?** Ask the model for the `sky` category and count
how many valid samples are actually about the sky — versus generating with **no
label** (which has no category preference; it just lands wherever the model's
unconditional prior does). If conditioning works, "sky" should jump up to nearly all.

**(b) The guidance dial — and its cost.** CFG *amplifies* the label by comparing
the model's "with-label" and "without-label" opinions:
$$\text{guided} = \text{cond} + w\,(\text{cond} - \text{uncond}).$$
$w=0$ is plain conditional; higher $w$ pushes harder toward the label. **But
guidance isn't free:** push too far and samples leave the data manifold and turn to
gibberish. On this *easy* toy, conditioning is already near-perfect, so raising $w$
mostly reveals that **downside** — a crucial practical lesson (over-guidance breaks
fluency, and people over-crank it constantly). When conditioning is *weak*
(ambiguous labels, class imbalance, an undertrained model), moderate $w$ instead
*raises* adherence — which is why CFG is standard. Watch the sweet spot below: low
$w$ keeps samples valid; high $w$ destroys them.
""")

code(r"""want = CAT_NAMES.index("sky")
def sky_rate(samples):
    '''Among VALID samples, fraction in the 'sky' category; and the valid fraction.'''
    cats = [is_valid(decode(r).rstrip()) for r in samples]
    valid = [c for c in cats if c is not None]
    return (np.mean([c == want for c in valid]) if valid else 0.0), len(valid) / len(cats)

# (a) does conditioning work? unconditional vs conditional, asking for 'sky'
uncond = sample_confidence(model, T=24, n=200, label=None, seed=4)
cond0  = sample_confidence(model, T=24, n=200, label=torch.full((200,), want, device=device),
                           cfg_w=0.0, seed=4)
print(f"'sky' among valid samples:   unconditional {sky_rate(uncond)[0]:.0%}"
      f"   ->   conditional (w=0) {sky_rate(cond0)[0]:.0%}")

# (b) the guidance dial and its cost
print("\nturning the guidance dial up (label='sky'):")
lab = torch.full((160,), want, device=device)
for w in [0.0, 1.0, 2.0, 4.0, 8.0]:
    frac_sky, valid = sky_rate(sample_confidence(model, T=24, n=160, label=lab, cfg_w=w, seed=4))
    tag = "   <- still valid" if valid >= 0.85 else ("   <- over-guided (fluency breaks)"
                                                     if valid < 0.6 else "")
    print(f"  w={w:>3}:  sky-of-valid {frac_sky:.0%}   |   valid {valid:.0%}{tag}")""")

# ===========================================================================
md(r"""## 7. The speed/quality trade-off (why confidence sampling matters)

Every sampling step is one run of the network (one **NFE**). Fewer steps = faster
generation. The whole point of confidence-based sampling is that it **holds up with
far fewer steps** than random unmasking. Let's measure validity vs. the number of
steps $T$ for both samplers.
""")

code(r"""Ts = [2, 4, 8, 16, 32, 64]
rand_q, conf_q = [], []
for T in Ts:
    lab = torch.randint(0, len(CAT_NAMES), (200,), device=device)
    rand_q.append(score(sample_random(model, T=T, n=200, label=lab, seed=7))[0])
    conf_q.append(score(sample_confidence(model, T=T, n=200, label=lab, seed=7))[0])
    print(f"T={T:>2} | random valid {rand_q[-1]:.0%}   confidence valid {conf_q[-1]:.0%}")

plt.figure(figsize=(6, 3.4))
plt.plot(Ts, rand_q, "o-", label="random unmasking (Lesson 1)")
plt.plot(Ts, conf_q, "s-", label="confidence unmasking (MaskGIT/LLaDA)")
plt.xscale("log", base=2); plt.xlabel("sampling steps T  (= NFE, fewer is faster)")
plt.ylabel("fraction valid"); plt.ylim(-0.02, 1.02); plt.legend()
plt.title("Confidence sampling stays good with fewer steps"); plt.tight_layout(); plt.show()""")

# ===========================================================================
md(r"""## 8. Recap & where the series goes

**You upgraded the Lesson-1 masking model into a practical, LLaDA-style one:**

- **Generalization, measured.** Trained on a *subset* of a grammar and showed the
  model invents **new, valid** sentences — real compositional learning, not memorizing.
- **Conditional generation + CFG.** One `label` embedding + label-dropout training
  gave a model that's both conditional and unconditional. Conditioning alone
  sharply raises category adherence (from its unconditional base rate to ~always);
  CFG is a **strength
  dial** with a **sweet spot** — moderate helps, but over-guidance pushes samples
  off the data manifold into gibberish.
- **Confidence-based sampling.** Unmasking the most-confident letters first
  (MaskGIT/LLaDA) stays accurate with **far fewer steps** than random unmasking —
  the practical key to fast generation.

**What real scale adds (the honest gap to LLaDA):**
- **Tokenizer + real data** instead of char-level toy grammar.
- **Size**: billions of params, trained on trillions of tokens — same objective,
  much more of it.
- **Semi-autoregressive / block generation** for *long, variable-length* text:
  generate one block at a time, diffusion within each block (see Exercise 3). This
  is how LLaDA and block-diffusion handle documents rather than fixed-length lines.
- **Guidance & remasking samplers** tuned for quality/speed at inference.

With those, masked diffusion LMs (LLaDA) become competitive with autoregressive
LLMs — while generating in parallel and infilling natively.

**Coming in Lesson 5 — SEDD (score-entropy discrete diffusion):** the most advanced
lesson. Instead of predicting the clean token (or the posterior), learn the
**ratios** of the data distribution directly — a discrete analogue of a "score."
This gave discrete diffusion its first decisive wins on likelihood.

---
### Exercises
1. **Guidance sweet spot.** Push $w$ to 10–20. At what point does in-category go up
   but *validity* collapse? Plot both vs. $w$ — that's the guidance trade-off.
2. **Confidence vs. random, conditional.** Redo the NFE plot (§7) with a fixed
   category label. Does guidance change the gap between the two samplers?
3. **Semi-autoregressive generation.** Split the sequence into 2 blocks; generate
   block 1 with diffusion, freeze it, then generate block 2 conditioned on it.
   This is the LLaDA/block-diffusion recipe for arbitrary length.
4. **Harder grammar.** Add a 4th category, or make validity depend on
   subject–verb agreement. Does the model still generalize? How much training does
   it need?
5. **Temperature × steps.** For confidence sampling, sweep temperature at small $T$.
   How do temperature and step count interact for quality vs. diversity?
""")

# ===========================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(ROOT, "04_scaling_diffusion_llm.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out_path, "with", len(cells), "cells")
