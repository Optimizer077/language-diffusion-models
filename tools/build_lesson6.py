"""Builds 06_unmasking_strategies.ipynb (Lesson 6) using nbformat.

Lesson 6 is the first of the "Part II / advanced" track: the DECODING ZOO. The
model is fixed (a masked diffusion LM from Lesson 1); what changes is *how* we
unmask at generation time — the order we reveal tokens (random / confidence /
entropy / margin) and, the headline idea, **remasking / self-correction**: letting
the sampler un-commit a token it now regrets. That directly fixes the
"irreversible commitment" weakness seen in Lessons 2 and 4.

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
md(r"""# Lesson 6 — Unmasking Strategies (the decoding zoo)
### Same model, wildly different results — it's all in *how you reveal*

The model is done training. Now comes a secret that matters more than most people
expect: **how you decode a masked diffusion model changes its output quality
dramatically** — with the *exact same weights*.

In Lesson 1 we revealed masked tokens in a **random** order. In Lesson 4 we
revealed the **most-confident** ones first (MaskGIT/LLaDA). This lesson lays out
the whole menu and adds the technique that fixes diffusion's biggest weakness:

> **Remasking / self-correction.** Every sampler so far *committed* a token and
> never looked back — so one early mistake was permanent (that's why Lesson 2's
> uniform kernel and Lesson 4's absorbing kernel struggled). What if the sampler
> could **un-commit** a token it later realizes is wrong, and try again? That one
> change — re-masking low-confidence tokens and re-predicting — is what makes
> diffusion decoding *robust*.

**What you'll build & compare (one fixed model, five samplers):**
1. **Random**, **confidence**, **entropy**, and **margin** unmasking orders.
2. **Remasking / self-correction** (a predictor–corrector loop).
3. A head-to-head of **quality vs. number of steps** — and a look at
   self-correction *catching a mistake in the act*.

> **Prereq:** Lessons 1 & 4 (masked diffusion + confidence sampling). We reuse
> Lesson 4's grammar corpus so "correct" is measurable.
""")

# ===========================================================================
md(r"""## 📖 New words in this lesson

| Term | In plain words |
|---|---|
| **unmasking order** | which blanks you choose to fill *first* each step. |
| **confidence** | the model's top probability at a position — how sure it is. |
| **entropy** | how *spread out* the model's guess is (low entropy = confident). |
| **margin** | gap between the top-1 and top-2 probabilities (big gap = decisive). |
| **reveal schedule** | how many blanks you commit per step (we use a cosine). |
| **committed / frozen** | a token that's been revealed and won't change again. |
| **remasking** | *un-committing* a token — turning it back into `[MASK]` to reconsider. |
| **self-correction** | using remasking to fix a token the model now thinks is wrong. |
| **predictor–corrector** | predict tokens, then correct the shaky ones — repeat. |

The model is Lesson 1's masking diffusion LM (continuous time $t$, keep-prob
$\alpha_t = 1-t$, $1/t$-weighted masked cross-entropy). We train it once, then
everything below only changes the **sampler**.
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
md(r"""## 1. The model (recap) — a masked diffusion LM on a grammar

We reuse Lesson 4's **grammar corpus** (`the <subject> <verb> the <object>`, three
categories) so we can *check* whether a generated sentence is correct — and, in
particular, whether its subject and object stay in the **same category** (mixing
them, like `"the cat chased the truck"`, is invalid). That category-consistency is
exactly the kind of thing an early mistake ruins — and self-correction can fix.

The model itself is Lesson 1's: a bidirectional transformer trained to fill masked
positions. This cell sets it up and trains it (a minute or so); nothing here is
new, so it's condensed.
""")

code(r"""CATS = {
    "animal":  (["cat","dog","fox","owl"],  ["chased","liked","watched","found"], ["mouse","bird","bone","fish"]),
    "machine": (["car","bus","van","cab"],  ["passed","towed","chased","met"],    ["truck","train","boat","bike"]),
    "sky":     (["sun","moon","star","cloud"], ["lit","hid","warmed","chased"],   ["hill","lake","town","field"]),
}
all_sents = [f"the {s} {v} the {o}" for (S,V,O) in CATS.values() for s,v,o in itertools.product(S,V,O)]
train_sents = [s for i,s in enumerate(all_sents) if i % 10 >= 3]
valid_set, train_set = set(all_sents), set(train_sents); held_out = valid_set - train_set

def is_valid(sent):
    tok = sent.split()
    if len(tok)!=5 or tok[0]!="the" or tok[3]!="the": return False
    s,v,o = tok[1],tok[2],tok[4]
    return any(s in S and v in V and o in O for (S,V,O) in CATS.values())

chars = sorted(set("".join(all_sents))); K = len(chars); MASK_ID = K; V_in = K + 1
stoi = {c:i for i,c in enumerate(chars)}; itos = {i:c for c,i in stoi.items()}; itos[MASK_ID] = "#"
L = max(len(s) for s in all_sents)
def encode(s): return torch.tensor([stoi[c] for c in (s + " "*(L-len(s)))[:L]])
def decode(t): return "".join(itos[int(i)] for i in t)
data = torch.stack([encode(s) for s in train_sents]).to(device)
print(f"{len(all_sents)} valid sentences | train {len(train_sents)} | K={K} | L={L}")

class TimeEmbedding(nn.Module):
    def __init__(self, d):
        super().__init__(); self.d=d; self.mlp=nn.Sequential(nn.Linear(d,d), nn.SiLU(), nn.Linear(d,d))
    def forward(self, t):
        half=self.d//2; fr=torch.exp(-math.log(10000)*torch.arange(half,device=t.device)/half)
        a=t[:,None]*fr[None,:]*1000.0; return self.mlp(torch.cat([a.sin(),a.cos()],-1))

class Denoiser(nn.Module):
    def __init__(self, V_in, K, L, d=128, nh=4, nl=3, ff=256, p=0.1):
        super().__init__()
        self.tok=nn.Embedding(V_in,d); self.pos=nn.Embedding(L,d); self.time=TimeEmbedding(d)
        lyr=nn.TransformerEncoderLayer(d,nh,ff,p,batch_first=True,activation="gelu",norm_first=True)
        self.enc=nn.TransformerEncoder(lyr,nl); self.norm=nn.LayerNorm(d); self.head=nn.Linear(d,K)
        self.register_buffer("pos_ids", torch.arange(L))
    def forward(self, x, t):
        h=self.tok(x)+self.pos(self.pos_ids)[None]+self.time(t)[:,None,:]
        return self.head(self.norm(self.enc(h)))

def forward_mask(x0, t):
    keep = torch.rand_like(x0, dtype=torch.float) < (1-t).unsqueeze(1)
    return torch.where(keep, x0, torch.full_like(x0, MASK_ID)), ~keep

def loss_fn(model, x0):
    B=x0.size(0); t=torch.rand(B,device=x0.device).clamp(min=1e-3)
    xt,m=forward_mask(x0,t); logits=model(xt,t)
    ce=F.cross_entropy(logits.reshape(-1,K), x0.reshape(-1), reduction="none").reshape(B,L)
    return ((ce*m.float()).sum(1)/m.float().sum(1).clamp(min=1) * (1/t)).mean()

model=Denoiser(V_in,K,L).to(device)
opt=torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01)
for step in range(1, 2501):
    idx=torch.randint(0,data.size(0),(64,),device=device)
    loss=loss_fn(model,data[idx]); opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    if step%500==0 or step==1: print(f"  step {step:4d} | loss {loss.item():.3f}")
print("trained.")""")

# ===========================================================================
md(r"""## 2. One sampler, five behaviors

Here is the whole zoo in a single function. Generation always does the same loop —
predict, then reveal some blanks — but two knobs change everything:

- **`order`** decides *which* blanks to reveal first, via a per-position **score**:
  - `random` — pick blanks at random (Lesson 1).
  - `confidence` — highest top-probability first (Lesson 4 / MaskGIT).
  - `entropy` — lowest-entropy (most peaked) distribution first.
  - `margin` — biggest top-1−top-2 gap first (most *decisive*).
- **`remask`** decides whether tokens are **frozen** once revealed (`False`, every
  sampler so far) or can be **un-committed and reconsidered** (`True` — the
  self-correcting predictor–corrector).

We use a **cosine reveal schedule**: the number of still-masked positions shrinks
from all → none over $T$ steps.
""")

code(r"""@torch.no_grad()
def sample(model, T, n=100, order="confidence", remask=False, temperature=1.0, seed=0, trace=False):
    torch.manual_seed(seed); model.eval()
    x = torch.full((n, L), MASK_ID, device=device)
    snaps = []
    for i in range(T):
        masked = (x == MASK_ID)
        t_vec = masked.float().mean(1).clamp(min=1e-3)          # noise level = fraction masked
        probs = (model(x, t_vec) / temperature).softmax(-1)     # (n,L,K)
        conf, _ = probs.max(-1)                                 # top prob per position
        pred = torch.multinomial(probs.reshape(-1, K), 1).reshape(n, L)   # sampled fill for blanks

        # "reveal score" for the blanks (higher = fill this blank sooner)
        if order == "entropy":   base = (probs * (probs + 1e-9).log()).sum(-1)     # = -entropy
        elif order == "margin":  tp = probs.topk(2, -1).values; base = tp[...,0] - tp[...,1]
        elif order == "random":  base = torch.rand_like(conf)
        else:                    base = conf                                       # confidence
        if order != "random":    # Gumbel-noised reveal order (MaskGIT/LLaDA), annealed — on blanks only
            g = -torch.log(-torch.log(torch.rand_like(base).clamp_min(1e-9)).clamp_min(1e-9))
            base = base + 0.6 * (1 - (i+1)/T) * g
        # committed positions: score by the model's SUPPORT for their current letter (no noise).
        # frozen -> committed never re-masked; remask -> a committed letter the model no longer
        # supports scores low and gets sent back to [MASK] to be reconsidered (self-correction).
        support = probs.gather(-1, x.clamp(0, K-1).unsqueeze(-1)).squeeze(-1)
        keep = torch.where(masked, base, support) if remask else base.masked_fill(~masked, float("inf"))

        n_mask = int(math.floor(L * math.cos(math.pi/2 * (i+1)/T)))   # how many stay masked after this step
        filled = torch.where(masked, pred, x)                  # blanks -> sampled pred; committed keep their letter
        if n_mask > 0:                                          # send the least-supported positions back to [MASK]
            idx = keep.topk(n_mask, dim=1, largest=False).indices
            filled.scatter_(1, idx, MASK_ID)
        x = filled
        if trace: snaps.append(x[0].clone())
    if (x == MASK_ID).any():                                    # fill any leftover with the greedy guess
        gm = model(x, torch.full((n,), 1e-3, device=device)).argmax(-1)
        x = torch.where(x == MASK_ID, gm, x)
    return (x, snaps) if trace else x

def valid_frac(x): return np.mean([is_valid(decode(r).rstrip()) for r in x])

print("quick look (T=16, confidence, no remask):")
for r in sample(model, T=16, n=6, order="confidence", seed=1): print("  ", repr(decode(r).rstrip()))""")

# ===========================================================================
md(r"""## 3. Does the *order* matter? (frozen samplers)

First, hold `remask=False` (tokens freeze once revealed, like every earlier lesson)
and compare the four unmasking orders across step counts. From a handful of steps
up, the **confident-first orders pull clearly ahead of random** (e.g. T=16 ≈ 46% vs
27% valid; T=32 ≈ 76% vs 38%). Two honest caveats you'll spot in the numbers: at
the *very* fewest steps everything is poor and random's scattered reveals can even
edge ahead; and on this simple corpus the three confidence-style scores
(confidence / entropy / margin) behave almost identically — they only diverge on
harder data with genuine near-ties.
""")

code(r"""Ts = [2, 4, 8, 16, 32]
orders = ["random", "confidence", "entropy", "margin"]
res = {o: [] for o in orders}
for o in orders:
    for T in Ts:
        res[o].append(valid_frac(sample(model, T=T, n=150, order=o, remask=False, seed=7)))
    print(f"{o:>10}: " + "  ".join(f"T{T}={v:.0%}" for T,v in zip(Ts,res[o])))

plt.figure(figsize=(6.2,3.6))
for o in orders: plt.plot(Ts, res[o], "o-", label=o)
plt.xscale("log", base=2); plt.xlabel("sampling steps T"); plt.ylabel("fraction valid")
plt.ylim(-0.02,1.02); plt.legend(); plt.title("Unmasking order (frozen / no remasking)")
plt.tight_layout(); plt.show()""")

# ===========================================================================
md(r"""## 4. The big one: remasking / self-correction

Now turn on `remask=True`. Instead of freezing a token forever, each step we
**re-predict every position and re-mask the least-confident ones** — so a token
that looked fine early but conflicts with what got filled in later can be
**un-committed and fixed**. This is a predictor–corrector loop (the idea behind
remasking samplers / ReMDM, and LLaDA's low-confidence remasking).

Let's pit **confidence + remasking** against the best frozen sampler.
""")

code(r"""conf_frozen = [valid_frac(sample(model, T=T, n=200, order="confidence", remask=False, seed=7)) for T in Ts]
conf_remask = [valid_frac(sample(model, T=T, n=200, order="confidence", remask=True,  seed=7)) for T in Ts]
for T,a,b in zip(Ts, conf_frozen, conf_remask):
    print(f"T={T:>2} | frozen {a:.0%}   |   +remasking {b:.0%}   ({'+' if b>=a else ''}{(b-a)*100:.0f} pts)")

plt.figure(figsize=(6.2,3.6))
plt.plot(Ts, conf_frozen, "o-", label="confidence, frozen")
plt.plot(Ts, conf_remask, "s-", label="confidence + remasking (self-correcting)")
plt.xscale("log", base=2); plt.xlabel("sampling steps T"); plt.ylabel("fraction valid")
plt.ylim(-0.02,1.02); plt.legend(); plt.title("Self-correction fixes early mistakes")
plt.tight_layout(); plt.show()""")

# ===========================================================================
md(r"""## 5. Watch self-correction happen

The plot shows *that* remasking helps; this shows *how*. We trace one sample and
flag any step where a position that was already a real letter gets **turned back
into `#`** — the model deciding "wait, that was wrong" and reconsidering it.
""")

code(r"""torch.manual_seed(3)
_, snaps = sample(model, T=20, n=1, order="confidence", remask=True, temperature=1.0, seed=3, trace=True)
prev = None; shown = 0
print("trace (>> marks a position that was un-committed for self-correction):\n")
for i, s in enumerate(snaps):
    line = decode(s)
    mark = ""
    if prev is not None:
        flips = [(prev[p] != MASK_ID) and (s[p] == MASK_ID) for p in range(L)]
        if any(flips):
            mark = "  >> re-masked pos " + ",".join(str(p) for p,f in enumerate(flips) if f)
    if i % 3 == 0 or mark:
        print(f"step {i:2d}: {line.rstrip()!r}{mark}"); shown += 1
    prev = s
print("\nfinal:", repr(decode(snaps[-1]).rstrip()))""")

# ===========================================================================
md(r"""## 6. Recap & what's next

**You saw that decoding is a first-class design choice:**

- The **unmasking order** (random → confidence/entropy/margin) matters a lot at
  low step counts — confident-first orders keep the sentence coherent with fewer
  network calls.
- **Remasking / self-correction** is the bigger lever: by letting the sampler
  **un-commit and fix** tokens, it repairs the early-mistake problem that frozen
  samplers (Lessons 1–5) can't — the same weakness behind Lesson 2's irreversible
  uniform kernel and Lesson 4's committed absorbing kernel.
- All of this used **one fixed model** — no retraining. Decoding is free quality.

**Connections.** This is the sampler side of **MaskGIT** and **LLaDA** (confidence
unmasking + low-confidence remasking), and the "corrector" half of
**predictor–corrector** samplers. It pairs with everything earlier: any of the
Lesson 1–5 models can be decoded this way.

**Coming in Lesson 7 — Parallelization & fast generation:** decoding fast *and* at
scale — parallel decoding, **block / semi-autoregressive** diffusion for
arbitrary-length text, and few-step distillation, benchmarked honestly against an
autoregressive baseline.

---
### Exercises
1. **Order × remasking.** Re-run §3's comparison with `remask=True` for *every*
   order. Does remasking shrink the gap between random and confidence?
2. **Corrector strength.** Add a "number of corrector sweeps" knob: after the last
   reveal step, keep re-masking the least-confident $k$ tokens and re-predicting
   for a few extra steps. How much does pure correction (no schedule) buy you?
3. **Temperature.** Sweep temperature for confidence+remasking at small $T$. Where
   is the quality/diversity sweet spot?
4. **A confidence metric bake-off.** Is `entropy` or `margin` ever better than
   `confidence` here? Construct a corpus where margin wins (hint: near-ties).
5. **Remasking cost.** Remasking re-runs the network on *all* positions each step.
   Plot quality vs. *total* network calls (not just $T$) — is it still a win?
""")

# ===========================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(ROOT, "06_unmasking_strategies.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out_path, "with", len(cells), "cells")
