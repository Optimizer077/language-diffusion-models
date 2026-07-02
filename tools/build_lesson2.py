"""Builds 02_d3pm_transition_kernels.ipynb (Lesson 2) using nbformat.

Lesson 2 generalizes Lesson 1's single absorbing/masking kernel to the full
D3PM framework: arbitrary categorical transition matrices, the closed-form
t-step marginal and posterior, the term-by-term ELBO, the hybrid loss, and a
head-to-head comparison of the UNIFORM and ABSORBING kernels.

Authoring the notebook from a script keeps cell sources as plain Python (easy to
diff/lint) instead of hand-edited notebook JSON.
"""
import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
cells = []
def md(src):   cells.append(new_markdown_cell(src))
def code(src): cells.append(new_code_cell(src))

# ===========================================================================
md(r"""[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Optimizer077/language-diffusion-models/blob/main/02_d3pm_transition_kernels.ipynb)

# Lesson 2 — General Transition Kernels (D3PM)
### From one kind of noise to *any* kind

Welcome back! In **Lesson 1** we built a "fill-in-the-blanks" language model.
The recipe was:

1. Take a clean sentence and **randomly blank out some letters** (replace them
   with a `[MASK]` symbol).
2. Train a transformer to **guess the blanked letters back**.
3. To generate new text, start from *all blanks* and let the model fill them in,
   a few at a time.

That blanking move — "turn a letter into `[MASK]`" — was the *only* kind of noise
we used. This lesson asks a bigger question:

> **What if the noise were something else?** What if, instead of blanking a
> letter, we replaced it with a *random* letter? Or nudged it toward a similar
> letter? Is "blanking" even the best choice?

The paper **D3PM** (Austin et al., 2021) gives a single framework that covers
*all* these choices at once. The trick is to describe "what the noise does" with a
small **probability table** called a **transition matrix**. Pick the table, and
you've picked your kind of diffusion.

We'll build and **compare two** kinds of noise on the same data:

| Kernel (= kind of noise) | One step of noise does... | The noise looks like | Starts generation from... |
|---|---|---|---|
| **Uniform** | replace a letter with a *random* letter | scrambled gibberish | random letters |
| **Absorbing** | replace a letter with `[MASK]` | blanked-out text | all `[MASK]` (Lesson 1!) |

> 🟢 **Don't be scared by the symbols.** This lesson has more math than Lesson 1,
> but every Greek letter and every formula is explained in plain English right
> next to it, and there's a glossary in the next cell. If you can read "table of
> probabilities" and "guess the original letter", you can follow all of it.

**By the end you'll understand:**
1. How *any* discrete noise is just a probability table $Q_t$ (with pictures).
2. How to compute, in one shot, "what does a clean letter look like after $t$
   steps of noise?" — and how to **run the noise backwards** to denoise.
3. What the model's training target (the "loss") really is, built up piece by
   piece — and why, for the blanking kernel, it turns out to be *exactly Lesson 1*.
4. A real, honest experiment comparing the two kinds of noise.

> **Prerequisite:** Lesson 1 (the blanking model). We reuse its toy data,
> transformer, and sampling idea.
""")

# ===========================================================================
md(r"""## 📖 A 60-second glossary (skim now, refer back as needed)

You don't need to memorize these — just know they're here.

| Term | In plain words |
|---|---|
| **token / letter** | one character of text (we work letter-by-letter). |
| **vocabulary** | the set of all possible letters (here: lowercase + space). |
| **state** | the value a position can hold: a letter, or (for blanking) `[MASK]`. |
| **categorical distribution** | a "weighted die": a list of probabilities over the possible letters that adds up to 1. |
| **$x_0$** | the **clean** (original) text. The subscript is *time / noise level*. |
| **$x_t$** | the text after **$t$ steps of noise**. Bigger $t$ = more corrupted. |
| **forward process** | adding noise: $x_0 \to x_1 \to \dots \to x_T$ (clean → garbage). |
| **reverse process** | removing noise: $x_T \to \dots \to x_0$ (garbage → clean). This is generation. |
| **transition matrix $Q_t$** | a table: "if this letter, then what's the chance of becoming each other letter next step?" |
| **kernel** | a *choice* of how the noise works = a choice of $Q_t$ (uniform or absorbing). |
| **schedule** | how *fast* we add noise as $t$ grows. |
| **marginal $q(x_t\mid x_0)$** | "given the clean letter, what might it look like after $t$ steps?" (skipping the in-between). |
| **posterior $q(x_{t-1}\mid x_t, x_0)$** | running the clock back one tick: "given now and the original, what was it one step ago?" |
| **prior $p(x_T)$** | what fully-noised text looks like (random letters, or all blanks). Where generation starts. |
| **KL divergence** | a number measuring *how different two weighted dice are* (0 = identical). |
| **loss / objective** | the number we make small during training; smaller = model fits the data better. |
| **ELBO** | the specific, computable loss for diffusion models (explained gently in §7). |

Greek letters we use: $\alpha_t$ ("alpha") = chance a letter **survives** one step;
$\beta_t$ ("beta") = chance it gets **corrupted** one step ($\alpha_t+\beta_t=1$);
$\bar\alpha_t$ ("alpha-bar") = chance it survives **all $t$ steps**.
""")

# ===========================================================================
md(r"""## 0. The one idea behind all discrete diffusion

Here's the whole framework in one picture. Imagine each letter playing a little
**board game**. At every step it rolls a (loaded) die that says where to move
next: "stay the same", "become an `e`", "become a blank", etc. The **transition
matrix** $Q_t$ is just *the table of those dice*, one row per starting letter:

$$\underbrace{q(x_t = j \mid x_{t-1} = i)}_{\text{prob. of becoming letter } j,\ \text{if currently letter } i} \;=\; [Q_t]_{ij} \quad (\text{row } i,\ \text{column } j).$$

So **row $i$** of the table is the die for "I'm currently letter $i$": it lists the
chance of turning into each possible letter next step. Because those are
probabilities for *one* die roll, **each row adds up to 1**. (Math people call
such a table *row-stochastic* — "every row is a valid probability distribution".)

That's it. Everything else is just consequences of this table:

- **"After $t$ steps"** = roll the dice $t$ times = **multiply $t$ tables
  together**: $\bar Q_t = Q_1 Q_2 \cdots Q_t$ (we'll see this is easy in closed form).
- **Denoising** = running the dice *backwards*, which needs **Bayes' rule**
  (don't worry, we explain it where it's used).
- **Different tables = different diffusions.** Blanking, random-replacement,
  nudge-to-similar... they're all just different $Q_t$. We do the two that matter
  most for text.

Lesson 1 was secretly already playing this game with the table "*keep the letter,
or jump to `[MASK]`*". Now we make the table something we can **choose**.

Let's set up and look at some actual tables.
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
md(r"""## 1. The data (same toy corpus as Lesson 1)

We use the same tiny set of simple sentences, one character at a time. Working
with toy data keeps training to a few seconds and makes the results easy to read.

Two small bookkeeping choices that keep the code clean:

- **Padding is just spaces.** All lines are stretched to the same length `L` by
  adding spaces on the end. Since space is already a normal character, we *don't*
  need a special padding symbol — every position is an ordinary letter.
- **"Data letters" vs. "states".** The model's job is always to guess one of the
  $K_{\text{data}}$ real letters. The **blanking (absorbing)** kernel needs one
  *extra* symbol, `[MASK]`, so its table has $K = K_{\text{data}}+1$ rows/columns.
  The **uniform** kernel has no blank, so $K = K_{\text{data}}$. The model never
  *predicts* `[MASK]` — a blank just means "I haven't been revealed yet".
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

chars = sorted(set("".join(raw_lines)))      # includes the space character
K_data = len(chars)                          # number of real data classes
MASK_ID = K_data                             # extra absorbing state (absorbing only)
V_in = K_data + 1                            # embedding vocab (mask slot unused by uniform)

stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
itos[MASK_ID] = "#"                          # render mask as '#'

L = max(len(s) for s in raw_lines)
def encode(s): return torch.tensor([stoi[c] for c in (s + " " * (L - len(s)))[:L]])
def decode(t): return "".join(itos[int(i)] for i in t)

data = torch.stack([encode(s) for s in raw_lines]).to(device)
print(f"K_data = {K_data} data classes | MASK_ID = {MASK_ID} | block length L = {L}")
print("classes:", repr("".join(chars)))
print("example:", repr(decode(data[0])))""")

# ===========================================================================
md(r"""## 2. The two noise tables $Q_t$

Time to write down the two dice-tables. First, two numbers that control "how much
noise this step adds":

- $\beta_t$ = **chance a letter gets corrupted** this step (the "corruption rate").
- $\alpha_t = 1-\beta_t$ = **chance it survives** unchanged ("keep rate").

(Subscript $t$ because we can use a different amount of noise at each step.)

**Uniform table** — "keep the letter, or replace it with a *random* letter":
$$[Q_t]_{ij} = \underbrace{\alpha_t\,\mathbb{1}[i{=}j]}_{\text{keep (stay letter } i)} + \underbrace{\tfrac{\beta_t}{K}}_{\text{jump to a random letter}}.$$
In words: with chance $\alpha_t$ you stay put; otherwise you jump to a letter
picked uniformly at random (each of the $K$ letters equally likely). The $I$ and
$\mathbf{1}\mathbf{1}^\top$ you may see elsewhere are just shorthand: $I$ = "stay"
(identity table), $\mathbf{1}\mathbf{1}^\top$ = "all entries equal" (the uniform
jump). If you keep rolling this die forever, every letter becomes equally likely —
the **end state of uniform noise is random letters**.

**Absorbing (blanking) table** — "keep the letter, or turn it into `[MASK]`":
$$[Q_t]_{ij} = \underbrace{\alpha_t\,\mathbb{1}[i{=}j{\neq}m]}_{\text{keep}} + \underbrace{\beta_t\,\mathbb{1}[j{=}m,\,i{\neq}m]}_{\text{become MASK}} + \underbrace{\mathbb{1}[i{=}j{=}m]}_{\text{MASK stays MASK}}.$$
With chance $\beta_t$ a letter becomes `[MASK]`; and crucially, **once something is
`[MASK]` it stays `[MASK]` forever** (that's what "absorbing" means — the last row
of the table is "100% stay `[MASK]`"). The **end state of blanking noise is all
blanks** — exactly Lesson 1's starting point.

Let's actually build these tables for a tiny alphabet and *look* at them. (At
training time we'll never build the full table — we'll use shortcut formulas — but
seeing the tables makes everything concrete.)
""")

code(r"""def Q_uniform(beta, K):
    return (1 - beta) * torch.eye(K) + beta / K * torch.ones(K, K)

def Q_absorbing(beta, K):
    m = K - 1
    Q = (1 - beta) * torch.eye(K)
    Q[:, m] += beta            # everyone leaks to MASK w.p. beta...
    Q[m] = 0.0; Q[m, m] = 1.0  # ...except MASK, which is absorbing.
    return Q

# Sanity: every row is a probability distribution (row-stochastic).
for name, Q in [("uniform", Q_uniform(0.3, 5)), ("absorbing", Q_absorbing(0.3, 6))]:
    assert torch.allclose(Q.sum(-1), torch.ones(Q.size(0))), name
    print(f"{name:9s} rows sum to 1  [ok]")

# Visualize the two kernels (small K) side by side.
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
for a, (name, Q) in zip(ax, [("uniform  Q_t (K=5)", Q_uniform(0.3, 5)),
                             ("absorbing  Q_t (K=6, last=MASK)", Q_absorbing(0.3, 6))]):
    im = a.imshow(Q, cmap="viridis", vmin=0, vmax=1)
    a.set_title(name, fontsize=10); a.set_xlabel("to state j"); a.set_ylabel("from state i")
    for (i, j), v in np.ndenumerate(Q.numpy()):
        a.text(j, i, f"{v:.2f}", ha="center", va="center",
               color="w" if v < 0.6 else "k", fontsize=7)
plt.tight_layout(); plt.show()
print("Uniform: mass on the diagonal (keep) + a little everywhere (resample).")
print("Absorbing: diagonal (keep) + a column into MASK; the MASK row is a self-loop.")""")

# ===========================================================================
md(r"""**How to read those pictures.** Each square is a probability; brighter = higher.

- **Uniform (left):** a bright **diagonal** (good chance of *staying* the same
  letter) plus a faint glow *everywhere else* (small chance of jumping to any
  other letter).
- **Absorbing (right):** a bright diagonal (keep) plus a bright **last column**
  (everything leaks into `[MASK]`). The bottom-right corner is a lone bright dot:
  the `[MASK]` row points only to itself — once a blank, always a blank.
""")

# ===========================================================================
md(r"""## 3. The schedule: how *fast* we add noise

We don't have to add the same amount of noise every step. The **schedule** sets
the pace. The cleanest way to think about pacing is the **survival probability**:

$$\bar\alpha_t = \alpha_1\,\alpha_2\cdots\alpha_t = \text{chance a letter is still its original self after } t \text{ steps.}$$

(It's just the keep-chances multiplied together — like compound interest, or
radioactive decay: each step it might "decay" into noise.) So:

- $\bar\alpha_0 = 1$: at the start, every letter is definitely itself.
- $\bar\alpha_t$ **shrinks** as $t$ grows: more steps, more likely it's been corrupted.
- $\bar\alpha_T \approx 0$: by the last step, essentially everything is noise.

We use a **cosine** schedule (it corrupts gently at first, then faster — this
tends to train better than a straight line; it comes from Nichol & Dhariwal's
image-diffusion work). From $\bar\alpha_t$ we can recover the per-step rates:
$\alpha_t = \bar\alpha_t / \bar\alpha_{t-1}$ and $\beta_t = 1-\alpha_t$.

> **Why force $\bar\alpha_T \approx 0$?** Because we want the *fully-noised* text to
> be pure noise with no trace of the original — random letters (uniform) or all
> blanks (absorbing). That makes the "starting point for generation" a fixed,
> known thing, and (as we'll see) makes one of the loss terms vanish.
""")

code(r"""T = 128  # number of diffusion steps

def make_schedule(T, s=0.008):
    t = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2
    abar = (f / f[0]).clamp(1e-6, 1.0)        # abar[0]=1, decreasing
    abar[-1] = abar[-1].clamp(max=1e-4)       # ensure near-total corruption at T
    return abar.to(device).float()            # length T+1, indexed by t=0..T

abar = make_schedule(T)                       # abar[t] = cumulative keep prob
alpha_step = abar[1:] / abar[:-1]             # alpha_t for t=1..T  (length T)
beta_step  = 1 - alpha_step

plt.figure(figsize=(6, 3))
plt.plot(range(T + 1), abar.cpu(), label=r"$\bar\alpha_t$ (cumulative keep)")
plt.plot(range(1, T + 1), beta_step.cpu(), label=r"$\beta_t$ (per-step corrupt)", alpha=0.7)
plt.xlabel("step t"); plt.ylabel("probability"); plt.legend(); plt.title("Cosine schedule")
plt.tight_layout(); plt.show()
print(f"abar_0={abar[0]:.3f}  abar_{T//2}={abar[T//2]:.3f}  abar_T={abar[T]:.2e}")""")

# ===========================================================================
md(r"""## 4. Jumping straight to step $t$ (the "marginal")

During training we'll constantly need to take a clean sentence and instantly make
its "$t$-steps-of-noise" version — **without** simulating all $t$ steps one by one.
The distribution of "a clean letter $x_0$ after $t$ steps" is called the
**marginal** $q(x_t \mid x_0)$ ("marginal" because we don't care about the
in-between letters $x_1,\dots,x_{t-1}$ — we've summed them out).

Multiplying $t$ copies of our structured tables happens to have a **shortcut
formula** (no big matrix multiply needed), and it's intuitive:

- **Uniform:** keep the original letter with probability $\bar\alpha_t$, otherwise
  it's a uniformly random letter.
- **Absorbing:** keep the original with probability $\bar\alpha_t$, otherwise it's
  `[MASK]`. *(This is identical to Lesson 1's masking — with $\bar\alpha_t$ as the
  keep-probability.)*

In both cases: **survive with prob $\bar\alpha_t$, else turn to noise.** Simple.

The next cell first *proves* the shortcut formula matches the slow
matrix-multiply (so you can trust it), then defines the fast sampler.
""")

code(r"""# Verify closed-form barQ_t == explicit product Q_1 @ ... @ Q_t (small K).
def cumprod_Q(Q_fn, K, upto, betas):
    P = torch.eye(K)
    for t in range(upto):
        P = P @ Q_fn(float(betas[t]), K)
    return P

for name, Q_fn, K in [("uniform", Q_uniform, 5), ("absorbing", Q_absorbing, 6)]:
    t = 40
    explicit = cumprod_Q(Q_fn, K, t, beta_step.cpu())
    ab = float(abar[t])
    if name == "uniform":
        closed = ab * torch.eye(K) + (1 - ab) / K * torch.ones(K, K)
    else:
        closed = ab * torch.eye(K); closed[:, K-1] += (1 - ab)
        closed[K-1] = 0.0; closed[K-1, K-1] = 1.0
    err = (explicit - closed).abs().max().item()
    print(f"{name:9s}  max|product - closed form|  = {err:.2e}   [ok]" )

def q_sample(x0, t, kernel):
    '''Corrupt x0 (B,L) at per-example step t (B,) -> (x_t, corrupted_bool).'''
    keep = torch.rand_like(x0, dtype=torch.float) < abar[t].unsqueeze(1)
    if kernel == "absorbing":
        x_t = torch.where(keep, x0, torch.full_like(x0, MASK_ID))
    else:  # uniform: resample over the K_data real classes
        unif = torch.randint(0, K_data, x0.shape, device=x0.device)
        x_t = torch.where(keep, x0, unif)
    return x_t, ~keep""")

# ===========================================================================
md(r"""### Seeing the two kinds of "noise" side by side

Here is the payoff of all that setup, and the clearest answer to *"what does the
noise actually look like?"* We corrupt the **same** sentence to increasing noise
levels with **both** kernels. Watch how uniform dissolves the text into random
gibberish, while absorbing slowly blanks it out into `#`s.
""")

code(r"""line = data[0:1]
print("original :", repr(decode(data[0])), "\n")
for frac in [0.15, 0.4, 0.7, 0.95]:
    t = torch.tensor([int(frac * T)], device=device)
    xu, _ = q_sample(line, t, "uniform")
    xa, _ = q_sample(line, t, "absorbing")
    print(f"t={int(frac*T):3d} (keep {float(abar[t]):.2f}) | uniform:   {decode(xu[0])!r}")
    print(f"            {' '*15} | absorbing: {decode(xa[0])!r}")""")

# ===========================================================================
md(r"""## 5. Running the clock backwards: the "posterior"

Adding noise is easy. **Generation is the reverse**: start from noise and walk
back to clean text. To take one step back, we'd love to answer:

> *"Given the noisy letter I see now ($x_t$), and supposing the original letter was
> $x_0$, what was the letter one step earlier ($x_{t-1}$)?"*

That backwards distribution is called the **posterior**, written
$q(x_{t-1}\mid x_t, x_0)$. ("Posterior" = an updated belief *after* taking
evidence into account — here the evidence is "I know where it ended up.")

We get it with **Bayes' rule**, which is just the everyday logic of combining two
pieces of information. Here the two pieces are: *(a)* how likely was the last step
$x_{t-1}\to x_t$, and *(b)* how likely was $x_0$ to reach $x_{t-1}$ in the first
$t-1$ steps. Multiply them, then normalize so the probabilities sum to 1:

$$q(x_{t-1}{=}k \mid x_t, x_0) \;\propto\; \underbrace{[Q_t]_{k,\,x_t}}_{(a)\ \text{one step } k\to x_t} \cdot \underbrace{[\bar Q_{t-1}]_{x_0,\,k}}_{(b)\ x_0\to k \text{ in } t{-}1 \text{ steps}}.$$

(The "$\propto$" means "proportional to" — compute the right-hand side for every
candidate previous-letter $k$, then divide by the total so it's a proper
distribution.) Plugging in our shortcut formulas gives clean results:

**Uniform** — a weighted die over all letters: it's the elementwise product of two
simple vectors (the "came from $x_t$" factor and the "could be reached from $x_0$"
factor), then normalized. The notebook does this for you.

**Absorbing** — *delightfully* simple, just two cases:
- **If $x_t$ is a real letter** (already un-blanked): then it was *never* blanked,
  so the previous letter was the same. The position is **frozen** — nothing to do.
- **If $x_t = $ `[MASK]`:** it either gets revealed to the true letter $x_0$ now,
  or stays blank. The chance of revealing **this** step is the **reveal
  probability**
$$r_t \;=\; \frac{\bar\alpha_{t-1}-\bar\alpha_t}{1-\bar\alpha_t}.$$

> 🧠 **What $r_t$ means:** of all the blanks still hanging around at step $t$, a
> fraction $r_t$ "come due" to be revealed this step. This is *literally* the
> reveal rule from Lesson 1's sampler.

Index conventions (which row, which column, $t$ vs $t-1$) are the #1 source of
bugs in this kind of code. So instead of trusting the formulas, the next cell
**checks them against brute-force Bayes** computed from the raw tables. If they
match to ~7 decimal places, we know we got it right.
""")

code(r"""def q_posterior_uniform(x0, xt, t):
    '''Closed-form q(x_{t-1}|x_t,x_0) for the uniform kernel -> (..., K_data).'''
    K = K_data
    a_t  = (abar[t] / abar[t - 1]).view(*([-1] + [1] * (x0.dim())))   # alpha_t
    ab_t1 = abar[t - 1].view(*([-1] + [1] * (x0.dim())))              # abar_{t-1}
    b_t = 1 - a_t
    f1 = a_t * F.one_hot(xt, K) + b_t / K                  # one step k->xt
    f2 = ab_t1 * F.one_hot(x0, K) + (1 - ab_t1) / K        # x0->k in t-1 steps
    post = f1 * f2
    return post / post.sum(-1, keepdim=True).clamp_min(1e-12)

def reveal_prob(t):
    '''Absorbing reveal probability r_t = (abar_{t-1}-abar_t)/(1-abar_t).'''
    return ((abar[t - 1] - abar[t]) / (1 - abar[t]).clamp_min(1e-12)).clamp(0, 1)

# ---- Brute-force validation against explicit matrices (the headline check) ----
def brute_posterior(Q_fn, K, xt, x0, t, betas):
    Qt = Q_fn(float(betas[t - 1]), K)
    Qbar_tm1 = cumprod_Q(Q_fn, K, t - 1, betas)
    unnorm = Qt[:, xt] * Qbar_tm1[x0, :]
    return unnorm / unnorm.sum()

betas_cpu = beta_step.cpu()
# Uniform: compare closed form vs brute force for many (t, x0, xt).
# (Query tensors live on `device`; we move the closed-form result to CPU before
#  comparing with the CPU brute-force result -- so this cell works on GPU too.)
max_err = 0.0
for t in [2, 17, 60, 110]:
    for x0 in range(K_data):
        for xt in range(K_data):
            bf = brute_posterior(Q_uniform, K_data, xt, x0, t, betas_cpu)
            cf = q_posterior_uniform(torch.tensor([x0], device=device),
                                     torch.tensor([xt], device=device),
                                     torch.tensor([t], device=device))[0].cpu()
            max_err = max(max_err, (bf - cf).abs().max().item())
print(f"UNIFORM   closed-form vs brute-force posterior: max err {max_err:.2e}  [ok]")

# Absorbing: when x_t = MASK, brute-force revert prob must equal r_t.
Kabs = K_data + 1
max_err = 0.0
for t in [2, 17, 60, 110]:
    for x0 in range(K_data):
        bf = brute_posterior(Q_absorbing, Kabs, MASK_ID, x0, t, betas_cpu)
        rt = float(reveal_prob(torch.tensor(t, device=device)))
        max_err = max(max_err, abs(bf[x0].item() - rt))
print(f"ABSORBING closed-form reveal prob vs brute force:  max err {max_err:.2e}  [ok]")""")

# ===========================================================================
md(r"""## 6. The model: a transformer that guesses the clean letters

There's a catch with the posterior from §5: it needs the *true* clean letter
$x_0$, which at generation time we **don't have** — that's the whole thing we're
trying to produce! So we train a neural network to **guess $x_0$** from the noisy
input.

The network is exactly Lesson 1's setup: a small **bidirectional transformer**
(it sees the whole sequence at once, left and right) that reads the corrupted
sentence $x_t$ and outputs, for each position, a **weighted die over the real
letters** — its guess of the original letter. We write this guess
$\tilde p_\theta(x_0 \mid x_t)$ (the $\theta$ just means "the network's
parameters").

The only differences from Lesson 1:
- The output has $K_{\text{data}}$ options (real letters only — it **never**
  guesses `[MASK]`; a blank isn't a letter, it's "not revealed yet").
- We tell the network the current noise level by feeding it $t/T$ (a number in
  $(0,1]$) through a small "time embedding".

> **The jargon for this:** because the network predicts the clean data $x_0$, this
> is called the **"$x_0$-parameterization"**. It's the easiest possible target —
> just "name the original letter", a plain classification problem. In the next
> section we turn that guess into an actual denoising step.
""")

code(r"""class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
    def forward(self, t):                       # t: (B,) in (0,1]
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        ang = t[:, None] * freqs[None, :] * 1000.0
        return self.mlp(torch.cat([ang.sin(), ang.cos()], dim=-1))

class DenoiserTransformer(nn.Module):
    def __init__(self, V_in, K_out, L, d_model=128, nhead=4, nlayers=3, ff=256, p=0.1):
        super().__init__()
        self.tok = nn.Embedding(V_in, d_model)
        self.pos = nn.Embedding(L, d_model)
        self.time = TimeEmbedding(d_model)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, p, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.norm = nn.LayerNorm(d_model); self.head = nn.Linear(d_model, K_out)
        self.register_buffer("pos_ids", torch.arange(L))
    def forward(self, x_t, t):                   # x_t:(B,L) long, t:(B,) in (0,1]
        h = self.tok(x_t) + self.pos(self.pos_ids)[None] + self.time(t)[:, None, :]
        return self.head(self.norm(self.encoder(h)))     # (B, L, K_data) logits

print("denoiser:", sum(p.numel() for p in
      DenoiserTransformer(V_in, K_data, L).parameters()), "params")""")

# ===========================================================================
md(r"""## 7. The training target (loss), built up gently

This is the heart of the lesson, so we'll go slowly. **Goal:** find a single
number (the *loss*) that, when made small, makes the model a good denoiser. Three
ideas get us there.

### Idea 1 — turn the model's guess into a real denoising step

The model gives us $\tilde p_\theta(x_0\mid x_t)$: "here's my guess of the original
letter." But denoising means producing $x_{t-1}$ (one step less noisy). We bridge
the two by **averaging the posterior over the model's guess**:

$$p_\theta(x_{t-1}\mid x_t) = \sum_{\text{each possible } x_0} \underbrace{q(x_{t-1}\mid x_t, x_0)}_{\text{the §5 backward step}} \times \underbrace{\tilde p_\theta(x_0\mid x_t)}_{\text{how much the model believes that } x_0}.$$

In plain words: *"for each letter the model thinks the original might be, take the
corresponding backward step, and blend them by how strongly the model believes
each."* This $p_\theta(x_{t-1}\mid x_t)$ is the model's actual reverse step.

### Idea 2 — measure how wrong each backward step is (KL divergence)

We know the *correct* backward step when we cheat and look at the true $x_0$: it's
the posterior $q(x_{t-1}\mid x_t, x_0)$ from §5. So training just means: **make the
model's reverse step $p_\theta$ close to the true posterior $q$.** The standard way
to measure "how different are two weighted dice" is the **KL divergence**
$D_{\mathrm{KL}}(q \,\|\, p)$ — a number that is $0$ when they're identical and
grows as they differ. That mismatch, summed over all the steps, is the loss:

$$\mathcal L = \underbrace{L_0}_{\substack{\text{final step:}\\ \text{did we get the clean text?}}} + \sum_{t=2}^{T}\underbrace{L_{t-1}}_{\substack{\text{step } t:\ \text{is the model's backward}\\ \text{step} = \text{the true posterior?}}} + \underbrace{L_T}_{\substack{\text{starting noise}\\ \text{matches the prior?}}}.$$

This sum has a famous name — the **ELBO** (Evidence Lower BOund). You don't need
its derivation; just the reading above: *"at every noise level, make the model's
one-step denoiser match the true one-step denoiser."* Three notes:

- $L_T$ has **no model parameters** in it (it only compares the fully-noised data
  to the prior). Because we forced $\bar\alpha_T\approx 0$, the fully-noised data
  *is* the prior, so $L_T \approx 0$. **We drop it.**
- $L_{t-1}$ is the KL between two dice — computed in log-space for numerical safety.
- $L_0$ is just "did the model name the right letter at the last step" — an
  ordinary cross-entropy.

### Idea 3 — don't sum all $T$ steps every time (Monte-Carlo)

Summing over all $T=128$ steps for every training example would be slow. Instead,
each step we **pick one random $t$**, compute just that term, and **multiply by
$T$**. On average this equals the full sum (it's an unbiased estimate) — the same
trick used in basically every diffusion model.

### Putting it together: the hybrid loss

D3PM trains with the ELBO **plus** a small, plain "just name the right letter"
cross-entropy bonus on the model's guess. This extra term steadies training (the
pure ELBO can be noisy). With a small weight $\lambda$:

$$\mathcal L_\lambda = \underbrace{\mathcal L_{\text{vb}}}_{\text{the ELBO above}} + \lambda\cdot\underbrace{\big(\text{cross-entropy of the } x_0\text{-guess}\big)}_{\text{a steadying hint}},\qquad \lambda = 0.01.$$

> ✨ **The beautiful part (we verify it in code):** for the **absorbing** kernel,
> the scary KL term $L_{t-1}$ simplifies *exactly* to $r_t \times$ (cross-entropy
> of the guess) on the blanked positions. That is **precisely Lesson 1's loss** —
> a reveal-weighted "guess the blank" cross-entropy. So all this machinery, for
> the blanking kernel, collapses back to the simple thing you already built. The
> uniform kernel doesn't simplify (nothing is ever frozen), so there we compute
> the full KL.
""")

code(r"""def reverse_dist_uniform(ptilde, xt, t):
    '''p_theta(x_{t-1}|x_t) = sum_{x0} q(x_{t-1}|x_t,x0) ptilde(x0)  (uniform, O(K)).'''
    K = K_data
    a_t  = (abar[t] / abar[t - 1]).view(-1, 1, 1); b_t = 1 - a_t
    ab_t1 = abar[t - 1].view(-1, 1, 1); ab_t = abar[t].view(-1, 1, 1)
    oh = F.one_hot(xt, K).float()
    Z = (ab_t * oh + (1 - ab_t) / K).clamp_min(1e-12)     # normalizer per x0 = barQ_t[x0,xt]
    w = ptilde / Z                                        # (B,L,K)
    mix = ab_t1 * w + (1 - ab_t1) / K * w.sum(-1, keepdim=True)
    f1 = a_t * oh + b_t / K
    p = f1 * mix
    return p / p.sum(-1, keepdim=True).clamp_min(1e-12)

# ---- Two invariants that catch most bugs ----
_x0 = data[:4]; _t = torch.randint(2, T + 1, (4,), device=device)
_xt, _ = q_sample(_x0, _t, "uniform")
_pt = torch.softmax(torch.randn(4, L, K_data, device=device), -1)
_rev = reverse_dist_uniform(_pt, _xt, _t)
print("uniform reverse sums to 1:",
      torch.allclose(_rev.sum(-1), torch.ones_like(_rev.sum(-1)), atol=1e-5), " [ok]")

# Absorbing: the KL term L_{t-1} provably collapses to r_t * cross-entropy.
# Verify on the 2-atom masked-position distribution: KL(q||p) == r_t * (-log ptilde[x0]).
t_ = torch.tensor([60], device=device); r = reveal_prob(t_)
c = 0.3                               # model's predicted prob on the true token
# build the 2-atom RESTRICTION over {true x0, MASK}: q={x0:r, MASK:1-r}, p={x0:r*c, MASK:1-r}.
# (These are just the two atoms with q>0; the full-vocab reverse IS normalized -- the other
#  data tokens carry the remaining mass r*(1-c) but have q=0, so they drop out of the KL.)
q2 = torch.tensor([float(r), 1 - float(r)])
p2 = torch.tensor([float(r) * c, 1 - float(r)])
kl = (q2 * (q2.clamp_min(1e-12).log() - p2.clamp_min(1e-12).log())).sum()
print(f"absorbing KL collapse: KL={kl:.4f}  vs  r*CE={float(r)*(-math.log(c)):.4f}  [ok]")""")

# ===========================================================================
md(r"""Now we wrap those ideas into one `d3pm_loss` function used for both kernels.
Read the comments — they map each line back to Idea 1/2/3 above. For **absorbing**
it uses the simple collapsed form (reveal-weighted cross-entropy on blanks); for
**uniform** it computes the genuine KL between the true posterior and the model's
reverse step.
""")

# ===========================================================================
code(r"""def d3pm_loss(model, x0, kernel, lam=0.01):
    '''Hybrid D3PM loss = L_vb (Monte-Carlo, x T) + lambda * CE.  Returns (loss, vb, ce).'''
    B = x0.size(0)
    t = torch.randint(1, T + 1, (B,), device=x0.device)         # t in 1..T
    xt, corrupted = q_sample(x0, t, kernel)
    logits = model(xt, t.float() / T)                           # (B,L,K_data)
    logp = F.log_softmax(logits, dim=-1)
    ce_tok = -logp.gather(-1, x0.unsqueeze(-1)).squeeze(-1)      # (B,L) per-token CE

    if kernel == "absorbing":
        # L_vb collapses to r_t-weighted CE on masked positions (= Lesson 1!).
        m = corrupted.float()
        ce_masked = (ce_tok * m).sum(1) / m.sum(1).clamp_min(1.0)
        vb = (T * reveal_prob(t) * ce_masked).mean()
        ce = ce_masked.mean()
    else:
        # Uniform: genuine KL( q(.|x_t,x0_true) || p_theta(.|x_t) ) over all positions.
        q_post = q_posterior_uniform(x0, xt, t)                 # (B,L,K)
        p_rev = reverse_dist_uniform(logp.exp(), xt, t)         # (B,L,K)
        kl = (q_post * (q_post.clamp_min(1e-12).log()
                        - p_rev.clamp_min(1e-12).log())).sum(-1)   # (B,L)
        vb = (T * kl.mean(1)).mean()
        ce = ce_tok.mean()
    return vb + lam * ce, vb.item(), ce.item()""")

# ===========================================================================
md(r"""## 8. Train both kernels

We train two identical models — one per kernel — on the toy corpus with the same
budget, so the comparison is fair. Watch the **cross-entropy (ce)** column drop:
that's the model getting better at naming the original letters. A minute or two on
CPU, seconds on GPU.
""")

code(r"""def train(kernel, steps=1200, batch=64, lr=2e-3):
    torch.manual_seed(SEED)
    model = DenoiserTransformer(V_in, K_data, L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    hist = []
    model.train()
    for step in range(1, steps + 1):
        idx = torch.randint(0, data.size(0), (batch,), device=device)
        loss, vb, ce = d3pm_loss(model, data[idx], kernel)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        hist.append((vb, ce))
        if step % 300 == 0 or step == 1:
            print(f"  [{kernel:9s}] step {step:4d} | vb {np.mean([h[0] for h in hist[-50:]]):7.3f}"
                  f" | ce {np.mean([h[1] for h in hist[-50:]]):.3f}")
    return model, np.array(hist)

print("Training absorbing kernel:")
model_abs, hist_abs = train("absorbing")
print("Training uniform kernel:")
model_uni, hist_uni = train("uniform")

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
for a, (name, h) in zip(ax, [("absorbing", hist_abs), ("uniform", hist_uni)]):
    a.plot(np.convolve(h[:, 1], np.ones(25)/25, "valid"))
    a.set_title(f"{name}: x0 cross-entropy"); a.set_xlabel("step"); a.set_ylabel("CE")
plt.tight_layout(); plt.show()""")

# ===========================================================================
md(r"""## 9. Generating text from each model

Generation runs the noise **backwards**. We start from pure noise ($x_T$, the
**prior**) and repeatedly take the model's reverse step, $t = T \to 1$, until we
reach clean text. ("Ancestral" sampling just means "sample each step in turn".)

- **Absorbing:** start from **all `[MASK]`**; each step, reveal some still-blank
  positions (with probability $r_t$) by drawing a letter from the model's guess.
  Once revealed, a position is **fixed**. *(Identical to Lesson 1's generator.)*
- **Uniform:** start from **random letters**; each step, resample *every* position
  from the reverse distribution. Nothing is ever fixed — the text keeps churning
  until it settles.

The `temperature` knob (like in any LM) trades variety for fidelity: lower =
safer/more repetitive, higher = more diverse/riskier.
""")

code(r"""@torch.no_grad()
def sample(model, kernel, n=10, temperature=0.9, seed=1):
    torch.manual_seed(seed); model.eval()
    if kernel == "absorbing":
        x = torch.full((n, L), MASK_ID, device=device)
    else:
        x = torch.randint(0, K_data, (n, L), device=device)
    for t in range(T, 0, -1):
        tt = torch.full((n,), t, device=device)
        ptilde = torch.softmax(model(x, tt.float() / T) / temperature, dim=-1)
        if kernel == "absorbing":
            sm = (x == MASK_ID)
            pred = torch.multinomial(ptilde.reshape(-1, K_data), 1).reshape(n, L)
            reveal = (torch.rand_like(x, dtype=torch.float) < reveal_prob(tt)[:, None]) & sm
            x = torch.where(reveal, pred, x)
        else:
            p_rev = reverse_dist_uniform(ptilde, x, tt)
            x = torch.multinomial(p_rev.reshape(-1, K_data), 1).reshape(n, L)
    if kernel == "absorbing" and (x == MASK_ID).any():       # fill any residual mask
        ptilde = torch.softmax(model(x, torch.zeros(n, device=device) + 1.0/T), -1)
        x = torch.where(x == MASK_ID, ptilde.argmax(-1), x)
    return x

print("=== ABSORBING samples ===")
for r in sample(model_abs, "absorbing"): print("  ", repr(decode(r).rstrip()))
print("\n=== UNIFORM samples ===")
for r in sample(model_uni, "uniform"):   print("  ", repr(decode(r).rstrip()))""")

# ===========================================================================
md(r"""## 10. Comparing the kernels: *revisable* vs. *irreversible*

Let's put a number on quality: the fraction of generated lines that exactly match
a training sentence (our corpus is tiny, so "reproduce the data" is the goal).
But read the result with the mechanism in mind — it's more interesting than a
simple "X beats Y".
""")

code(r"""train_set = set(decode(d).rstrip() for d in data)
def exact_frac(model, kernel, n=200):
    s = sample(model, kernel, n=n, seed=7)
    hits = sum(decode(r).rstrip() in train_set for r in s)
    return hits / n

fa = exact_frac(model_abs, "absorbing"); fu = exact_frac(model_uni, "uniform")
print(f"exact-match rate | absorbing: {fa:.0%}   uniform: {fu:.0%}")
print("\nUniform can REVISE every position at every step, so on this tiny memorization")
print("toy it often locks onto clean lines. Absorbing COMMITS each token the instant")
print("it is unmasked and never revisits it -- one bad draw leaves a permanent typo.")
print("See the discussion below for why absorbing is still the modern default.")

# The bridge: absorbing's reveal schedule matches Lesson 1's (t-s)/t intuition.
ts = torch.arange(1, T + 1, device=device)
print(f"\nabsorbing reveal prob r_t: r_1={float(reveal_prob(ts[0:1])):.3f} (full),"
      f" r_{T//2}={float(reveal_prob(ts[T//2:T//2+1])):.3f},"
      f" r_T={float(reveal_prob(ts[-1:])):.3f}")""")

# ===========================================================================
md(r"""## 11. Recap & what's next

**What you built — the whole D3PM framework, from scratch:**

- **Noise = a probability table $Q_t$.** Pick the table, pick your diffusion. We
  did **uniform** (random-replace) and **absorbing** (blank-out).
- **Shortcut formulas** for "jump to step $t$" (the marginal) and "step back one"
  (the posterior) — each **checked against brute-force Bayes** so you can trust them.
- **The loss, demystified:** turn the model's "guess the original letter" into a
  reverse step, then make that reverse step match the true one (KL), at every noise
  level (the ELBO), estimated cheaply (Monte-Carlo), with a small steadying hint
  (the hybrid cross-entropy).
- **A satisfying punchline:** for the blanking kernel, that whole loss *collapses*
  to Lesson 1's reveal-weighted "guess the blank" — so **Lesson 1 was a special
  case of D3PM all along.**

**The key takeaway — *revisable* vs. *irreversible* generation:** the uniform
kernel re-decides *every* letter at *every* step, so it can fix its own mistakes —
which is why on this tiny toy it often scores **higher**. The absorbing kernel
**commits a letter the moment it's revealed and never touches it again**, so one
unlucky draw leaves a permanent typo. So why do real diffusion LLMs (MDLM, LLaDA)
use absorbing anyway?

- Its training target is **far simpler** — it *is* Lesson 1's cross-entropy (we
  proved it), not a full KL over the whole vocabulary every step.
- A `[MASK]` is an explicit *"fix me here"* signal; a random letter hides *where*
  the corruption even is.
- It gives **better likelihoods on real text at scale** (D3PM's actual finding).
- Its one weakness — irreversibility — is **fixable** with smarter generators
  (confidence-based unmasking à la MaskGIT/LLaDA, "remasking", or more steps),
  which recover and surpass uniform. Uniform's edge here is a small-data artifact:
  re-deciding every letter over a big vocabulary every step doesn't scale.

**Where this is heading.** The continuous-time limit of the absorbing kernel is
**MDLM**, which (scaled up) is **LLaDA**. A different refinement — learning the
*ratios* of the distribution instead of the posterior — is **SEDD** (a later lesson).

**Coming in Lesson 3 — Diffusion in *continuous* space (Diffusion-LM):** instead of
corrupting discrete letters with tables, we'll turn letters into vectors, add
ordinary **Gaussian noise** (like image diffusion), denoise in that continuous
space, and round back to letters. A completely different way to diffuse language.

---
### Exercises (each builds on the code above)
1. **A third kernel.** D3PM also has a "nudge to a *similar* state" table (good for
   ordered data). Build a banded $Q_t$ that prefers swapping a letter for a nearby
   one, and see how the corruption looks.
2. **Schedule ablation.** Replace the cosine schedule with a straight line
   ($\beta_t$ constant). How do the loss curves and samples change for each kernel?
3. **Turn off the hint.** Set $\lambda = 0$ (drop the steadying cross-entropy). Is
   training noisier? Now set $\lambda$ huge — the absorbing model becomes *exactly*
   Lesson 1; explain why.
4. **Fewer steps.** Plot exact-match rate vs. $T \in \{8,16,32,64,128\}$ for both
   kernels. Which one falls apart faster when it has fewer chances to revise/reveal?
5. **Smarter unmasking.** Swap the absorbing sampler's *random* reveal for
   "reveal the letters the model is most confident about first" (MaskGIT/LLaDA
   style; Lesson 1, Exercise 1). Does the gap to uniform close?
""")

# ===========================================================================
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.13"},
}
# Write the notebook to the repository root (one level up from tools/).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
out_path = os.path.join(ROOT, "02_d3pm_transition_kernels.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", out_path, "with", len(cells), "cells")
