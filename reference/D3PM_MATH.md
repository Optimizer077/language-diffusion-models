# D3PM Math Reference

A compact, implementation-oriented derivation of the discrete-diffusion math used
in **Lesson 2** (the D3PM framework), covering the **uniform** and **absorbing**
kernels. Every formula here is validated numerically in two places:

- [`tools/d3pm_reference_check.py`](../tools/d3pm_reference_check.py) — a
  brute-force oracle that builds the explicit transition matrices and checks the
  closed forms against them.
- Inline cells in [`02_d3pm_transition_kernels.ipynb`](../02_d3pm_transition_kernels.ipynb)
  (closed-form posterior vs. brute-force Bayes; reverse distribution normalizes;
  the absorbing KL-collapse identity).

Notation: row-stochastic transition matrices, $q(x_t{=}j \mid x_{t-1}{=}i) = [Q_t]_{ij}$.
A state is a one-hot **row** vector $x$; one forward step is $x_{t-1} Q_t$.
$K$ = number of states. $K_{\text{data}}$ = number of real data classes. For the
absorbing kernel there is one extra `[MASK]` state $m$, so $K = K_{\text{data}}+1$;
the uniform kernel has no mask state, so $K = K_{\text{data}}$. The denoiser only
ever predicts over the $K_{\text{data}}$ data classes.

---

## 1. Schedule

Let $\beta_t \in (0,1]$ be the per-step corruption rate, $\alpha_t = 1-\beta_t$ the
keep rate, and the **cumulative keep (survival) probability**

$$\bar\alpha_t = \prod_{s=1}^{t}\alpha_s = \prod_{s=1}^{t}(1-\beta_s),\qquad \bar\alpha_0 = 1.$$

We use a **cosine** schedule on $\bar\alpha_t$ (Nichol–Dhariwal):
$\bar\alpha_t \propto \cos^2\!\big(\tfrac{\pi}{2}\tfrac{t/T+s}{1+s}\big)$ with $s=0.008$,
then recover $\alpha_t = \bar\alpha_t/\bar\alpha_{t-1}$, $\beta_t = 1-\alpha_t$.
We force $\bar\alpha_T \approx 0$ (fully corrupted at the last step) so the prior
term $L_T$ vanishes. A simple linear $\beta_t$ also works.

> **Pitfall.** $\bar\alpha_t$ is the *keep* probability and must **decrease** in
> $t$ ($\bar\alpha_0=1 \to \bar\alpha_T\approx 0$). $\bar\alpha_0 = 1$ exactly, or
> the whole schedule shifts by one.

---

## 2. Transition matrices $Q_t$ and the cumulative $\bar Q_t$

### Uniform ($K = K_{\text{data}}$)

$$[Q_t]_{ij} = \alpha_t\,\mathbb 1[i{=}j] + \tfrac{\beta_t}{K},
\qquad
[\bar Q_t]_{ij} = \bar\alpha_t\,\mathbb 1[i{=}j] + \tfrac{1-\bar\alpha_t}{K}.$$

The cumulative form follows because $Q_t = \alpha_t I + (1-\alpha_t)U$ with
$U=\tfrac1K\mathbf 1\mathbf 1^\top$ idempotent ($U^2=U$); products telescope.
Stationary distribution: uniform.

**Direct sampler** $q(x_t\mid x_0)$: keep $x_0$ w.p. $\bar\alpha_t$, else draw a
token uniformly over all $K$ classes (*including possibly $x_0$ itself* — so
$q(x_t{=}x_0)=\bar\alpha_t+\tfrac{1-\bar\alpha_t}{K}$, not $\bar\alpha_t$).

### Absorbing ($K = K_{\text{data}}+1$, mask state $m$)

$$[Q_t]_{ij} = \alpha_t\,\mathbb 1[i{=}j{\neq}m] + \beta_t\,\mathbb 1[j{=}m,\,i{\neq}m] + \mathbb 1[i{=}j{=}m].$$

The $m$-th row is $e_m$ (mask is **absorbing**). Cumulative:

$$[\bar Q_t]_{ij} = \bar\alpha_t\,\mathbb 1[i{=}j{\neq}m] + (1-\bar\alpha_t)\,\mathbb 1[j{=}m,\,i{\neq}m] + \mathbb 1[i{=}j{=}m].$$

Stationary distribution: point mass on $m$.

**Direct sampler** $q(x_t\mid x_0)$: keep $x_0$ w.p. $\bar\alpha_t$, else `[MASK]`.
This is **identical to Lesson 1's** `forward_mask` with keep-probability $\bar\alpha_t$.

> **Pitfall.** Always overwrite row $m$ with $e_m$. The naive
> $\alpha_t I + \beta_t \mathbf 1 e_m^\top$ leaves the mask row non-absorbing and
> lets mask "decay" back into a token.

---

## 3. Forward posterior $q(x_{t-1}\mid x_t, x_0)$

By Bayes, with the **single-step** $Q_t$ and the **$(t{-}1)$-cumulative** $\bar Q_{t-1}$:

$$q(x_{t-1}{=}k \mid x_t, x_0) \;\propto\; [Q_t]_{k,\,x_t}\,[\bar Q_{t-1}]_{x_0,\,k},$$

normalized over $k$ (the normalizer is $[\bar Q_t]_{x_0,x_t} = q(x_t\mid x_0)$).

> **Pitfall.** The numerator uses the **column** $x_t$ of $Q_t$, i.e.
> $[Q_t]_{k,x_t}$, and $\bar Q_{t-1}$ (not $\bar Q_t$). Transposing $Q_t$ or using
> the wrong cumulative index silently yields a wrong-but-normalized distribution
> — verify against brute force.

### Uniform (categorical over $K$ states)

$$q(x_{t-1}{=}k\mid x_t,x_0)\;\propto\;
\underbrace{\Big(\alpha_t\mathbb 1[k{=}x_t]+\tfrac{\beta_t}{K}\Big)}_{f_1(k)}\,
\underbrace{\Big(\bar\alpha_{t-1}\mathbb 1[k{=}x_0]+\tfrac{1-\bar\alpha_{t-1}}{K}\Big)}_{f_2(k)}.$$

Implement as the elementwise product of two length-$K$ vectors, then normalize.

### Absorbing (two cases on the observed $x_t$)

- **$x_t \neq m$ (already revealed):** $q(x_{t-1}\mid x_t,x_0)=\delta(x_{t-1}{=}x_t)$.
  A revealed token was never masked, so it is **frozen**.
- **$x_t = m$ (still masked):** reverts to $x_0$ or stays masked:

$$\boxed{\,r_t \;=\; \frac{\bar\alpha_{t-1}-\bar\alpha_t}{1-\bar\alpha_t}\,},\qquad
q(x_{t-1}{=}x_0\mid m,x_0)=r_t,\quad q(m\mid m,x_0)=1-r_t.$$

This $r_t$ is **exactly Lesson 1's reveal probability** $(t-s)/t$ under the linear
schedule. (Derivation: numerator for $x_0$ is $\beta_t\bar\alpha_{t-1}=\bar\alpha_{t-1}-\bar\alpha_t$;
for $m$ is $1-\bar\alpha_{t-1}$; normalizer is $1-\bar\alpha_t$.)

**Boundary $t=1$:** $\bar Q_0 = I$, so the posterior collapses to $\delta(x_0)$ in
both kernels; $r_1=1$ (deterministic recovery). The $t=1$ reverse is handled by
the reconstruction term $L_0$.

---

## 4. The x0-parameterized reverse and the ELBO

The network outputs $\tilde p_\theta(x_0\mid x_t) = \mathrm{softmax}$ over the
$K_{\text{data}}$ data classes (never predicts $m$). The reverse transition is the
posterior averaged under the predicted clean token:

$$p_\theta(x_{t-1}\mid x_t) \;=\; \sum_{x_0}\, q(x_{t-1}\mid x_t,x_0)\,\tilde p_\theta(x_0\mid x_t).$$

The negative variational bound telescopes to

$$\mathcal L = \underbrace{L_0}_{-\log p_\theta(x_0\mid x_1)} + \sum_{t=2}^{T}\underbrace{L_{t-1}}_{D_{\mathrm{KL}}(q(x_{t-1}\mid x_t,x_0)\,\|\,p_\theta(x_{t-1}\mid x_t))} + \underbrace{L_T}_{\text{prior, const}}.$$

- **$L_T = D_{\mathrm{KL}}(q(x_T\mid x_0)\,\|\,p(x_T))$** has no parameters and
  $\to 0$ once $\bar\alpha_T\approx 0$ (then $q(x_T\mid x_0)$ *is* the prior:
  uniform for uniform, all-mask for absorbing). Dropped.
- **$L_{t-1}$** is a KL between two categoricals — compute in log space, clamp
  with $\varepsilon$.
- **$L_0$** is a cross-entropy of the $x_0$-head at $t=1$.

**Monte-Carlo estimator.** Sample one $t\sim\mathcal U\{1..T\}$ per example,
evaluate that term, and **multiply by $T$** (unbiased for the sum over $t$). Don't
double-count $t{=}1$: it is $L_0$, not an $L_{t-1}$ KL.

### Uniform reverse, $O(K)$ closed form

With $w(x_0) = \tilde p_\theta(x_0)/Z(x_0)$, $Z(x_0)=[\bar Q_t]_{x_0,x_t}=\bar\alpha_t\mathbb 1[x_0{=}x_t]+\tfrac{1-\bar\alpha_t}{K}$, and $W=\sum_{x_0} w(x_0)$:

$$p_\theta(x_{t-1}{=}k\mid x_t) \;\propto\; f_1(k)\,\Big[\bar\alpha_{t-1}\,w(k) + \tfrac{1-\bar\alpha_{t-1}}{K}\,W\Big],\qquad f_1(k)=\alpha_t\mathbb 1[k{=}x_t]+\tfrac{\beta_t}{K}.$$

### Absorbing: the KL collapses to weighted cross-entropy

At a still-masked position the true posterior is the 2-atom $\{x_0: r_t,\ m: 1-r_t\}$
and the model's reverse is $\{x_0: r_t\tilde p_\theta(x_0),\ \dots,\ m: 1-r_t\}$. The KL is

$$L_{t-1} \;=\; r_t\,\big(-\log \tilde p_\theta(x_0\mid x_t)\big) \;=\; r_t\cdot \mathrm{CE}.$$

Revealed positions contribute $0$. So **the entire absorbing ELBO is
$r_t$-weighted cross-entropy on masked positions** — i.e. exactly Lesson 1's
objective (Lesson 1's $1/t$ weighting is the continuous-time limit of $T\,r_t$, the
reveal *rate* — which is exactly what the code multiplies, `T * reveal_prob(t)`). This
identity is checked numerically in the notebook.

---

## 5. Hybrid loss

D3PM trains the **hybrid** objective (Austin et al. 2021, Eq. 5):

$$\mathcal L_\lambda = \mathcal L_{\text{vb}} + \lambda\,\mathbb E_{t}\big[-\log\tilde p_\theta(x_0\mid x_t)\big],\qquad \lambda = 0.01\ \text{(text/absorbing)}.$$

The auxiliary cross-entropy directly supervises the $x_0$-head and stabilizes the
high-variance bound. For absorbing it is scored on masked positions; for uniform,
on all positions. The target is always a **data class** — never feed $m$ as a CE
target.

---

## 6. Sampling (ancestral)

Initialize $x_T$ from the prior, then for $t=T,\dots,1$ draw
$x_{t-1}\sim p_\theta(x_{t-1}\mid x_t)$.

- **Uniform:** $x_T\sim$ uniform tokens; each step resamples *every* position from
  the reverse mixture (nothing frozen). → **revisable**.
- **Absorbing:** $x_T=$ all `[MASK]`; revealed positions are copied through;
  masked positions reveal w.p. $r_t$, drawing the token from $\tilde p_\theta$.
  Identical to Lesson 1. → **irreversible** (a committed token never changes).

The revisable/irreversible distinction explains why the uniform kernel can *win*
on tiny memorization tasks (it fixes its mistakes), while absorbing — preferred at
scale for its simpler objective and better likelihoods — needs smarter samplers
(confidence-based unmasking / remasking) to overcome irreversibility.

---

## 7. Invariants worth asserting (used in the checks)

1. Rows of $Q_t$ and $\bar Q_t$ sum to 1 (both kernels).
2. Closed-form $\bar Q_t$ equals the explicit product $Q_1\cdots Q_t$ (small $K$).
3. $q(x_t\mid x_0)$ and the posterior are valid distributions (nonneg, sum 1).
4. **Closed-form posterior == brute-force Bayes** $[Q_t]_{k,x_t}[\bar Q_{t-1}]_{x_0,k}$ normalized, for all $(x_0,x_t,t)$.
5. $t{=}1$ posterior collapses to $\delta(x_0)$; $r_1 = 1$.
6. Absorbing reduces to Lesson 1 (matched schedule → identical reveal statistics).
7. Absorbing $L_{t-1} = r_t\cdot\mathrm{CE}$; revealed positions contribute 0.
8. Reverse mixture $p_\theta(x_{t-1}\mid x_t)$ sums to 1.
9. $L_T \to 0$ as $\bar\alpha_T\to 0$.

---

### References
- Austin, Johnson, Ho, Tarlow, van den Berg (2021), *Structured Denoising Diffusion Models in Discrete State-Spaces* (D3PM). https://arxiv.org/abs/2107.03006
- Sahoo et al. (2024), *Simple and Effective Masked Diffusion Language Models* (MDLM). https://arxiv.org/abs/2406.07524
- Nie et al. (2025), *Large Language Diffusion Models* (LLaDA). https://arxiv.org/abs/2502.09992
