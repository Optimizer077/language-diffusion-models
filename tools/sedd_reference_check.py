"""Ground-truth oracle for the SEDD math (Lesson 5 validation).

SEDD (Lou, Meng, Ermon 2023) learns the *concrete score* — the ratios
p_t(y)/p_t(x) between a state x and its single-token neighbours y — instead of the
posterior. This script verifies, by brute force on a tiny toy, the three facts the
lesson relies on:

  1. The forward (uniform) marginal q(x_t|x_0) is a valid distribution.
  2. The denoising score-entropy term  f(s) = s - r*log(s)  is minimised at s = r
     (so training pushes the network's score toward the target ratio).
  3. The SEDD identity: the posterior-weighted average of the per-x0 target ratios
     equals the true marginal ratio p_t(y)/p_t(x).  This is why a network trained
     with the (x0-dependent) denoising target recovers the true marginal score.

Run: python sedd_reference_check.py
"""
import itertools
import numpy as np

np.random.seed(0)


def p_marginal(a, x0, N):
    """q(x_t = . | x_0), uniform kernel: keep x0 w.p. a, else uniform over N."""
    v = np.full(N, (1 - a) / N)
    v[x0] += a
    return v


def seq_marginal(a, x0_seq, x_seq, N):
    """q(x_t = x_seq | x_0 = x0_seq) = product over independent positions."""
    return np.prod([p_marginal(a, x0i, N)[xi] for x0i, xi in zip(x0_seq, x_seq)])


def check_marginal(N=5):
    for a in [0.9, 0.5, 0.1]:
        for x0 in range(N):
            v = p_marginal(a, x0, N)
            assert np.isclose(v.sum(), 1.0) and (v >= 0).all()
    print("  [ok] uniform marginal q(x_t|x_0) is a valid distribution")


def check_dse_minimizer():
    # f(s) = s - r*log s  has f'(s) = 1 - r/s = 0  ->  s = r  (and f''>0).
    for r in [0.05, 0.5, 1.0, 3.0, 12.0]:
        s_grid = np.linspace(1e-3, 20, 400000)
        f = s_grid - r * np.log(s_grid)
        s_star = s_grid[np.argmin(f)]
        assert abs(s_star - r) < 1e-2, (r, s_star)
    print("  [ok] denoising score-entropy term  s - r*log s  is minimised at s = r")


def check_sedd_identity(N=4, L=3, M=5):
    """Posterior-avg of per-x0 ratios == true marginal ratio p_t(y)/p_t(x)."""
    data = [tuple(np.random.randint(0, N, size=L)) for _ in range(M)]   # toy dataset
    prior = 1.0 / M
    max_err = 0.0
    for a in [0.8, 0.4, 0.15]:
        # true marginal over ALL sequences: p_t(x) = mean_x0 q(x|x0)
        def p_t(x_seq):
            return prior * sum(seq_marginal(a, x0, x_seq, N) for x0 in data)

        # pick a few current states x_t and a single-position flip -> y
        for _ in range(30):
            x = tuple(np.random.randint(0, N, size=L))
            i = np.random.randint(L)
            b = np.random.randint(N)
            y = tuple(b if j == i else x[j] for j in range(L))

            true_ratio = p_t(y) / p_t(x)

            # posterior p(x0 | x_t=x)  ∝  q(x|x0) * prior
            w = np.array([seq_marginal(a, x0, x, N) for x0 in data]) * prior
            post = w / w.sum()
            # per-x0 target ratio (only position i matters; others cancel)
            per_x0 = np.array([p_marginal(a, x0[i], N)[b] / p_marginal(a, x0[i], N)[x[i]]
                               for x0 in data])
            identity = float((post * per_x0).sum())

            max_err = max(max_err, abs(identity - true_ratio))
    assert max_err < 1e-9, max_err
    print(f"  [ok] SEDD identity  E_post[ per-x0 ratio ] == p_t(y)/p_t(x)  (max err {max_err:.1e})")


if __name__ == "__main__":
    print("=== SEDD math oracle ===")
    check_marginal()
    check_dse_minimizer()
    check_sedd_identity()
    print("\nALL SEDD ORACLE CHECKS PASSED")
