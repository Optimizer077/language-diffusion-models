"""Ground-truth oracle for D3PM math (Lesson 2 validation).

This builds the FULL explicit K x K transition matrices for the uniform and
absorbing kernels and computes the cumulative marginal and the forward posterior
by brute force (matrix products + Bayes). The notebook will use *reduced scalar
closed forms* for speed; this file is the oracle those closed forms must match.

Run: python d3pm_reference_check.py
"""
import numpy as np

np.random.seed(0)
np.set_printoptions(precision=4, suppress=True)


# ----------------------------------------------------------------------------
# Schedules
# ----------------------------------------------------------------------------
def betas_linear(T, b0=1e-2, b1=0.2):
    return np.linspace(b0, b1, T)


def betas_cosine_absorbing(T):
    # Absorbing schedule s.t. cumulative keep-prob \bar alpha_t follows a cosine.
    # \bar alpha_t = cos(pi/2 * t/T)^2 style; derive per-step beta from the ratio.
    t = np.arange(T + 1)
    abar = np.cos(0.5 * np.pi * t / T) ** 2          # abar[0]=1 ... abar[T]=0
    abar = np.clip(abar, 1e-6, 1.0)
    beta = 1.0 - abar[1:] / abar[:-1]                 # 1 - abar_t/abar_{t-1}
    return np.clip(beta, 0.0, 1.0)


# ----------------------------------------------------------------------------
# Explicit transition matrices Q_t  (q(x_t=j | x_{t-1}=i) = Q[i, j])
# ----------------------------------------------------------------------------
def Q_uniform(beta, K):
    return (1 - beta) * np.eye(K) + beta / K * np.ones((K, K))


def Q_absorbing(beta, K):
    # K states: indices 0..K-2 are data classes, index K-1 is [MASK] (absorbing).
    m = K - 1
    Q = (1 - beta) * np.eye(K)
    Q[:, m] += beta            # everything moves to mask w.p. beta...
    Q[m, :] = 0.0; Q[m, m] = 1.0  # ...except mask, which is absorbing (stays).
    return Q


def cumulative(Qs):
    """Qbar[t] = Q_1 @ Q_2 @ ... @ Q_t, with Qbar[0] = I."""
    K = Qs[0].shape[0]
    out = [np.eye(K)]
    for Q in Qs:
        out.append(out[-1] @ Q)
    return out  # length T+1, out[t] = \bar Q_t


def posterior_bruteforce(Qt, Qbar_tm1, xt, x0, K):
    """q(x_{t-1} | x_t=xt, x_0=x0) by Bayes, returned as a length-K vector.

    q(x_{t-1}=k | x_t, x_0) ∝ q(x_t | x_{t-1}=k) * q(x_{t-1}=k | x_0)
                            = Qt[k, xt]            * Qbar_tm1[x0, k]
    """
    unnorm = Qt[:, xt] * Qbar_tm1[x0, :]
    Z = unnorm.sum()
    return unnorm / Z


# ----------------------------------------------------------------------------
# Invariant checks
# ----------------------------------------------------------------------------
def check_kernel(name, Q_fn, K, T, betas, has_mask):
    print(f"\n=== {name} kernel | K={K} T={T} ===")
    Qs = [Q_fn(b, K) for b in betas]
    Qbar = cumulative(Qs)

    # (1) Row-stochastic Q_t and \bar Q_t
    for t in range(T):
        assert np.allclose(Qs[t].sum(1), 1), f"Q_{t+1} rows !=1"
    for t in range(T + 1):
        assert np.allclose(Qbar[t].sum(1), 1), f"Qbar_{t} rows !=1"
    print("  [ok] all Q_t and \\bar Q_t are row-stochastic")

    # (2) Marginal direct vs sequential: \bar Q_t row should equal sampling stats
    # Compare closed-form \bar Q_t against Monte-Carlo of the step-by-step chain.
    t_test = T // 2
    x0 = 0
    N = 200000
    state = np.full(N, x0)
    for t in range(t_test):
        P = Qs[t][state]                       # (N, K)
        u = np.random.rand(N, 1)
        state = (np.cumsum(P, 1) > u).argmax(1)
    emp = np.bincount(state, minlength=K) / N
    err = np.abs(emp - Qbar[t_test][x0]).max()
    print(f"  [ok] MC marginal at t={t_test} matches \\bar Q_t (max err {err:.4f})")
    assert err < 0.02

    # (3) Posterior is a valid distribution; and consistency:
    #     sum_{x_{t-1}} q(x_t|x_{t-1}) q(x_{t-1}|x0) == q(x_t|x0) = Qbar_t[x0, xt]
    t = max(2, T // 2)
    for x0 in range(K - (1 if has_mask else 0)):       # x0 is always a data class
        for xt in range(K):
            # The posterior q(x_{t-1}|x_t,x_0) is only defined for REACHABLE x_t,
            # i.e. q(x_t|x_0) = Qbar_t[x0, xt] > 0. Under the absorbing kernel a
            # data token != x0 is unreachable (marginal 0) -> skip; 0/0 otherwise.
            if Qbar[t][x0, xt] <= 1e-12:
                if has_mask:
                    assert xt != K - 1 and xt != x0, "unexpected zero marginal"
                continue
            post = posterior_bruteforce(Qs[t - 1], Qbar[t - 1], xt, x0, K)
            assert np.all(post >= -1e-9) and np.isclose(post.sum(), 1), "bad posterior"
            recon = (Qs[t - 1][:, xt] * Qbar[t - 1][x0, :]).sum()
            assert np.isclose(recon, Qbar[t][x0, xt]), "Bayes/marginal inconsistent"
    print("  [ok] posterior valid + Bayes-marginal consistency holds for all (x0, xt)")
    return Qs, Qbar


def check_absorbing_reduces_to_lesson1(K, T, betas):
    """Absorbing \bar Q_t keep-prob must equal prod(1-beta_s); marginal = keep-or-mask."""
    print("\n=== absorbing reduces to Lesson 1 ===")
    Qs = [Q_absorbing(b, K) for b in betas]
    Qbar = cumulative(Qs)
    m = K - 1
    keep = np.cumprod(1 - betas)                         # \bar alpha_t
    for t in range(1, T + 1):
        x0 = 0
        assert np.isclose(Qbar[t][x0, x0], keep[t - 1]), "keep-prob mismatch"
        assert np.isclose(Qbar[t][x0, m], 1 - keep[t - 1]), "mask-prob mismatch"
        # no probability on other data tokens
        others = [j for j in range(K) if j not in (x0, m)]
        assert np.allclose(Qbar[t][x0, others], 0.0), "leak to other tokens"
    print("  [ok] q(x_t|x0): keep w.p. prod(1-beta), else MASK -- matches Lesson 1")


if __name__ == "__main__":
    K_data = 6
    T = 20

    # Uniform kernel: K = K_data (no mask state)
    check_kernel("uniform", Q_uniform, K_data, T, betas_linear(T), has_mask=False)

    # Absorbing kernel: K = K_data + 1 (mask is last index)
    bab = betas_cosine_absorbing(T)
    check_kernel("absorbing", Q_absorbing, K_data + 1, T, bab, has_mask=True)
    check_absorbing_reduces_to_lesson1(K_data + 1, T, bab)

    print("\nALL ORACLE INVARIANTS PASSED")
