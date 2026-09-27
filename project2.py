"""Project 2 - Ill-conditioned least squares (family C): thermocouple calibration.

Fit T(E) = sum_k c_k * phi_k(E) to noisy (voltage, temperature) data by minimizing
f(c) = 1/(2m) ||V c - y||^2,  Hessian H = V^T V / m.
Run:  python project2.py   (writes figs/*.png and prints every number used in README.md)
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numpy.polynomial import chebyshev as C
from pathlib import Path

rng = np.random.default_rng(0)
FIG = Path(__file__).parent / "figs"
FIG.mkdir(exist_ok=True)

# --- Data: NIST ITS-90 type-K inverse polynomial, 0..500 degC (E in mV, 0..20.644) ---
NIST_K = [0.0, 2.508355e1, 7.860106e-2, -2.503131e-1, 8.315270e-2,
          -1.228034e-2, 9.804036e-4, -4.413030e-5, 1.057734e-6, -1.052755e-8]
E_MAX = 20.644
m = 200
E = np.linspace(0.0, E_MAX, m)
T_true = np.polynomial.polynomial.polyval(E, NIST_K)
assert abs(np.polynomial.polynomial.polyval(E_MAX, NIST_K) - 500) < 0.1  # sanity check of coefficients
y = T_true + rng.normal(0, 0.05, m)  # 0.05 degC measurement noise


def design(d, basis):
    """Design matrix for degree d. 'raw' = monomials in mV, 'cheb' = Chebyshev on t in [-1,1]."""
    if basis == "raw":
        return np.vander(E, d + 1, increasing=True)
    t = 2 * E / E_MAX - 1
    return C.chebvander(t, d)


def hessian(V):
    return V.T @ V / m


def jacobi(H):
    """Symmetric Jacobi rescaling D^-1/2 H D^-1/2 (the family-F test)."""
    s = 1 / np.sqrt(np.diag(H))
    return H * np.outer(s, s), s


def kappa(H):
    w = np.linalg.eigvalsh(H)
    return w[-1] / w[0]


def kappa_V(V, scaled=False):
    """kappa(V^T V) from singular values of V: accurate even when H is numerically singular."""
    if scaled:
        V = V / np.linalg.norm(V, axis=0)  # same as Jacobi rescaling of H
    s = np.linalg.svd(V, compute_uv=False)
    return (s[0] / s[-1]) ** 2


def gd(H, g, x0, xstar, tol=1e-8, cap=100_000):
    """Gradient descent with step 1/L on f = 1/2 x^T H x - g^T x. Returns (iters, gap history)."""
    L = np.linalg.eigvalsh(H)[-1]
    x = x0.copy()
    gap = lambda x: 0.5 * (x - xstar) @ H @ (x - xstar)
    g0 = gap(x0)
    hist = [1.0]
    for k in range(1, cap + 1):
        x -= (H @ x - g) / L
        r = gap(x) / g0
        hist.append(r)
        if r <= tol:
            return k, np.array(hist)
    return None, np.array(hist)  # None = did not converge within cap


def cg(H, g, x0, xstar, tol=1e-8, cap=100_000):
    x = x0.copy()
    r = g - H @ x
    p = r.copy()
    gap = lambda x: 0.5 * (x - xstar) @ H @ (x - xstar)
    g0 = gap(x0)
    hist = [1.0]
    for k in range(1, cap + 1):
        Hp = H @ p
        a = (r @ r) / (p @ Hp)
        x = x + a * p
        r_new = r - a * Hp
        p = r_new + (r_new @ r_new) / (r @ r) * p
        r = r_new
        rel = max(gap(x) / g0, 1e-18)
        hist.append(rel)
        if rel <= tol:
            return k, np.array(hist)
    return None, np.array(hist)


def problem(d, basis, scaled):
    """Quadratic data (H, g, x*) in the chosen coordinates; x* from a stable QR least-squares solve."""
    V = design(d, basis)
    H = hessian(V)
    s = np.ones(d + 1)
    if scaled:
        H, s = jacobi(H)
        V = V * s
    g = V.T @ y / m
    xstar = np.linalg.lstsq(V, y, rcond=None)[0]
    return H, g, xstar


# ---------------- D2: intrinsic-kappa test (growth with degree + survival after rescaling) --------------
degrees = range(1, 13)
rows = []
for d in degrees:
    Vr, Vc = design(d, "raw"), design(d, "cheb")
    rows.append((d, kappa_V(Vr), kappa_V(Vr, True), kappa_V(Vc, True)))
print("\nD2  kappa vs degree d")
print(f"{'d':>3} {'raw monomial':>14} {'Jacobi-scaled':>14} {'Chebyshev':>10}")
for d, kr, kj, kc in rows:
    print(f"{d:>3} {kr:14.3e} {kj:14.3e} {kc:10.2f}")
R = np.array(rows)
plt.figure(figsize=(6, 4))
plt.semilogy(R[:, 0], R[:, 1], "o-", label="raw monomials (mV)")
plt.semilogy(R[:, 0], R[:, 2], "s-", label="monomials, Jacobi-rescaled")
plt.semilogy(R[:, 0], R[:, 3], "^-", label="Chebyshev basis (remedy)")
plt.axvline(9, color="gray", ls=":", lw=1)
plt.text(9.1, 1e30, "NIST degree", color="gray")
plt.xlabel("polynomial degree d")
plt.ylabel(r"$\kappa(H)$")
plt.title("D2: $\\kappa$ grows with d and survives diagonal rescaling")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "d2_kappa_vs_degree.png", dpi=150)

# collinearity of the top two scaled monomial columns at d = 9
V9 = design(9, "raw")
a, b = V9[:, 8], V9[:, 9]
cosang = a @ b / np.linalg.norm(a) / np.linalg.norm(b)
print(f"\ncos(angle) between E^8 and E^9 columns: {cosang:.6f}  (angle = {np.degrees(np.arccos(min(cosang, 1))):.3f} deg)")

# ---------------- D1: spectrum at the NIST degree d = 9 ----------------
d = 9
plt.figure(figsize=(6, 4))
for label, V in [("raw monomials", design(d, "raw")),
                 ("Jacobi-scaled monomials", design(d, "raw") / np.linalg.norm(design(d, "raw"), axis=0)),
                 ("Chebyshev (Jacobi-scaled)", design(d, "cheb") / np.linalg.norm(design(d, "cheb"), axis=0))]:
    w = np.linalg.svd(V, compute_uv=False) ** 2 / m  # eigenvalues of H = V^T V / m
    plt.semilogy(range(1, d + 2), w, "o-", label=f"{label}, $\\kappa$={w[0] / w[-1]:.1e}")
plt.xlabel("eigenvalue index")
plt.ylabel(r"$\lambda_i(H)$")
plt.title("D1: Hessian spectrum, d = 9")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG / "d1_spectrum_d9.png", dpi=150)

# ---------------- D3/D4: iterations to tolerance vs degree ----------------
CAP = 100_000
print(f"\nD3/D4  iterations to reach (f-f*)/(f0-f*) <= 1e-8 (cap {CAP:,})")
print(f"{'d':>3} {'GD scaled mono':>15} {'CG scaled mono':>15} {'GD Chebyshev':>13} {'CG Chebyshev':>13}")
iter_rows = []
for d in range(1, 10):
    out = []
    for basis, method in [("raw", gd), ("raw", cg), ("cheb", gd), ("cheb", cg)]:
        H, g, xs = problem(d, basis, True)
        k, _ = method(H, g, np.zeros(d + 1), xs, cap=CAP)
        out.append(k)
    iter_rows.append((d, *out))
    fmt = lambda k: f">{CAP:,}" if k is None else f"{k:,}"
    print(f"{d:>3} {fmt(out[0]):>15} {fmt(out[1]):>15} {fmt(out[2]):>13} {fmt(out[3]):>13}")

# convergence curves + wall clock at d = 5 (monomial GD still finishes within cap for small d)
for d in (5, 9):
    plt.figure(figsize=(6, 4))
    for label, basis, method in [("GD, monomials (Jacobi-scaled)", "raw", gd),
                                 ("CG, monomials (Jacobi-scaled)", "raw", cg),
                                 ("GD, Chebyshev", "cheb", gd)]:
        H, g, xs = problem(d, basis, True)
        t0 = time.perf_counter()
        k, hist = method(H, g, np.zeros(d + 1), xs, cap=CAP)
        dt = time.perf_counter() - t0
        print(f"d={d} {label:32s} iters={k if k else '>' + str(CAP)}  time={dt * 1e3:.1f} ms  final gap={hist[-1]:.2e}")
        plt.semilogy(np.arange(len(hist)), hist, label=label)
    plt.axhline(1e-8, color="k", ls=":", lw=1)
    plt.xscale("symlog", linthresh=10)
    plt.xlim(left=0)
    plt.xlabel("iteration k")
    plt.ylabel(r"$(f(c_k)-f^*)/(f(c_0)-f^*)$")
    plt.title(f"D3/D4: convergence, degree d = {d}")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / f"d3d4_convergence_d{d}.png", dpi=150)

# ---------------- D3: 2-variable picture (d = 1, straight-line calibration) ----------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (title, basis, scaled) in zip(axes, [("raw monomials [1, E]", "raw", False),
                                             ("Jacobi-scaled [1, E]", "raw", True),
                                             ("Chebyshev [T0, T1]", "cheb", True)]):
    H, g, xs = problem(1, basis, scaled)
    L = np.linalg.eigvalsh(H)[-1]
    x = xs + np.array([1.0, -1.0]) * (np.abs(xs).max() + 1) * 0.8  # same relative offset start
    path = [x.copy()]
    for _ in range(40):
        x = x - (H @ x - g) / L
        path.append(x.copy())
    path = np.array(path)
    r = np.abs(path - xs).max() * 1.2
    u, v = np.meshgrid(np.linspace(-r, r, 200), np.linspace(-r, r, 200))
    Z = 0.5 * (H[0, 0] * u ** 2 + 2 * H[0, 1] * u * v + H[1, 1] * v ** 2)
    ax.contour(u + xs[0], v + xs[1], Z, levels=np.geomspace(Z[Z > 0].min() * 10, Z.max(), 15), cmap="viridis")
    ax.plot(path[:, 0], path[:, 1], "r.-", ms=4, lw=1)
    ax.plot(*xs, "k*", ms=12)
    ax.set_title(f"{title}\n$\\kappa$={kappa(H):.1f}, 40 GD steps")
    ax.set_xlabel("$c_0$")
    ax.set_ylabel("$c_1$")
plt.tight_layout()
plt.savefig(FIG / "d3_paths_d1.png", dpi=150)

# ---------------- Why it matters: coefficient accuracy via normal equations ----------------
for basis in ("raw", "cheb"):
    V = design(9, basis)
    c_ne = np.linalg.solve(V.T @ V, V.T @ y)
    c_qr = np.linalg.lstsq(V, y, rcond=None)[0]
    print(f"d=9 {basis:4s}: normal-equation vs QR fit, max |T diff| = {np.abs(V @ (c_ne - c_qr)).max():.2e} degC, "
          f"fit RMS = {np.sqrt(np.mean((V @ c_qr - y) ** 2)):.4f} degC")
print(f"\nfigures written to {FIG}")
