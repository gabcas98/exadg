#!/usr/bin/env python3
"""offline test of the coarse correction P = I + Z E^{-1} Z^T M (I - S), E = Z^T M S Z, with Z the
invariant subspace of the k eigenvalues of S farthest from one (Table tab:coarse_offline).
usage: coarse_space_offline.py case   (run from the repository root; reads the dense S and M written
by SpectrumDiagnosticsFile = output/spd_<case>/S with SpectrumDiagnosticsInterval = 5: S_000003.txt
at step 20, S_000000.txt at step 5)
columns: exact modes of S; modes of step 5 with E of step 20; correction of step 5 (frozen)"""
import sys, numpy as np
def load(f):
    A = np.loadtxt(f); n = A.shape[1]; return A[:n], A[n:]
def gmres_its(A, b, M, tol=1e-5, maxit=300):
    # GMRES in the M inner product, x0 = 0, iterations until ||b - A x||_M <= tol ||b||_M
    nrm = lambda v: np.sqrt(v @ M @ v)
    beta = nrm(b); V = [b / beta]; H = np.zeros((maxit + 1, maxit))
    for k in range(maxit):
        w = A @ V[k]
        for i in range(k + 1):
            H[i, k] = V[i] @ M @ w; w = w - H[i, k] * V[i]
        H[k + 1, k] = nrm(w)
        e = np.zeros(k + 2); e[0] = beta
        y, *_ = np.linalg.lstsq(H[:k + 2, :k + 1], e, rcond=None)
        res = np.linalg.norm(H[:k + 2, :k + 1] @ y - e)
        if res <= tol * beta or H[k + 1, k] < 1e-300: return k + 1
        V.append(w / H[k + 1, k])
    return maxit
def cgnr_its(A, b, M, tol=1e-5, maxit=2000):
    # CG on the normal equations A* A x = A* b with A* = M^{-1} A^T M (M-adjoint), stop on ||b - A x||_M
    Minv = np.linalg.inv(M); As = lambda v: Minv @ (A.T @ (M @ v)); nrm = lambda v: np.sqrt(v @ M @ v)
    x = np.zeros_like(b); r = b.copy(); z = As(r); p = z.copy(); zz = z @ M @ z; nb = nrm(b)
    for k in range(maxit):
        q = A @ p; alpha = zz / (q @ M @ q); x += alpha * p; r -= alpha * q
        if nrm(r) <= tol * nb: return k + 1
        z = As(r); zz_new = z @ M @ z; p = z + (zz_new / zz) * p; zz = zz_new
    return maxit
def kappa(A, M):
    w, V = np.linalg.eigh(M); Mh = V @ np.diag(np.sqrt(w)) @ V.T; Mih = V @ np.diag(1 / np.sqrt(w)) @ V.T
    s = np.linalg.svd(Mh @ A @ Mih, compute_uv=False); return s[0] / s[-1]
def modes(S, k):
    if k == 0: return np.zeros((S.shape[0], 0))
    lam, V = np.linalg.eig(S); idx = np.argsort(-np.abs(lam - 1))[:k]
    Z = V[:, idx]
    # real basis of the (possibly complex) invariant subspace
    Z = np.real(np.hstack([Z.real, Z.imag])); q, r = np.linalg.qr(Z); keep = np.abs(np.diag(r)) > 1e-10 * np.abs(r[0, 0])
    return q[:, keep][:, :max(k, 0)] if k > 0 else np.zeros((S.shape[0], 0))
def coarse_P(S, M, Z):
    n = S.shape[0]
    if Z.shape[1] == 0: return np.eye(n)
    E = Z.T @ M @ S @ Z; Ei = np.linalg.inv(E)
    return np.eye(n) - Z @ Ei @ Z.T @ M @ S + Z @ Ei @ Z.T @ M
def main():
    case = sys.argv[1]; d = f"output/spd_{case}"
    S, M = load(f"{d}/S_000003.txt"); S_old, _ = load(f"{d}/S_000000.txt")
    rng = np.random.default_rng(1); B = rng.standard_normal((S.shape[0], 3))
    print(f"{case}: n = {S.shape[0]}")
    print(f"{'k':>3s} | {'exact Z: gmres cgnr kappa':>26s} | {'old Z, new E: gmres cgnr':>25s} | {'frozen P: gmres cgnr':>21s}")
    for k in [0, 1, 2, 4, 8, 12, 16, 24]:
        row = []
        for Z, Sref in [(modes(S, k), S), (modes(S_old, k), S), (modes(S_old, k), S_old)]:
            P = coarse_P(Sref, M, Z); A = S @ P
            g = np.mean([gmres_its(A, B[:, j], M) for j in range(3)]); c = np.mean([cgnr_its(A, B[:, j], M) for j in range(3)])
            row.append((g, c, kappa(A, M)))
        print(f"{k:3d} | {row[0][0]:7.1f} {row[0][1]:7.1f} {row[0][2]:8.1f} | {row[1][0]:9.1f} {row[1][1]:9.1f}      | {row[2][0]:8.1f} {row[2][1]:8.1f}")
    
if __name__ == '__main__':
    main()