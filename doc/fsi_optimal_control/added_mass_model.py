#!/usr/bin/env python3
"""
Single-degree-of-freedom model of the converged (monolithic) coupled FSI time step, used in the
section "Consistency of the coupling" of the report.

  structure:  m_s ((1-a_m) a_{n+1} + a_m a_n) + k d_{n+1-a_f} = load at t_{n+1-a_f}
              with the Newmark relations of the generalized-alpha family,
  fluid:      added mass only, f = -m_a D_t(u) with the BDF2 derivative of the interface
              velocity u given to the fluid (u = v_{n+1}, as in ExaDG).

The load at the mid point is either f_{n+1} (end-of-step load, the original behaviour of ExaDG) or
(1 - a_f) f_{n+1} + a_f f_n (consistent mid-point load). The script prints the spectral radius of
the amplification matrix (estimated from the asymptotic growth of the recurrence; values within
about 3e-4 of one are not resolved) against Omega = sqrt(k / (m_s + m_a)) dt for the added-mass
ratios mu = m_a / m_s.

usage: python3 added_mass_model.py   (pure Python, no dependencies)
"""

import math
import random


def integrator_parameters(kind, rho):
    """alpha_m, alpha_f, gamma, beta for the spectral radius rho at infinite frequency"""
    if kind == "newmark":  # ExaDG: first-order Newmark with gamma from rho
        gamma = 0.5 + (1 - rho) / (1 + rho)
        return 0.0, 0.0, gamma, 0.25 * (gamma + 0.5) ** 2
    if kind == "bossak":
        am, af = (rho - 1) / (rho + 1), 0.0
    elif kind == "genalpha":
        am, af = (2 * rho - 1) / (rho + 1), rho / (rho + 1)
    elif kind == "hht":
        am, af = 0.0, (1 - rho) / (1 + rho)
    else:
        raise ValueError(kind)
    return am, af, 0.5 - am + af, 0.25 * (1 - am + af) ** 2


def solve(A, B):
    """A^{-1} B by Gauss-Jordan elimination with partial pivoting"""
    n, m = len(A), len(B[0])
    M = [A[i][:] + B[i][:] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0.0:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [[M[i][n + j] / M[i][i] for j in range(m)] for i in range(n)]


def growth_rate(T, steps=4000, tail=2000):
    random.seed(1)
    x = [random.uniform(-1, 1) for _ in range(len(T))]
    s = 0.0
    for it in range(steps):
        x = [sum(T[i][j] * x[j] for j in range(len(T))) for i in range(len(T))]
        norm = math.sqrt(sum(v * v for v in x))
        if it >= steps - tail:
            s += math.log(norm)
        x = [v / norm for v in x]
    return math.exp(s / tail)


def spectral_radius(Omega, mu, kind, rho, midpoint_load=False, dt=1.0):
    am, af, gamma, beta = integrator_parameters(kind, rho)
    ms, ma = 1.0, mu
    k = Omega ** 2 * (ms + ma) / dt ** 2
    # old state x = (d_n, v_n, a_n, u_n, u_{n-1}, f_n), unknowns y = (d, v, a, u, f) at t_{n+1}
    A = [[0.0] * 5 for _ in range(5)]
    B = [[0.0] * 6 for _ in range(5)]
    # structure equation at the mid point
    A[0][2] = (1 - am) * ms
    B[0][2] = -am * ms
    A[0][0] = (1 - af) * k
    B[0][0] = -af * k
    if midpoint_load:
        A[0][4] = -(1 - af)
        B[0][5] = af
    else:
        A[0][4] = -1.0
    # Newmark relations
    A[1][0], A[1][2] = 1.0, -dt ** 2 * beta
    B[1][0], B[1][1], B[1][2] = 1.0, dt, dt ** 2 * (0.5 - beta)
    A[2][1], A[2][2] = 1.0, -dt * gamma
    B[2][1], B[2][2] = 1.0, dt * (1 - gamma)
    # interface velocity of the fluid: u = v_{n+1}
    A[3][3], A[3][1] = 1.0, -1.0
    # added-mass force with the BDF2 derivative of u
    A[4][4], A[4][3] = 1.0, ma * 3 / (2 * dt)
    B[4][3], B[4][4] = ma * 4 / (2 * dt), -ma / (2 * dt)
    Y = solve(A, B)
    T = [Y[0][:], Y[1][:], Y[2][:], Y[3][:], [0.0] * 6, Y[4][:]]
    T[4][3] = 1.0
    return growth_rate(T)


if __name__ == "__main__":
    omegas = [0.2, 0.45, 0.6, 1.0, 1.5, 2.0, 3.0]
    cases = [("Bossak rho=0.8", "bossak", 0.8, False),
             ("Bossak rho=0", "bossak", 0.0, False),
             ("GenAlpha rho=0.8, end-of-step load", "genalpha", 0.8, False),
             ("GenAlpha rho=0.8, mid-point load", "genalpha", 0.8, True),
             ("HHT rho=0.8, end-of-step load", "hht", 0.8, False),
             ("HHT rho=0.8, mid-point load", "hht", 0.8, True)]
    print(" " * 46 + "Omega " + " ".join(f"{o:7.2f}" for o in omegas))
    for label, kind, rho, mid in cases:
        for mu in [0.1, 0.5, 2.0]:
            r = [spectral_radius(o, mu, kind, rho, mid) for o in omegas]
            print(f"{label:38s} mu={mu:3.1f}  " + " ".join(f"{x:7.4f}" for x in r))
