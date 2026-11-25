#!/usr/bin/env python3
"""
Compare the contact output of the setup "Cattaneo" (2d) with the Cattaneo-Mindlin solution.

Usage: cattaneo_comparison.py <E> <nu> <R> <mu> <output_prefix>

The normal load P and the tangential load Q (per unit length) are taken from the computed contact
force. Hertz: a = sqrt(4 P R / (pi E*)), p0 = 2 P / (pi a), E* = E / (1 - nu^2).
Cattaneo-Mindlin (Q < mu P): stick zone |x| < c with c = a sqrt(1 - Q / (mu P)) and tangential
traction q(x) = mu p0 sqrt(1 - x^2/a^2) - mu p0 (c/a) sqrt(1 - x^2/c^2) (second term for |x| < c).

Since friction is active also during the indentation, the computed traction contains an odd part
(frictional indentation), which is not part of the Cattaneo-Mindlin solution; the even part is
compared separately.
"""

import glob
import math
import sys


def load(filename):
    with open(filename) as f:
        return [[float(v) for v in line.split()] for line in f
                if line.strip() and not line.startswith("#")]


def compare(E, nu, R, mu, prefix):
    E_star = E / (1.0 - nu**2)

    # columns: time force_0 force_1 active_area max_pressure min_gap max_tangential stick_area
    last = load(prefix + ".txt")[-1]
    Q = abs(last[1])
    P = abs(last[2])
    a_num = 0.5 * last[3]
    c_num = 0.5 * last[7]

    a = math.sqrt(4.0 * P * R / (math.pi * E_star))
    p0 = 2.0 * P / (math.pi * a)
    c = a * math.sqrt(max(1.0 - Q / (mu * P), 0.0))

    def q_analytical(x):
        q = mu * p0 * math.sqrt(max(1.0 - (x / a)**2, 0.0))
        if abs(x) < c:
            q -= mu * p0 * (c / a) * math.sqrt(1.0 - (x / c)**2)
        return q

    # columns: X_0 X_1 gap pressure t_T_0 t_T_1
    profile_file = sorted(glob.glob(prefix + "_profile_*.txt"))[-1]
    profile = sorted((row[0], row[3], row[4]) for row in load(profile_file))

    # Decomposition into the even part (compared with Cattaneo-Mindlin) and the odd part of the
    # tangential traction (residual traction of the frictional indentation, which is not part of
    # the Cattaneo-Mindlin solution); the mesh is symmetric with respect to x = 0.
    traction = {round(x, 12): t for x, p, t in profile}
    xs = sorted(traction)
    even, odd = [], []
    for x in xs:
        if x <= 0.0:
            continue
        x_mirror = min(xs, key=lambda y: abs(y + x))
        even.append((x, 0.5 * (traction[x] + traction[x_mirror])))
        odd.append((x, 0.5 * (traction[x] - traction[x_mirror])))

    def rms(values):
        return math.sqrt(sum(v**2 for v in values) / len(values)) / (mu * p0) if values else 0.0

    err_even = rms([t - q_analytical(x) for x, t in even if x < 0.8 * a])
    err_even_stick = rms([t - q_analytical(x) for x, t in even if x < 0.8 * c])
    rms_odd = rms([t for x, t in odd if x < 0.8 * a])
    errors = [abs(t) - q_analytical(x) for x, p, t in profile if abs(x) < 0.8 * a]
    err_l2 = math.sqrt(sum(e**2 for e in errors) / len(errors)) / (mu * p0)

    return dict(P=P, Q=Q, ratio=Q / (mu * P), a=a, a_num=a_num, c=c, c_num=c_num, p0=p0,
                err_l2=err_l2, err_even=err_even, err_even_stick=err_even_stick, rms_odd=rms_odd,
                even=even, profile=profile, q_analytical=q_analytical)


def main():
    E, nu, R, mu = (float(v) for v in sys.argv[1:5])
    res = compare(E, nu, R, mu, sys.argv[5])
    print(f"P = {res['P']:.6e}  Q = {res['Q']:.6e}  Q/(mu P) = {res['ratio']:.4f}")
    print(f"contact half-width a : computed {res['a_num']:.6e}  Hertz {res['a']:.6e}")
    print(f"stick half-width c   : computed {res['c_num']:.6e}  Cattaneo {res['c']:.6e}  "
          f"rel. error {(res['c_num'] - res['c']) / res['c']:+.2e}")
    print(f"tangential traction  : L2 error {res['err_l2']:.3e} (relative to mu p0, |x| < 0.8a)")
    print(f"  even part          : L2 error {res['err_even']:.3e} (|x| < 0.8a), "
          f"{res['err_even_stick']:.3e} (stick zone |x| < 0.8c)")
    print(f"  odd part           : rms {res['rms_odd']:.3e} (frictional indentation, |x| < 0.8a)")


if __name__ == "__main__":
    main()
