#!/usr/bin/env python3
"""
Compare the contact output of the setup "Hertz" with the Hertz solution.

Usage: hertz_comparison.py <dim> <E> <nu> <R> <output_prefix>

with output_prefix = OutputDirectory + OutputName + "_contact", e.g.
output/contact_rigid_obstacle/hertz_contact. The load P is taken from the computed contact force,
accounting for the symmetry (half domain in 2d, quarter domain in 3d), and the Hertz contact
radius a and maximum pressure p0 are computed for this load:

  2d (cylinder, plane strain, P per unit length): a = sqrt(4 P R / (pi E*)),  p0 = 2 P / (pi a)
  3d (sphere):                                     a = (3 P R / (4 E*))^(1/3), p0 = 3 P / (2 pi a^2)

with E* = E / (1 - nu^2), and p(r) = p0 sqrt(1 - r^2 / a^2).
"""

import glob
import math
import sys


def load(filename):
    with open(filename) as f:
        return [[float(v) for v in line.split()] for line in f
                if line.strip() and not line.startswith("#")]


def main():
    dim, E, nu, R, prefix = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(
        sys.argv[4]), sys.argv[5]

    E_star = E / (1.0 - nu**2)

    last = load(prefix + ".txt")[-1]
    force_y = last[2]
    active_area = last[1 + dim]
    max_pressure = last[2 + dim]

    if dim == 2:
        P = 2.0 * abs(force_y)
        a = math.sqrt(4.0 * P * R / (math.pi * E_star))
        p0 = 2.0 * P / (math.pi * a)
        a_num = active_area  # half-width of the computed half domain
    else:
        P = 4.0 * abs(force_y)
        a = (3.0 * P * R / (4.0 * E_star))**(1.0 / 3.0)
        p0 = 3.0 * P / (2.0 * math.pi * a**2)
        a_num = math.sqrt(4.0 * active_area / math.pi)  # quarter disk of area pi a^2 / 4

    profile_file = sorted(glob.glob(prefix + "_profile_*.txt"))[-1]
    errors = []
    r_center, p_center = float("inf"), 0.0
    for row in load(profile_file):
        r = abs(row[0]) if dim == 2 else math.hypot(row[0], row[2])
        if r < r_center:
            r_center, p_center = r, row[-1]
        if r < 2.0 * a:
            p_hertz = p0 * math.sqrt(1.0 - (r / a)**2) if r < a else 0.0
            errors.append(row[-1] - p_hertz)

    # pressure at the quadrature point closest to the center compared to p(r_center)
    p_center_hertz = p0 * math.sqrt(1.0 - (r_center / a)**2)

    # discrete L2 / max error of the pressure at the quadrature points with r < 2a, relative to p0
    err_l2 = math.sqrt(sum(e**2 for e in errors) / len(errors)) / p0
    err_max = max(abs(e) for e in errors) / p0

    print(f"load P                 = {P:.6e}")
    print(f"contact radius a       : computed {a_num:.6e}  Hertz {a:.6e}  "
          f"rel. error {(a_num - a) / a:+.3e}")
    print(f"center pressure        : computed {p_center:.6e}  Hertz {p_center_hertz:.6e}  "
          f"rel. error {(p_center - p_center_hertz) / p_center_hertz:+.3e}  (r = {r_center:.2e})")
    print(f"max. pressure          : computed {max_pressure:.6e}  Hertz p0 {p0:.6e}  "
          f"ratio {max_pressure / p0:.3f}")
    print(f"pressure profile error : L2 {err_l2:.3e}  max {err_max:.3e}  (relative to p0, r < 2a)")


if __name__ == "__main__":
    main()
