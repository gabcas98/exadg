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


def compare(dim, E, nu, R, prefix):
    """Returns the computed and Hertz quantities for the output files with the given prefix."""
    E_star = E / (1.0 - nu**2)

    last = load(prefix + ".txt")[-1]
    force_y = last[2]
    active_area = last[1 + dim]
    max_pressure = last[2 + dim]
    min_gap = last[3 + dim]

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
    profile = []
    errors = []
    r_center, p_center = float("inf"), 0.0
    for row in load(profile_file):
        r = abs(row[0]) if dim == 2 else math.hypot(row[0], row[2])
        # columns: X_0, ..., X_{dim-1}, gap, pressure (, tangential traction)
        pressure = row[dim + 1]
        profile.append((r, pressure))
        if r < r_center:
            r_center, p_center = r, pressure
        if r < 2.0 * a:
            p_hertz = p0 * math.sqrt(1.0 - (r / a)**2) if r < a else 0.0
            errors.append(pressure - p_hertz)

    return dict(P=P,
                a=a,
                a_num=a_num,
                p0=p0,
                max_pressure=max_pressure,
                min_gap=min_gap,
                r_center=r_center,
                p_center=p_center,
                p_center_hertz=p0 * math.sqrt(1.0 - (r_center / a)**2),
                # discrete L2 / max error of the pressure at the quadrature points with r < 2a,
                # relative to p0
                err_l2=math.sqrt(sum(e**2 for e in errors) / len(errors)) / p0,
                err_max=max(abs(e) for e in errors) / p0,
                profile=sorted(profile))


def main():
    dim, E, nu, R, prefix = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(
        sys.argv[4]), sys.argv[5]

    c = compare(dim, E, nu, R, prefix)

    print(f"load P                 = {c['P']:.6e}")
    print(f"contact radius a       : computed {c['a_num']:.6e}  Hertz {c['a']:.6e}  "
          f"rel. error {(c['a_num'] - c['a']) / c['a']:+.3e}")
    print(f"center pressure        : computed {c['p_center']:.6e}  "
          f"Hertz {c['p_center_hertz']:.6e}  "
          f"rel. error {(c['p_center'] - c['p_center_hertz']) / c['p_center_hertz']:+.3e}  "
          f"(r = {c['r_center']:.2e})")
    print(f"max. pressure          : computed {c['max_pressure']:.6e}  Hertz p0 {c['p0']:.6e}  "
          f"ratio {c['max_pressure'] / c['p0']:.3f}")
    print(f"pressure profile error : L2 {c['err_l2']:.3e}  max {c['err_max']:.3e}  "
          f"(relative to p0, r < 2a)")


if __name__ == "__main__":
    main()
