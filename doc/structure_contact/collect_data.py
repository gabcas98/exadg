#!/usr/bin/env python3
"""
Collects the results of the contact studies into data files for the report.

Usage: collect_data.py <run_directory>

where <run_directory> is the build directory of the application structure/contact_rigid_obstacle in
which the studies A_*, B_*, C_*, D_* were run (see run_studies.sh). The data files are written to
the subdirectory data/ next to this script.
"""

import bisect
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.join(HERE, "..", "..", "applications", "structure", "contact_rigid_obstacle"))
from hertz_comparison import compare, load  # noqa: E402

E, NU, R = 1.0, 0.3, 1.0


def log_info(run_dir, name):
    text = open(os.path.join(run_dir, name + ".log")).read()
    dofs = int(re.search(r"Degrees of freedom\s*=\s*(\d+)", text).group(1))
    newton = [int(n) for n in re.findall(r"Nonlinear iterations:\s*(\d+)", text)]
    linear = [float(n) for n in re.findall(r"Linear iterations \(avg\):\s*([\d.]+)", text)]
    return dofs, newton, linear


def prefix(run_dir, name):
    return os.path.join(run_dir, "output", "contact_rigid_obstacle", name + "_contact")


def interpolate(profile, r):
    rs = [p[0] for p in profile]
    i = bisect.bisect_left(rs, r)
    if i <= 0:
        return profile[0][1]
    if i >= len(rs):
        return profile[-1][1]
    (r0, p0), (r1, p1) = profile[i - 1], profile[i]
    return p0 + (p1 - p0) * (r - r0) / (r1 - r0) if r1 > r0 else p0


def profile_difference(c, ref, r_max=2.0):
    """L2 difference of the pressure profiles (r < r_max a_ref), relative to p0 of the reference."""
    diff = [p - interpolate(ref["profile"], r) for r, p in c["profile"] if r < r_max * ref["a"]]
    return math.sqrt(sum(d**2 for d in diff) / len(diff)) / ref["p0"]


def hertz_interior_error(c, r_max=0.8):
    """L2 error with respect to the Hertz profile for r < r_max a, relative to p0."""
    diff = [p - c["p0"] * math.sqrt(1.0 - (r / c["a"])**2) for r, p in c["profile"]
            if r < r_max * c["a"]]
    return math.sqrt(sum(d**2 for d in diff) / len(diff)) / c["p0"]


def write(filename, header, rows):
    with open(os.path.join(HERE, "data", filename), "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(" ".join(f"{v:.6e}" if isinstance(v, float) else str(v) for v in row) + "\n")


def main():
    run_dir = sys.argv[1]
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)

    # A: indentation study
    rows = []
    for d, L in [("3e-2", 7), ("1e-2", 7), ("3e-3", 8), ("1e-3", 9), ("3e-4", 9), ("1e-4", 10)]:
        c = compare(2, E, NU, R, prefix(run_dir, "A_d" + d))
        dofs, newton, _ = log_info(run_dir, "A_d" + d)
        rows.append((float(d), c["a"] / R, c["a"] / (0.5 / 2**L),
                     abs(c["p_center"] - c["p_center_hertz"]) / c["p_center_hertz"],
                     abs(c["a_num"] - c["a"]) / c["a"], c["err_l2"], hertz_interior_error(c),
                     c["P"], dofs, sum(newton)))
    write("indentation.dat",
          "delta a_over_R a_over_h err_p0 err_a err_l2 err_l2_int P dofs newton", rows)

    # B: mesh study (reference: finest mesh)
    levels = [5, 6, 7, 8, 9, 10, 11]
    results = {L: compare(2, E, NU, R, prefix(run_dir, f"B_L{L}")) for L in levels}
    ref = results[levels[-1]]
    rows = []
    for L in levels[:-1]:
        c = results[L]
        dofs, newton, _ = log_info(run_dir, f"B_L{L}")
        h = 0.5 / 2**L
        rows.append((L, h, h / ref["a"], dofs, abs(c["P"] - ref["P"]) / ref["P"],
                     profile_difference(c, ref), profile_difference(c, ref, 0.8),
                     abs(c["a_num"] - ref["a_num"]) / ref["a_num"], c["err_l2"],
                     hertz_interior_error(c), sum(newton)))
    write("mesh.dat",
          "L h h_over_a dofs err_P err_profile err_profile_int err_a err_hertz_l2 "
          "err_hertz_int newton", rows)

    # pressure profile of the finest mesh and a coarse mesh, normalized with the Hertz solution
    for L in [6, levels[-1]]:
        c = results[L]
        write(f"profile_L{L}.dat", "r_over_a p_over_p0",
              [(r / c["a"], p / c["p0"]) for r, p in c["profile"] if r < 1.5 * c["a"]])

    # C: penalty study (reference: largest penalty)
    penalties = ["1e2", "1e3", "1e4", "1e5", "1e6", "1e7"]
    results = {e: compare(2, E, NU, R, prefix(run_dir, "C_e" + e)) for e in penalties}
    ref = results[penalties[-1]]
    rows = []
    for e in penalties:
        c = results[e]
        _, newton, linear = log_info(run_dir, "C_e" + e)
        err_P = abs(c["P"] - ref["P"]) / ref["P"]
        err_profile = profile_difference(c, ref)
        err_profile_int = profile_difference(c, ref, 0.8)
        rows.append((float(e), -c["min_gap"] / 1e-3, err_P, err_profile, err_profile_int,
                     hertz_interior_error(c), sum(newton), max(linear)))
    write("penalty.dat",
          "eps penetration_over_delta err_P err_profile err_profile_int err_hertz_int newton "
          "max_linear", rows)

    # D: patch tests (exact: force p L (W), area L (W), pressure p, gap -p / eps)
    eps = 1000.0
    rows = []
    for name, dim, p in [("D_2d_q2_amg", 2, 0.1), ("D_2d_q3_mg", 2, 0.1), ("D_2d_q2_mb", 2, 0.1),
                         ("D_2d_q2_large", 2, 0.15), ("D_3d_q2_mg", 3, 0.1),
                         ("D_3d_q3_large", 3, 0.15)]:
        last = load(prefix(run_dir, name) + ".txt")[-1]
        area = 2.0
        exact = [p * area, area, p, -p / eps]
        computed = [last[2], last[1 + dim], last[2 + dim], last[3 + dim]]
        err = max(abs(c - x) / abs(x) for c, x in zip(computed, exact))
        dofs, newton, _ = log_info(run_dir, name)
        rows.append((name, dim, p, dofs, err, sum(newton)))
    write("patch.dat", "name dim p dofs max_rel_err newton", rows)

    # E: Nitsche mesh study (reference: finest mesh)
    results = {L: compare(2, E, NU, R, prefix(run_dir, f"E_L{L}")) for L in levels}
    ref = results[levels[-1]]
    rows = []
    for L in levels[:-1]:
        c = results[L]
        dofs, newton, _ = log_info(run_dir, f"E_L{L}")
        h = 0.5 / 2**L
        rows.append((L, h, h / ref["a"], dofs, abs(c["P"] - ref["P"]) / ref["P"],
                     profile_difference(c, ref), profile_difference(c, ref, 0.8),
                     abs(c["a_num"] - ref["a_num"]) / ref["a_num"], c["err_l2"],
                     hertz_interior_error(c), sum(newton)))
    write("mesh_nitsche.dat",
          "L h h_over_a dofs err_P err_profile err_profile_int err_a err_hertz_l2 "
          "err_hertz_int newton", rows)

    # F: Nitsche parameter study (reference: gamma_0 E = 10), compared to the penalty solution
    # with the largest penalty on the same mesh
    penalty_ref = compare(2, E, NU, R, prefix(run_dir, "C_e1e7"))
    nitsche_ref = compare(2, E, NU, R, prefix(run_dir, "F_g10"))
    rows = []
    for g in ["0.3", "1", "3", "10", "30", "100", "300"]:
        name = "F_g" + g
        if "FAILED" in open(os.path.join(run_dir, name + ".log")).read() or \
                "Could not solve quasi static problem" in open(
                    os.path.join(run_dir, name + ".log")).read():
            continue
        c = compare(2, E, NU, R, prefix(run_dir, name))
        _, newton, linear = log_info(run_dir, name)
        rows.append((float(g), abs(c["P"] - nitsche_ref["P"]) / nitsche_ref["P"],
                     profile_difference(c, nitsche_ref, 0.8),
                     abs(c["P"] - penalty_ref["P"]) / penalty_ref["P"],
                     profile_difference(c, penalty_ref, 0.8), hertz_interior_error(c),
                     -c["min_gap"] / 1e-3, sum(newton), max(linear)))
    write("gamma.dat",
          "gamma0 err_P err_profile_int err_P_penalty err_profile_int_penalty err_hertz_int "
          "penetration_over_delta newton max_linear", rows)

    # G: Nitsche patch tests (exact: force p L (W), area L (W), pressure p, gap 0)
    rows = []
    for name, dim, p in [("G_2d_q2_amg", 2, 0.1), ("G_2d_q3_mg", 2, 0.1),
                         ("G_3d_q3_large", 3, 0.15)]:
        last = load(prefix(run_dir, name) + ".txt")[-1]
        exact = [p * 2.0, 2.0, p]
        computed = [last[2], last[1 + dim], last[2 + dim]]
        err = max(abs(c - x) / abs(x) for c, x in zip(computed, exact))
        dofs, newton, _ = log_info(run_dir, name)
        rows.append((name, dim, p, dofs, err, abs(last[3 + dim]), sum(newton)))
    write("patch_nitsche.dat", "name dim p dofs max_rel_err abs_gap newton", rows)

    # 3d Hertz (delta = 1e-3): average linear iterations per load step (L = 6). The runs "before"
    # were computed before the fix that passes the time (load factor) to the multigrid levels.
    series = [("h3d_d1e-3_L6", "penalty_before"), ("N_h3d_L6", "nitsche_before"),
              ("fix_h3d_d1e-3_L6", "penalty_k5"), ("fix_N_h3d_L6", "nitsche_k5"),
              ("z_pen_e1e5_pj8", "penalty_k8"), ("z_nit_g10_pj8", "nitsche_k8"),
              ("amg_Nitsche", "nitsche_amg")]
    columns = [log_info(run_dir, name)[2] for name, _ in series]
    n = min(len(c) for c in columns)
    write("mg3d.dat", "step " + " ".join(label for _, label in series),
          [(i + 1, *[c[i] for c in columns]) for i in range(n)])

    # 3d Hertz: multigrid performance for different contact stiffnesses and smoother degrees
    def solve_time(name):
        text = open(os.path.join(run_dir, name + ".log")).read()
        return sum(float(t) for t in re.findall(r"Wall time \[s\]:\s*([\d.e+-]+)", text))

    rows = []
    for name, level, method, param, smoother in [
        ("h3d_d1e-3_L6", 6, "penalty", "1e5", "J5 (before fix)"),
        ("m_pen_e1e4", 6, "penalty", "1e4", "J5"),
        ("fix_h3d_d1e-3_L6", 6, "penalty", "1e5", "J5"),
        ("z_pen_e1e5_pj8", 6, "penalty", "1e5", "J8"),
        ("z_pen_e1e6_pj5", 6, "penalty", "1e6", "J5"),
        ("z_pen_e1e6_pj8", 6, "penalty", "1e6", "J8"),
        ("N_h3d_L6", 6, "Nitsche", "10", "J5 (before fix)"),
        ("fix_N_h3d_L6", 6, "Nitsche", "10", "J5"),
        ("z_nit_g10_pj8", 6, "Nitsche", "10", "J8"),
        ("m_nit_g100", 6, "Nitsche", "100", "J5"),
        ("z_nit_g100_pj8", 6, "Nitsche", "100", "J8"),
        ("amg_Nitsche", 6, "Nitsche", "10", "AMG"),
        ("y_pen_pj5", 7, "penalty", "1e5", "J5"),
        ("y_pen_pj8", 7, "penalty", "1e5", "J8"),
        ("y_pen_pj12", 7, "penalty", "1e5", "J12"),
        ("y_nit_pj5", 7, "Nitsche", "10", "J5"),
        ("y_nit_pj8", 7, "Nitsche", "10", "J8"),
        ("y_nit_pj12", 7, "Nitsche", "10", "J12"),
    ]:
        dofs, newton, linear = log_info(run_dir, name)
        text = open(os.path.join(run_dir, name + ".log")).read()
        total = sum(int(n) for n in re.findall(r"Linear iterations \(tot\):\s*(\d+)", text))
        rows.append((level, dofs, method, param, '"' + smoother + '"', sum(newton), total,
                     max(linear), solve_time(name)))
    with open(os.path.join(HERE, "data", "mg_table.csv"), "w") as f:
        f.write("L,dofs,method,param,smoother,newton,linear_total,linear_max,time\n")
        for r in rows:
            f.write(",".join(str(v).strip('"') if not isinstance(v, float) else f"{v:.1f}"
                             for v in r) + "\n")

if __name__ == "__main__":
    main()
