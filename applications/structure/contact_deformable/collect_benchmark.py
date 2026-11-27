#!/usr/bin/env python3
"""
Collects the results of run_benchmark.sh into a Markdown table.

usage: collect_benchmark.py <output directory of run_benchmark.sh>
"""

import glob
import os
import re
import sys


def parse(path):
    text = open(path).read()
    r = {"name": os.path.basename(path)[:-4]}

    def first(pattern, cast=float, default=None):
        m = re.search(pattern, text, re.M)
        return cast(m.group(1)) if m else default

    def op(name):
        # "  <name>   <calls>   <time> s" of the cost tables
        m = re.search(r"^\s+" + re.escape(name) + r"\s+(\d+)\s+([0-9.e+-]+) s", text, re.M)
        return (int(m.group(1)), float(m.group(2))) if m else (0, 0.0)

    r["dofs"] = first(r"Degrees of freedom\s+=\s+(\d+)", int)
    r["newton"] = sum(int(x) for x in re.findall(r"^\s+Nonlinear iterations:\s+(\d+)", text, re.M))
    r["krylov"] = sum(int(x) for x in re.findall(r"^\s+Linear iterations \(tot\):\s+(\d+)", text, re.M))
    r["steps"] = len(re.findall(r"^\s+Nonlinear iterations:\s+(\d+)", text, re.M))
    r["failed_steps"] = len(re.findall(r"Could not solve non-linear problem", text))
    r["solve"] = first(r"^\s+Solve\s+([0-9.e+-]+) s", float)
    r["total"] = first(r"^Elasticity\s+([0-9.e+-]+) s", float)
    r["residual"] = op("residual evaluations")
    r["linearization"] = op("linearizations")
    r["vmult"] = op("Jacobian-vector products")
    r["build"] = op("list build")
    r["check"] = op("validity check")
    r["queries"] = op("queries (search + projections)")
    r["projections"] = op("projections")[0]
    r["assembly"] = op("residual assembly")
    r["lin"] = op("linearization (reused search)")
    r["reused"] = op("reused")[0]
    r["vmult_contact"] = op("Jacobian-vector products (contact)")
    r["candidates"] = first(r"mean candidates per Verlet query\s+=\s+([0-9.e+-]+)", float, 0.0)
    r["proj_per_query"] = first(r"mean projections per query\s+=\s+([0-9.e+-]+)", float, 0.0)
    r["ratio"] = first(r"max 2 D_max / skin \(valid lists\)\s+=\s+([0-9.e+-]+)", float, 0.0)
    m = re.search(r"slave points / master faces / vertices = (\d+) / (\d+) / (\d+)", text)
    r["surface"] = tuple(int(x) for x in m.groups()) if m else None
    r["cutoff"] = first(r"ties / cutoff hits\s+=\s+\d+ / (\d+)", int, 0)
    r["ties"] = first(r"ties / cutoff hits\s+=\s+(\d+) /", int, 0)
    forces = re.findall(r"Contact force on slave:\s+\S+\s+(\S+)\s*$", text, re.M)
    r["reaction"] = float(forces[-1]) if forces else None
    return r


def main():
    directory = sys.argv[1]
    runs = {}
    for path in sorted(glob.glob(os.path.join(directory, "*_L*_*.log"))):
        r = parse(path)
        if r["solve"] is None:
            print("incomplete run:", path, file=sys.stderr)
            continue
        runs[r["name"]] = r

    print("| case | DoFs | slave pts / master faces | search | Newton (steps, failed) | Krylov | "
          "solve [s] | residual: n, t [s] | vmult: n, t [s] | queries t [s] | projections / query | "
          "builds, t [s] | checks, t [s] | contact vmult t [s] | lin. reused | rest of solve [s] | reaction F_y |")
    print("|" + "---|" * 17)
    for name, r in runs.items():
        surfaces, level, search = name.split("_")
        print(f"| {surfaces} {level} | {r['dofs']} | {r['surface'][0]} / {r['surface'][1]} | {search} | "
              f"{r['newton']} ({r['steps']}, {r['failed_steps']}) | {r['krylov']} | {r['solve']:.3g} | "
              f"{r['residual'][0]}, {r['residual'][1]:.3g} | {r['vmult'][0]}, {r['vmult'][1]:.3g} | "
              f"{r['queries'][1]:.3g} | {r['proj_per_query']:.1f} | "
              f"{r['build'][0]}, {r['build'][1]:.2g} | {r['check'][0]}, {r['check'][1]:.2g} | "
              f"{r['vmult_contact'][1]:.3g} | {r['reused']}/{r['lin'][0]} | "
              f"{r['solve'] - r['residual'][1] - r['linearization'][1] - r['vmult'][1]:.3g} | {r['reaction']:.10e} |")

    # Exhaustive vs. Verlet: identical paths and the time attributable to the search
    print()
    print("| case | identical Newton/Krylov, reaction file bitwise | search time E -> V [s] | search share of residual E -> V | "
          "list build + checks [s] | solve E / V [s] |")
    print("|---|---|---|---|---|---|")
    for name, r in runs.items():
        if not name.endswith("_Exhaustive"):
            continue
        v = runs.get(name.replace("_Exhaustive", "_Verlet"))
        if v is None:
            continue
        def reaction_file(run):
            path = os.path.join(directory, "output", "contact_deformable", run + "_contact_deformable.txt")
            return open(path).read() if os.path.exists(path) else None
        bitwise = reaction_file(name) is not None and reaction_file(name) == reaction_file(name.replace("_Exhaustive", "_Verlet"))
        same = bitwise and (r["newton"] == v["newton"] and r["krylov"] == v["krylov"] and r["reaction"] == v["reaction"]
                and r["failed_steps"] == v["failed_steps"])
        search_e = r["queries"][1]
        search_v = v["queries"][1] + v["build"][1] + v["check"][1]
        print(f"| {name[:-11]} | {'yes' if same else 'NO'} | {search_e:.3g} -> {search_v:.3g} | "
              f"{100 * search_e / r['residual'][1]:.1f}% -> {100 * search_v / v['residual'][1]:.1f}% | "
              f"{v['build'][1] + v['check'][1]:.2g} | {r['solve']:.3g} / {v['solve']:.3g} |")


if __name__ == "__main__":
    main()
