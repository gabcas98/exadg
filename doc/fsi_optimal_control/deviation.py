#!/usr/bin/env python3
"""max deviation of the tip displacement of runs from a reference run (linear interpolation in
time onto the reference times), relative to max |d| of the reference over the common interval.
usage: deviation.py ref run1 [run2 ...]   (reads output/<run>/run.log from the cwd)"""
import re, sys, bisect
def series(run):
    t, x, y = [], [], []
    for line in open(f'output/{run}/run.log'):
        m = re.search(r'monitor point: t = (\S+), d = (\S+) (\S+), v', line)
        if m:
            t.append(float(m.group(1))); x.append(float(m.group(2))); y.append(float(m.group(3)))
    return t, x, y
def interp(t, f, s):
    i = bisect.bisect_left(t, s)
    if i == 0: return f[0]
    if i >= len(t): return f[-1]
    a = (s - t[i-1]) / (t[i] - t[i-1]); return (1-a)*f[i-1] + a*f[i]
ref = sys.argv[1]; tr, xr, yr = series(ref)
print(f"{'run':24s} {'t_end':>7s} {'dev d_x':>10s} {'dev d_y':>10s}")
for run in sys.argv[2:]:
    t, x, y = series(run)
    te = min(t[-1], tr[-1]); idx = [i for i in range(len(tr)) if tr[i] >= t[0] and tr[i] <= te]
    sx = max(abs(xr[i]) for i in idx); sy = max(abs(yr[i]) for i in idx)
    dx = max(abs(interp(t, x, tr[i]) - xr[i]) for i in idx) / sx
    dy = max(abs(interp(t, y, tr[i]) - yr[i]) for i in idx) / sy
    print(f"{run:24s} {te:7.3f} {dx:10.2e} {dy:10.2e}")
