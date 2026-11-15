#!/usr/bin/env python3
# Summary of short benchmark runs: python3 output/bench.py run1 run2 ...
# fK/step: fluid Krylov iterations per time step (state solves + linearized solves of S, S*),
# the dominant, timing-noise-free cost measure
import re, sys, os
print('%-18s %5s %8s %6s %5s %6s %5s %6s %7s %8s %8s %16s' % ('run','steps','wall[s]','s/step','it','states','S','S*','Krylov','fK/state','fK/step','tip dx(end)'))
for r in sys.argv[1:]:
    f = 'output/%s/run.log' % r
    if not os.path.exists(f): print(r,'missing'); continue
    txt = open(f).read()
    T = re.findall(r'monitor point: t = (\S+), d = (\S+) ', txt)
    try: w = open('output/%s/wall_time.txt' % r).read().split()[-2]
    except Exception: w = 'run'
    g = lambda k: (re.search(k + r'\s+(\S+)', txt) or [None, '-'])[1]
    it = g('Optimisation iterations'); st = g('State evaluations'); S = g(r'Tangent solves \(S\)'); Sa = g(r'Adjoint solves \(S\*\)'); kr = g('Krylov iterations')
    if it == '-': it = g('Partitioned iterations'); st = it
    cs = g(r'\n  Coupled system'); lk = g('Linearized fluid Krylov iterations')
    try:
        fk_state = float(cs); fk = float(st) * fk_state + (float(lk) if lk != '-' else 0.0)
    except ValueError: fk_state = fk = float('nan')
    crash = 'CRASH' if 'An error occurred' in txt else ''
    n = len(T)
    try: sps = float(w) / n
    except Exception: sps = float('nan')
    print('%-18s %5d %8s %6.2f %5s %6s %5s %6s %7s %8.1f %8.1f %16s %s' % (r, n, w, sps, it, st, S, Sa, kr, fk_state, fk, T[-1][1].rstrip(',') if T else '-', crash))
