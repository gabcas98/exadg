"""interface_forces.py t0 t1 run1 [run2 ...]: interface forces (fluid on its boundary, x and y) averaged over
[t0, t1] and the tip displacement at the last time <= t1, from output/<run>/run.log"""
import re, sys, os
t0, t1 = float(sys.argv[1]), float(sys.argv[2])
num = r'([-0-9.eE+]+)'
pat = re.compile(r'FSI interface forces: t = ' + num + r', F_pressure = ' + num + ' ' + num +
                 r', F_viscous = ' + num + ' ' + num + r'(?:, F_penalty = ' + num + ' ' + num + ')?')
mon = re.compile(r'monitor point: t = ' + num + r', d = ' + num + ' ' + num)
print(f"{'run':24s} {'Fp_x':>11s} {'Fv_x':>11s} {'Fpen_x':>11s} {'Fv+pen_x':>11s} {'Fp_y':>11s} {'Fv_y':>11s} {'Fpen_y':>11s} {'d_x(t1)':>11s} {'d_y(t1)':>11s} n")
for run in sys.argv[3:]:
    acc = [0.0]*6; n = 0; dx = dy = float('nan')
    for line in open(os.path.join('output', run, 'run.log')):
        m = pat.search(line)
        if m:
            t = float(m.group(1))
            if t0 <= t <= t1:
                v = [float(m.group(i)) if m.group(i) else 0.0 for i in range(2, 8)]
                acc = [a + b for a, b in zip(acc, v)]; n += 1
        m = mon.search(line)
        if m and float(m.group(1)) <= t1 + 1e-12:
            dx, dy = float(m.group(2)), float(m.group(3))
    if n == 0:
        print(f"{run:24s} no data"); continue
    a = [x / n for x in acc]
    print(f"{run:24s} {a[0]:11.4e} {a[2]:11.4e} {a[4]:11.4e} {a[2]+a[4]:11.4e} {a[1]:11.4e} {a[3]:11.4e} {a[5]:11.4e} {dx:11.4e} {dy:11.4e} {n}")
