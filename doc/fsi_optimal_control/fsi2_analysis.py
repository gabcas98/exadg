#!/usr/bin/env python3
"""
Periodic-regime analysis of the FSI2 tip displacement (point A = (0.6, 0.2)).

usage: python3 doc/fsi_optimal_control/fsi2_analysis.py <out_dir> <window> run1 [run2 ...]

<window> is either a duration (e.g. 5: the last 5 seconds of each run) or an absolute time
interval (e.g. 12.5:15).

Reads output/<run>/run.log ("FSI monitor point" lines) and writes to <out_dir>:
  <run>_signal.dat   t ux uy (downsampled, for plots)
  <run>_periods.dat  per period of uy (between upward zero crossings):
                     t_mid  uy_mean  uy_amp  ux_mean  ux_amp  f_uy
  summary.txt        mean +- amplitude [frequency] over the window,
                     compared with the reference values of Turek & Hron (2006), FSI2.
"""
import os
import re
import sys

# Turek & Hron (2006), FSI2, point A: mean +- amplitude [frequency]
REFERENCE = {'ux': (-1.4580e-2, 1.2440e-2, 3.8), 'uy': (1.2300e-3, 8.0600e-2, 2.0)}


def read_series(run):
    t, ux, uy = [], [], []
    for line in open(os.path.join('output', run, 'run.log')):
        m = re.search(r'monitor point: t = ([0-9.eE+-]+), d = (\S+) (\S+), v', line)
        if m:
            t.append(float(m.group(1)))
            ux.append(float(m.group(2)))
            uy.append(float(m.group(3)))
    return t, ux, uy


def upward_crossings(t, y, hysteresis):
    """times of upward zero crossings (linear interpolation); the signal has to drop below
    -hysteresis before the next crossing counts"""
    crossings, armed = [], False
    for i in range(1, len(t)):
        if y[i - 1] < -hysteresis:
            armed = True
        if armed and y[i - 1] < 0.0 <= y[i]:
            s = -y[i - 1] / (y[i] - y[i - 1])
            crossings.append(t[i - 1] + s * (t[i] - t[i - 1]))
            armed = False
    return crossings


def periods(t, ux, uy):
    amp_scale = max(abs(v) for v in uy)
    zc = upward_crossings(t, uy, 1e-3 * amp_scale)
    rows = []
    j = 0
    for a, b in zip(zc[:-1], zc[1:]):
        idx = [i for i in range(len(t)) if a <= t[i] < b]
        if len(idx) < 4 or b - a > 1.0:  # no oscillation yet (start-up)
            continue
        ys = [uy[i] for i in idx]
        xs = [ux[i] for i in idx]
        rows.append((0.5 * (a + b), 0.5 * (max(ys) + min(ys)), 0.5 * (max(ys) - min(ys)),
                     0.5 * (max(xs) + min(xs)), 0.5 * (max(xs) - min(xs)), 1.0 / (b - a)))
    return rows


def local_maxima_frequency(t, y, t0, min_distance):
    peaks = []
    for i in range(1, len(t) - 1):
        if t[i] >= t0 and y[i] > y[i - 1] and y[i] >= y[i + 1]:
            if not peaks or t[i] - peaks[-1] > min_distance:
                peaks.append(t[i])
    if len(peaks) < 2:
        return float('nan')
    return (len(peaks) - 1) / (peaks[-1] - peaks[0])


def main():
    out_dir, window, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    lines = ['%-14s %-34s %-34s' % ('run', 'ux [1e-3]: mean +- amp [f]', 'uy [1e-3]: mean +- amp [f]')]
    r = REFERENCE
    lines.append('%-14s %8.2f +- %6.2f [%4.2f Hz]          %8.2f +- %6.2f [%4.2f Hz]' %
                 ('Turek-Hron', 1e3 * r['ux'][0], 1e3 * r['ux'][1], r['ux'][2],
                  1e3 * r['uy'][0], 1e3 * r['uy'][1], r['uy'][2]))
    for run in runs:
        t, ux, uy = read_series(run)
        # downsampled signal
        step = max(1, len(t) // 4000)
        with open(os.path.join(out_dir, run + '_signal.dat'), 'w') as f:
            f.write('t ux uy\n')
            for i in range(0, len(t), step):
                f.write('%.6e %.6e %.6e\n' % (t[i], ux[i], uy[i]))
        rows = periods(t, ux, uy)
        with open(os.path.join(out_dir, run + '_periods.dat'), 'w') as f:
            f.write('t uymean uyamp uxmean uxamp fuy\n')
            for row in rows:
                f.write(' '.join('%.6e' % v for v in row) + '\n')
        # evaluation window
        if ':' in window:
            t0, t1 = (float(v) for v in window.split(':'))
        else:
            t0, t1 = t[-1] - float(window), t[-1]
        last = [row for row in rows if t0 <= row[0] <= t1]
        if last:
            n = len(last)
            uym = sum(x[1] for x in last) / n
            uya = sum(x[2] for x in last) / n
            uxm = sum(x[3] for x in last) / n
            uxa = sum(x[4] for x in last) / n
            fy = sum(x[5] for x in last) / n
            ti = [i for i in range(len(t)) if t[i] <= t1]
            fx = local_maxima_frequency([t[i] for i in ti], [ux[i] for i in ti], t0,
                                        0.5 / (2.0 * fy))
            lines.append('%-14s %8.2f +- %6.2f [%4.2f Hz]          %8.2f +- %6.2f [%5.3f Hz]   (%d periods, t in [%.1f, %.1f])' %
                         (run, 1e3 * uxm, 1e3 * uxa, fx, 1e3 * uym, 1e3 * uya, fy, n, t0, t1))
        else:
            lines.append('%-14s no complete period in [%.1f, %.1f]' % (run, t0, t1))
    open(os.path.join(out_dir, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
