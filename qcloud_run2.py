#!/usr/bin/env python
"""qcloud_run2.py — driver: trial -> full VSCR cloud experiment on a real
OriginQ chip (default WK_C180) via qcloud_vscr_new.py (pyqpanda3 interface).

Stages (each stage stops at the first successful variant):
  trial : 16 circuits (psi=0, p=0.05, 1 frame) x 2000 shots
          (a) batch    (b) --no-batch    (c) --no-batch --late-measure
  full  : 128 circuits (psi=0,+, p=0.02/0.08, 2 frames each) x 16 branches
          (a) batch 4000 shots  (b) no-batch 4000  (c) batch 2000
          (+ --late-measure on every attempt if trial needed it)
Outputs: qcloud_run2.log (nohup), raw_trial_*.json, qcloud_raw_full.json,
qcloud_results_<tag>.json, figures/fig_qcloud_hardware_<tag>.png.

Launch:  cd /home/yqc/github/QEC
         (nohup ./qenv/bin/python -u qcloud_run2.py > qcloud_run2.log 2>&1 &)
"""
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = os.path.join(ROOT, 'qenv', 'bin', 'python')
SCRIPT = os.path.join(ROOT, 'qcloud_vscr_new.py')
with open(os.path.join(ROOT, '.originq_token')) as f:
    TOK = f.read().strip()
CHIP = os.environ.get('QEC_CHIP', 'WK_C180')


def log(msg):
    print(f'[{datetime.now():%m-%d %H:%M:%S}] {msg}', flush=True)


def run(name, extra):
    cmd = [PY, '-u', SCRIPT, '--mode', 'cloud', '--chip-id', CHIP,
           '--token', TOK] + extra
    shown = ' '.join(a if a != TOK else 'TOKEN' for a in cmd[3:])
    log(f'RUN {name}: {shown}')
    t0 = time.time()
    try:
        rc = subprocess.call(cmd, cwd=ROOT)
    except Exception as exc:                       # noqa: BLE001
        log(f'{name}: exception {exc}')
        return False
    log(f'{name}: exit={rc} after {time.time() - t0:.0f}s')
    return rc == 0


def main():
    log(f'driver start; chip={CHIP}')
    trial_common = ['--shots', '2000', '--p-values', '0.05',
                    '--frames', '1', '--states', '0']
    batch_ok = late_ok = False
    if run('trial-batch', trial_common + ['--dump-raw', 'raw_trial_batch.json',
                                          '--chunk', '32', '--tag', 'trial_batch']):
        batch_ok = True
    elif run('trial-nobatch', trial_common + ['--no-batch', '--tag', 'trial_nb',
                                              '--dump-raw', 'raw_trial_nb.txt']):
        batch_ok = False
    elif run('trial-late', trial_common + ['--no-batch', '--late-measure',
                                           '--tag', 'trial_late',
                                           '--dump-raw', 'raw_trial_late.txt']):
        batch_ok, late_ok = False, True
    else:
        log('TRIAL FAILED on all paths; aborting (check token / quota / chip).')
        sys.exit(1)
    log(f'trial OK (batch={batch_ok}, late_measure={late_ok}); '
        f'FULL experiment starts in 30 s')
    time.sleep(30)

    full_common = ['--p-values', '0.02,0.08', '--frames', '2', '--states', '0,+',
                   '--dump-raw', 'qcloud_raw_full.json']
    if late_ok:
        full_common = full_common + ['--late-measure']
    if batch_ok:
        attempts = [
            ('full-batch-4000', full_common + ['--shots', '4000', '--chunk', '32',
                                               '--tag', 'full_wk4000']),
            ('full-nobatch-4000', full_common + ['--shots', '4000', '--no-batch',
                                                 '--tag', 'full_wk4000nb']),
            ('full-batch-2000', full_common + ['--shots', '2000', '--chunk', '32',
                                               '--tag', 'full_wk2000']),
        ]
    else:
        attempts = [
            ('full-nobatch-4000', full_common + ['--shots', '4000', '--no-batch',
                                                 '--tag', 'full_wk4000nb']),
            ('full-nobatch-2000', full_common + ['--shots', '2000', '--no-batch',
                                                 '--tag', 'full_wk2000nb']),
        ]
    for name, extra in attempts:
        if run(name, extra):
            log(f'FULL EXPERIMENT COMPLETED via {name}')
            sys.exit(0)
    log('FULL EXPERIMENT FAILED on all attempts')
    sys.exit(1)


if __name__ == '__main__':
    main()