#!/usr/bin/env python
"""Minimal end-to-end probe: ONE complete data point (psi=0, p=0.02, frame 1)
= 16 branch circuits x 100 shots = 1600 shots total, late-measure protocol.

Purpose:
  1. test whether any real-chip quota is available after recharge
     (if zero, submission is rejected with no cost);
  2. validate the FULL pipeline end-to-end on real data:
     submit -> detect_layout -> aggregate -> F_hw vs F_ideal.
"""
import sys, os, json
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import qcloud_vscr_new as q

q.setup()
phi = q.load_angles('depolarizing')
q.PHI_GLOBAL = phi
plan = q.build_plan(['0'], [0.02], 1)
gates, meta = q.plan_progs(plan, phi, late_measure=True)
print(f'[probe] {len(gates)} circuits x 100 shots', flush=True)
with open(os.path.join(ROOT, '.originq_token')) as f:
    tok = f.read().strip()
res = q.submit_cloud(gates, tok, chip_id='WK_C180', shots=100, batch=True,
                     dump_raw=os.path.join(ROOT, 'raw_probe_100.json'),
                     chunk=16, job_timeout=3600, use_options=True)
print('[probe] results received', flush=True)
anc, rev = q.detect_layout(meta, res)
agg = q.aggregate(plan, meta, res, 100, anc, rev)
for key, entries in sorted(agg.items()):
    for (fr, F_hw, err, F_ideal, sel_frac) in entries:
        print(f'  psi={key[0]} p={key[1]} frame={fr}: F_hw={F_hw:.3f}+-{err:.3f} '
              f'F_ideal={F_ideal:.3f} sel_frac={sel_frac:.3f}')
print('[probe] PIPELINE OK')
