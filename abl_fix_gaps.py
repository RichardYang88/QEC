"""abl_fix_gaps.py — repair the max_branch_gap column of the E3 SDP results.

Root cause (fixed at source): VSCRWarm registered PHI_DEC WITHOUT cloning,
so the E2 cold-start ablation's `phi_dec.zero_()` zeroed the global decoder
angle table mid-run; the cf_dec tables used for the branch-gap column were
then computed against an identity recovery.  All other outputs (SDP
ceilings, F_dec references, E1/E2/E4/E5, figures) consume m.C_SYNDS or
saved npz files and are unaffected.  This script recomputes the gap column
from the intact decoder angles and updates vscr_paper_abl_results.npz +
paper_numbers.json in place.
"""
import json
import numpy as np
import torch

import ssvr_qec as m
import vscr_paper as vp

PHI = vp.decoder_angles()          # fresh, provably intact
assert torch.allclose(PHI, vp.PHI_DEC), 'module PHI_DEC not intact'
R_dec = m.recovery_unitary_batch(PHI)

POINTS = [('depolarizing', 0.05), ('depolarizing', 0.10),
          ('depolarizing', 0.15), ('amplitude_damping', 0.10),
          ('mixed', 0.10), ('coherent', 0.15), ('coherent', 0.30)]
SHORT = {'depolarizing': 'dep', 'amplitude_damping': 'ad',
         'mixed': 'mixed', 'coherent': 'coh'}

d = np.load('vscr_paper_abl_results.npz', allow_pickle=True)
sdp = d['sdp'].item()
num = json.load(open('paper_numbers.json'))

for noise, p in POINTS:
    key = f'{SHORT[noise]}_{p}'
    cf_dec, w = vp.cf_table(R_dec, noise, p, M=200)
    Fs = sdp[key]['F_s_max']
    gaps = [Fs[s] - cf_dec[s] for s in range(16) if w[s] > 1e-3]
    sdp[key]['max_branch_gap'] = float(max(gaps))
    sdp[key]['cf_dec_branch'] = [float(x) for x in cf_dec]
    sdp[key]['w_branch'] = [float(x) for x in w]
    print(f'{key}: max_branch_gap={sdp[key]["max_branch_gap"]:.3e} '
          f'F_cptp={sdp[key]["F_cptp"]:.6f} F_dec={sdp[key]["F_dec"]:.6f} '
          f'dF={sdp[key]["F_gap_bar"]:+.2e}', flush=True)

num['abl']['sdp'] = sdp

# coherent Pauli-class detail (for the text): which branch deviates
pl = num['abl']['pauli_lookup']['coh_0.15']
gains = pl['gain_over_I']
s_star = int(np.argmax(gains))
p_s = sdp['coh_0.15']['p_s'][s_star]
print(f'coh Pauli search: best deviating branch s={s_star} '
      f'class={pl["best_class"][s_star]} gain={gains[s_star]:.4f} '
      f'weight p_s={p_s:.2e}')
num['abl']['coh_pauli_detail'] = {
    'branch': s_star, 'class': pl['best_class'][s_star],
    'gain': float(gains[s_star]), 'weight': float(p_s),
    'n_deviating': int(sum(1 for b in pl['best_class'] if b != 'I'))}

np.savez('vscr_paper_abl_results.npz', **{k: d[k] for k in d.files
                                          if k != 'sdp'}, sdp=sdp)
json.dump(num, open('paper_numbers.json', 'w'), indent=2)
print('updated vscr_paper_abl_results.npz + paper_numbers.json')
