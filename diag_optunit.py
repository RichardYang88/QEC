#!/usr/bin/env python3
"""
diag_optunit.py — exact optimal UNITARY recovery per syndrome branch.

For branch s reduce the problem to the 2x2 operators A_k = W_s^dag K_k V.  The
Haar-averaged (unnormalised) branch fidelity of a recovery whose action on the
branch is the 2x2 unitary G is

    F_s(G) = [ sum_k |Tr(G A_k)|^2 + Tr(sum_k A_k^dag A_k) ] / 6 ,

whose second term is G-independent, so the optimum over UNITARY recoveries is

    max_{G unitary} J(G) = sum_k |Tr(G A_k)|^2 .

For a single Kraus operator (the coherent channel) this has the closed form
J_max = ||A||_*^2 (nuclear norm squared) attained at the polar factor of A, so
those branches are PROVABLY global; multi-Kraus branches are solved by
multi-start L-BFGS-B over the Lie algebra u(2) plus a Nelder-Mead polish
(`vscr_paper_abl._max_unitary_J`).

This is the information-theoretic optimum INSIDE the VSCR ansatz family, and so
the natural target a trained warm start should reach.  Comparing it with the
SDP CPTP ceiling answers a different and sharper question than "does VSCR beat
the decoder": it says how much of the decoder->optimal gap is reachable by a
UNITARY, syndrome-resolved recovery at all.

Outputs a table to stdout and writes `diag_optunit.json` (and, with --headroom,
`recovery/headroom_table.json`).

Usage: python diag_optunit.py [--quick] [--headroom]
"""
import os, sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
# See the long note in ssvr_qec.py: this host intermittently SIGSEGV/SIGILLs in
# tiny complex128 kernels when a process is free to MIGRATE across its hybrid
# P-/E-core cluster. Importing ssvr_qec pins us to one CPU, which fixes it
# (measured: unpinned core-dumps within ~2e4 calls; pinned runs 4e5 clean).
# OPENBLAS_CORETYPE is for determinism only -- it is NOT the fix, since the
# faults reproduce under every coretype including AVX-only SANDYBRIDGE.
os.environ.setdefault('OPENBLAS_CORETYPE', 'HASWELL')

import json
import numpy as np                                            # noqa: E402
import torch                                                  # noqa: E402
torch.set_num_threads(1)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssvr_qec as m                                          # noqa: E402
import vscr_paper as vp                                       # noqa: E402
import vscr_paper_abl as ab                                   # noqa: E402

POINTS = [('depolarizing', 0.10), ('depolarizing', 0.30),
          ('amplitude_damping', 0.06), ('amplitude_damping', 0.10),
          ('amplitude_damping', 0.30), ('mixed', 0.10),
          ('coherent', 0.06), ('coherent', 0.10),
          ('coherent', 0.15), ('coherent', 0.30)]
QUICK = [('amplitude_damping', 0.06), ('coherent', 0.15)]


def main(argv):
    quick = '--quick' in argv
    points = QUICK if quick else POINTS
    print(f"{'channel':22s} {'p':>5s} {'F_decoder':>12s} {'F_opt-unit':>12s} "
          f"{'gain':>11s} {'F_CPTP':>12s} {'frac of gap':>12s} "
          f"{'max br. gap':>12s}")
    rows = []
    worst_mc = 0.0
    for noise, p in points:
        r = ab.opt_unitary_ceiling(noise, p)
        F_d, F_u = float(r['F_dec']), float(r['F_unit'])
        ceil = ab.sdp_ceiling(noise, p)['F_cptp']
        frac = ((F_u - F_d) / (ceil - F_d)) if (ceil - F_d) > 1e-12 else float('nan')
        cf_dec = np.asarray(r['cf_dec'], dtype=float)
        cf_unit = np.asarray(r['cf_unit'], dtype=float)
        ps = np.asarray(r['p_s'], dtype=float)
        max_branch = float(np.max(cf_unit - cf_dec))
        print(f'{noise:22s} {p:5.2f} {F_d:12.9f} {F_u:12.9f} '
              f'{F_u - F_d:+11.2e} {ceil:12.9f} {frac:12.4f} '
              f'{max_branch:12.2e}', flush=True)
        rows.append({'channel': noise, 'p': float(p),
                     'F_dec': F_d, 'F_unit': F_u,
                     'headroom': F_u - F_d, 'F_cptp': float(ceil),
                     'frac_of_cptp_gap': (None if np.isnan(frac) else float(frac)),
                     'cf_dec': cf_dec.tolist(), 'cf_unit': cf_unit.tolist(),
                     'p_s': ps.tolist()})
        # the analytic decoder branch-sum must match Monte-Carlo
        F_mc = ab._decoder_ref_F(noise, p, n=200)
        worst_mc = max(worst_mc, abs(F_mc - F_d))
        assert abs(F_mc - F_d) < 3e-4, (noise, p, F_mc, F_d)

    print(f'\n[ok] analytic decoder branch-sum matches Monte-Carlo(200) to '
          f'<3e-4 (worst {worst_mc:.2e})')
    # A unitary recovery can never beat the CPTP optimum.
    for row in rows:
        assert row['F_unit'] <= row['F_cptp'] + 1e-9, row
    print('[ok] F_opt-unit <= F_CPTP on every point (weak duality respected)')

    with open('diag_optunit.json', 'w') as f:
        json.dump(rows, f, indent=1)
    print('wrote diag_optunit.json')
    if '--headroom' in argv:
        os.makedirs('recovery', exist_ok=True)
        with open('recovery/headroom_table.json', 'w') as f:
            json.dump(rows, f, indent=1)
        print('wrote recovery/headroom_table.json')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
