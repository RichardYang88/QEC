#!/usr/bin/env python
"""
qcloud_verify_options.py — LOW-COST (single batch task, ~1600 shots total)
validation of the real-chip submission options BEFORE spending quota on the
full experiment.

Submits 4 circuits with QCloudOptions(amend=True, mapping=True,
optimization=True, specified_block=best_qubit_blocks(9)):

   A  mid-measure,  frame IIIII, branch s=0   -> expect key '000000000' dominant
   B  mid-measure,  frame IIXII, branch s=12   -> expect key '110000000'
                                                 (or reversed '000000011')
   C  late-measure variant of A
   D  late-measure variant of B

Circuit B/D discriminate the bitstring orientation (ancilla-first vs reversed)
because their expected key is NOT a palindrome.

Environment overrides:
   VERIFY_CHIP   (default WK_C180)
   VERIFY_SHOTS  (default 400 per circuit)
   VERIFY_BLOCK  (optional comma-separated physical block; default auto-query)
"""
import sys, os, json

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import qcloud_vscr_new as q

CHIP = os.environ.get('VERIFY_CHIP', 'WK_C180')
SHOTS = int(os.environ.get('VERIFY_SHOTS', '400'))
BLOCK = [int(x) for x in os.environ.get('VERIFY_BLOCK', '').replace(' ', '').split(',') if x] or None


def syndrome_of(frame):
    for si, bits in enumerate(q.m.SYND_BITS):
        if q.m.SYND_TABLE[bits] == frame:
            return si
    return 0


def main():
    q.setup()
    phi = q.load_angles('depolarizing')
    q.PHI_GLOBAL = phi

    cases = []
    for late in (False, True):
        for frame in ('IIIII', 'IIXII'):
            s = syndrome_of(frame)
            gates = q.build_branch_circuit(q.ENC5, '0', frame, s, phi,
                                           late_measure=late)
            cases.append(dict(kind='late' if late else 'mid', frame=frame,
                              s=s, gates=gates,
                              F_ideal_frame=q.simulator_fidelity_frame('0', frame, phi)))

    gate_lists = [c['gates'] for c in cases]
    with open(os.path.join(ROOT, '.originq_token')) as f:
        token = f.read().strip()

    print(f'[verify] chip={CHIP} shots/circuit={SHOTS} block={BLOCK or "auto"}',
          flush=True)
    results = q.submit_cloud(gate_lists, token, chip_id=CHIP, shots=SHOTS,
                             batch=True,
                             dump_raw=os.path.join(ROOT, 'raw_verify_options.json'),
                             chunk=8, job_timeout=3600,
                             use_options=True, block=BLOCK)

    meta = [('0', 0.0, c['frame'], c['s']) for c in cases]
    print('\n===== per-circuit diagnostics =====', flush=True)
    verdict = []
    for c, counts in zip(cases, results):
        tot = sum(counts.values()) or 1
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        anc = format(c['s'], '04b')
        exp_fwd = anc + '00000'
        exp_rev = exp_fwd[::-1]
        pf = counts.get(exp_fwd, 0) / tot
        pr = counts.get(exp_rev, 0) / tot
        verdict.append(max(pf, pr))
        print(f"[{c['kind']}] frame={c['frame']} s={c['s']}({anc}) total={tot} "
              f"F_ideal_frame={c['F_ideal_frame']:.3f}")
        print(f'   top3 = {top}')
        print(f"   P(expected '{exp_fwd}') = {pf:.3f}   P(reversed '{exp_rev}') = {pr:.3f}")
        for sl, rev, name in [(slice(0, 4), False, 'anc@[0:4]'),
                              (slice(0, 4), True, 'anc@[0:4]rev'),
                              (slice(5, 9), False, 'anc@[5:9]'),
                              (slice(5, 9), True, 'anc@[5:9]rev')]:
            n_sel, n_hit = q.counts_to_branch_stats(counts, c['s'], sl, rev)
            fs = n_hit / n_sel if n_sel else float('nan')
            print(f'   {name:12s}: sel={n_sel:4d} hit={n_hit:4d} F_s={fs:.3f}')

    print('\n===== detect_layout over all 4 circuits =====', flush=True)
    anc_slice, rev = q.detect_layout(meta, results)
    print(f'chosen: anc_slice={anc_slice} rev={rev}')

    print('\nmax expected-key probability per circuit:',
          ['%.3f' % v for v in verdict], flush=True)
    if max(verdict) > 0.15:
        print('VERDICT: options-based submission looks CORRECT '
              '(expected key dominant) -> safe to run full experiment')
    else:
        print('VERDICT: still garbage-like -> do NOT run the full experiment; '
              'inspect raw_verify_options.json')


if __name__ == '__main__':
    main()