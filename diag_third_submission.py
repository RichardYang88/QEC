#!/usr/bin/env python3
"""Root-cause diagnosis of the REJECTED 2026-10-24T14:18 suppl submission.

Run:  ./qenv/bin/python diag_third_submission.py

WHY THIS FILE EXISTS.  hw_mixedness.json['suppl'][2] used to record "Root cause
not established -- either transient device degradation between 12:09 and 14:18, or
a compilation/mapping fault in this submission".  That hedge is false in both
directions: the cause WAS established on 2026-10-25 by a 0.6252 s cal9screen, and
the compilation branch is not merely unevidenced but excluded.  The claim now
appears in hw_mixedness.py's SUPPL_MIN_OUTCOME_ENTROPY_RATIO comment, in the
rejection note, and in the manuscript, so it needs to be reproducible rather than
asserted.  Every number those three places quote is recomputed here from committed
artifacts alone and ASSERTED, so this script exits nonzero if any of them drifts.

It buys nothing: no submission, no shots, no network.  Inputs are
  qcloud_raw_suppl_WK_C180_20261024T{120330,120934,141813}Z.json[.meta.json]
  qcloud_raw_cal9screen_WK_C180_20261024T120512Z.json[.meta.json]
  qcloud_raw_cal9screen_WK_C180_20261025T022633Z.json[.meta.json]
  hw_mixedness.json
plus hw_mixedness's own circuit builder, ideal simulator and key conversions --
so the ideal references are the same code path the submission used, not a copy.

WHAT IT ESTABLISHES, and the one thing it does not:
  * compilation/mapping fault EXCLUDED -- the rejected submission's recorded spec
    is field-for-field identical to the valid same-block submission two hours
    earlier, and its ideal matches the 2026-09 archive to 5.44e-15;
  * apparatus fault CONFIRMED -- the block measured NOT USABLE 12 h later with 4
    of 9 readout channels dead (2 of them data), and the rejected counts show ~5
    of 9 measured bits reporting a constant, zero shots on every ideal outcome,
    TV distance 1.0000, and entropy that FELL while the answer went wrong;
  * the near-identical z is the statistic's floor for an empty expected-syndrome
    bin, not evidence that four different circuits behaved alike;
  * NOT established: WHICH physical channels were dead in that job.  mapping=True
    leaves logical->physical routing to the server per circuit and no dump records
    the assignment, and the constant-reporting bit set moves between circuits
    (empty intersection).  Sections below assert that too, so the manuscript
    cannot upgrade "consistent with" into an identification.
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import hw_mixedness as H                              # noqa: E402
import qcloud_vscr_new as q                           # noqa: E402

NQ = q.NQ                                             # 9
LABELS = ['C', 'D', 'A', 'B']                         # SUPPL_GRID order = dump order
SEP_BLOCK = [147, 155, 157, 164, 165, 166, 174, 175, 176]
RANK_BLOCK = [52, 61, 62, 70, 71, 79, 80, 88, 89]
DUMPS = {'12:03 valid (Sep, 2/4)': '120330',
         '12:09 valid (ranked, 4/4)': '120934',
         '14:18 REJECTED (Sep)': '141813'}
DET_LO, DET_HI = 0.02, 0.98        # the thresholds the manuscript quotes

ok = True


def chk(cond, msg):
    global ok
    ok = ok and bool(cond)
    print(('  OK   ' if cond else '  FAIL ') + msg)


def suppl_path(tag):
    return 'qcloud_raw_suppl_WK_C180_20261024T%sZ.json' % tag


def screen_path(tag):
    return 'qcloud_raw_cal9screen_WK_C180_%sZ.json' % tag


def vendor_maps(g):
    """canonical key -> vendor key, and logical qubit -> vendor bit position."""
    c2v = {}
    for canon in H.exact_probs(g, n=NQ):
        c2v[canon] = q.sim_key_to_vendor_key(
            H.canonical_to_native_key(canon, g, nq=NQ), nq=NQ)
    l2p = {}
    for j in range(NQ):
        v = c2v['0' * j + '1' + '0' * (NQ - 1 - j)]
        ones = [p for p, ch in enumerate(v) if ch == '1']
        assert len(ones) == 1, (j, v)
        l2p[j] = ones[0]
    return c2v, l2p


def ideal_vendor(g):
    c2v, l2p = vendor_maps(g)
    iv = {}
    for canon, p in H.exact_probs(g, n=NQ).items():
        iv[c2v[canon]] = iv.get(c2v[canon], 0.0) + p
    return iv, l2p


def marg1(counts, l2p):
    tot = float(sum(counts.values()))
    return [sum(v for k, v in counts.items() if k[l2p[j]] == '1') / tot
            for j in range(NQ)]


def det_bits(m):
    return sorted(j for j, x in enumerate(m) if x <= DET_LO or x >= DET_HI)


def gcd_all(vals):
    g = 0
    for v in vals:
        g = math.gcd(g, int(v))
    return g


def norm_entropy(counts):
    tot = float(sum(counts.values()))
    h = -sum((v / tot) * math.log2(v / tot) for v in counts.values() if v)
    return h / math.log2(min(tot, 512.0)), h


print('=== diag_third_submission: root cause of the rejected 2026-10-24T14:18 '
      'suppl submission ===')
q.setup()
PHI = H.load_angle_set('cold')
GATES, META = H.build_suppl(PHI)
assert [m['label'] for m in META] == LABELS
print()

# ---------------------------------------------------------------------------
print('--- 1. CLIENT-SIDE SPEC IDENTITY: the compilation/mapping branch -------')
_mA = json.load(open(suppl_path('120330') + '.meta.json'))
_mB = json.load(open(suppl_path('141813') + '.meta.json'))
chk(_mA['block'] == _mB['block'] == SEP_BLOCK,
    'the rejected submission and the valid 12:03 one ran on the SAME block %s'
    % (SEP_BLOCK,))
chk(_mA['instruction_counts'] == _mB['instruction_counts'] == [203, 204, 203, 204],
    'identical recorded instruction counts %s' % (_mB['instruction_counts'],))
chk([m['n_cnot'] for m in _mA['meta']] == [m['n_cnot'] for m in _mB['meta']]
    == [82] * 4,
    'identical entangling content: 82 two-qubit gates in every circuit')
chk(_mA['mapping'] == _mB['mapping'] and _mA['shots'] == _mB['shots'] == 20000,
    'identical mapping flag (%s) and shot count (%d)'
    % (_mB['mapping'], _mB['shots']))
chk([m['frame'] for m in _mA['meta']] == [m['frame'] for m in _mB['meta']]
    and [m['measure'] for m in _mA['meta']] == [m['measure'] for m in _mB['meta']],
    'identical Pauli frames and measurement depths')
_ide = max(m['ideal_matches_2026_09_to'] for m in _mB['meta'])
chk(_ide < 1e-14,
    "the rejected submission's ideal references match the 2026-09 archive to "
    '%.2e, so what was SENT was the archived circuit' % _ide)
chk(all(m['matches_2026_09_inventory'] for m in _mB['meta']),
    'and it says so itself: matches_2026_09_inventory is true on all four')
_meas = [sorted((t[1], t[2]) for t in g if t[0] == 'MEAS') for g in GATES]
chk(all(_m == _meas[0] for _m in _meas),
    'the MEAS map is identical in all four circuits (mid- and late-measure '
    'alike), so no per-circuit remapping happens on the client side')
_l2p0 = vendor_maps(GATES[0])[1]
chk([_l2p0[j] for j in range(NQ)] == [8, 7, 6, 5, 4, 3, 2, 1, 0],
    'logical qubit j lands at vendor key position 8-j, i.e. the layout '
    '[a3 a2 a1 a0][d4 d3 d2 d1 d0] that sim_key_to_vendor_key documents')

# ---------------------------------------------------------------------------
print()
print('--- 2. THE REJECTED COUNTS vs THE EXACT IDEAL ------------------------')
_rows = {}
for _name, _tag in DUMPS.items():
    _dump = json.load(open(suppl_path(_tag)))
    _rows[_tag] = []
    for _i, _obs in enumerate(_dump):
        _iv, _l2p = ideal_vendor(GATES[_i])
        _tot = float(sum(_obs.values()))
        _kb, _pb = max(_iv.items(), key=lambda kv: kv[1])
        _sup = {k for k, p in _iv.items() if p > 1e-9}
        _tv = 0.5 * sum(abs(_obs.get(k, 0) / _tot - _iv.get(k, 0.0))
                        for k in set(_obs) | set(_iv))
        _er, _hb = norm_entropy(_obs)
        _s0 = max(_obs, key=_obs.get)
        _h2 = sum(v for k, v in _obs.items()
                  if sum(a != b for a, b in zip(k, _s0)) <= 2) / _tot
        _rows[_tag].append(dict(
            label=LABELS[_i], tv=_tv, mode=_kb, mode_p=_pb,
            hits=_obs.get(_kb, 0),
            ov=sum(v for k, v in _obs.items() if k in _sup) / _tot,
            er=_er, bits=_hb, distinct=len(_obs), total=int(_tot),
            gcd=gcd_all(_obs.values()), det=det_bits(marg1(_obs, _l2p)),
            ham2=_h2))
    print('  %s' % _name)
    for _r in _rows[_tag]:
        print('    %s TV=%.4f shots_on_ideal_mode=%-4d overlap=%.4f entropy=%.3f '
              '(%.2f bits) distinct=%-4d gcd=%d n_det=%d %s P(ham<=2)=%.3f'
              % (_r['label'], _r['tv'], _r['hits'], _r['ov'], _r['er'],
                 _r['bits'], _r['distinct'], _r['gcd'], len(_r['det']),
                 _r['det'], _r['ham2']))

_R, _V9, _V3 = _rows['141813'], _rows['120934'], _rows['120330']
chk(all(r['tv'] > 0.9999 for r in _R),
    'rejected: total-variation distance to the exact ideal is 1.0000 on all four '
    'circuits')
chk(all(r['hits'] == 0 for r in _R),
    'rejected: ZERO shots land on the ideal outcome in all four circuits, which '
    'is exactly where its F_s of 0.0000 comes from')
chk(all(r['ov'] < 5e-4 for r in _R) and min(r['ov'] for r in _V9) > 6e-3,
    'rejected: overlap with the p>1e-9 ideal support is %.4f-%.4f against '
    '%.4f-%.4f for the valid complete run'
    % (min(r['ov'] for r in _R), max(r['ov'] for r in _R),
       min(r['ov'] for r in _V9), max(r['ov'] for r in _V9)))
chk([r['hits'] for r in _V3] == [418, 48],
    'the valid same-block run of two hours earlier put 418 and 48 shots on those '
    'same two ideal modes')
chk(all(len(r['det']) == 5 for r in _R),
    'rejected: exactly 5 of the 9 measured bits report a constant in EVERY '
    'circuit at p1<=%.2f or >=%.2f' % (DET_LO, DET_HI))
chk(all(len(r['det']) == 0 for r in _V9 + _V3),
    'valid: not one bit reports a constant in any of the %d valid circuits, so '
    'the determinism is the fault and not a property of these circuits'
    % len(_V9 + _V3))
_sets = [set(r['det']) for r in _R]
chk(not set.intersection(*_sets),
    'rejected: the constant-reporting set MOVES between circuits (intersection '
    'empty: %s), so this dump cannot identify WHICH physical channels died'
    % [sorted(s) for s in _sets])
chk(all(r['gcd'] == 1 for r in _R),
    'rejected: counts are coprime on every circuit, excluding a rescaled or '
    'subsampled histogram')
chk(max(r['bits'] for r in _R) < min(r['bits'] for r in _V9 + _V3),
    'rejected: entropy FELL to %.2f-%.2f bits from the %.2f-%.2f bits of the '
    'valid runs while the answer went wrong -- the opposite of a device getting '
    'noisier, and what excludes re-routing (adds depth, raises entropy) and a bit '
    'permutation (preserves entropy)'
    % (min(r['bits'] for r in _R), max(r['bits'] for r in _R),
       min(r['bits'] for r in _V9 + _V3), max(r['bits'] for r in _V9 + _V3)))
chk(all(0.27 <= r['er'] <= 0.30 for r in _R)
    and all(0.78 <= r['er'] <= 0.83 for r in _V9),
    'the normalised ratios reproduce the guard\'s calibration: rejected '
    '%.3f-%.3f, valid %.3f-%.3f, against a cut of %.2f sitting in the empty gap'
    % (min(r['er'] for r in _R), max(r['er'] for r in _R),
       min(r['er'] for r in _V9), max(r['er'] for r in _V9),
       H.SUPPL_MIN_OUTCOME_ENTROPY_RATIO))
chk(all(r['ham2'] > 0.97 for r in _R) and all(r['ham2'] < 0.52 for r in _V9),
    'rejected: %.3f-%.3f of all shots sit within Hamming distance 2 of the mode, '
    'against %.3f-%.3f for the valid complete run'
    % (min(r['ham2'] for r in _R), max(r['ham2'] for r in _R),
       min(r['ham2'] for r in _V9), max(r['ham2'] for r in _V9)))

# ---------------------------------------------------------------------------
print()
print('--- 3. THE TWO READOUT SCREENS: cause and control ---------------------')
_anc, _rev = H.layout_from_meas(H.science_meas_layer())
_screens = {}
for _tag, _blk in (('20261024T120512', RANK_BLOCK),
                   ('20261025T022633', SEP_BLOCK)):
    _base = screen_path(_tag)
    _mrec = json.load(open(_base + '.meta.json'))
    assert _mrec['block'] == _blk, (_tag, _mrec['block'], _blk)
    _got = [(r, m) for r, m in zip(json.load(open(_base)), _mrec['meta']) if r]
    _c9 = H._analyze_cal9(_got, _anc, _rev)
    _screens[_tag] = _c9
    _s = _c9['screen']
    print('  %s block=%s' % (_tag, _blk))
    print('    block_usable=%s broken=%d (data %d, ancilla %d) mean_fid=%.4f'
          % (_s['block_usable'], _s['n_broken_data'] + _s['n_broken_ancilla'],
             _s['n_broken_data'], _s['n_broken_ancilla'],
             _c9['mean_readout_fidelity']))
    for _lab in (['data%d' % i for i in range(q.NQ_DATA)]
                 + ['anc%d' % i for i in range(q.NQ_ANC)]):
        _v = _c9['per_qubit'][_lab]
        print('      %-6s q%-2d e01=%.3f e10=%.3f contrast=%+.3f fid=%.4f %s'
              % (_lab, _v['logical_qubit'], _v['e01'], _v['e10'], _v['contrast'],
                 _v['readout_fidelity'], 'BROKEN' if _v['broken'] else ''))

_pre, _post = _screens['20261024T120512'], _screens['20261025T022633']
chk(_pre['screen']['block_usable'] is True
    and _pre['screen']['n_broken_data'] + _pre['screen']['n_broken_ancilla'] == 0,
    'the 12:05:12Z screen CLEARED the ranked block (0 of 9 broken, mean readout '
    'fidelity %.4f) four minutes before the valid 12:09 submission used it -- '
    'recomputed here because no verdict for it had ever been stored'
    % _pre['mean_readout_fidelity'])
chk(_post['screen']['block_usable'] is False
    and _post['screen']['n_broken_data'] == 2
    and _post['screen']['n_broken_ancilla'] == 2,
    'the 2026-10-25T02:26:33Z screen returns the September block NOT USABLE with '
    '4 of 9 channels dead, 2 of them DATA channels')
_pb = sorted(v['readout_fidelity'] for v in _post['per_qubit'].values()
             if v['broken'])
_ph = sorted(v['readout_fidelity'] for v in _post['per_qubit'].values()
             if not v['broken'])
chk(len(_pb) == 4 and len(_ph) == 5 and max(_pb) < min(_ph),
    'the dead channels (%.3f-%.3f) are separated from the passing ones '
    '(%.3f-%.3f) by a gap, so "dead" reports a bimodal measurement rather than a '
    'threshold chosen to fit' % (_pb[0], _pb[-1], _ph[0], _ph[-1]))
_dead = sorted(v['logical_qubit'] for v in _post['per_qubit'].values()
               if v['broken'])
chk(_dead == [2, 3, 5, 8],
    'the dead channels are logical qubits %s = data2, data3, anc0, anc3' % _dead)
chk(_post['screen']['primary_estimator_valid'] is False,
    'and the screen declares the primary estimator INVALID on that block, so no '
    'number of additional shots could have rescued the third submission')


# ---------------------------------------------------------------------------
print()
print("--- 4. THE z STATISTIC'S FLOOR, AND THE RECORD ITSELF -----------------")
_n = 20000
_exp = _n / 16.0
_sd = math.sqrt(_exp * (1 - 1.0 / 16))
_floor = (0 - _exp) / _sd
print('  n=%d  exp=%.1f  sd=%.4f  ->  z(empty bin) = %.4f' % (_n, _exp, _sd, _floor))
chk(abs(_floor + 36.5148) < 1e-4,
    'an empty expected-syndrome bin gives z = %.4f, i.e. -36.51 to two decimals'
    % _floor)
_mx = json.load(open('hw_mixedness.json'))
_rec = [e for e in _mx['suppl'] if e.get('rejected')][0]
_zR = [_rec['per_circuit'][L]['syndrome_z_vs_code_blind'] for L in LABELS]
chk(all(_floor - 1e-9 <= z <= _floor + 0.30 for z in _zR),
    'all four rejected z values %s sit at that floor, so they are identical '
    'because the correct syndrome was unattainable and NOT because the circuits '
    'are alike' % ['%.2f' % z for z in _zR])
chk(_rec['source_dump'].endswith('20261024T141813Z.json')
    and _rec['block'] == SEP_BLOCK and _rec['degenerate'] is True,
    'the record under diagnosis is the 14:18 September-block submission')
_rk = _mx['cal9_screen_ranked_20261024']
chk(_rk['block_usable'] is True and _rk['billing'] is None
    and _rk['block'] == RANK_BLOCK,
    "the control screen's verdict is now stored in the artifact and marked as "
    'recomputed, so it cost nothing and cannot be mistaken for a new purchase')
chk('Root cause not established' not in _rec['rejection_note']
    and 'APPARATUS FAULT' in _rec['rejection_note'],
    'the rejection note states the measured cause instead of the old hedge')
chk('device or compilation failure' not in json.dumps(_rec)
    and 'APPARATUS FAULT' in _rec['rejection_reason']
    and _rec['rejection_reason'] == _rec['comparison']['reason'],
    'and so does the generated rejection reason, identically in both its copies')

print()
print('=== DIAGNOSIS: apparatus fault on block %s CONFIRMED; client-side '
      'compilation/mapping fault EXCLUDED; the identity of the dead physical '
      'channels in that job remains UNDETERMINED and is reported as such ==='
      % (SEP_BLOCK,))
print('=== %s ===' % ('ALL CHECKS PASSED' if ok else 'FAILURES PRESENT'))
raise SystemExit(0 if ok else 1)

