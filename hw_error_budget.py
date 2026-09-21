"""
hw_error_budget.py — effective hardware error budget for the WK_C180 run.

Why this file exists
--------------------
An experimental section has to let a reviewer decide whether an observed fidelity
is limited by the METHOD or by the HARDWARE, and that normally means device
figures of merit: gate fidelities, coherence times, readout error.  The OriginQ
cloud channel does expose exactly that metadata -- `ChipInfo.single_qubit_info()`
carries readout fidelity, single-gate fidelity, T1, T2 and frequency per qubit,
`double_qubits_info()` the two-qubit gate fidelity per edge, and
`get_single_gate_timing()`/`get_double_gate_timing()` the instruction durations --
but it sits behind the same authorisation as a job submission, and this account's
key is refused (401 Unauthorized, re-checked 2026-09-21); no snapshot was captured
at run time either.  So no vendor number can be quoted, and this file derives the
strongest budget the data ON FILE supports, labelling what each entry includes.

What is derived, and how
------------------------
1. Gate counts, exactly.  The four branch circuits are rebuilt offline from the v1
   angle snapshot (no cloud, no shots, no QPU time) and counted by section and by
   type.  That is what a reader needs to turn any future vendor calibration into a
   predicted budget -- and it corrects the "~160-gate circuit" the manuscript said.
2. Effective per-ancilla-bit syndrome error rates, model-free.  The ideal syndrome
   of each circuit is deterministic (branch weight >= 0.9999957), so the marginal
   of each measured ancilla bit IS the probability that bit is wrong.  No
   independence assumption and no fit; it lumps pre-measurement physical error
   together with assignment error, which is the honest limit of this data.
3. The independence defect: the measured all-four-correct rate against the product
   of those marginals.  Ratio 1 means independent bit errors, below 1 means
   correlated (common-cause) errors, which is what a decoder prior would need.
4. Model adequacy: the observed syndrome Hamming-weight distribution against the
   symmetric Bin(4, p) at the mean rate, as a total-variation distance.
5. The conditional data-register error 1 - F_s, with the artifact's Wilson interval.
6. The factorisation P(expected bitstring) = P(correct syndrome) * F_s.  This is an
   identity of how the post-selection is defined, not a model, and it is asserted.
7. Algorithm-vs-device attribution: the identical circuit, code and angles put
   through the exact statevector reference give 0.9999957, so the device accounts
   for all but a few parts in 10^6 of the observed loss.
8. The implied mean per-instruction error under an independent-error model, quoted
   as model-dependent, and the mid-vs-late contrast that prices the idle wait.

Run:
    ./qenv/bin/python hw_error_budget.py                     # offline, ~20 s
    ./qenv/bin/python hw_error_budget.py --try-calibration   # also re-attempt the
                                                             # vendor metadata query
Outputs:
    hw_error_budget.json
"""
import argparse
import collections
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

IN_JSON = os.path.join(ROOT, 'hw_feasibility_numbers.json')
OUT_JSON = os.path.join(ROOT, 'hw_error_budget.json')
CASES = [('A', 'mid', 'IIIII', 0), ('B', 'mid', 'IIXII', 12),
         ('C', 'late', 'IIIII', 0), ('D', 'late', 'IIXII', 12)]
# The 9-qubit physical block the run was pinned to (QCLOUD_SETUP.md sec.6.1).
BLOCK = [147, 155, 157, 164, 165, 166, 174, 175, 176]
ONE_Q = ('RX', 'RY', 'RZ', 'H', 'X', 'Y', 'Z', 'S', 'T', 'SX', 'U1', 'U2', 'U3')
TWO_Q = ('CNOT', 'CZ', 'ISWAP', 'SWAP', 'CR', 'CRK', 'SQISWAP')
ANCILLA_BITS = ('a0', 'a1', 'a2', 'a3')   # s = a0*8 + a1*4 + a2*2 + a3


def bits(s):
    """Syndrome integer -> (a0, a1, a2, a3) in the layout the run established."""
    return [(s >> 3) & 1, (s >> 2) & 1, (s >> 1) & 1, s & 1]


def hamming(s, ref):
    """Number of ancilla bits by which read syndrome s differs from ref."""
    return sum(1 for i in range(4) if bits(s)[i] != bits(ref)[i])


# --------------------------------------------------------------------------
# 1) exact gate inventory, rebuilt offline from the v1 angle snapshot
# --------------------------------------------------------------------------
def gate_inventory(phi):
    """Count the compiled branch circuits by section and by gate type.

    Rebuilt with qcloud_vscr_new.build_branch_circuit -- the same function that
    produced the submitted programs -- so the counts are of the circuits that ran,
    not of an idealised sketch of them.  Nothing here touches the cloud.
    """
    import qcloud_vscr_new as q
    q.setup()
    out = {}
    for nm, kind, frame, s in CASES:
        gates = q.build_branch_circuit(q.ENC5, '0', frame, s, phi,
                                       late_measure=(kind == 'late'))
        by_type = dict(collections.Counter(g[0] for g in gates))
        n1 = sum(v for k, v in by_type.items() if k in ONE_Q)
        n2 = sum(v for k, v in by_type.items() if k in TWO_Q)
        nmeas = by_type.get('MEAS', 0)
        sections = {
            'encoder': list(q.ENC5),
            'frame': q.frame_gates(frame),
            'extraction': q.extraction_gates(),
            'recovery': q.recovery_gates(phi[s]),
            'decoder': q.invert_gates(list(q.ENC5)),
            'measure_ancilla': q.extraction_measures(),
            'measure_data': q.data_measures(),
        }
        sec_counts = {k: len(v) for k, v in sections.items()}
        prep, val = q.TEST_STATES['0']
        out[nm] = {
            'kind': kind, 'frame': frame, 's': s,
            'instructions': len(gates) + len(prep) + len(val),
            'single_qubit_gates': n1 + len(prep) + len(val),
            'two_qubit_gates': n2,
            'measurements': nmeas,
            'by_type': by_type,
            'by_section': sec_counts,
            'unitary_gates': n1 + n2 + len(prep) + len(val),
        }
    return out


# --------------------------------------------------------------------------
# 2)-5) the syndrome and data channels, model-free, from the run's own marginals
# --------------------------------------------------------------------------
def syndrome_channel(rec):
    """Effective per-bit syndrome error rates from the measured marginals.

    The ideal branch weight is >= 0.9999957 (asserted in _finish), so the ideal
    syndrome is deterministic and P(bit i wrong) is simply the marginal of the
    measured bit against it.  `joint` is the measured all-four-correct rate,
    `product_of_marginals` what independence would predict, and `weight_tv` how far
    the Hamming-weight histogram is from the symmetric Bin(4, mean/4) model.
    """
    dist = {int(k): float(v) for k, v in rec['syndrome_dist_hw'].items()}
    ref = rec['s']
    n = rec['shots_used']
    marg = [sum(v for s, v in dist.items() if bits(s)[i] != bits(ref)[i])
            for i in range(4)]
    joint = sum(v for s, v in dist.items() if s == ref)
    prod = math.prod(1.0 - e for e in marg)
    weights = collections.Counter()
    for s, v in dist.items():
        weights[hamming(s, ref)] += v
    p_sym = sum(marg) / 4.0
    binom = {w: math.comb(4, w) * p_sym ** w * (1 - p_sym) ** (4 - w)
             for w in range(5)}
    return {
        'injected_bits': bits(ref),
        'per_bit_error': {ANCILLA_BITS[i]: marg[i] for i in range(4)},
        'per_bit_wrong_counts': {ANCILLA_BITS[i]: int(round(marg[i] * n))
                                 for i in range(4)},
        'mean_hamming_weight': sum(marg),
        'symmetric_rate_ml': p_sym,
        'joint_all_correct': joint,
        'joint_all_correct_counts': int(round(joint * n)),
        'product_of_marginals': prod,
        'independence_ratio': (joint / prod) if prod else float('nan'),
        'weight_distribution': {str(w): weights.get(w, 0.0) for w in range(5)},
        'weight_tv_vs_symmetric_binomial':
            0.5 * sum(abs(weights.get(w, 0.0) - binom[w]) for w in range(5)),
        'worst_bit': ANCILLA_BITS[max(range(4), key=lambda i: marg[i])],
        'worst_bit_error': max(marg),
        'n_worse_than_random': sum(1 for e in marg if e > 0.5),
    }


def data_channel(rec):
    """Conditional data-register error given a correctly read syndrome."""
    st = rec['branch_stats']
    if not st['n_sel']:
        return {'defined': False,
                'note': 'sel=0, so the conditional error is undefined, not 1'}
    lo, hi = st['F_s_lo95'], st['F_s_hi95']
    return {'defined': True, 'n_sel': st['n_sel'], 'n_hit': st['n_hit'],
            'F_s': st['F_s'], 'error': 1.0 - st['F_s'],
            'error_lo95': 1.0 - hi, 'error_hi95': 1.0 - lo,
            'ideal_reference': rec['P_expected_key_ideal']}


# --------------------------------------------------------------------------
# 6)-8) attribution, implied per-instruction error, and the mid-vs-late contrast
# --------------------------------------------------------------------------
def resolve_token():
    """$ORIGINQ_TOKEN, then ./.originq_token -- the convention list_devices.py uses."""
    tok = os.environ.get('ORIGINQ_TOKEN', '').strip()
    if tok:
        return tok
    path = os.path.join(ROOT, '.originq_token')
    if os.path.exists(path):
        tok = open(path).read().strip()
        if tok:
            return tok
    raise RuntimeError('no API token: set ORIGINQ_TOKEN or create %s' % path)


def try_vendor_calibration(chip_id='WK_C180'):
    """Re-attempt the FREE calibration metadata query.  Never submits a circuit.

    Either the per-qubit figures of merit for the pinned block come back, or the
    refusal is recorded verbatim with a timestamp, so the artifact always states
    which of the two happened rather than leaving the reader to guess why the
    error budget is effective rather than calibrated.
    """
    rec = {'attempted_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'chip_id': chip_id, 'block': BLOCK, 'available': False,
           'spends_qpu_time': False}
    try:
        import qcloud_vscr_new as q
        svc = q._make_qcloud_service(resolve_token())
        try:
            be = svc.backend(chip_id)
        except TypeError:
            be = svc.backend(chip_id=chip_id)
        ci = be.chip_info()
        single = {str(x.get_qubit_id()): {
            'readout_fidelity': x.get_readout_fidelity(),
            'single_gate_fidelity': x.get_single_gate_fidelity(),
            't1_us': x.get_t1(), 't2_us': x.get_t2(),
            'frequency_hz': x.get_frequency()} for x in ci.single_qubit_info()}
        double = {}
        for info in ci.double_qubits_info():
            qs = [int(u) for u in info.get_qubits()]
            if len(qs) == 2:
                double['%d-%d' % tuple(qs)] = info.get_fidelity()
        rec.update({
            'available': True,
            'single_gate_timing_ns': ci.get_single_gate_timing(),
            'double_gate_timing_ns': ci.get_double_gate_timing(),
            'basic_gates': list(ci.get_basic_gates()),
            'qubits_num': ci.qubits_num(),
            'block_qubits': {str(b): single.get(str(b)) for b in BLOCK},
            'block_edges': {k: v for k, v in double.items()
                            if all(int(u) in BLOCK for u in k.split('-'))},
        })
    except Exception as exc:                       # refusal, network, or API drift
        rec['error'] = repr(exc)[:400]
        rec['note'] = ('vendor figures of merit are behind the same authorisation '
                       'as a job submission; the effective rates in this artifact '
                       'are what the run itself supports')
    return rec


def budget(rec, syn, dat, inv):
    """Split the observed loss into its measured channels and attribute it."""
    hw = rec['P_expected_key_hw']
    ideal = rec['P_expected_key_ideal']
    loss_total = 1.0 - hw
    loss_method = 1.0 - ideal
    b = {
        'end_to_end_expected_key': hw,
        'ideal_reference_expected_key': ideal,
        'syndrome_channel_loss': 1.0 - syn['joint_all_correct'],
        'conditional_data_error': (dat['error'] if dat['defined'] else None),
        'conditional_data_error_ci95': (
            [dat['error_lo95'], dat['error_hi95']] if dat['defined'] else None),
        'factorisation_check': syn['joint_all_correct'] * (
            dat['F_s'] if dat['defined'] else 0.0),
        'factorisation_residual': abs(
            hw - syn['joint_all_correct'] * (dat['F_s'] if dat['defined'] else 0.0)),
        'method_share_of_loss': (loss_method / loss_total) if loss_total else None,
        'device_share_of_loss': (
            (loss_total - loss_method) / loss_total) if loss_total else None,
        'implied_mean_instruction_error': (
            1.0 - hw ** (1.0 / inv['instructions'])) if hw > 0 else None,
    }
    return b


def readout_asymmetry(hw):
    """Per-ancilla directional asymmetry, which identifies the relaxing qubit.

    The identity circuits (A mid, C late) inject s=0, so every ancilla's ideal bit
    is 0 and the measured marginal IS the effective 0->1 rate.  The injected
    circuits (B mid, D late) put s=12 = a0a1a2a3 = 1100 on the ancillas, so for a0
    and a1 the measured marginal IS the effective 1->0 rate.  Comparing the two
    directions per bit gives the bias of the effective channel without fitting
    anything.

    Caveat that must travel with the numbers: the two directions come from
    DIFFERENT circuits, so this is an asymmetry of the effective end-to-end
    channel, not a calibrated assignment matrix.  What makes it informative is the
    contrast between the two measurement schemes -- the same bit pair is compared
    before and after the ancilla idle time grows by the recovery+decode section.
    """
    e = {nm: syndrome_channel(hw['circuits'][nm])['per_bit_error']
         for nm in 'ABCD'}
    out = {'note': ('0->1 rates from the identity circuits (ideal bits all 0) and '
                    '1->0 rates from the injected circuits (ideal a0=a1=1); the '
                    'two directions come from different circuits, so this is an '
                    'effective-channel asymmetry, not an assignment matrix'),
           'per_bit': {}}
    for i, nm in enumerate(ANCILLA_BITS):
        ideal1 = bits(hw['circuits']['D']['s'])[i]
        rec = {'ideal_bit_in_injected_circuit': ideal1}
        # the identity circuits score 0->1 for every bit; the injected circuits
        # score 1->0 only for the bits they actually set to 1 (a0 and a1 here)
        rec['zero_to_one_late'] = e['C'][nm]
        rec['one_to_zero_late'] = e['D'][nm] if ideal1 else None
        rec['zero_to_one_mid'] = e['A'][nm]
        rec['one_to_zero_mid'] = e['B'][nm] if ideal1 else None
        if ideal1:
            rec['bias_late'] = rec['one_to_zero_late'] - rec['zero_to_one_late']
            rec['bias_mid'] = rec['one_to_zero_mid'] - rec['zero_to_one_mid']
            rec['bias_swing_on_deferral'] = rec['bias_late'] - rec['bias_mid']
        out['per_bit'][nm] = rec
    scored = [nm for nm in ANCILLA_BITS
              if out['per_bit'][nm].get('bias_late') is not None]
    out['bits_scored'] = scored
    out['relaxation_biased_bits_late'] = [
        nm for nm in scored if out['per_bit'][nm]['bias_late'] > 0]
    return out


def deferred_readout_contrast(hw):
    """What deferring the ancilla readout to the end of the circuit costs, per bit.

    The injected pair (B mid, D late) runs the same frame and the same branch; the
    only structural difference is where the four ancilla measurements sit.  A bit
    whose ideal value is 1 and which idles through the recovery+decode can only
    relax, so an increase confined to such a bit is the relaxation signature rather
    than an interpretation of it.
    """
    b = hw['circuits']['B']
    d = hw['circuits']['D']
    sb, sd = syndrome_channel(b), syndrome_channel(d)
    per_bit = {}
    for i, nm in enumerate(ANCILLA_BITS):
        per_bit[nm] = {'ideal_bit': bits(d['s'])[i],
                       'mid_error': sb['per_bit_error'][nm],
                       'late_error': sd['per_bit_error'][nm],
                       'delta_late_minus_mid':
                           sd['per_bit_error'][nm] - sb['per_bit_error'][nm]}
    worse = [nm for nm, v in per_bit.items() if v['delta_late_minus_mid'] > 0]
    return {
        'per_bit': per_bit,
        'bits_that_get_worse_when_deferred': worse,
        'mean_hamming_weight_mid': sb['mean_hamming_weight'],
        'mean_hamming_weight_late': sd['mean_hamming_weight'],
        'note': ('confounded by the fact that mid-circuit measurement is itself '
                 'dysfunctional on this device, so the three bits that improve are '
                 'not evidence that deferring helps them'),
    }


# --------------------------------------------------------------------------
# artifact + invariants
# --------------------------------------------------------------------------
def _finish(payload, wall_s):
    """Write the artifact, but only after every invariant below holds.

    These are the checks that keep the budget honest: the gate inventory has to
    add up, the per-bit rates have to reproduce the counts the run actually
    reported, the factorisation has to be an identity rather than an
    approximation, and a refused calibration query has to leave its reason on file
    instead of vanishing.
    """
    inv, circ = payload['gate_inventory'], payload['circuits']
    fails = []

    def need(cond, msg):
        if not cond:
            fails.append(msg)

    for nm in 'ABCD':
        g, c = inv[nm], circ[nm]
        syn, bud = c['syndrome_channel'], c['budget']
        rec = payload['_hw']['circuits'][nm]
        need(g['single_qubit_gates'] + g['two_qubit_gates'] + g['measurements']
             == g['instructions'],
             '%s: gate inventory does not add up (%d+%d+%d != %d)'
             % (nm, g['single_qubit_gates'], g['two_qubit_gates'],
                g['measurements'], g['instructions']))
        need(sum(g['by_section'].values()) == g['instructions'],
             '%s: section counts do not sum to the instruction count' % nm)
        need(g['unitary_gates'] == g['single_qubit_gates'] + g['two_qubit_gates'],
             '%s: unitary gate count inconsistent' % nm)
        need(all(0.0 <= v <= 1.0 for v in syn['per_bit_error'].values()),
             '%s: an effective per-bit error rate is outside [0,1]' % nm)
        need(abs(syn['mean_hamming_weight']
                 - sum(syn['per_bit_error'].values())) < 1e-12,
             '%s: mean Hamming weight is not the sum of the per-bit rates' % nm)
        need(abs(sum(syn['weight_distribution'].values()) - 1.0) < 1e-12,
             '%s: the Hamming-weight histogram is not normalised' % nm)
        need(abs(syn['joint_all_correct']
                 - rec['branch_stats']['n_sel'] / rec['shots_used']) < 1e-12,
             '%s: the all-four-correct rate is not n_sel/n_total from the run' % nm)
        need(syn['weight_tv_vs_symmetric_binomial'] < 0.15,
             '%s: the symmetric independent-bit model is off by %.3f in TV, too '
             'far to quote as a description of the syndrome channel'
             % (nm, syn['weight_tv_vs_symmetric_binomial']))
        need(rec['P_expected_key_ideal'] > 0.9999,
             '%s: the ideal branch weight %.7f no longer makes the ideal syndrome '
             'deterministic, so the marginals stop being error rates'
             % (nm, rec['P_expected_key_ideal']))
        if rec['branch_stats']['n_sel']:
            need(bud['factorisation_residual'] < 1e-12,
                 '%s: P(expected key) != P(correct syndrome)*F_s (residual %.2e)'
                 % (nm, bud['factorisation_residual']))
            need(bud['device_share_of_loss'] > 0.999,
                 '%s: the device no longer accounts for >99.9%% of the loss' % nm)
            if bud['implied_mean_instruction_error'] is None:
                need(rec['P_expected_key_hw'] == 0.0,
                     '%s: no implied per-instruction error, so the expected '
                     'bitstring rate must be zero' % nm)
            else:
                need(0.0 < bud['implied_mean_instruction_error'] < 0.05,
                     '%s: implied per-instruction error %.4f outside a physical '
                     'range' % (nm, bud['implied_mean_instruction_error']))
        else:
            need(rec['P_expected_key_hw'] == 0.0,
                 '%s: sel=0 but the expected bitstring was still observed' % nm)
            need(c['data_channel']['defined'] is False,
                 '%s: the conditional data error must stay undefined at sel=0' % nm)
    need(inv['D']['instructions'] - inv['C']['instructions'] == 1
         == inv['B']['instructions'] - inv['A']['instructions'],
         'the injected circuits differ from the identity ones by exactly the '
         'single frame gate')
    con = payload['deferred_readout_contrast']
    need(con['bits_that_get_worse_when_deferred'] == ['a1'],
         'deferring the ancilla readout is claimed to hurt only a1, but the '
         'artifact says %s' % con['bits_that_get_worse_when_deferred'])
    need(con['per_bit']['a1']['ideal_bit'] == 1
         and con['per_bit']['a1']['late_error'] > 0.5,
         'a1 is not the relaxing bit it is claimed to be (ideal %s, late error '
         '%.3f)' % (con['per_bit']['a1']['ideal_bit'],
                    con['per_bit']['a1']['late_error']))
    cal = payload['vendor_calibration']
    need('available' in cal and (cal['available'] or cal.get('error')),
         'the calibration record must say either what was measured or why not')
    asy = payload['readout_asymmetry']
    need(asy['bits_scored'] == ['a0', 'a1'],
         'only the two ancillas the injected circuit sets to 1 can be scored in '
         'both directions, got %s' % asy['bits_scored'])
    need(asy['relaxation_biased_bits_late'] == ['a1'],
         'a1 is claimed to be the unique relaxation-biased ancilla, but the late '
         'pair scores %s' % asy['relaxation_biased_bits_late'])
    a1, a0 = asy['per_bit']['a1'], asy['per_bit']['a0']
    need(a1['bias_mid'] < 0.0 < a1['bias_late'],
         'deferring the readout is claimed to flip a1 from readout-biased to '
         'relaxation-biased, but its bias goes %.3f -> %.3f'
         % (a1['bias_mid'], a1['bias_late']))
    need(a1['bias_swing_on_deferral'] > 0.4,
         'the a1 asymmetry swing on deferral is only %.3f, too small to carry the '
         'relaxation attribution' % a1['bias_swing_on_deferral'])
    need(a0['bias_late'] < 0.0 and a0['bias_late'] < a1['bias_late'],
         'a0 must stay biased the other way, or a1 is not singled out')
    payload = {k: v for k, v in payload.items() if k != '_hw'}
    payload['meta']['wall_s'] = round(wall_s, 2)
    if fails:
        for f in fails:
            print('  INVARIANT FAIL: ' + f)
        raise SystemExit('%d invariant(s) failed; artifact not written'
                         % len(fails))
    with open(OUT_JSON, 'w') as fh:
        json.dump(payload, fh, indent=1, allow_nan=False)
    print('wrote %s' % OUT_JSON)


def main():
    ap = argparse.ArgumentParser(
        description='effective hardware error budget for the WK_C180 run')
    ap.add_argument('--try-calibration', action='store_true',
                    help='re-attempt the free vendor metadata query (no shots, no '
                         'QPU time) and record the outcome either way')
    ap.add_argument('--chip-id', default='WK_C180')
    args = ap.parse_args()
    t0 = time.time()

    hw = json.load(open(IN_JSON))
    import qcloud_vscr_new as q
    angles_file = os.environ.get('VSCR_ANGLES_FILE') or 'vscr_angles_dep.npz'
    phi = q.load_angles('depolarizing')
    inv = gate_inventory(phi)
    for nm in 'ABCD':
        g = inv[nm]
        print('[gates] %s %-4s instructions=%3d (1Q=%3d 2Q=%3d meas=%d) %s'
              % (nm, g['kind'], g['instructions'], g['single_qubit_gates'],
                 g['two_qubit_gates'], g['measurements'], g['by_section']))

    if args.try_calibration:
        cal = try_vendor_calibration(args.chip_id)
    else:
        # Keep the last recorded outcome rather than blanking it: an offline
        # re-run must not silently turn "the query was refused at <time>" into
        # "we did not look", which is what section 16 of the audit forbids.
        cal = {}
        if os.path.exists(OUT_JSON):
            try:
                cal = json.load(open(OUT_JSON)).get('vendor_calibration') or {}
            except Exception:
                cal = {}
        if not (cal.get('available') or cal.get('error')):
            cal = {'available': False, 'attempted_utc': None,
                   'error': 'not re-attempted on this run',
                   'note': ('vendor metadata sits behind the job-submission '
                            'authorisation; pass --try-calibration to '
                            're-attempt')}
    print('[calibration] available=%s %s' % (cal['available'], cal.get('error', '')))

    circ = {}
    for nm, kind, frame, s in CASES:
        rec = hw['circuits'][nm]
        syn = syndrome_channel(rec)
        dat = data_channel(rec)
        circ[nm] = {'kind': kind, 'frame': frame, 's': s,
                    'shots_used': rec['shots_used'],
                    'syndrome_channel': syn, 'data_channel': dat,
                    'budget': budget(rec, syn, dat, inv[nm])}
        print('[budget] %s per-bit error %s  joint=%.4f (prod=%.4f, ratio=%.3f)'
              '  worst=%s %.3f  TV(binom)=%.4f'
              % (nm, ['%.3f' % syn['per_bit_error'][b] for b in ANCILLA_BITS],
                 syn['joint_all_correct'], syn['product_of_marginals'],
                 syn['independence_ratio'], syn['worst_bit'],
                 syn['worst_bit_error'],
                 syn['weight_tv_vs_symmetric_binomial']))

    con = deferred_readout_contrast(hw)
    asy = readout_asymmetry(hw)
    for nm in asy['bits_scored']:
        r = asy['per_bit'][nm]
        print('[asymmetry] %s 0->1 late=%.3f mid=%.3f | 1->0 late=%.3f mid=%.3f '
              '| bias late=%+.3f mid=%+.3f swing=%+.3f'
              % (nm, r['zero_to_one_late'], r['zero_to_one_mid'],
                 r['one_to_zero_late'], r['one_to_zero_mid'],
                 r['bias_late'], r['bias_mid'], r['bias_swing_on_deferral']))
    d = circ['D']
    payload = {
        'meta': {
            'task': 'effective hardware error budget, WK_C180 feasibility run',
            'generated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'host': os.uname().nodename, 'python': sys.version.split()[0],
            'numpy': np.__version__,
            'inputs': {'counts': 'hw_feasibility_numbers.json',
                       'angles': angles_file,
                       'angles_mtime_utc': time.strftime(
                           '%Y-%m-%dT%H:%M:%SZ',
                           time.gmtime(os.path.getmtime(angles_file))),
                       'circuits': 'rebuilt offline by qcloud_vscr_new'},
            'qpu_time_spent': 0,
            'estimator': ('model-free marginals against a deterministic ideal '
                          'syndrome; no independence assumption enters the '
                          'per-bit rates, and where one is used it is reported '
                          'as a ratio against the measurement'),
        },
        'vendor_calibration': cal,
        'gate_inventory': inv,
        'circuits': circ,
        'deferred_readout_contrast': con,
        'readout_asymmetry': asy,
        '_hw': hw,
    }
    payload['headline'] = {
        'instructions_D': inv['D']['instructions'],
        'unitary_gates_D': inv['D']['unitary_gates'],
        'single_qubit_gates_D': inv['D']['single_qubit_gates'],
        'two_qubit_gates_D': inv['D']['two_qubit_gates'],
        'worst_bit_D': d['syndrome_channel']['worst_bit'],
        'worst_bit_error_D': d['syndrome_channel']['worst_bit_error'],
        'n_bits_worse_than_random_D': d['syndrome_channel']['n_worse_than_random'],
        'syndrome_channel_loss_D': d['budget']['syndrome_channel_loss'],
        'conditional_data_error_D': d['budget']['conditional_data_error'],
        'conditional_data_error_ci95_D': d['budget']['conditional_data_error_ci95'],
        'independence_ratio_D': d['syndrome_channel']['independence_ratio'],
        'weight_tv_D': d['syndrome_channel']['weight_tv_vs_symmetric_binomial'],
        'method_share_of_loss_D': d['budget']['method_share_of_loss'],
        'device_share_of_loss_D': d['budget']['device_share_of_loss'],
        'method_loss_D': 1.0 - hw['circuits']['D']['P_expected_key_ideal'],
        'implied_mean_instruction_error_D':
            d['budget']['implied_mean_instruction_error'],
        'implied_mean_instruction_error_C':
            circ['C']['budget']['implied_mean_instruction_error'],
        'a1_error_mid_injected': con['per_bit']['a1']['mid_error'],
        'a1_error_late_injected': con['per_bit']['a1']['late_error'],
        'a1_delta_deferred': con['per_bit']['a1']['delta_late_minus_mid'],
        'a1_zero_to_one_late': asy['per_bit']['a1']['zero_to_one_late'],
        'a1_one_to_zero_late': asy['per_bit']['a1']['one_to_zero_late'],
        'a1_bias_late': asy['per_bit']['a1']['bias_late'],
        'a1_bias_mid': asy['per_bit']['a1']['bias_mid'],
        'a0_bias_late': asy['per_bit']['a0']['bias_late'],
        'ideal_branch_weight_D': hw['circuits']['D']['P_expected_key_ideal'],
    }
    _finish(payload, time.time() - t0)

    print('\n=== headline ===')
    for k, v in payload['headline'].items():
        print('  %-34s %s' % (k, ('%.6g' % v) if isinstance(v, float) else v))


if __name__ == '__main__':
    main()
