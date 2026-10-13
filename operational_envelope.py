#!/usr/bin/env python3
"""operational_envelope.py -- what the certified headroom COSTS, and what a device
must deliver for that cost to be worth paying.

WHY THIS MODULE EXISTS
----------------------
The paper certifies a non-Pauli headroom Delta (a per-branch optimal-CPTP ceiling
minus the decoder fidelity) and then compiles a recovery that captures part of it.
Both statements are exact and both are about an IDEAL machine.  A reviewer is
entitled to ask the only question that connects them to the hardware section: what
does the compiled recovery cost in instructions, and how good must those
instructions be for the captured headroom to survive the cost?  Nothing in the
repository answered that in instruction-error units, so the hardware section could
be read as "we ran it and it was worse", which is neither the claim nor the
interesting fact.  The interesting fact is a THRESHOLD, and it is computable from
artifacts that already exist.

THREE NUMBERS ALREADY ON DISK, AND THE ONE LINK THAT WAS MISSING
----------------------------------------------------------------
  hw_error_budget.json    gate_inventory.D.by_section.recovery = 90 instructions
                          for one compiled branch recovery, out of 204 for the
                          whole circuit; headline.implied_mean_instruction_error_D
                          = 1.04e-2, measured FROM THE RUN ITSELF (no vendor
                          calibration was available for WK_C180).
  storage_rounds.json     recovery_noise rows: the advantage over the Pauli decoder
                          as a function of a depolarizing layer of strength lam
                          composed with each recovery, bisected at R=40 to give
                          lam_star (sign change) and lam_half (half magnitude),
                          plus the universal erosion law adv(lam)/adv(0) ~ e^{-lam R}
                          verified to 8.8e-3 over lam <= 0.04 and shared by all
                          eight non-degenerate configurations to 2.3e-3.
  paper_numbers.json      the captured headroom Delta at the operating point.

The missing link is the dictionary between lam -- a depolarizing strength on the
whole 5-qubit register, which is what the round map can absorb -- and eps, a
per-INSTRUCTION error, which is what a device quotes.  The compiled recovery is
N_rec = 90 instructions long, so the strictest reading is lam = N_rec * eps: every
instruction fault is charged as a full-register depolarizing event and nothing is
corrected.  That gives

    eps_half(R) = lam_half / N_rec = ln2 / (R * N_rec)      (half the advantage)
    eps_star    = lam_star / N_rec                          (sign change)

and, independently of the lam model altogether, an arm-difference bound: the
learned branch pays N_rec instructions where a frame-tracked Pauli lookup pays
none, so under a plain independent-error model the two arms differ by about
N_rec * eps to first order, and break-even needs

    eps_be = Delta / N_rec .

Both ends are CONSERVATIVE, in opposite directions, and the truth is in between: a
distance-3 code corrects most single-qubit faults inside the recovery, so charging
each one as an uncorrectable register-wide depolarizing event over-states the
damage; while the arm-difference bound charges the full N_rec instructions even
though most of the circuit is shared with the decoder arm and cancels in the
difference only to first order.  We therefore report a BRACKET and check that the
device sits outside the LOOSE end of it, so that the verdict does not depend on
which end is closer to the truth.  _selftest_model asserts the ordering
eps_be <= eps_half(R=1) on every non-degenerate row and fails loudly if a future
artifact ever inverts it, because an inverted bracket would mean one of the two
models had been misread.

WHAT IS NOT CLAIMED
-------------------
This is not a fault-tolerance threshold for the code, and not a simulation of the
compiled circuit under a realistic gate-noise model; both would need a
circuit-level Monte Carlo over 204 instructions with a calibrated noise model that
the platform refuses to supply.  It is an order-of-magnitude envelope built from
the run's own measured instruction error and the paper's own measured erosion law,
and it is reported as such.
"""
# ---------------------------------------------------------------------------
# SHOT BUDGET -- THE SECOND HALF OF THE ENVELOPE, AND THE ONE THAT KILLS THE
# EXPERIMENT EVEN IF THE GATES WERE FREE.
# An advantage of Delta is a claim about a MEAN.  Resolving it at z sigma with a
# Bernoulli estimator of per-shot variance v needs N >= (z^2 v)/Delta^2 shots per
# arm, and on this protocol those shots must first survive syndrome
# post-selection, so the REQUESTED budget is that divided by the end-to-end yield
# the feasibility run measured.  Both v and the yield come from artifacts: v is
# bounded by 1/4 for a Bernoulli estimator, the yield from
# hw_feasibility_numbers.json.  The executed run's own Wilson half-width is
# recomputed here from the same counts the audit checks, so the comparison
# "achieved +-0.067 against a signal of +1.6e-3" is a computation, not a remark.
# ---------------------------------------------------------------------------
import argparse
import datetime as _dt
import json
import math
import os
import platform
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Single-threaded BLAS, set before numpy is imported, for the same reason
# scaling_analysis.py gives.  This module only reads artifacts and does scalar
# arithmetic, but it imports numpy for the figure grid, and a pinned thread count
# costs nothing.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')

import numpy as np                                              # noqa: E402

OUT = os.path.join(ROOT, 'operational_envelope.json')
FIG = os.path.join(ROOT, 'paper', 'figures', 'fig_operational_envelope')
EB_PATH = os.path.join(ROOT, 'hw_error_budget.json')
SR_PATH = os.path.join(ROOT, 'storage_rounds.json')
PN_PATH = os.path.join(ROOT, 'paper_numbers.json')
HW_PATH = os.path.join(ROOT, 'hw_feasibility_numbers.json')

Z95 = 1.959963984540054   # two-sided 95% normal quantile, as in the audit
Z3 = 3.0                  # the resolution target quoted in the paper
LN2 = math.log(2.0)
DEP_PAULI_LOOKUP = 0        # a frame-tracked Pauli correction costs no instructions


# ---------------------------------------------------------------------------
# Wilson interval, recomputed rather than quoted, so the shot-budget arithmetic is
# anchored to the same statistic the paper already prints.
# ---------------------------------------------------------------------------
def wilson95(k, n):
    """Two-sided 95% Wilson score interval for k successes out of n."""
    if not n or n <= 0:
        return (float('nan'), float('nan'), float('nan'))
    p = k / n
    z2 = Z95 * Z95
    den = 1.0 + z2 / n
    ctr = (p + z2 / (2 * n)) / den
    half = (Z95 / den) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (max(0.0, ctr - half), min(1.0, ctr + half), half)


def load():
    """Read the four artifacts this module is allowed to depend on."""
    with open(EB_PATH) as f:
        eb = json.load(f)
    with open(SR_PATH) as f:
        sr = json.load(f)
    with open(PN_PATH) as f:
        pn = json.load(f)
    with open(HW_PATH) as f:
        hw = json.load(f)
    return eb, sr, pn, hw


# ---------------------------------------------------------------------------
# 1. the instruction inventory of one compiled branch
# ---------------------------------------------------------------------------
def inventory(eb, circuit='D'):
    """The compiled circuit's own gate count, split by section.

    `D` is the circuit the headline fidelity comes from (late-measure, injected
    X_2, branch s=12); the audit already pins that choice, so the envelope is
    priced on the same circuit the paper reports.
    """
    gi = eb['gate_inventory'][circuit]
    sec = dict(gi['by_section'])
    unitary_sections = sum(sec.values())
    return {
        'circuit': circuit,
        'kind': gi['kind'],
        'frame': gi['frame'],
        'by_section': sec,
        'unitary_section_sum': unitary_sections,
        'instructions': gi['instructions'],
        'measurements': gi['measurements'],
        'unitary_gates': gi['unitary_gates'],
        'two_qubit_gates': gi['two_qubit_gates'],
        'single_qubit_gates': gi['single_qubit_gates'],
        # the learned part of the branch: what a Pauli lookup would replace
        'n_recovery': sec['recovery'],
        'n_pauli_lookup': DEP_PAULI_LOOKUP,
        'delta_instructions': sec['recovery'] - DEP_PAULI_LOOKUP,
        # sections both arms pay, i.e. the part that cancels in the difference
        'n_shared': gi['instructions'] - sec['recovery'] + DEP_PAULI_LOOKUP,
        'n_data_qubits': 5,
    }


# ---------------------------------------------------------------------------
# 2. the envelope, from the measured erosion law
# ---------------------------------------------------------------------------
def envelope_rows(sr, n_rec, delta_r1=None):
    """Every recovery_noise row, converted into per-instruction units.

    A row is degenerate when the advantage at lam=0 is zero -- depolarizing and
    mixed noise, where the decoder is already the family optimum -- and
    storage_rounds.py records that explicitly instead of bisecting round-off.  We
    keep those rows in the artifact with null thresholds, because "there is no
    envelope to draw here" is itself part of the answer, and it is the answer for
    exactly the channel a Pauli-frame hardware benchmark can emulate.

    `advantage_lam0` in this artifact is the advantage at R=R_last (40), NOT at
    R=1; `_selftest_cross_artifact` pins that reading against the `summary` block,
    which carries advantage_R1 and advantage_R40 separately.  The arm-difference
    break-even therefore has two versions and they differ by the compounding
    factor: a single-round benchmark has to pay for its extra instructions out of
    the R=1 headroom, a memory experiment out of the R=40 one.
    """
    out = []
    # advantage_R1 per configuration, from the summary block at eta=0, so the
    # single-round and memory curves in the figure are each drawn from the
    # advantage that actually belongs to them.
    r1 = {}
    for s in sr['summary']:
        if abs(float(s['eta'])) < 1e-12:
            r1[(s['code'], s['channel'], round(float(s['p']), 4),
                s['recovery'])] = s['advantage_R1']
    for r in sr['recovery_noise']:
        degenerate = r['threshold_status'] != 'ok'
        k = (r['code'], r['channel'], round(float(r['p']), 4), r['recovery'])
        row = {
            'code': r['code'], 'n': r['n'], 'channel': r['channel'],
            'p': r['p'], 'recovery': r['recovery'], 'R_last': r['R_last'],
            'advantage_lam0': r['advantage_lam0'],
            'advantage_R1': r1.get(k),
            'advantage_R': r['R_last'],
            'lam_star': r['lam_star'], 'lam_half': r['lam_half'],
            'eps_per_qubit_star': r['eps_per_qubit_star'],
            'eps_per_qubit_half': r['eps_per_qubit_half'],
            'threshold_status': r['threshold_status'],
            'degenerate': degenerate,
        }
        if not degenerate:
            # strict dictionary: one full-register depolarizing event per
            # instruction fault, nothing corrected.
            row['eps_instr_star'] = r['lam_star'] / n_rec
            row['eps_instr_half_at_R_last'] = r['lam_half'] / n_rec
            # the universal one-line rule lam_half ~ ln2/R, re-expressed per
            # instruction; it needs no knowledge of the channel at all.
            for R in (1, 5, 10, 40):
                row['eps_instr_half_R%d' % R] = LN2 / (float(R) * n_rec)
            # arm-difference break-even: the learned branch pays n_rec
            # instructions where a frame-tracked lookup pays none.  The R=1
            # version is the strict end of the bracket for a single-round
            # benchmark; the R_last version is what a memory experiment gets.
            row['eps_instr_breakeven_Rlast'] = r['advantage_lam0'] / n_rec
            adv1 = row['advantage_R1'] if row['advantage_R1'] else delta_r1
            row['eps_instr_breakeven'] = (adv1 / n_rec) if adv1 else None
        out.append(row)
    return out



# ---------------------------------------------------------------------------
# 3. the device, measured from its own run
# ---------------------------------------------------------------------------
def device_point(eb, env_rows, n_rec):
    """Where the platform sits relative to the envelope it is being asked to meet."""
    h = eb['headline']
    eps_D = h['implied_mean_instruction_error_D']
    eps_C = h['implied_mean_instruction_error_C']
    ref = [r for r in env_rows if not r['degenerate'] and r['recovery'] == 'warm']
    if not ref:
        ref = [r for r in env_rows if not r['degenerate']]
    # the tightest requirement over all non-degenerate learned rows, so the
    # verdict is not cherry-picked from the friendliest channel
    def tight(k):
        vals = [r[k] for r in ref if r.get(k) is not None]
        return min(vals) if vals else None
    return {
        'chip': eb['vendor_calibration'].get('chip_id'),
        'eps_instr_measured_D': eps_D,
        'eps_instr_measured_C': eps_C,
        'source': 'implied by the run itself under an independent-error model; no '
                  'vendor calibration accompanied that run',
        'eps_instr_breakeven_required': tight('eps_instr_breakeven'),
        'eps_instr_half_R1_required': tight('eps_instr_half_R1'),
        'eps_instr_half_R40_required': tight('eps_instr_half_R40'),
        'eps_instr_star_required': min(r['eps_instr_star'] for r in ref)
        if ref else None,
        'factor_above_breakeven': eps_D / tight('eps_instr_breakeven')
        if tight('eps_instr_breakeven') else None,
        'factor_above_half_R1': eps_D / tight('eps_instr_half_R1')
        if tight('eps_instr_half_R1') else None,
        'factor_above_half_R40': eps_D / tight('eps_instr_half_R40')
        if tight('eps_instr_half_R40') else None,
        'factor_above_star': eps_D / min(r['eps_instr_star'] for r in ref)
        if ref else None,
        'n_rec': n_rec,
        # the naive uncorrected cost of the extra instructions at the measured
        # error rate: what the recovery section alone does to one branch.
        'recovery_survival_at_measured': (1.0 - eps_D) ** n_rec,
        'recovery_loss_at_measured': 1.0 - (1.0 - eps_D) ** n_rec,
        'whole_circuit_survival_at_measured': (1.0 - eps_D) ** h['instructions_D'],
        'verdict': None,     # set by verdict(); derived, never asserted
    }


def verdict(dev):
    """The envelope verdict, computed from the factors rather than written in."""
    f_loose = dev['factor_above_half_R1']
    f_strict = dev['factor_above_breakeven']
    if f_loose is None:
        return 'undetermined'
    if f_loose > 1.0:
        return 'outside_envelope_even_at_the_loose_end'
    if f_strict is not None and f_strict > 1.0:
        return 'inside_the_loose_end_outside_the_strict_end'
    return 'inside_envelope'


# ---------------------------------------------------------------------------
# 4. readout erosion of the same envelope
# ---------------------------------------------------------------------------
def readout_erosion(sr, n_rec, n_minus_k=4):
    """The certified headroom is multiplied by (1-eta)^(n-k) per round.

    That law is exact for the Pauli decoder -- its cross-branch blocks
    V^dag R_s' W_s vanish identically -- and holds for the learned recovery to
    5e-11, so it erodes the NUMERATOR of the envelope while leaving the
    instruction cost untouched: the required eps falls with it.  We report the
    factor at the etas the storage artifact already scanned.  The device's own
    ancilla readout error is far outside that grid, which is the point: the grid
    is the optimistic end of what a superconducting platform delivers.
    """
    rows = []
    for eta in list(sr['config']['eta_grid']) + [0.05, 0.10, 0.25]:
        fac = (1.0 - eta) ** n_minus_k
        rows.append({
            'eta': eta,
            'in_storage_grid': eta in sr['config']['eta_grid'],
            'headroom_factor': fac,
            'eps_instr_breakeven_scaled': None,   # filled per channel by caller
            'eps_instr_half_R40_scaled': LN2 / (40.0 * n_rec) * fac,
        })
    return {
        'n_minus_k': n_minus_k,
        'law': 'F(eta,R) = (1-eta)^((n-k)R) F(0,R) exactly for the decoder; the '
               'learned recovery violates it only at the 5e-11 level',
        'eta_rows': rows,
        'note': 'readout error shrinks the certified headroom but not the '
                'instruction cost, so it tightens the envelope monotonically',
    }


# ---------------------------------------------------------------------------
# 5. the shot budget
# ---------------------------------------------------------------------------
def shot_budget(hw, pn, deltas=None):
    """How many shots a Delta-sized advantage needs, against how many were run.

    The estimator is a Bernoulli mean on a post-selected branch, so its per-shot
    variance is F(1-F), bounded by 1/4.  We quote the bound rather than the
    optimistic value: at F_s = 0.80 the true variance is 0.16 and the bound costs a
    factor 1.6 in shots, which is noise next to the three orders of magnitude the
    answer spans.  The requested budget is then divided by the end-to-end yield the
    feasibility run measured, because on this protocol a shot only counts if its
    syndrome is read correctly AND its data register lands on the logical state.

    Both channels with certified headroom are priced.  The headline number is the
    CHEAPEST of them -- the largest Delta, amplitude damping -- so that the quoted
    budget is the most favourable one available and the conclusion cannot be an
    artifact of having picked the harder channel.
    """
    circuits = hw['circuits']
    tot_use = sum(int(c.get('shots_used', 0)) for c in circuits.values())
    tot_req = int(hw['shots_requested']) * len(circuits)
    tot_sel = sum(int(c['branch_stats']['n_sel']) for c in circuits.values()
                  if 'branch_stats' in c)
    tot_hit = sum(int(c['branch_stats']['n_hit']) for c in circuits.values()
                  if 'branch_stats' in c)
    D = circuits['D']['branch_stats']
    lo, hi, half = wilson95(D['n_hit'], D['n_sel'])
    if deltas is None:
        deltas = {
            'amplitude_damping': (pn['ad_p010']['VSCR warm (ours)']
                                  - pn['ad_p010']['Perfect-code decoder']),
        }
        if 'coh_p010' in pn and 'VSCR warm (ours)' in pn['coh_p010']:
            deltas['coherent'] = (pn['coh_p010']['VSCR warm (ours)']
                                  - pn['coh_p010']['Perfect-code decoder'])
    v = 0.25                                   # Bernoulli bound
    yld = (tot_hit / tot_req) if tot_req else None
    per_channel = {}
    for ch, d in sorted(deltas.items()):
        per_arm = (Z3 ** 2) * v / d ** 2
        per_channel[ch] = {
            'delta': d,
            'shots_per_arm_3sigma': per_arm,
            'shots_both_arms_3sigma': 2 * per_arm,
            'shots_requested_if_yield_applies':
                (2 * per_arm / yld) if yld else None,
            'shortfall_factor_vs_executed': per_arm / tot_req if tot_req else None,
        }
    head_ch = max(per_channel, key=lambda c: per_channel[c]['delta'])
    head = per_channel[head_ch]
    return {
        'headline_channel': head_ch,
        'delta_target': head['delta'],
        'per_shot_variance_bound': v,
        'z': Z3,
        'shots_per_arm_3sigma': head['shots_per_arm_3sigma'],
        'shots_both_arms_3sigma': head['shots_both_arms_3sigma'],
        'end_to_end_yield': yld,
        'shots_requested_if_yield_applies': head['shots_requested_if_yield_applies'],
        'per_channel': per_channel,
        'executed_shots_requested': tot_req,
        'executed_shots_used': tot_use,
        'executed_post_selected': tot_sel,
        'executed_data_hits': tot_hit,
        'executed_branch_D': {'n_hit': D['n_hit'], 'n_sel': D['n_sel'],
                              'F_s': D['F_s'], 'wilson95': [lo, hi]},
        'executed_wilson_halfwidth': half,
        'halfwidth_over_delta': (half / head['delta']) if head['delta'] else None,
        'shortfall_factor_per_arm': head['shortfall_factor_vs_executed'],
        'verdict': 'underpowered' if (half and head['delta']
                                      and half > head['delta']) else 'ok',
    }



# ---------------------------------------------------------------------------
# self-tests: every one of these is a cross-check against an EXISTING artifact,
# not against a re-derivation of this module's own model.
# ---------------------------------------------------------------------------
def _selftest_inventory(inv, eb):
    """The section counts must reproduce the instruction count the artifact reports."""
    gi = eb['gate_inventory'][inv['circuit']]
    return [
        ('section_sum_equals_instructions',
         inv['unitary_section_sum'] == gi['instructions'],
         inv['unitary_section_sum'], gi['instructions']),
        ('section_sum_minus_measurements_equals_unitary_gates',
         inv['unitary_section_sum'] - inv['measurements'] == gi['unitary_gates'],
         inv['unitary_section_sum'] - inv['measurements'], gi['unitary_gates']),
        ('measurements_split_ancilla_plus_data',
         inv['by_section']['measure_ancilla'] + inv['by_section']['measure_data']
         == inv['measurements'],
         inv['by_section']['measure_ancilla'] + inv['by_section']['measure_data'],
         inv['measurements']),
        ('recovery_section_read_from_the_artifact',
         inv['n_recovery'] == gi['by_section']['recovery'],
         inv['n_recovery'], gi['by_section']['recovery']),
        ('learned_branch_costs_more_than_a_frame_update',
         inv['delta_instructions'] > 0, inv['delta_instructions'], 0),
    ]


def _selftest_model(sr, rows):
    """Two readings of the same artifact must agree, or one of them is wrong.

    (a) eps_per_qubit_half in the artifact is lam_half/n, which is the evidence
        that the dictionary this module relies on is linear in lam;
    (b) the universal rule lam_half ~ ln2/R must reproduce the bisected lam_half
        to the tolerance the paper quotes for it;
    (c) the bracket must be ordered: the arm-difference break-even is the strict
        end and the R=1 half-advantage budget the loose end.  An inverted bracket
        would mean one of the two models had been misread.
    """
    worst_a = worst_b = worst_c = 0.0
    n_checked = 0
    for r in rows:
        if r['degenerate']:
            continue
        n_checked += 1
        worst_a = max(worst_a, abs(r['eps_per_qubit_half'] * r['n']
                                   - r['lam_half']))
        worst_b = max(worst_b, abs(LN2 / r['R_last'] - r['lam_half'])
                      / r['lam_half'])
        worst_c = max(worst_c, r['eps_instr_breakeven'] / r['eps_instr_half_R1'])
    return [
        ('eps_per_qubit_half_is_lam_half_over_n', worst_a < 1e-12, worst_a, 1e-12),
        ('universal_lam_half_rule_reproduces_the_bisection', worst_b < 5e-2,
         worst_b, 5e-2),
        ('bracket_ordered_strict_below_loose', worst_c < 1.0, worst_c, 1.0),
        ('non_degenerate_rows_present', n_checked > 0, n_checked, 1),
    ]


def _selftest_cross_artifact(sr, rows, pn):
    """Pin what `advantage_lam0` means, and link it to the paper's own headroom.

    storage_rounds.py and vscr_paper.py are separate programs with separate
    estimators; if they disagree about what the learned recovery buys, one of the
    two artifacts is stale and every number built on top of them is unverifiable.
    The first check is the one that matters most for THIS module: `advantage_lam0`
    is the advantage at R=R_last, not at R=1, and reading it as a single-round
    number would make the break-even instruction error fourteen times too loose.
    """
    out = []
    key = ('5,1,3', 'amplitude_damping', 0.10, 'warm')
    srow = None
    for s in sr['summary']:
        if (s['code'], s['channel'], round(float(s['p']), 4),
                s['recovery']) == (key[0], key[1], key[2], key[3]) \
                and abs(float(s['eta'])) < 1e-12:
            srow = s
            break
    nrow = None
    for r in rows:
        if (r['code'], r['channel'], round(float(r['p']), 4),
                r['recovery']) == key and not r['degenerate']:
            nrow = r
            break
    delta_pn = (pn['ad_p010']['VSCR warm (ours)']
                - pn['ad_p010']['Perfect-code decoder'])
    if srow is None or nrow is None:
        return [('storage_summary_and_recovery_noise_rows_found', False,
                 (srow is not None, nrow is not None), True)]
    out.append(('storage_summary_and_recovery_noise_rows_found', True, True, True))
    d40 = abs(nrow['advantage_lam0'] - srow['advantage_R40'])
    out.append(('advantage_lam0_is_the_R_last_advantage_not_R1', d40 < 1e-9,
                d40, 1e-9))
    d1 = abs(srow['advantage_R1'] - delta_pn)
    rel1 = d1 / abs(delta_pn)
    # The two programs do NOT share an estimator: storage_rounds.py uses the exact
    # Haar closed form on the refined branch table, vscr_paper.py's ad_p010 is the
    # 200-state benchmark average that Fig.~2 plots.  They agree to 3.4e-3
    # relative, which is the cross-check; a relative disagreement above 1% would
    # mean one of the two artifacts had gone stale, which is what this catches.
    out.append(('storage_advantage_R1_matches_paper_headroom', rel1 < 1e-2,
                rel1, 1e-2))
    out.append(('advantage_compounds_with_rounds',
                srow['advantage_R40'] > srow['advantage_R1'] > 0,
                (srow['advantage_R1'], srow['advantage_R40']), None))
    return out



def _selftest_wilson(hw, sb):
    """The Wilson interval recomputed here must be the one the paper prints."""
    ref = hw['summary'].get('late_D_F_s_wilson95')
    if not ref:
        return [('wilson_recomputed_matches_artifact', True, 'no reference', None)]
    got = sb['executed_branch_D']['wilson95']
    d = max(abs(got[0] - ref[0]), abs(got[1] - ref[1]))
    return [('wilson_recomputed_matches_artifact', d < 5e-3, d, 5e-3)]


def _selftest_verdicts(dev, sb):
    """The verdicts must be the ones the factors imply, and must be consistent."""
    return [
        ('device_verdict_is_derived_not_asserted',
         dev['verdict'] == verdict(dev), dev['verdict'], verdict(dev)),
        ('shot_verdict_follows_from_halfwidth_vs_delta',
         (sb['verdict'] == 'underpowered')
         == (sb['executed_wilson_halfwidth'] > sb['delta_target']),
         sb['verdict'], sb['executed_wilson_halfwidth'] > sb['delta_target']),
        ('halfwidth_over_delta_is_the_ratio_of_the_two',
         abs(sb['halfwidth_over_delta']
             - sb['executed_wilson_halfwidth'] / sb['delta_target']) < 1e-9,
         sb['halfwidth_over_delta'], None),
    ]


def run_selftests(eb, sr, pn, hw, inv, rows, dev, sb):
    groups = [_selftest_inventory(inv, eb), _selftest_model(sr, rows),
              _selftest_cross_artifact(sr, rows, pn), _selftest_wilson(hw, sb),
              _selftest_verdicts(dev, sb)]
    out, nfail = [], 0
    for g in groups:
        for name, passed, got, want in g:
            out.append({'check': name, 'passed': bool(passed),
                        'got': float(got) if isinstance(got, float) else got,
                        'tol': want})
            nfail += 0 if passed else 1
    return {'checks': out, 'n_checks': len(out), 'n_failures': nfail,
            'passed': nfail == 0}


def build():
    """Assemble the whole artifact.  Separated from main() so the audit and the
    figure can both call it without re-parsing argv."""
    eb, sr, pn, hw = load()
    inv = inventory(eb)
    # the single-round headroom the paper reports at the operating point; a
    # hardware benchmark is one round, so this -- not the compounded R=40 value --
    # is what the instruction overhead has to be paid out of.
    delta_r1 = (pn['ad_p010']['VSCR warm (ours)']
                - pn['ad_p010']['Perfect-code decoder'])
    rows = envelope_rows(sr, inv['n_recovery'], delta_r1=delta_r1)
    dev = device_point(eb, rows, inv['n_recovery'])
    dev['verdict'] = verdict(dev)
    ro = readout_erosion(sr, inv['n_recovery'])
    # One Delta per channel, taken from the SAME source as the envelope rows
    # (storage_rounds' exact Haar closed form at R=1) so that ED Table 15a, 15b and
    # 15c cannot drift apart by the 3.4e-3 relative that separates the two
    # estimators; `_selftest_cross_artifact` is what ties that source to the
    # headroom vscr_paper.py reports.
    def _adv1(channel, p=0.10):
        for r in rows:
            if (r['code'] == '5,1,3' and r['channel'] == channel
                    and r['recovery'] == 'warm' and not r['degenerate']
                    and abs(float(r['p']) - p) < 1e-12 and r['advantage_R1']):
                return r['advantage_R1']
        return None
    deltas = {ch: d for ch, d in
              (('amplitude_damping', _adv1('amplitude_damping')),
               ('coherent', _adv1('coherent'))) if d}
    delta_r1 = deltas.get('amplitude_damping', delta_r1)
    # the per-eta break-even, scaled by the readout law, on the channel the
    # hardware protocol would have to use if it used a non-Pauli one
    for e in ro['eta_rows']:
        e['eps_instr_breakeven_scaled'] = (
            delta_r1 * e['headroom_factor'] / inv['n_recovery'])
    sb = shot_budget(hw, pn, deltas=deltas)
    st = run_selftests(eb, sr, pn, hw, inv, rows, dev, sb)
    return {
        'meta': {
            'task': 'operational envelope: instruction-error and shot budgets for '
                    'the certified non-Pauli headroom',
            'generated_utc': _dt.datetime.now(_dt.timezone.utc)
                             .strftime('%Y-%m-%dT%H:%M:%SZ'),
            'host': platform.node(),
            'python': sys.version.split()[0],
            'numpy': np.__version__,
            'inputs': {
                'instruction_inventory': 'hw_error_budget.json',
                'erosion_law': 'storage_rounds.json (recovery_noise)',
                'headroom': 'paper_numbers.json',
                'shot_counts': 'hw_feasibility_numbers.json',
            },
            'qpu_time_spent': 0,
            'note': 'recomputes no physics; every input is an existing artifact',
        },
        'model': {
            'strict_dictionary': 'lam = N_rec * eps: every instruction fault in the '
                                 'recovery section charged as a full-register '
                                 'depolarizing event, nothing corrected',
            'eps_half_R': 'ln2 / (R * N_rec), from the universal rule '
                          'lam_half ~ ln2/R',
            'eps_star': 'lam_star / N_rec, the sign-change budget',
            'eps_breakeven': 'Delta / N_rec, the arm-difference bound under a plain '
                             'independent-error model',
            'bracket': 'eps_breakeven is the strict end and eps_half(R=1) the loose '
                       'end; both are conservative, in opposite directions, and the '
                       'verdict is taken against the loose end',
            'not_claimed': 'a fault-tolerance threshold for the code, or a '
                           'circuit-level simulation of the compiled recovery under '
                           'a calibrated noise model',
        },
        'inventory': inv,
        'envelope': rows,
        'device': dev,
        'readout_erosion': ro,
        'shot_budget': sb,
        'selftests': st,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--selftest', action='store_true',
                    help='run the consistency self-tests and exit non-zero on '
                         'failure')
    ap.add_argument('--no-figure', action='store_true')
    ap.add_argument('--json', default=OUT)
    a = ap.parse_args(argv)

    art = build()
    st = art['selftests']

    if a.selftest:
        for c in st['checks']:
            print('%-4s %-52s got=%s tol=%s'
                  % ('OK' if c['passed'] else 'FAIL', c['check'],
                     str(c['got'])[:44], c['tol']))
        print('=== %d/%d passed ==='
              % (st['n_checks'] - st['n_failures'], st['n_checks']))
        return 0 if st['passed'] else 1

    with open(a.json, 'w') as f:
        json.dump(art, f, indent=1)
    print('wrote %s' % a.json)
    inv, dev, sb = art['inventory'], art['device'], art['shot_budget']
    print('  recovery section: %d instructions of %d in the compiled branch'
          % (inv['n_recovery'], inv['instructions']))
    for k in ('eps_instr_breakeven', 'eps_instr_half_R1', 'eps_instr_half_R40',
              'eps_instr_star'):
        vals = sorted(r[k] for r in art['envelope']
                      if not r['degenerate'] and r.get(k) is not None)
        if vals:
            print('  %-22s %.3e .. %.3e' % (k, vals[0], vals[-1]))
    print('  measured eps_instr (circuit D) %.4e' % dev['eps_instr_measured_D'])
    print('  factor above eps_breakeven    %.0f' % dev['factor_above_breakeven'])
    print('  factor above eps_half(R=1)    %.2f' % dev['factor_above_half_R1'])
    print('  factor above eps_half(R=40)   %.1f' % dev['factor_above_half_R40'])
    print('  verdict: %s' % dev['verdict'])
    print('  shots/arm for 3 sigma on Delta: %.3e (yield %.4f -> requested %.3e)'
          % (sb['shots_per_arm_3sigma'], sb['end_to_end_yield'],
             sb['shots_requested_if_yield_applies']))
    print('  executed: %d requested, %d post-selected, Wilson half-width %.4f '
          'against Delta=%.2e (ratio %.0f) -> %s'
          % (sb['executed_shots_requested'], sb['executed_post_selected'],
             sb['executed_wilson_halfwidth'], sb['delta_target'],
             sb['halfwidth_over_delta'], sb['verdict']))
    print('  selftests: %d/%d passed'
          % (st['n_checks'] - st['n_failures'], st['n_checks']))
    if not st['passed']:
        print('SELFTESTS FAILED', file=sys.stderr)
        return 1
    if not a.no_figure:
        make_figure(art)
    return 0


def make_figure(art):
    """Two panels: the envelope in instruction-error units, and the shot budget.

    Panel (a) is the whole argument on one axis -- the net advantage of the learned
    recovery over the Pauli decoder as a function of per-instruction error, for the
    two channels that have certified headroom, at R=1 and R=40, with the measured
    device error as a vertical line.  Panel (b) is the statistical half: the 95%
    half-width against shots, the signal it has to resolve, and where the executed
    run actually landed.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    inv, dev, sb = art['inventory'], art['device'], art['shot_budget']
    n_rec = inv['n_recovery']
    sel = [r for r in art['envelope']
           if not r['degenerate'] and r['recovery'] == 'warm'
           and r['code'] == '5,1,3' and abs(r['p'] - 0.10) < 1e-12]
    colors = {'amplitude_damping': '#1f77b4', 'coherent': '#d62728'}

    eps = np.logspace(-6, -1.2, 400)
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.0))
    for r in sel:
        c = colors.get(r['channel'], '#333333')
        lbl = r['channel'].replace('_', ' ')
        for R, ls in ((1, '-'), (r['R_last'], '--')):
            adv0 = r['advantage_R1'] if R == 1 else r['advantage_lam0']
            if not adv0:
                continue
            ax[0].plot(eps, adv0 * np.exp(-n_rec * eps * R), ls, color=c, lw=1.6,
                       label=r'%s, $R=%d$' % (lbl, R))
    ax[0].axhline(0.0, color='k', lw=0.6)
    ax[0].axvline(dev['eps_instr_measured_D'], color='0.35', lw=1.4, ls=':')
    cands = ([r['advantage_lam0'] for r in sel]
             + [r['advantage_R1'] for r in sel if r['advantage_R1']])
    top = max(cands) * 1.15 if cands else 1.0
    ax[0].set_ylim(-0.15 * top, top)
    ax[0].annotate(r'measured $1.04\times10^{-2}$',
                   xy=(dev['eps_instr_measured_D'], 0.45 * top),
                   xytext=(dev['eps_instr_measured_D'] * 0.06, 0.55 * top),
                   fontsize=8, color='0.25',
                   arrowprops=dict(arrowstyle='->', color='0.35', lw=0.9))
    ax[0].set_xscale('log')
    ax[0].set_xlabel(r'per-instruction error $\varepsilon_{\rm instr}$')
    ax[0].set_ylabel('net advantage over the Pauli decoder')
    ax[0].set_title(r'(a) operational envelope, $N_{\rm rec}=%d$ instructions'
                    % n_rec, fontsize=10)
    ax[0].legend(fontsize=7.5, loc='lower left')
    ax[0].grid(alpha=0.25, which='both', lw=0.5)

    shots = np.logspace(2, 9, 300)
    ax[1].loglog(shots, Z95 * 0.5 / np.sqrt(shots), color='#1f77b4', lw=1.6,
                 label=r'95% half-width, $v=1/4$')
    _dm, _de = ('%.1e' % sb['delta_target']).split('e')
    ax[1].axhline(sb['delta_target'], color='#d62728', lw=1.4, ls='--',
                  label=r'signal $\Delta=%s\times10^{%d}$' % (_dm, int(_de)))
    n_ex = sb['executed_branch_D']['n_sel']
    ax[1].plot([n_ex], [sb['executed_wilson_halfwidth']], 'ko', ms=6,
               label='executed run (%d shots)' % n_ex)
    ax[1].axvline(sb['shots_per_arm_3sigma'], color='0.5', lw=1.0, ls=':')
    ax[1].set_xlabel('shots per arm')
    ax[1].set_ylabel('statistical half-width')
    ax[1].set_title('(b) shot budget', fontsize=10)
    ax[1].legend(fontsize=7.5, loc='lower left')
    ax[1].grid(alpha=0.25, which='both', lw=0.5)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(FIG + '.' + ext, dpi=200)
    plt.close(fig)
    print('wrote %s.{pdf,png}' % FIG)


if __name__ == '__main__':
    raise SystemExit(main())







