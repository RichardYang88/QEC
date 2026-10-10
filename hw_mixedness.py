"""
hw_mixedness.py -- MEASURE, on a real superconducting QPU, the mixedness of the
logical state the device actually delivers, and test on hardware the one half of
the mixed-input boundary that hardware can resolve.

Why this file exists
--------------------
`mixed_input_boundary.device_prep` prices the quadratic training loss's over-read
at a device Bloch radius r = 0.9375 with 39% of the encoded weight outside the
code space.  Both numbers are DERIVED, not measured: they come from feeding the
mean per-instruction error that `hw_error_budget.py` implies from the WK_C180
feasibility counts through the compiled 36-gate encoder as an isotropic
depolarizing channel.  The manuscript says so, and ED Table 10d is labelled
"model-implied" for exactly that reason.  This script replaces the label with a
measurement, on the device's own noise and with no injected error.

What is measured (plan: QCLOUD_SETUP.md section 7)
--------------------------------------------------
M1  device mixedness.  A 5-qubit tomography circuit
        prep(psi) on q4 -> U_enc (36) -> U_enc^dag (36) -> R_B on q4 -> MEAS q0..q4
    is run for psi in {0, 1, +, -} and B in {X, Y, Z}.  U_enc^dag maps the code
    space back onto |0000> (x) q4, so "q0..q3 all zero" IS the code-space
    projector V V^dag that device_prep applies analytically, and q4 carries the
    logical state.  Hence, from the same counts:
        w   = P(q0..q3 = 0000)                      code-space weight
        <s_B>_cs = P(q4=0|cs) - P(q4=1|cs)          conditioned Bloch component
        r   = |(<s_X>, <s_Y>, <s_Z>)|               Bloch radius, purity (1+r^2)/2
    Every one of these is reported raw AND after readout mitigation with an
    assignment matrix measured by this run itself (M4), because the vendor
    calibration API returns 401 for this account and the manuscript records that
    refusal verbatim.

M2  the LINEAR score decomposes exactly.  rho_in = I/2 has two different pure
    ensembles, {|0>,|1>} and {|+>,|->}.  The unconditional logical <Z_L> is a
    LINEAR functional of rho_in, so
        L_Z = (<Z_L>_|0> + <Z_L>_|1>)/2 = Tr[Z_L M(I/2)] = L_X
    holds for ANY channel M -- exactly, not approximately.  The individual terms
    span ~1.7 while the two averages must agree, so this is a hardware statement
    with real content, and it is the half of `mixed_input_boundary.selftests`
    item (d) that licences the existing population / <Z_L> benchmarks.  The
    conditional (post-selected) version is NOT linear and is reported only as a
    leakage-asymmetry diagnostic, never as the M2 claim.

M3  the isotropic-depolarizing surrogate is tested at the hardware's own depth.
    `model_mixedness` re-implements device_prep's model on an ARBITRARY compiled
    gate list (independently, as a 32x32 density matrix, so the audit can
    re-derive it).  One parameter, p_instr, is fitted to the MEASURED code-space
    weight; the model then PREDICTS r with nothing left to fit.  Agreement inside
    2 sigma supports the surrogate; disagreement is reported as a measurement of
    where it fails, which is also a result and is consistent with the manuscript
    already calling r an order of magnitude.

M4  the 5-bit readout assignment matrix, measured by this run: e01 from |00000>,
    e10 from X^(x5).  Bracketed by a repeat of the |00000> circuit so the drift
    is a recorded systematic rather than an assumption.

M5  depth scan.  The same circuit with rounds=2 (enc,dec,enc,dec = 150
    instructions, 72 CNOT) separates a PER-GATE loss law (one p_instr explains
    both depths) from a PER-TIME one (it does not).  Two of its four circuits are
    null controls: <X> for psi=|0> and <Z> for psi=|+> must vanish at any depth,
    so a significant value is evidence of a phase/calibration problem and not of
    mixedness.

What this run deliberately does NOT measure
-------------------------------------------
The over-read delta^T C delta itself.  At the predicted radius it is
(1-r^2)(F_Haar - c0) ~ 0.007 on the depolarizing channel, while a branch-summed
hardware fidelity has sigma(F) ~ sqrt(F(1-F)/N_eff) with N_eff ~ 0.15 x shots on
this device's syndrome readout, i.e. sigma(F) ~ 0.033 at 1000 shots and
sigma(F^2) ~ 0.05 -- seven times the effect.  Resolving 0.001 would need
N_eff ~ 4e5 per branch, ~5e7 shots in total.  So the run measures the RADIUS that
prices the over-read, and the linear exactness; the over-read number itself stays
with the analytic result that is audited to 1e-14.  This is the same shot-budget
bound ED Table 3's note already states.

Stages (4 cloud jobs, priority order, submitted ONE AT A TIME, each dumped
incrementally; the run plan and the abort rules are QCLOUD_SETUP.md 8)
----------------------------------------------------------------
    probe  :  8 circuits x  300 shots, 5 qubits -- 2+1 readout calibration (the
              repeat brackets the drift) and 5 single-X fingerprints that determine
              the returned bit order uniquely;  2 400 shots
    tomo   : 19 circuits x 8000 shots, 5 qubits -- M1 M2 M3 M5, plus its OWN three
              calibration circuits so it never assumes the canary's block is still
              its own;  152 000 shots
    bench  : 34 circuits x 2000 shots, 9 qubits -- mid/late A-B, then the full
              16-branch sum at p=0 for psi in {0,+} (2 x 16), late-measure, on the
              paper's warm-start angles;  68 000 shots
    ladder :  4 circuits x 4000 shots, 5 qubits -- the DEPTH_SCAN grid at rounds=4
              (293 instructions / 144 CNOT), analysed together with tomo as one
              depth curve;  16 000 shots
Total 65 circuits, 238 400 shots.

The metering unit is NOT documented by the vendor and the two candidate models
differ ~50x in what a 300 s allocation buys: metered per shot, 300 s is ~21 000
shots and this plan is 11x over budget; metered per job wall-clock (queue included),
shots are nearly free and the FOUR-JOB structure is the budget.  Nothing in the
artifacts decides between them -- the 2026-09 run executed ~162 000 shots and then
ran out, but the size of that budget in seconds was never recorded.  So `probe` is
a canary sized to cost ~34 s even under the pessimistic model, its wall clock picks
the shot count for every later job (QCLOUD_SETUP.md 8.4), and a QPU-time refusal
halves --shots and retries down to SHOTS_FLOOR rather than losing the run, with
every attempt recorded.  A refusal is free, so the halving ladder is itself a
measurement of the cost model.  `--mode cloud --stage all` is therefore rejected:
the abort rules are a human decision, not something to run unattended.

Shot sizing is not a guess: `--mode simulate` runs the whole chain on synthetic
counts from a device built out of quoted vendor fidelities, through the SAME
analysis path the cloud data takes (including the tomo job's own calibration), so
the achievable error bars are known before any shots are bought.  At 8000 shots on
a 0.9993/0.9831/0.9612 device it recovers w = 0.3415 (truth 0.3396) and r =
0.83-0.87 (truth 0.854-0.881) with a 95% CI half-width of 0.0278 on r, every null
control inside 1 sigma, and |L_Z - L_X| bounded to 4.7% of the term span; over 6
sampling seeds the M2 z-score has mean +0.50 and rms 1.21, i.e. pure shot noise.
The same study on 0.9988/0.9790/0.9260 gives a CI half-width of 0.0489 and needs
47 918 shots for +-0.02 -- 3.1x more -- which is the arithmetic behind the backend
choice.  Halving the shots widens the CI on r by sqrt(2); that is the whole
trade-off, it is a CLI flag, and `--mode plan` prints the table.

Run:
    ./qenv/bin/python hw_mixedness.py --mode plan         # the budget, no account
    ./qenv/bin/python hw_mixedness.py --mode verify       # exact, no account
    ./qenv/bin/python hw_mixedness.py --mode selftest     # the 9 anchors, no account
    ./qenv/bin/python hw_mixedness.py --mode sample       # CPUQVM dry run of the
                                                          # whole analysis chain
    ./qenv/bin/python hw_mixedness.py --mode simulate \
        --vendor 0.9993,0.9831,0.9612 --shots 8000 --seeds 6   # power study
    ./qenv/bin/python hw_mixedness.py --mode preflight \
        --chip-ids WK_C180_2,WK_C180                    # FREE cloud queries
    bash launch_mixedness.sh probe    # then read wall_s and apply the abort rules
    bash launch_mixedness.sh tomo
    bash launch_mixedness.sh bench
    bash launch_mixedness.sh ladder
    ./qenv/bin/python hw_mixedness.py --mode analyze --stage all   # offline, no token
Outputs:
    hw_mixedness.json, qcloud_raw_{probe,mixedness,bench}.json,
    figures/fig_hw_mixedness_<tag>.png
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import qcloud_vscr_new as q                                       # noqa: E402

NQ5 = 5
DIM5 = 2 ** NQ5
PAULI5 = [q._gate_matrix(('X', 0)), q._gate_matrix(('Y', 0)),
          q._gate_matrix(('Z', 0))]

# tomography grid: 4 inputs x 3 bases = 12 circuits, plus the depth-2 scan
TOMO_STATES = ('0', '1', '+', '-')
BASES = ('X', 'Y', 'Z')
BASIS_ROT = {'Z': [], 'X': [('H', 4)],
             'Y': [('RZ', 4, -math.pi / 2), ('H', 4)]}
# (psi, basis) pairs run at rounds=2.  The first two are the SIGNAL (for psi=0 the
# Bloch vector is along z so <Z> alone gives r; for psi=+ it is along x so <X>
# does), the last two are NULL CONTROLS that must vanish at any depth.
DEPTH_SCAN = (('0', 'Z'), ('+', 'X'), ('0', 'X'), ('+', 'Z'))
IDEAL_BLOCH = {'0': (0.0, 0.0, 1.0), '1': (0.0, 0.0, -1.0),
               '+': (1.0, 0.0, 0.0), '-': (-1.0, 0.0, 0.0),
               '+i': (0.0, 1.0, 0.0), '-i': (0.0, -1.0, 0.0)}
BASIS_AXIS = {'X': 0, 'Y': 1, 'Z': 2}   # which Bloch component basis B reads

STAGE_SHOTS = {'probe': 300, 'tomo': 8000, 'bench': 2000, 'ladder': 4000}
# QPU time on this platform is metered in a unit the vendor does not document, and
# the 2026-09 run ran out mid-sweep (see QCLOUD_SETUP.md 6.2).  So a refusal is
# treated as information rather than as failure: the stage is retried at half the
# shots, down to SHOTS_FLOOR, and every attempt is recorded in the artifact.
#
# The probe default is deliberately 300, not 1500: it is the CANARY whose only job
# is to measure the cost model (QCLOUD_SETUP.md 8.1), and the two candidate models
# differ by ~50x in what 300 s buys.  At the pessimistic 14 ms/shot implied by the
# 2026-09 timeline, 8 circuits x 300 shots = 2400 shots costs ~34 s, i.e. ~11% of
# the budget, while still returning an assignment matrix whose e_ij are good to
# +-0.011 (at e~0.04), a unique bit order, and the platform's shots shortfall.  The
# tomo stage then repeats the three calibration circuits at its own (much higher)
# shot count, so the published assignment matrix is not the canary's.
#
# The probe stays on FIVE qubits and therefore cannot carry the mid/late A/B pair:
# a submission has one block, and on a 9-qubit block the returned bitstrings are
# 9 bits wide, which would silently invalidate the 5-position layout likelihood and
# the 5x5 assignment matrix.  Whether a 5-qubit circuit submitted inside a 9-qubit
# block comes back 5 or 9 bits wide is NOT verifiable offline, so the plan never
# depends on it.  The A/B pair is circuits 1-2 of the bench stage instead.
SHOTS_FLOOR = 1000
QPU_REFUSAL_MARKERS = ('qpu time', 'insufficient', 'not enough', 'quota')
# seeds for the power study's repeat draws; the first is the one every other
# function defaults to, so a single-seed run and a scan share their first draw
SEED_BANK = (20260921, 11, 202, 3003, 40004, 777777, 5, 606, 8080, 91,
             1234, 24680, 13579, 99991, 424242, 8888)
STAGE_NQ = {'probe': NQ5, 'tomo': NQ5, 'bench': q.NQ, 'ladder': NQ5}
STAGE_RAW = {'probe': 'qcloud_raw_probe.json',
             'tomo': 'qcloud_raw_mixedness.json',
             'bench': 'qcloud_raw_bench.json',
             'ladder': 'qcloud_raw_ladder.json'}
# The ladder stage re-runs the DEPTH_SCAN grid at deeper rounds, so the measured
# radius becomes a curve in depth rather than two points.  rounds=4 is 293
# instructions / 144 CNOT and rounds=8 is 581 / 288 (both verified constructible);
# 8 is off by default because it is the most expensive circuit in the whole plan
# and the least essential to any of M1-M5.
LADDER_ROUNDS = (4,)
# Every stage is submitted on ONE fixed physical block, passed with --block, so the
# readout assignment matrix measured by the canary applies to the tomography.  The
# 5-qubit stages and the 9-qubit bench necessarily use different blocks (a
# submission has one block), and which two were used is recorded in the artifact
# rather than assumed to be comparable.
STAGE_JOB = {'probe': 1, 'tomo': 2, 'bench': 3, 'ladder': 4}
# the bench stage reuses the frame the WK_C180 feasibility run used, so the new
# full-branch point and the old 4-circuit point are the same physical situation
BENCH_FRAME = 'IIIII'
BENCH_S_INJECTED = 12
BENCH_FRAME_INJECTED = 'IIXII'
OUT_JSON = os.path.join(ROOT, 'hw_mixedness.json')
LAYOUT_FLOOR = 1e-9        # floor on ideal probabilities in the layout likelihood


# ---------------------------------------------------------------------
# 1) circuits
# ---------------------------------------------------------------------
def setup():
    """Synthesise and phase-verify the encoder once (delegates to qcloud_vscr_new)."""
    return q.setup()


def enc_dec_gates(rounds=1):
    """U_enc then U_enc^dag, `rounds` times: 72 unitary instructions per round."""
    enc = q.ENC5
    dec = q.invert_gates(enc)
    out = []
    for _ in range(rounds):
        out += list(enc) + list(dec)
    return out


def build_tomo(psi, basis, rounds=1):
    """The M1 tomography circuit: prep(psi) -> (enc,dec)^rounds -> R_B -> MEAS q0..q4.

    Deliberately does NOT apply TEST_STATES' validation rotation: `val` maps the
    ideal output to |0>, which would erase the sign that M2's <Z_L> needs."""
    prep = q.TEST_STATES[psi][0]
    gates = list(prep) + enc_dec_gates(rounds) + list(BASIS_ROT[basis])
    gates += [('MEAS', i, i) for i in range(NQ5)]
    return gates


def build_cal(kind):
    """Readout-assignment circuits: 'zero' = |00000>, 'one' = X^(x5)|00000>."""
    gates = [('X', i) for i in range(NQ5)] if kind == 'one' else []
    return gates + [('MEAS', i, i) for i in range(NQ5)]


def build_fingerprint(b):
    """X on qubit b only, then measure: an ideal outcome with a single '1', whose
    observed position identifies b.

    These five circuits exist because of a real degeneracy the self-tests caught.
    Every tomography circuit ideally ends in |0000> (x) q4, so its ideal
    distribution is supported on just {'00000','00001'}: the likelihood pins WHICH
    observed position carries q4, and says nothing at all about how q0..q3 are
    arranged among the remaining four.  That is harmless for every estimator in
    this file -- the code-space condition is "the other four are all zero", which
    is permutation invariant, and readout mitigation is applied per observed
    position -- but it is not a determined layout, and an undetermined layout
    should not be written into an artifact as if it were one.  Five depth-2
    circuits remove the ambiguity outright."""
    return [('X', b)] + [('MEAS', i, i) for i in range(NQ5)]


def build_bench_ab(phi):
    """The mid/late A-B pair: the exact circuit whose mid-circuit measurement
    returned 0 hits out of 735 on WK_C180, rebuilt for this chip."""
    out, meta = [], []
    for kind, late in (('mid', False), ('late', True)):
        out.append(q.build_branch_circuit(q.ENC5, '0', BENCH_FRAME_INJECTED,
                                          BENCH_S_INJECTED, phi, late_measure=late))
        meta.append(dict(kind='ab', measure=kind, psi='0', p=0.0,
                         frame=BENCH_FRAME_INJECTED, s=BENCH_S_INJECTED))
    return out, meta


def build_bench(phi, states=('0', '+')):
    """Full 16-branch sum at p=0 (no injected error) for each state, late-measure.

    p=0 rather than the paper's 0.02/0.08 sweep because (i) it is deterministic --
    a sampled Pauli frame depends on RNG call order -- (ii) it pairs with the tomo
    stage, whose mixedness is also the device's own, and (iii) the ideal reference
    is then exactly `simulator_fidelity_frame(psi, 'IIIII', phi)`, so the gap is
    pure device noise."""
    out, meta = [], []
    for psi in states:
        for s in range(16):
            out.append(q.build_branch_circuit(q.ENC5, psi, BENCH_FRAME, s, phi,
                                              late_measure=True))
            meta.append(dict(kind='bench', measure='late', psi=psi, p=0.0,
                             frame=BENCH_FRAME, s=s))
    return out, meta


def stage_circuits(stage, phi=None):
    """(gate_lists, meta) for one cloud job.  meta entries are plain dicts so they
    survive the JSON dump and can be rebuilt identically in --mode analyze."""
    if stage == 'probe':
        out = [build_cal('zero'), build_cal('one'), build_cal('zero')]
        meta = [dict(kind='cal', cal='zero', rep=0), dict(kind='cal', cal='one', rep=0),
                dict(kind='cal', cal='zero', rep=1)]
        for b in range(NQ5):
            out.append(build_fingerprint(b))
            meta.append(dict(kind='fingerprint', qubit=b))
        return out, meta
    if stage == 'tomo':
        out, meta = [], []
        for r in (1, 2):
            pairs = ([(p, b) for p in TOMO_STATES for b in BASES] if r == 1
                     else list(DEPTH_SCAN))
            for psi, basis in pairs:
                out.append(build_tomo(psi, basis, rounds=r))
                meta.append(dict(kind='tomo', psi=psi, basis=basis, rounds=r))
        # The tomography job carries its OWN three calibration circuits.  A
        # submission has one block, and the block is whatever the platform ranks
        # best at submission time, so a job that depends on a calibration measured
        # in an earlier job silently assumes the two landed on the same physical
        # qubits.  Carrying them costs three 5-10 instruction circuits and makes
        # the headline M1/M2/M3 numbers self-contained on whichever chip ran them.
        out += [build_cal('zero'), build_cal('one'), build_cal('zero')]
        meta += [dict(kind='cal', cal='zero', rep=0), dict(kind='cal', cal='one', rep=0),
                 dict(kind='cal', cal='zero', rep=1)]
        return out, meta
    if stage == 'ladder':
        # The DEPTH_SCAN grid at deeper rounds: for psi=|0> the Bloch vector lies
        # along z and for psi=|+> along x, so (0,Z) and (+,X) are the radius at any
        # depth, while (0,X) and (+,Z) remain null controls at any depth.  Same
        # meta kind as the tomo stage, so --stage all merges them into one depth
        # curve and analyze_tomo's loops pick up every rounds value present.
        out, meta = [], []
        for r in LADDER_ROUNDS:
            for psi, basis in DEPTH_SCAN:
                out.append(build_tomo(psi, basis, rounds=r))
                meta.append(dict(kind='tomo', psi=psi, basis=basis, rounds=r))
        return out, meta
    if stage == 'bench':
        assert phi is not None, 'the bench stage needs the trained angles'
        g1, m1 = build_bench_ab(phi)
        g2, m2 = build_bench(phi)
        return g1 + g2, m1 + m2
    raise ValueError(stage)


# ---------------------------------------------------------------------
# 2) exact ideal references (no account, no sampling)
# ---------------------------------------------------------------------
def canonical_key(idx, n=NQ5):
    """Basis index -> bitstring with char j = qubit j.  In qcloud_vscr_new's kron
    order qubit 0 is the MOST significant bit, so qubit j sits at n-1-j."""
    return ''.join(str((idx >> (n - 1 - j)) & 1) for j in range(n))


def unitary_of(gates, n=NQ5):
    """Exact unitary of a gate list, measurements dropped."""
    u = [g for g in gates if g[0] != 'MEAS']
    return q.gates_to_unitary(u, n=n)


def exact_probs(gates, n=NQ5):
    """Ideal output distribution of `gates` applied to |0...0>, as a dict keyed by
    the CANONICAL bitstring (char j = qubit j)."""
    U = unitary_of(gates, n)
    p = np.abs(U[:, 0]) ** 2
    return {canonical_key(i, n): float(p[i]) for i in range(2 ** n)}


def ideal_bloch_of(gates, n=NQ5):
    """(code-space weight, Bloch vector) of the exact output state, read the way
    the hardware estimator reads it: project on q0..q3 = 0000, renormalise, then
    take the three Pauli expectations of q4.  Used to check that the estimator and
    the state-vector agree before either is trusted with real counts."""
    U = unitary_of(gates, n)
    psi = U[:, 0]
    P = np.abs(psi) ** 2
    cs = [i for i in range(2 ** n) if (i >> 1) & 15 == 0]     # q0..q3 == 0
    w = float(sum(P[i] for i in cs))
    rl = np.array([[psi[cs[a]] * np.conj(psi[cs[b]]) for b in (0, 1)]
                   for a in (0, 1)])
    rl = rl / w
    nvec = np.array([float(np.trace(Pm @ rl).real) for Pm in PAULI5])
    return w, nvec



# ---------------------------------------------------------------------
# 3) bitstring layout, detected by maximum likelihood against the exact ideal
#    distributions.  The old qcloud_vscr_new.detect_layout scored four hardcoded
#    hypotheses by data-hit count, which is enough for a 9-cbit circuit whose
#    ideal outcome is a single key.  A 5-cbit tomography circuit has a richer
#    ideal distribution, so the layout can be pinned over ALL 5! = 120
#    assignments of observed character -> qubit, with a likelihood margin that
#    goes into the artifact the way ED Table 4's evidence table does.
# ---------------------------------------------------------------------
def str_to_idx(key):
    return int(key, 2)


def idx_to_str(i, n=NQ5):
    return format(i, '0%db' % n)


def _remap(key, sigma):
    """Observed string -> canonical string (char b = qubit b) under the hypothesis
    that observed character j came from qubit sigma[j]."""
    out = ['0'] * len(key)
    for j, ch in enumerate(key):
        out[sigma[j]] = ch
    return ''.join(out)


def layout_hypotheses(n=NQ5):
    return [tuple(s) for s in itertools.permutations(range(n))]


def detect_layout_ml(results, ideals, n=NQ5, top=5, tie_tol=1e-6, verbose=True):
    """Score every permutation by the multinomial log-likelihood of the observed
    counts under the exact ideal distributions.

    `ideals[i]` is exact_probs() of circuit i, keyed by the CANONICAL string.
    Returns (sigma, evidence).

    Ties are reported, not silently broken.  On the tomography circuits alone the
    ideal distributions cannot separate the four code-space positions (see
    build_fingerprint), so the tie set is expected to have 4! = 24 members there;
    what MUST be determined is the position of the logical qubit, because that is
    the only thing any estimator in this file actually reads.  `evidence` records
    whether the tie set determines it, and the function raises if it does not, so
    an ambiguous layout can never reach the artifact looking like a resolved one.
    The representative returned is the lexicographically smallest member of the tie
    set, which makes the choice reproducible rather than arbitrary."""
    scores = []
    for sigma in layout_hypotheses(n):
        ll = 0.0
        for counts, ideal in zip(results, ideals):
            for key, cnt in counts.items():
                if cnt <= 0 or len(key) != n:
                    continue
                ll += cnt * math.log(max(ideal.get(_remap(key, sigma), 0.0),
                                         LAYOUT_FLOOR))
        scores.append((ll, sigma))
    scores.sort(key=lambda t: (-t[0], t[1]))
    best_ll, _ = scores[0]
    tie = [sg for ll, sg in scores if best_ll - ll <= tie_tol]
    rest = [t for t in scores if t[1] not in tie]
    runner_ll, runner = rest[0] if rest else (float('-inf'), None)
    pos_of = {sg: [j for j in range(n) if sg[j] == n - 1][0] for sg in tie}
    determines_logical = len(set(pos_of.values())) == 1
    determines_codespace = len({frozenset(j for j in range(n)
                                          if sg[j] != n - 1) for sg in tie}) == 1
    if not determines_logical:
        raise RuntimeError(
            'the ideal distributions do not determine which observed position '
            'carries the logical qubit: %d tied hypotheses disagree on it (%s). '
            'Add build_fingerprint() circuits to this job.'
            % (len(tie), sorted(set(pos_of.values()))))
    best = min(tie)
    ev = dict(n_hypotheses=len(scores), best_loglik=best_ll,
              best_sigma=list(best), logical_position=pos_of[best],
              n_tied=len(tie), tie_set=[list(s) for s in sorted(tie)],
              tie_determines_logical=bool(determines_logical),
              tie_determines_codespace=bool(determines_codespace),
              tie_tol=tie_tol,
              runner_up_loglik=runner_ll,
              runner_up_sigma=(list(runner) if runner is not None else None),
              margin=(best_ll - runner_ll) if rest else None,
              top=[dict(loglik=s, sigma=list(sg)) for s, sg in scores[:top]])
    if verbose:
        print('[layout] sigma=%s loglik=%.1f  %d tied hypotheses (all agree the '
              'logical qubit is at observed position %d)  margin to the rest=%s'
              % (list(best), best_ll, len(tie), pos_of[best],
                 ('%.1f' % ev['margin']) if rest else 'n/a'), flush=True)
    return best, ev


# ---------------------------------------------------------------------
# 4) readout assignment matrix, measured by this run, and its mitigation
# ---------------------------------------------------------------------
def assignment_from_cal(counts_zero, counts_one, n=NQ5):
    """Per-OBSERVED-POSITION flip rates: e01[j] = P(char j reads 1 | true 0),
    e10[j] = P(char j reads 0 | true 1).  Position-based on purpose: mitigation
    needs no layout, only the later interpretation of positions does."""
    def marg(counts, val):
        tot = float(sum(counts.values()))
        return [sum(c for key, c in counts.items()
                    if len(key) == n and key[j] == val) / tot if tot > 0
                else float('nan') for j in range(n)]
    return dict(e01=marg(counts_zero, '1'), e10=marg(counts_one, '0'),
                n_zero=int(sum(counts_zero.values())),
                n_one=int(sum(counts_one.values())))


def assignment_matrix(e01, e10, n=NQ5):
    """The 2^n x 2^n independent-bit assignment matrix A with p_obs = A p_true.
    Kronecker order matches int(key, 2): position j is bit n-1-j."""
    A = np.array([[1.0]])
    for j in range(n):
        A = np.kron(A, np.array([[1.0 - e01[j], e10[j]],
                                 [e01[j], 1.0 - e10[j]]]))
    return A


def mitigate(probs_vec, e01, e10, n=NQ5, clip=True):
    """Invert the assignment matrix on a probability vector indexed by int(key, 2).

    Returns (p_true, diagnostics).  Linear inversion of a measured assignment
    matrix can leave the simplex; that is reported rather than hidden, and the
    projection `clip` applies is named in the diagnostics.  Kept for the
    distribution-level diagnostic -- every ESTIMATOR in this file goes through
    `observable_weights` instead, because clipping a noisy inversion is biased and
    the bias shows up as a spurious several-sigma violation of M2."""
    A = assignment_matrix(e01, e10, n)
    b = np.asarray(probs_vec, dtype=float)
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    diag = dict(negativity_before_clip=float(-min(0.0, x.min())),
                max_refit_residual=float(np.abs(A @ x - b).max()),
                clipped=bool(clip), sum_before_clip=float(x.sum()))
    if clip:
        x = np.clip(x, 0.0, None)
        s = x.sum()
        if s > 0:
            x = x / s
    return x, diag


def observable_weights(O_true, e01, e10, n=NQ5):
    """The observed-side weight vector of a diagonal observable: O~ = A^{-T} O.

    Then <O>_true = sum_j O~[j] p_obs[j] = the sample mean of O~ over the shots,
    which is EXACTLY unbiased for any per-bit readout channel, needs no inversion of
    a distribution, and never needs clipping.  Its sampling variance follows from
    the same weights, so the mitigated numbers get proper standard errors instead of
    borrowing the raw ones."""
    A = assignment_matrix(e01, e10, n)
    return np.linalg.solve(A.T, np.asarray(O_true, dtype=float))


def true_bit_masks(sigma, n=NQ5):
    """For each qubit b, the bitmask of ITS TRUE value inside a position-space index
    (the index int(key, 2) of a string whose character j came from qubit sigma[j])."""
    return [1 << (n - 1 - j) for j, b in sorted(
        [(j, b) for j, b in enumerate(sigma)], key=lambda t: t[1])]


def observables_for(sigma, n=NQ5):
    """The four true-side diagonal observables this experiment needs, as vectors over
    position-space indices: the code-space indicator, the logical Z, their product
    (the numerator of the conditioned component), and the all-ones trace."""
    masks = true_bit_masks(sigma, n)
    cs_mask = 0
    for b in range(n - 1):
        cs_mask |= masks[b]
    idx = np.arange(2 ** n)
    o_cs = ((idx & cs_mask) == 0).astype(float)
    o_z = np.where((idx & masks[n - 1]) == 0, 1.0, -1.0)
    return dict(cs=o_cs, z=o_z, num=o_z * o_cs, one=np.ones(2 ** n))


def expectation(counts_vec, weights):
    """(mean, standard error) of a mitigated observable from a count vector.

    The standard error is the exact i.i.d. sampling error of the sample mean of the
    weights, sqrt((<O~^2> - <O~>^2)/N) -- no binomial approximation, and valid for
    the conditioned ratio's numerator and denominator alike."""
    N = float(counts_vec.sum())
    if N <= 0:
        return float('nan'), float('nan')
    p = counts_vec / N
    m = float(weights @ p)
    var = float((weights ** 2) @ p) - m * m
    return m, math.sqrt(max(var, 0.0) / N)


def wilson(k, n, z=1.96):
    """Wilson score interval -- the same estimator ED Table 3 and the abstract use
    for the hardware branch fidelity."""
    if n <= 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1.0 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0))
    return ((c - h) / d, (c + h) / d)


# ---------------------------------------------------------------------
# 5) estimators: code-space weight, Bloch components, radius
# ---------------------------------------------------------------------
def qubit_masks(sigma, n=NQ5):
    """For a layout hypothesis sigma (observed char j came from qubit sigma[j]),
    the bitmask of each qubit inside an index parsed by int(key, 2)."""
    inv = {b: j for j, b in enumerate(sigma)}
    return [1 << (n - 1 - inv[b]) for b in range(n)]


def cs_split(sigma, n=NQ5):
    """(code-space index list, q4 bitmask).  The code space after decoding is
    'q0..q3 all zero', which IS the projector V V^dag that device_prep applies."""
    masks = qubit_masks(sigma, n)
    cs_mask = 0
    for b in range(n - 1):
        cs_mask |= masks[b]
    cs = [i for i in range(2 ** n) if (i & cs_mask) == 0]
    return cs, masks[n - 1]


def counts_to_vec(counts, n=NQ5):
    v = np.zeros(2 ** n)
    for key, c in counts.items():
        if len(key) == n:
            v[str_to_idx(key)] += float(c)
    return v


def _weights(O, e01, e10, n, mitigated):
    if not mitigated:
        return {k: O[k] for k in O}
    return {k: observable_weights(O[k], e01, e10, n) for k in O}


def _ratio_stats(v, W):
    """(w, num, sigma_cs) with exact sampling errors, including the correlation
    between the numerator and the post-selection denominator -- they come from the
    same shots, and dropping that covariance understates the error on the ratio."""
    N = float(v.sum())
    p = v / N if N > 0 else v
    w = float(W['cs'] @ p)
    num = float(W['num'] @ p)
    var_w = max(float((W['cs'] ** 2) @ p) - w * w, 0.0) / max(N, 1)
    var_n = max(float((W['num'] ** 2) @ p) - num * num, 0.0) / max(N, 1)
    cov = (float((W['num'] * W['cs']) @ p) - num * w) / max(N, 1)
    if w > 1e-12:
        s = num / w
        var_s = (var_n / w ** 2 - 2 * num * cov / w ** 3
                 + num ** 2 * var_w / w ** 4)
        se = math.sqrt(max(var_s, 0.0))
    else:
        s, se = float('nan'), float('nan')
    z = float(W['z'] @ p)
    se_z = math.sqrt(max(float((W['z'] ** 2) @ p) - z * z, 0.0) / max(N, 1))
    return dict(w=w, w_stderr=math.sqrt(var_w), num=num, sigma_cs=s,
                sigma_cs_stderr=se, sigma_uncond=z, sigma_uncond_stderr=se_z,
                norm_check=float(W['one'] @ p))


def read_tomo(counts, sigma, n=NQ5, e01=None, e10=None):
    """Everything one tomography circuit yields, raw and readout-mitigated.

    Both variants are sample means of a diagonal observable, so both are unbiased
    and both carry their own exact sampling error; the mitigated one uses
    O~ = A^{-T} O rather than an inverted-and-clipped distribution (see
    observable_weights).  `sigma_cs` is the conditioned Bloch component that M1
    reads; `sigma_uncond` is the UNCONDITIONAL logical <Z_L> that M2 needs, which
    must not be post-selected because post-selection is not a linear functional of
    the input state."""
    O = observables_for(sigma, n)
    v = counts_to_vec(counts, n)
    out = dict(n_used=int(v.sum()), n_requested=None)
    mit = e01 is not None and e10 is not None
    out['raw'] = _ratio_stats(v, _weights(O, e01, e10, n, False))
    k = int(round(out['raw']['w'] * out['n_used']))
    out['raw']['w_wilson95'] = list(wilson(k, out['n_used']))
    out['raw']['w_ci95'] = [out['raw']['w'] - 1.96 * out['raw']['w_stderr'],
                            out['raw']['w'] + 1.96 * out['raw']['w_stderr']]
    if mit:
        out['mitigated'] = _ratio_stats(v, _weights(O, e01, e10, n, True))
        out['mitigated']['w_wilson95'] = None
        out['mitigated']['w_ci95'] = [
            out['mitigated']['w'] - 1.96 * out['mitigated']['w_stderr'],
            out['mitigated']['w'] + 1.96 * out['mitigated']['w_stderr']]
        p = v / v.sum() if v.sum() > 0 else v
        _pm, diag = mitigate(p, e01, e10, n)
        out['mitigation_diagnostics'] = diag
    return out


def bootstrap_radius(counts_by_basis, sigma, n=NQ5, e01=None, e10=None,
                     n_boot=2000, seed=20260921, variant='mitigated'):
    """Monte-Carlo CI on the Bloch radius r = |(<s_X>,<s_Y>,<s_Z>)| and on the mean
    code-space weight.  Resamples each basis circuit multinomially from its own
    observed counts and reruns the estimator, so the ratio's correlation, the
    mitigation and the norm of the three components all propagate together."""
    rng = np.random.default_rng(seed)
    O = observables_for(sigma, n)
    W = _weights(O, e01, e10, n, variant == 'mitigated')
    obs = {}
    for B, counts in counts_by_basis.items():
        v = counts_to_vec(counts, n)
        obs[B] = (v, max(v.sum(), 1.0))
    rs, ws = [], []
    for _ in range(n_boot):
        comps, wsum = [], []
        for B in BASES:
            v, tot = obs[B]
            s = rng.multinomial(int(tot), v / tot).astype(float)
            st = _ratio_stats(s, W)
            comps.append(st['sigma_cs'] if st['sigma_cs'] == st['sigma_cs'] else 0.0)
            wsum.append(st['w'])
        rs.append(float(np.linalg.norm(comps)))
        ws.append(float(np.mean(wsum)))
    rs, ws = np.array(rs), np.array(ws)

    def pct(a, f):
        return float(np.percentile(a, f))
    return dict(radius=pct(rs, 50), radius_ci95=[pct(rs, 2.5), pct(rs, 97.5)],
                radius_mean=float(rs.mean()), radius_std=float(rs.std(ddof=1)),
                w_mean=pct(ws, 50), w_ci95=[pct(ws, 2.5), pct(ws, 97.5)],
                w_std=float(ws.std(ddof=1)),
                n_boot=n_boot, seed=seed, variant=variant)


# ---------------------------------------------------------------------
# 6) the isotropic-depolarizing model at the HARDWARE's own depth, and the
#    one-parameter fit that turns M1 into a test of the surrogate (M3)
# ---------------------------------------------------------------------
def _kraus(p):
    I2 = np.eye(2, dtype=complex)
    return [math.sqrt(1.0 - p) * I2] + [math.sqrt(p / 3.0) * s for s in PAULI5]


def codespace_projector():
    """32x2 orthonormal basis of the [[5,1,3]] code space, from ssvr_qec's own
    logical vectors.  Only its column SPACE matters: w = Tr(V^dag rho V) and
    r^2 = 2 Tr(rho_L^2) - 1 are both invariant under a change of basis inside the
    code space, which is what makes the anchor below a statement about physics
    rather than about a convention."""
    import ssvr_qec as m
    V = np.stack([m.ZERO_L.numpy(), m.ONE_L.numpy()], axis=1)
    G = V.conj().T @ V
    assert np.abs(G - np.eye(2)).max() < 1e-12, G
    return V


def _propagate(gates, p_instr=None, p1=None, p2=None, psi_vec=None, n=NQ5):
    """rho after `gates`, with a depolarizing channel after every instruction on
    every qubit it touches.  Shared by model_mixedness and model_output_probs so the
    two can never drift apart."""
    if psi_vec is None:
        psi_vec = np.zeros(2 ** n, dtype=complex)
        psi_vec[0] = 1.0
    rho = np.outer(psi_vec, np.asarray(psi_vec, dtype=complex).conj())
    for g in gates:
        if g[0] == 'MEAS':
            continue
        if g[0] == 'CNOT':
            U = q.embed2(q.CNOT4, g[1], g[2], n=n)
            qs, p = [g[1], g[2]], (p2 if p_instr is None else p_instr)
        else:
            U = q.embed1(q._gate_matrix(g), g[1], n=n)
            qs, p = [g[1]], (p1 if p_instr is None else p_instr)
        rho = U @ rho @ U.conj().T
        if p > 0.0:
            # one qubit's channel at a time, exactly as device_prep does it.
            # Applying the union of the Kraus sets as a single flat list would
            # treat "depolarise a" and "depolarise b" as ALTERNATIVES instead of
            # composing them, and would multiply Tr(rho) by 2 per two-qubit gate.
            for qb in qs:
                K = [q.embed1(k, qb, n=n) for k in _kraus(p)]
                rho = sum(k @ rho @ k.conj().T for k in K)
            tr = float(np.trace(rho).real)
            assert abs(tr - 1.0) < 1e-9, tr
    return rho


def model_output_probs(gates, p1, p2, psi_vec=None, n=NQ5):
    """The modelled output distribution over canonical bitstrings -- what a device
    with these fidelities would hand back, before readout error."""
    rho = _propagate(gates, p1=p1, p2=p2, psi_vec=psi_vec, n=n)
    d = np.diag(rho).real
    return {canonical_key(i, n): float(d[i]) for i in range(2 ** n)}


def model_mixedness(gates, p_instr=None, p1=None, p2=None, psi_vec=None,
                    proj='decode', n=NQ5):
    """Propagate rho through `gates` with a depolarizing channel after every
    instruction, on every qubit that instruction touches, then project.

    This is `mixed_input_boundary.device_prep`'s model, re-implemented on an
    ARBITRARY compiled gate list and as an explicit 32x32 density matrix, so it
    can be run at the depth the hardware circuit actually has and so the audit can
    re-derive it independently.  Rates: a single `p_instr` (device_prep's choice,
    and what the M3 fit produces), or per-type `p1`/`p2` from vendor single- and
    two-qubit fidelities.  proj='decode' projects onto q0..q3=0 after the gate
    list has decoded; proj='iso' applies V V^dag, which is what device_prep does
    for an encoder-only list."""
    assert (p_instr is not None) != (p1 is not None and p2 is not None), \
        'give either p_instr or both of p1 and p2'
    rho = _propagate(gates, p_instr=p_instr, p1=p1, p2=p2, psi_vec=psi_vec, n=n)
    if proj == 'decode':
        cs = [0, 1]                       # q0..q3 == 0, q4 free (q4 is the LSB)
        w = float(np.trace(rho[np.ix_(cs, cs)]).real)
        rl = rho[np.ix_(cs, cs)]
    elif proj == 'iso':
        V = codespace_projector()
        rl = V.conj().T @ rho @ V
        w = float(np.trace(rl).real)
    else:
        raise ValueError(proj)
    assert w > 0.0, 'the model predicts zero code-space weight'
    rl = rl / w
    nvec = np.array([float(np.trace(Pm @ rl).real) for Pm in PAULI5])
    r = float(np.linalg.norm(nvec))
    purity = float(np.trace(rl @ rl).real)
    # the UNCONDITIONAL logical <Z_L>, taken before any projection: this is the
    # linear functional M2 uses, and it is linear in rho_in by construction.
    Z4 = q.embed1(PAULI5[2], n - 1, n=n)
    sz = float(np.trace(Z4 @ rho).real)
    return dict(radius=r, codespace_weight=w, purity=purity, leakage=1.0 - w,
                bloch=nvec.tolist(), n_instructions=len(gates), proj=proj,
                p_instr=p_instr, p1=p1, p2=p2, sigma_z_uncond=sz)


def prep_vec(psi, n=NQ5):
    """The physical |0...0>-based state vector that prep(psi) on q4 produces."""
    U = q.gates_to_unitary(list(q.TEST_STATES[psi][0]), n=n)
    return U[:, 0]


def fit_p_instr(gates, w_target, psi_vec, proj='decode', n=NQ5,
                lo=0.0, hi=0.25, iters=80):
    """The single p_instr whose modelled code-space weight equals the measured one.

    Bisection: w(p) is continuous and strictly decreasing on [0, 0.25] (asserted
    at the endpoints), so the solution is unique.  This is the same inversion
    hw_error_budget.py performs to get the implied mean instruction error, only
    here the measured quantity is a code-space weight rather than a syndrome
    marginal, and the fitted rate is then used to PREDICT r -- one fitted
    parameter, one independent prediction."""
    w_lo = model_mixedness(gates, p_instr=lo, psi_vec=psi_vec, proj=proj, n=n)
    w_hi = model_mixedness(gates, p_instr=hi, psi_vec=psi_vec, proj=proj, n=n)
    # w(lo) is exactly 1 for a noiseless gate list, and a device good enough to
    # land there -- or a noiseless dry run -- must return p_instr = lo rather than
    # fail an assertion, so the endpoint is handled explicitly.
    if w_target >= w_lo['codespace_weight'] - 1e-12:
        return lo
    assert w_lo['codespace_weight'] > w_target > w_hi['codespace_weight'], \
        (w_lo['codespace_weight'], w_target, w_hi['codespace_weight'])
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        wm = model_mixedness(gates, p_instr=mid, psi_vec=psi_vec, proj=proj,
                             n=n)['codespace_weight']
        if wm > w_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------
# 7) self-tests -- every estimator this module uses, anchored before it touches
#    a single real count.  No account, no QPU time, no artifact required.
# ---------------------------------------------------------------------
def _counts_from_probs(probs, N):
    """Deterministic integer counts from an exact distribution (rounding error
    ~1/N, which is what lets test (b) use a tight tolerance)."""
    out = {}
    for key, p in probs.items():
        c = int(round(p * N))
        if c > 0:
            out[key] = c
    return out


def _tomo_pairs():
    return ([(psi, B, 1) for psi in TOMO_STATES for B in BASES]
            + [(psi, B, 2) for psi, B in DEPTH_SCAN])


def selftests(verbose=True):
    res = {}

    def say(msg):
        if verbose:
            print('    ' + msg, flush=True)

    # (a) every tomography circuit is IDEALLY a perfect readout of the input
    #     Bloch vector, at both depths -- including the two null controls, whose
    #     ideal component is exactly zero.  The circuit ends with R_B, so the
    #     component that carries the signal is the Z component of the ROTATED
    #     output state, and the two transverse ones must vanish; that is exactly
    #     what the hardware estimator reads as <sigma_B>.  Only the Z component of
    #     the rotated state is the measured quantity; the two transverse ones carry
    #     the input's other Bloch components, so the invariants to assert are that
    #     the Z component is exact, the state is still pure (|n| = 1) and the
    #     code-space weight is exactly 1.
    worst_ideal = 0.0
    for psi, B, rounds in _tomo_pairs():
        gates = build_tomo(psi, B, rounds)
        w, nvec = ideal_bloch_of(gates)
        want = IDEAL_BLOCH[psi][BASIS_AXIS[B]]
        worst_ideal = max(worst_ideal, abs(w - 1.0), abs(nvec[2] - want),
                          abs(np.linalg.norm(nvec) - 1.0))
    assert worst_ideal < 1e-12, worst_ideal
    res['ideal_bloch_residual'] = worst_ideal
    say('(a) %d tomography circuits: exact code-space weight 1 and exact ideal '
        'Bloch component (2 of them null), worst residual %.2e'
        % (len(_tomo_pairs()), worst_ideal))

    # (b) the COUNT-based estimator agrees with the state-vector estimator on the
    #     same ideal distribution, so a hardware number is not an artifact of how
    #     the counts are sliced.
    sigma_id = tuple(range(NQ5))
    worst_est = 0.0
    for psi, B, rounds in _tomo_pairs():
        gates = build_tomo(psi, B, rounds)
        counts = _counts_from_probs(exact_probs(gates), 10 ** 7)
        rec = read_tomo(counts, sigma_id)['raw']
        worst_est = max(worst_est, abs(rec['w'] - 1.0))
        want = IDEAL_BLOCH[psi][BASIS_AXIS[B]]
        worst_est = max(worst_est, abs(rec['sigma_cs'] - want),
                        abs(rec['sigma_uncond'] - want))
    assert worst_est < 1e-6, worst_est
    res['estimator_vs_statevector'] = worst_est
    say('(b) count estimator vs exact state vector over all %d circuits: worst '
        'residual %.2e' % (len(_tomo_pairs()), worst_est))

    # (c) ANCHOR: the re-implemented model reproduces the AUDITED device_prep
    #     numbers in mixed_input_boundary.json at the recorded p_instr, on the
    #     encoder-only gate list with the V V^dag projection device_prep uses.
    mb = json.load(open(os.path.join(ROOT, 'mixed_input_boundary.json')))
    dp = mb['device_prep']
    enc = q.ENC5
    assert len(enc) == dp['n_encoder_instructions'], len(enc)
    worst_anchor = 0.0
    for b, label in ((0, '0_L'), (1, '1_L')):
        pv = np.zeros(DIM5, dtype=complex)
        pv[b] = 1.0
        got = model_mixedness(enc, p_instr=dp['p_instr'], psi_vec=pv, proj='iso')
        ref = dp[label]
        for key in ('radius', 'codespace_weight', 'purity', 'leakage'):
            worst_anchor = max(worst_anchor, abs(got[key] - ref[key]))
    assert worst_anchor < 1e-9, worst_anchor
    res['anchor_device_prep'] = worst_anchor
    say('(c) model re-implementation vs the AUDITED device_prep block at '
        'p_instr=%.12f: worst |diff| on r, w, purity, leakage = %.2e'
        % (dp['p_instr'], worst_anchor))

    # (d) readout mitigation is an exact inverse of the assignment matrix it is
    #     given, and it recovers (w, <sigma>) of a distribution that was pushed
    #     through a synthetic readout channel.
    rng = np.random.default_rng(4242)
    worst_mit = 0.0
    for _ in range(6):
        e01 = list(rng.uniform(0.01, 0.09, NQ5))
        e10 = list(rng.uniform(0.01, 0.09, NQ5))
        gates = build_tomo('0', 'Z', 1)
        ptrue = np.zeros(DIM5)
        for key, pr in exact_probs(gates).items():
            ptrue[str_to_idx(key)] = pr
        pobs = assignment_matrix(e01, e10) @ ptrue
        back, diag = mitigate(pobs, e01, e10)
        worst_mit = max(worst_mit, float(np.abs(back - ptrue).max()),
                        diag['negativity_before_clip'])
        cs, _m4 = cs_split(tuple(range(NQ5)))
        wt = float(sum(ptrue[i] for i in cs))
        wb = float(sum(back[i] for i in cs))
        worst_mit = max(worst_mit, abs(wt - wb))
        # the estimator this file actually uses: O~ = A^{-T} O, which must be
        # EXACTLY unbiased on the noisy distribution -- no inversion of p, no clip,
        # hence no bias, which is what the clip-and-renormalise version had
        O = observables_for(tuple(range(NQ5)))
        W = _weights(O, e01, e10, NQ5, True)
        for key in ('cs', 'z', 'num', 'one'):
            worst_mit = max(worst_mit,
                            abs(float(W[key] @ pobs) - float(O[key] @ ptrue)))
        # and in the SYMMETRIC special case e01 = e10 = e it must reduce to the
        # familiar contrast division by (1 - 2e).  It does NOT reduce to a division
        # by (1 - e01 - e10) when the readout is asymmetric -- that was wrong on
        # first writing this test, and the A^{-T} O machinery is what makes the
        # asymmetric case exact instead of approximate.
        m4 = cs_split(tuple(range(NQ5)))[1]
        e_sym = 0.05
        A_sym = assignment_matrix([e_sym] * NQ5, [e_sym] * NQ5)
        po_sym = A_sym @ ptrue
        p0 = float(sum(po_sym[i] for i in range(DIM5) if (i & m4) == 0))
        p1 = float(sum(po_sym[i] for i in range(DIM5) if (i & m4) != 0))
        analytic = (p0 - p1) / (1.0 - 2.0 * e_sym)
        W_sym = _weights(O, [e_sym] * NQ5, [e_sym] * NQ5, NQ5, True)
        worst_mit = max(worst_mit, abs(analytic - float(O['z'] @ ptrue)),
                        abs(analytic - float(W_sym['z'] @ po_sym)),
                        abs(float(W_sym['z'] @ po_sym) - float(W['z'] @ pobs)))
    assert worst_mit < 1e-12, worst_mit
    res['mitigation_roundtrip'] = worst_mit
    say('(d) readout mitigation: the distribution inversion is exact, and the '
        'O~ = A^-T O estimator actually used here is exactly unbiased on a noisy '
        'distribution and reduces to the analytic contrast division for one bit -- '
        'worst residual %.2e' % worst_mit)

    # (e) M2's mathematical content, tested on the model at a NONZERO noise rate:
    #     the unconditional <Z_L> is a linear functional of rho_in, so the two pure
    #     ensembles of I/2 must give the same average exactly, while the individual
    #     terms are O(1).  If this ever fails the hardware claim is meaningless.
    worst_lin, min_span = 0.0, None
    for p in (0.0, 0.005, 0.02, 0.05):
        terms = {}
        # the prep is carried by psi_vec here, so the gate list must NOT also
        # contain it -- build_tomo() folds prep into the list and is used with
        # psi_vec=None instead.  Doing both applies the prep twice.
        g = enc_dec_gates(1) + [('MEAS', i, i) for i in range(NQ5)]
        for psi in ('0', '1', '+', '-'):
            terms[psi] = model_mixedness(g, p_instr=p, proj='decode',
                                         psi_vec=prep_vec(psi))['sigma_z_uncond']
        Lz = 0.5 * (terms['0'] + terms['1'])
        Lx = 0.5 * (terms['+'] + terms['-'])
        worst_lin = max(worst_lin, abs(Lz - Lx))
        # non-vacuity is only claimed where the terms are still O(1): at p=0.05
        # the unconditional <Z_L> has depolarised to ~0.01 for EVERY input, so two
        # zeros agreeing would prove nothing.  The same guard is applied to the
        # hardware numbers, where the span is recorded in the artifact.
        if p <= 0.005:
            span = max(terms.values()) - min(terms.values())
            min_span = span if min_span is None else min(min_span, span)
    assert worst_lin < 1e-12, worst_lin
    assert min_span > 0.5, min_span
    res['linear_ensemble_residual'] = worst_lin
    res['linear_term_span_min'] = min_span
    say('(e) linear score over the two pure ensembles of I/2 at p_instr up to 0.05: '
        'worst |L_Z - L_X| = %.2e while the individual <Z_L> terms span >= %.3f'
        % (worst_lin, min_span))

    # (f) layout detection: (f1) on the tomography circuits ALONE the ideal
    #     distributions leave the four code-space positions undetermined -- 4! tied
    #     hypotheses -- but they DO pin the logical position, and every estimator
    #     in this file must be invariant across the tie set.  (f2) adding the five
    #     fingerprint circuits removes the tie outright.
    rng2 = np.random.default_rng(99)

    def synth(circuits, sigma_true, shots=4000):
        ideals, results = [], []
        for g in circuits:
            idl = exact_probs(g)
            ideals.append(idl)
            v = np.zeros(DIM5)
            for key, pr in idl.items():
                obs = ['0'] * NQ5
                for j in range(NQ5):
                    obs[j] = key[sigma_true[j]]
                v[str_to_idx(''.join(obs))] += pr
            results.append(dict(zip((idx_to_str(i) for i in range(DIM5)),
                                    rng2.multinomial(shots, v))))
        return ideals, results

    tomo_gates = [build_tomo(psi, B, r) for psi, B, r in _tomo_pairs()]
    fp_gates = [build_fingerprint(b) for b in range(NQ5)]
    smallest_margin, worst_invariance = None, 0.0
    for sigma_true in [(0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (2, 0, 4, 1, 3)]:
        ideals, results = synth(tomo_gates, sigma_true)
        sigma_hat, ev = detect_layout_ml(results, ideals, verbose=False)
        assert ev['n_tied'] == 24, ev['n_tied']
        assert ev['tie_determines_logical'] and ev['tie_determines_codespace']
        want_pos = [j for j in range(NQ5) if sigma_true[j] == NQ5 - 1][0]
        assert ev['logical_position'] == want_pos, (ev, want_pos)
        # every estimator must agree across the whole tie set
        ref = read_tomo(results[0], tuple(sigma_hat))
        for sg in ev['tie_set']:
            got = read_tomo(results[0], tuple(sg))
            for var in ('raw',):
                for key in ('w', 'sigma_cs', 'sigma_uncond'):
                    worst_invariance = max(
                        worst_invariance,
                        abs(got[var][key] - ref[var][key]))
        ideals2, results2 = synth(tomo_gates + fp_gates, sigma_true)
        sigma_hat2, ev2 = detect_layout_ml(results2, ideals2, verbose=False)
        assert tuple(sigma_hat2) == tuple(sigma_true), (sigma_hat2, sigma_true)
        assert ev2['n_tied'] == 1, ev2['n_tied']
        smallest_margin = ev2['margin'] if smallest_margin is None else min(
            smallest_margin, ev2['margin'])
    assert worst_invariance < 1e-12, worst_invariance
    assert smallest_margin > 100.0, smallest_margin
    res['layout_tie_invariance'] = worst_invariance
    res['layout_margin_min'] = smallest_margin
    say('(f) layout: tomo circuits alone leave exactly 24 tied hypotheses (the four '
        'code-space positions) but pin the logical one, and w / <sigma_cs> / '
        '<sigma_uncond> are identical across all 24 to %.2e; adding the 5 '
        'fingerprint circuits breaks the tie uniquely in all 3 planted orders, '
        'smallest margin %.0f' % (worst_invariance, smallest_margin))

    # (g) the M3 inversion: a planted p_instr is recovered from the model's own
    #     code-space weight, which is exactly what the fit does with a measured one
    worst_fit = 0.0
    for p_true in (0.005, 0.02, 0.06):
        g = enc_dec_gates(1)
        w = model_mixedness(g, p_instr=p_true, psi_vec=prep_vec('0'),
                            proj='decode')['codespace_weight']
        worst_fit = max(worst_fit,
                        abs(fit_p_instr(g, w, prep_vec('0')) - p_true))
    assert worst_fit < 1e-6, worst_fit
    res['fit_p_instr_residual'] = worst_fit
    say('(g) one-parameter fit of p_instr from the code-space weight recovers a '
        'planted rate to %.2e' % worst_fit)

    # (h) the four cloud stages have the size and architecture the budget assumes
    sizes = {s: len(stage_circuits(s, phi=np.zeros((16, 60)))[0])
             for s in ('probe', 'tomo', 'bench', 'ladder')}
    assert sizes == {'probe': 8, 'tomo': 19, 'bench': 34,
                     'ladder': 4 * len(LADDER_ROUNDS)}, sizes
    res['stage_sizes'] = sizes
    res['n_circuits_total'] = sum(sizes.values())
    res['shots_total'] = sum(sizes[s] * STAGE_SHOTS[s] for s in sizes)
    say('(h) stage circuit counts %s = %d circuits, %d shots at the default '
        'per-stage budgets' % (sizes, res['n_circuits_total'], res['shots_total']))

    # (i) the tomo job carries its own calibration, so it never has to assume that
    #     the canary's physical block is still this job's block (QCLOUD_SETUP.md 8.2)
    _g, tm = stage_circuits('tomo')
    kinds = [m['kind'] for m in tm]
    assert kinds.count('cal') == 3 and kinds.count('tomo') == 16, kinds
    assert [m['cal'] for m in tm if m['kind'] == 'cal'] == ['zero', 'one', 'zero']
    # and the ladder is the SAME grid at strictly deeper rounds, with the same meta
    # kind, which is what lets --stage all merge the two into one depth curve
    _gl, ml = stage_circuits('ladder')
    assert all(m['kind'] == 'tomo' for m in ml), ml
    assert sorted({m['rounds'] for m in ml}) == sorted(LADDER_ROUNDS)
    assert min(m['rounds'] for m in ml) > 2
    assert {(m['psi'], m['basis']) for m in ml} == set(DEPTH_SCAN)
    merged = sorted({m['rounds'] for m in tm + ml if m['kind'] == 'tomo'})
    assert merged == [1, 2] + sorted(LADDER_ROUNDS), merged
    # the two signal circuits the radius is read from must be present at every depth,
    # otherwise analyze_tomo's M3 fit skips that depth silently
    tomo_keys = {(m['psi'], m['basis'], m['rounds'])
                 for m in tm + ml if m['kind'] == 'tomo'}
    for r in merged:
        assert ('0', 'Z', r) in tomo_keys, r
    res['ladder_rounds'] = list(LADDER_ROUNDS)
    res['merged_depths'] = merged
    say('(i) the tomo job carries its own 3 calibration circuits; the ladder adds '
        'rounds %s over the %s grid, so the merged depth curve is %s and (|0>,Z) '
        'is present at every depth' % (list(LADDER_ROUNDS),
                                       [list(p) for p in DEPTH_SCAN], merged))
    return res


# ---------------------------------------------------------------------
# 8) free pre-flight queries, and the three cloud stages
# ---------------------------------------------------------------------
def _now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _backend(chip_id, token):
    svc = q._make_qcloud_service(token)
    try:
        return svc, svc.backend(chip_id)
    except TypeError:
        return svc, svc.backend(chip_id=chip_id)


def preflight(chip_ids, token=None, verbose=True):
    """FREE cloud queries only -- submits nothing and spends no QPU time.

    Three things must be settled before any shots are bought: which chips are
    online and how long their queues are, which physical block each stage should
    be pinned to (5 qubits for probe/tomo, 9 for bench -- qcloud_vscr_new's own
    auto-query hardcodes 9, so a 5-qubit stage must pin its block explicitly), and
    whether the vendor calibration metadata is still behind the 401 that
    hw_error_budget.py records.  Every failure is captured verbatim rather than
    raised, so the artifact always says which of the two happened."""
    out = dict(attempted_utc=_now(), chip_ids=list(chip_ids),
               spends_qpu_time=False, backends=None, blocks={}, calibration={},
               errors=[])
    try:
        import hw_error_budget as heb
        tok = token or heb.resolve_token()
    except Exception as exc:
        out['errors'].append('token: %r' % (exc,))
        return out
    try:
        svc, _ = _backend(chip_ids[0], tok)
        bl = svc.backends()
        out['backends'] = [b if isinstance(b, dict) else str(b)
                           for b in (bl if isinstance(bl, (list, tuple)) else [bl])]
    except Exception as exc:
        out['errors'].append('backends(): %r' % (exc,))
    for chip in chip_ids:
        for nq in (NQ5, q.NQ):
            try:
                _svc, be = _backend(chip, tok)
                bb = be.best_qubit_blocks(nq)
                blocks = list(bb.qubit_blocks) if bb.qubit_blocks else []
                out['blocks']['%s|%d' % (chip, nq)] = (
                    [[int(x) for x in b] for b in blocks[:3]])
            except Exception as exc:
                out['errors'].append('best_qubit_blocks(%d) on %s: %r'
                                     % (nq, chip, exc))
    try:
        import hw_error_budget as heb
        for chip in chip_ids:
            out['calibration'][chip] = heb.try_vendor_calibration(chip)
    except Exception as exc:
        out['errors'].append('vendor calibration: %r' % (exc,))
    if verbose:
        print('[preflight] blocks: %s' % json.dumps(out['blocks']), flush=True)
        for e in out['errors']:
            print('[preflight] %s' % e, flush=True)
    return out


def query_best_block(chip_id, nq, token):
    """The top-ranked physical block of size nq on `chip_id` (free query)."""
    _svc, be = _backend(chip_id, token)
    bb = be.best_qubit_blocks(nq)
    blocks = list(bb.qubit_blocks) if bb.qubit_blocks else []
    return [int(x) for x in blocks[0]] if blocks else None


def run_stage(stage, token, chip_id, shots=None, block=None, chunk=None,
              job_timeout=21600, use_options=True, dump_raw=None, phi=None,
              verbose=True):
    """Submit one stage as ONE job (chunk defaults to the whole stage) and dump the
    raw counts incrementally, so a mid-run quota refusal loses nothing."""
    shots = shots or STAGE_SHOTS[stage]
    dump_raw = dump_raw or os.path.join(ROOT, STAGE_RAW[stage])
    gates, meta = stage_circuits(stage, phi)
    nq = STAGE_NQ[stage]
    if block is None and use_options:
        try:
            block = query_best_block(chip_id, nq, token)
        except Exception as exc:
            print('[stage %s] best_qubit_blocks(%d) unavailable (%s); the server '
                  'will choose qubits' % (stage, nq, exc), flush=True)
    if verbose:
        inv = {}
        for g in gates:
            inv[len(g)] = inv.get(len(g), 0) + 1
        print('[stage %s] %d circuits on %d qubits, %d shots each, block=%s, '
              'instruction counts %s' % (stage, len(gates), nq, shots, block, inv),
              flush=True)
    t0 = time.time()
    results = q.submit_cloud(gates, token, chip_id=chip_id, shots=shots,
                             batch=True, dump_raw=dump_raw,
                             chunk=chunk or len(gates), job_timeout=job_timeout,
                             use_options=use_options, block=block)
    wall = time.time() - t0
    assert len(results) == len(gates), (len(results), len(gates))
    used = [int(sum(r.values())) for r in results]
    if verbose:
        print('[stage %s] received %d circuits in %.0f s wall; shots used %d/%d '
              'per circuit (min %d, max %d)'
              % (stage, len(results), wall, shots * len(gates), sum(used),
                 min(used), max(used)), flush=True)
    return dict(results=results, meta=meta, gates=gates, shots_requested=shots,
                shots_used=used, chip_id=chip_id, block=block, wall_s=wall,
                dump_raw=dump_raw, stage=stage)


# ---------------------------------------------------------------------
# 9) analysis: raw counts -> the five measured statements
# ---------------------------------------------------------------------
def analyze_probe(results, meta, shots_requested=None, n=NQ5, verbose=True):
    """Readout assignment matrix (M4), its drift, and the bit order (from the five
    fingerprint circuits, which determine it uniquely)."""
    zeros = [r for r, m in zip(results, meta) if m.get('cal') == 'zero']
    ones = [r for r, m in zip(results, meta) if m.get('cal') == 'one']
    fps = [(r, m['qubit']) for r, m in zip(results, meta)
           if m.get('kind') == 'fingerprint']
    assert len(zeros) == 2 and len(ones) == 1 and len(fps) == n, \
        (len(zeros), len(ones), len(fps))
    cal = assignment_from_cal(zeros[0], ones[0], n)
    cal2 = assignment_from_cal(zeros[1], ones[0], n)
    drift = [abs(a - b) for a, b in zip(cal['e01'], cal2['e01'])]
    fp_results = [r for r, _ in fps]
    fp_ideals = [exact_probs(build_fingerprint(b)) for _, b in fps]
    sigma, ev = detect_layout_ml(fp_results, fp_ideals, n, verbose=verbose)
    out = dict(assignment=cal, assignment_repeat_e01=cal2['e01'],
               e01_drift=drift, e01_drift_max=max(drift),
               n_used=[int(sum(r.values())) for r in results],
               shots_requested=shots_requested,
               layout_sigma=list(sigma), layout_evidence=ev,
               # the readout error of the logical qubit, in the position the
               # layout puts it -- the single number that sets the contrast loss
               logical_position=ev['logical_position'],
               logical_e01=cal['e01'][ev['logical_position']],
               logical_e10=cal['e10'][ev['logical_position']],
               mean_readout_error=0.5 * (sum(cal['e01']) / n
                                         + sum(cal['e10']) / n))
    if verbose:
        print('[probe] e01=%s\n[probe] e10=%s\n[probe] drift max=%.4f  readout '
              'fidelity=%.4f' % ([round(x, 4) for x in cal['e01']],
                                 [round(x, 4) for x in cal['e10']],
                                 max(drift), 1 - out['mean_readout_error']),
              flush=True)
    return sigma, out


def analyze_tomo(results, meta, sigma, cal, shots_requested=None, n=NQ5,
                 n_boot=2000, seed=20260921, vendor=None, verbose=True):
    """M1 mixedness, M2 linear exactness, M3 the surrogate test, M5 the depth scan.

    `vendor`, if given as (f1q, f2q, fread), adds the prediction that the vendor's
    own aggregate fidelities give through the same model -- the arithmetic behind a
    backend choice, recorded in the artifact rather than in a chat message."""
    e01, e10 = cal['e01'], cal['e10']
    circuits, by_key = {}, {}
    for rec, m in zip(results, meta):
        if m.get('kind') != 'tomo':
            continue      # the stage's own calibration circuits, handled separately
        key = '%s|%s|r%d' % (m['psi'], m['basis'], m['rounds'])
        out = read_tomo(rec, tuple(sigma), n, e01, e10)
        out['n_requested'] = shots_requested
        out['ideal'] = dict(zip(
            ('w', 'sigma'),
            (1.0, IDEAL_BLOCH[m['psi']][BASIS_AXIS[m['basis']]])))
        circuits[key] = out
        by_key[(m['psi'], m['basis'], m['rounds'])] = (rec, out, m)
    # every depth the submitted circuits actually reached, so the ladder stage
    # extends the curve instead of being analysed as if it were depth 1 and 2
    rounds_present = sorted({k[2] for k in by_key})
    assert 1 in rounds_present, (
        'the ladder stage is the SAME estimator at deeper rounds and must be '
        'analysed together with the depth-1/2 tomography, not on its own '
        '(rounds present = %s)' % (rounds_present,))

    per_input = {}
    for psi in TOMO_STATES:
        counts_by_basis = {B: by_key[(psi, B, 1)][0] for B in BASES}
        comps = {}
        for var in ('raw', 'mitigated'):
            comps[var] = [by_key[(psi, B, 1)][1][var]['sigma_cs'] for B in BASES]
        boot = {var: bootstrap_radius(counts_by_basis, tuple(sigma), n,
                                      e01 if var == 'mitigated' else None,
                                      e10 if var == 'mitigated' else None,
                                      n_boot=n_boot, seed=seed, variant=var)
                for var in ('raw', 'mitigated')}
        w = {var: float(np.mean([by_key[(psi, B, 1)][1][var]['w']
                                 for B in BASES])) for var in ('raw', 'mitigated')}
        rec = dict(bloch={v: comps[v] for v in comps}, w=w,
                   purity={v: (1 + comps[v][0] ** 2 + comps[v][1] ** 2
                               + comps[v][2] ** 2) / 2 for v in comps},
                   bootstrap=boot)
        for v in comps:
            rec['radius_' + v] = float(np.linalg.norm(comps[v]))
        per_input[psi] = rec

    # M2: the two pure ensembles of rho_in = I/2 must give the same LINEAR score
    terms, ses = {}, {}
    for psi in TOMO_STATES:
        zc = by_key[(psi, 'Z', 1)][1]
        terms[psi] = dict(raw=zc['raw']['sigma_uncond'],
                          mitigated=zc['mitigated']['sigma_uncond'])
        ses[psi] = dict(raw=zc['raw']['sigma_uncond_stderr'],
                        mitigated=zc['mitigated']['sigma_uncond_stderr'])
    lin = {}
    for var in ('raw', 'mitigated'):
        Lz = 0.5 * (terms['0'][var] + terms['1'][var])
        Lx = 0.5 * (terms['+'][var] + terms['-'][var])
        se = 0.5 * math.sqrt(sum(ses[p][var] ** 2 for p in TOMO_STATES))
        vals = [terms[p][var] for p in TOMO_STATES]
        lin[var] = dict(terms={p: terms[p][var] for p in TOMO_STATES},
                        L_Z=Lz, L_X=Lx, diff=Lz - Lx, stderr=se,
                        ci95=[Lz - Lx - 1.96 * se, Lz - Lx + 1.96 * se],
                        term_span=max(vals) - min(vals),
                        ci_over_span=(3.92 * se) / max(max(vals) - min(vals), 1e-12),
                        non_vacuous=bool(max(vals) - min(vals) > 0.5))

    # M1: the headline device mixedness, as the range over the four inputs
    radii = {v: [per_input[p]['radius_' + v] for p in TOMO_STATES]
             for v in ('raw', 'mitigated')}
    ws = {v: [per_input[p]['w'][v] for p in TOMO_STATES] for v in ('raw', 'mitigated')}
    boot_mit = [per_input[p]['bootstrap']['mitigated'] for p in TOMO_STATES]
    mixedness = dict(
        radius_raw=radii['raw'], radius_mitigated=radii['mitigated'],
        radius_min=min(radii['mitigated']), radius_max=max(radii['mitigated']),
        radius_ci95_widest=[min(b['radius_ci95'][0] for b in boot_mit),
                            max(b['radius_ci95'][1] for b in boot_mit)],
        radius_stderr_max=max(b['radius_std'] for b in boot_mit),
        w_raw=ws['raw'], w_mitigated=ws['mitigated'],
        w_mean_mitigated=float(np.mean(ws['mitigated'])),
        leakage_mean_mitigated=1.0 - float(np.mean(ws['mitigated'])),
        purity_mitigated=[per_input[p]['purity']['mitigated']
                          for p in TOMO_STATES],
        purity_consistency_max=float(np.max(np.abs(
            [per_input[p]['purity']['mitigated']
             - (1 + per_input[p]['radius_mitigated'] ** 2) / 2
             for p in TOMO_STATES]))),
        r_below_one=bool(max(radii['mitigated']) < 1.0),
        ci_upper_below_one=bool(max(b['radius_ci95'][1] for b in boot_mit) < 1.0))

    # M3: fit ONE p_instr to the measured code-space weight, then let the model
    #     predict r with nothing left to choose.  Done at both depths: if a single
    #     rate explains both, the loss is per-gate; if not, time is contributing.
    #     At rounds=2 only the (psi=0, Z) and (psi=+, X) circuits are signal -- for
    #     those inputs the Bloch vector lies along the measured axis, so that single
    #     component IS the radius.
    model = {}
    for rounds in rounds_present:
        if ('0', 'Z', rounds) not in by_key:
            continue      # a depth without the (|0>, Z) signal circuit cannot be fitted
        gates = enc_dec_gates(rounds)
        w_meas = float(by_key[('0', 'Z', rounds)][1]['mitigated']['w'])
        if rounds == 1:
            r_meas = per_input['0']['radius_mitigated']
            se = per_input['0']['bootstrap']['mitigated']['radius_std']
        else:
            # beyond depth 1 only the DEPTH_SCAN grid is submitted, and for |0>_L the
            # Bloch vector lies along z, so that one component IS the radius and its
            # own standard error is the right one to scale the residual by
            r_meas = abs(float(by_key[('0', 'Z', rounds)][1]['mitigated']['sigma_cs']))
            se = float(by_key[('0', 'Z', rounds)][1]['raw']['sigma_cs_stderr'])
        try:
            p_fit = fit_p_instr(gates, w_meas, prep_vec('0'))
        except AssertionError as exc:
            model['rounds%d' % rounds] = dict(
                w_measured=w_meas, r_measured=r_meas, fit=None,
                note='the measured code-space weight is outside the range the '
                     'isotropic model can produce on [0, 0.25]: %s' % (exc,))
            continue
        pred = model_mixedness(gates, p_instr=p_fit, psi_vec=prep_vec('0'))
        model['rounds%d' % rounds] = dict(
            w_measured=w_meas, p_instr_fitted=p_fit,
            predicted=pred, r_measured=r_meas, r_predicted=pred['radius'],
            residual=pred['radius'] - r_meas,
            residual_in_sigma=((pred['radius'] - r_meas) / se) if se > 0 else None,
            # with se = 0 (a noiseless dry run) fall back to an absolute tolerance
            # rather than reporting a disagreement that is exactly zero
            agrees_within_2sigma=bool(
                (abs(pred['radius'] - r_meas) <= 2 * se) if se > 0
                else (abs(pred['radius'] - r_meas) < 1e-9)),
            n_instructions=len(gates))
    if model.get('rounds1', {}).get('p_instr_fitted') is not None and \
            model.get('rounds2', {}).get('p_instr_fitted') is not None:
        p1_, p2_ = model['rounds1']['p_instr_fitted'], model['rounds2']['p_instr_fitted']
        model['per_gate_vs_per_time'] = dict(
            p_instr_depth1=p1_, p_instr_depth2=p2_, ratio=p2_ / p1_ if p1_ else None,
            consistent_with_per_gate=bool(abs(p2_ - p1_) <= 0.2 * max(p1_, 1e-12)),
            note='a ratio near 1 means one rate per instruction explains both '
                 'depths (gate-like loss); a ratio well above 1 means the deeper '
                 'circuit loses more per instruction, i.e. time-like decoherence '
                 'is contributing')
    if vendor:
        model['vendor_prediction'] = vendor_prediction(*vendor)

    # M5: the depth scan itself, including the two null controls and how many
    #     standard errors each sits from zero.  Written over every depth actually
    #     submitted (the ladder stage adds rounds=4 and optionally 8), with the
    #     r1/r2 keys kept because the report, the figure and selftest (g) read them.
    scan = {}
    for psi, B in DEPTH_SCAN:
        depths = [r for r in rounds_present if (psi, B, r) in by_key]
        if not depths:
            continue
        d1 = by_key[(psi, B, 1)][1]
        null = IDEAL_BLOCH[psi][BASIS_AXIS[B]] == 0.0
        se1 = d1['raw']['sigma_cs_stderr']
        per_depth = {}
        for r in depths:
            dd = by_key[(psi, B, r)][1]
            per_depth[r] = dict(
                sigma_cs_mitigated=dd['mitigated']['sigma_cs'],
                sigma_cs_raw=dd['raw']['sigma_cs'],
                stderr=dd['raw']['sigma_cs_stderr'], w=dd['mitigated']['w'],
                null_sigma=(dd['raw']['sigma_cs'] / dd['raw']['sigma_cs_stderr']
                            if (null and dd['raw']['sigma_cs_stderr'] > 0) else None))
        d2 = by_key[(psi, B, 2)][1] if (psi, B, 2) in by_key else None
        se2 = d2['raw']['sigma_cs_stderr'] if d2 else None
        scan['%s|%s' % (psi, B)] = dict(
            null_control=null, depths=depths, by_rounds=per_depth,
            sigma_cs_r1_mitigated=d1['mitigated']['sigma_cs'],
            sigma_cs_r2_mitigated=(d2['mitigated']['sigma_cs'] if d2 else None),
            ratio_r2_over_r1=((d2['mitigated']['sigma_cs'] / d1['mitigated']['sigma_cs'])
                              if (d2 and abs(d1['mitigated']['sigma_cs']) > 1e-12)
                              else None),
            sigma_cs_r1_raw=d1['raw']['sigma_cs'],
            sigma_cs_r2_raw=(d2['raw']['sigma_cs'] if d2 else None),
            stderr_r1=se1, stderr_r2=se2,
            null_sigma_r1=(d1['raw']['sigma_cs'] / se1) if (null and se1 > 0) else None,
            null_sigma_r2=((d2['raw']['sigma_cs'] / se2)
                           if (null and d2 and se2 > 0) else None),
            w_r1=d1['mitigated']['w'],
            w_r2=(d2['mitigated']['w'] if d2 else None))
    nulls = [v for v in scan.values() if v['null_control']]
    all_null_sigmas = [abs(p['null_sigma']) for v in nulls
                       for p in v['by_rounds'].values() if p['null_sigma'] is not None]
    scan_summary = dict(
        n_null_controls=len(nulls),
        depths=sorted({r for v in scan.values() for r in v['depths']}),
        max_null_sigma=max(all_null_sigmas) if all_null_sigmas else 0.0,
        nulls_within_3sigma=bool(all(s < 3.0 for s in all_null_sigmas)),
        note='the null controls measure a Bloch component that is exactly zero '
             'ideally; a value beyond a few standard errors is a phase or '
             'calibration problem, not evidence of mixedness')
    out = dict(circuits=circuits, per_input=per_input, mixedness=mixedness,
               linear_ensemble=lin, model=model, depth_scan=scan,
               depth_scan_summary=scan_summary,
               layout_sigma=list(sigma), assignment=cal,
               shots_requested=shots_requested)
    return out, _tomo_report(out)


def _tomo_report(out):
    """The printable summary; returned separately so --mode analyze and the
    offline audit both print exactly what the artifact claims."""
    lines = []
    per_input, lin, model = out['per_input'], out['linear_ensemble'], out['model']
    lines.append('[tomo] M1 device mixedness (readout-mitigated):')
    for p_ in TOMO_STATES:
        b = per_input[p_]['bootstrap']['mitigated']
        lines.append('   |%s>_L  r = %.4f [%.4f, %.4f]  purity = %.4f  w = %.4f'
                     % (p_, per_input[p_]['radius_mitigated'], b['radius_ci95'][0],
                        b['radius_ci95'][1], per_input[p_]['purity']['mitigated'],
                        per_input[p_]['w']['mitigated']))
    for var in ('raw', 'mitigated'):
        L = lin[var]
        lines.append('[tomo] M2 (%s): L_Z = %+.5f  L_X = %+.5f  diff = %+.5f '
                     '+-%.5f  CI95 [%+.4f, %+.4f]  term span %.3f  non-vacuous=%s'
                     % (var, L['L_Z'], L['L_X'], L['diff'], L['stderr'],
                        L['ci95'][0], L['ci95'][1], L['term_span'],
                        L['non_vacuous']))
    for k in ('rounds1', 'rounds2'):
        M = model.get(k, {})
        if M.get('p_instr_fitted') is None:
            lines.append('[tomo] M3 %s: %s' % (k, M.get('note', 'missing')))
            continue
        sig = M['residual_in_sigma']
        lines.append('[tomo] M3 %s: w_meas %.4f -> p_instr %.5f -> r_pred %.4f vs '
                     'r_meas %.4f (%s sigma, agrees=%s)'
                     % (k, M['w_measured'], M['p_instr_fitted'], M['r_predicted'],
                        M['r_measured'],
                        ('%.1f' % sig) if sig is not None and sig == sig else 'n/a',
                        M['agrees_within_2sigma']))
    if 'per_gate_vs_per_time' in model:
        t = model['per_gate_vs_per_time']
        rat = t['ratio']
        lines.append('[tomo] M5 per-gate vs per-time: p_instr(depth2)/p_instr(depth1)'
                     ' = %s -> per-gate consistent = %s'
                     % (('%0.3f' % rat) if rat is not None
                        else 'undefined (both depths fitted to zero noise)',
                        t['consistent_with_per_gate']))
    s = out['depth_scan_summary']
    mns = s['max_null_sigma']
    lines.append('[tomo] M5 null controls: max %s sigma, all within 3 sigma = %s'
                 % (('%0.1f' % mns) if mns == mns else 'n/a',
                    s['nulls_within_3sigma']))
    return lines


def vendor_prediction(f1q, f2q, fread, rounds=1):
    """What a vendor's aggregate fidelities predict for THIS circuit, through the
    same model -- the arithmetic behind a backend choice, recorded in the artifact
    rather than left in a chat message.

    Single-qubit error on 1Q instructions, two-qubit error on CNOTs, then the readout
    error as a contrast factor on the logical bit and as a survival factor on the
    four code-space bits the post-selection requires."""
    gates = enc_dec_gates(rounds)
    d = model_mixedness(gates, p1=1.0 - f1q, p2=1.0 - f2q, psi_vec=prep_vec('0'))
    n_cnot = sum(1 for g in gates if g[0] == 'CNOT')
    contrast = 1.0 - 2.0 * (1.0 - fread)
    return dict(f1q=f1q, f2q=f2q, fread=fread, rounds=rounds,
                n_instructions=len(gates), n_cnot=n_cnot,
                two_qubit_survival=f2q ** n_cnot,
                codespace_weight=d['codespace_weight'], radius=d['radius'],
                radius_as_measured=d['radius'] * contrast,
                logical_contrast=contrast,
                codespace_bits_all_read_correct=fread ** 4,
                combined_signal=d['radius'] * contrast * fread ** 4)


def analyze_bench(results, meta, phi, shots_requested=None, verbose=True):
    """The 9-qubit stage: the mid/late A-B, then the full 16-branch sum at p=0.

    Reuses qcloud_vscr_new's own layout detection and branch statistics so the new
    full-branch point and the existing 4-circuit feasibility point are computed by
    literally the same code -- a difference between them is then a difference in the
    device or the data, not in the analysis."""
    ab = [(r, m) for r, m in zip(results, meta) if m.get('kind') == 'ab']
    bm = [(r, m) for r, m in zip(results, meta) if m.get('kind') == 'bench']
    ab_out = {}
    for r, m in ab:
        n_tot = int(sum(r.values()))
        tup = [(m['psi'], m['p'], m['frame'], m['s'])]
        anc, rev = q.detect_layout(tup, [r])
        n_sel, n_hit = q.counts_to_branch_stats(r, m['s'], anc, rev)
        lo, hi = wilson(n_hit, n_sel)
        ab_out[m['measure']] = dict(
            n_total=n_tot, n_selected=n_sel, n_hit=n_hit,
            selected_fraction=n_sel / n_tot if n_tot else None,
            F_s=(n_hit / n_sel) if n_sel else None, F_s_wilson95=[lo, hi],
            anc_slice=[anc.start, anc.stop], rev=rev,
            top3=sorted(r.items(), key=lambda kv: -kv[1])[:3])
    if len(ab_out) == 2:
        ab_out['verdict'] = dict(
            mid_selected_fraction=ab_out['mid']['selected_fraction'],
            late_selected_fraction=ab_out['late']['selected_fraction'],
            mid_measure_works=bool(
                (ab_out['mid']['selected_fraction'] or 0)
                > 0.5 * (ab_out['late']['selected_fraction'] or 0)),
            note='mid-circuit measurement returned 0 selected shots out of 735 on '
                 'WK_C180; this is the same circuit re-run on this chip, and the '
                 'verdict decides which variant a future full sweep should use')
    tuples = [(m['psi'], m['p'], m['frame'], m['s']) for r, m in bm]
    counts = [r for r, m in bm]
    anc, rev = q.detect_layout(tuples, counts)
    per_branch, per_state = {}, {}
    for (psi, p, fr, s), r in zip(tuples, counts):
        n_tot = int(sum(r.values()))
        n_sel, n_hit = q.counts_to_branch_stats(r, s, anc, rev)
        lo, hi = wilson(n_hit, n_sel)
        per_branch['%s|s%d' % (psi, s)] = dict(
            psi=psi, s=s, n_total=n_tot, n_selected=n_sel, n_hit=n_hit,
            P_s=n_sel / n_tot if n_tot else None,
            F_s=(n_hit / n_sel) if n_sel else None, F_s_wilson95=[lo, hi])
        d = per_state.setdefault(psi, dict(hits=0, selected=0, total=0))
        d['hits'] += n_hit
        d['selected'] += n_sel
        d['total'] += n_tot
    bench = {}
    for psi, d in per_state.items():
        N = d['total'] / 16.0
        F = d['hits'] / N if N > 0 else None
        lo, hi = wilson(d['hits'], int(round(N)))
        F_id = q.simulator_fidelity_frame(psi, BENCH_FRAME, phi)
        bench[psi] = dict(shots_total=d['total'], n_selected=d['selected'],
                          n_hit=d['hits'], effective_N=N, F_hw=F,
                          F_hw_wilson95=[lo, hi], F_ideal_same_frame=F_id,
                          gap=F - F_id if F is not None else None,
                          selected_fraction=d['selected'] / d['total']
                          if d['total'] else None)
    out = dict(ab_test=ab_out, per_branch=per_branch, per_state=bench,
               anc_slice=[anc.start, anc.stop], rev=rev,
               shots_requested=shots_requested)
    if verbose:
        for k, v in ab_out.items():
            if isinstance(v, dict) and 'selected_fraction' in v:
                print('[bench] A/B %s: selected %d/%d (%.3f), F_s=%s %s'
                      % (k, v['n_selected'], v['n_total'], v['selected_fraction'],
                         ('%.3f' % v['F_s']) if v['F_s'] is not None else 'n/a',
                         ['%.3f' % x for x in v['F_s_wilson95']]), flush=True)
        for psi, v in bench.items():
            print('[bench] psi=%s p=0: F_hw = %.4f [%.4f, %.4f] vs ideal %.6f '
                  '(gap %+.4f) from %d hits / %d shots'
                  % (psi, v['F_hw'], v['F_hw_wilson95'][0], v['F_hw_wilson95'][1],
                     v['F_ideal_same_frame'], v['gap'], v['n_hit'],
                     v['shots_total']), flush=True)
    return out


# ---------------------------------------------------------------------
# 10) artifact, figure, and the offline dry run
# ---------------------------------------------------------------------
def load_artifact(path=OUT_JSON):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_artifact(art, path=OUT_JSON):
    art['updated_utc'] = _now()
    with open(path, 'w') as f:
        json.dump(art, f, indent=1, default=str, ensure_ascii=False)
    print('[out] %s' % path, flush=True)
    return path


def stage_record(run):
    """The bookkeeping half of a stage: what was submitted, where, and how many shots
    actually came back -- the last of which was ~92% of the request on WK_C180, and
    is the denominator every statistic in this file uses."""
    return dict(stage=run['stage'], chip_id=run['chip_id'], block=run['block'],
                shots_requested=run['shots_requested'],
                shots_used=run['shots_used'],
                shots_used_total=int(sum(run['shots_used'])),
                shots_expected_total=run['shots_requested'] * len(run['results']),
                shots_return_fraction=(sum(run['shots_used'])
                                       / max(run['shots_requested']
                                             * len(run['results']), 1)),
                wall_s=run['wall_s'], dump_raw=run['dump_raw'],
                n_circuits=len(run['results']),
                instruction_counts=sorted({len(g) for g in run['gates']}),
                attempts=run.get('attempts', []),
                meta=run['meta'])


def dry_run(shots=4000, n_boot=200, seed=7, vendor=None, verbose=True):
    """Sample every probe and tomo circuit on the local CPUQVM and push the counts
    through the ENTIRE analysis chain.  Spends no QPU time and needs no account, and
    it is the only way to learn before buying shots whether the artifact assembles.

    The sampled device is noiseless, so every answer must come out at its ideal value
    within Monte-Carlo error -- w = 1, r = 1, L_Z = L_X = 0, each null control at
    zero -- and a deviation beyond 4/sqrt(shots) is a bug in the chain, not a
    property of a device."""
    pg, pm = stage_circuits('probe')
    pres = [q.cpuqvm_sample(g, shots) for g in pg]
    sigma, probe_out = analyze_probe(pres, pm, shots, verbose=verbose)
    tg, tm = stage_circuits('tomo')
    tres = [q.cpuqvm_sample(g, shots) for g in tg]
    tomo_out, lines = analyze_tomo(tres, tm, sigma, probe_out['assignment'], shots,
                                   n_boot=n_boot, seed=seed, vendor=vendor,
                                   verbose=False)
    if verbose:
        for ln in lines:
            print(ln, flush=True)
    tol = 4.0 / math.sqrt(shots)
    chk = {}
    for psi in TOMO_STATES:
        rec = tomo_out['per_input'][psi]
        chk['w_%s' % psi] = abs(rec['w']['mitigated'] - 1.0)
        chk['r_%s' % psi] = abs(rec['radius_mitigated'] - 1.0)
    L = tomo_out['linear_ensemble']['mitigated']
    chk['L_Z'], chk['L_X'], chk['diff'] = abs(L['L_Z']), abs(L['L_X']), abs(L['diff'])
    for k, v in tomo_out['depth_scan'].items():
        if v['null_control']:
            chk['null_%s_r1' % k] = abs(v['sigma_cs_r1_mitigated'])
            chk['null_%s_r2' % k] = abs(v['sigma_cs_r2_mitigated'])
    worst = max(chk.values())
    if verbose:
        if tomo_out['model'].get('vendor_prediction'):
            print('[dry-run] what the quoted vendor fidelities predict for this '
                  'circuit: %s' % json.dumps(tomo_out['model']['vendor_prediction']),
                  flush=True)
        print('[dry-run] noiseless CPUQVM, %d shots/circuit: worst deviation from '
              'the ideal values = %.4f (tolerance %.4f)' % (shots, worst, tol),
              flush=True)
        print('[dry-run] per-quantity: %s'
              % json.dumps({k: round(v, 5) for k, v in chk.items()}), flush=True)
    assert worst < tol, (worst, tol, chk)
    return dict(probe=probe_out, tomo=tomo_out, worst_deviation=worst,
                tolerance=tol, shots=shots, checks=chk)


def _to_observed(probs_canonical, sigma):
    """Re-key a canonical distribution (char b = qubit b) into the observed order
    under the layout hypothesis sigma (observed char j came from qubit sigma[j])."""
    out = {}
    for key, p in probs_canonical.items():
        obs = ''.join(key[sigma[j]] for j in range(len(key)))
        out[obs] = out.get(obs, 0.0) + p
    return out


def _sample(probs, shots, rng, e01=None, e10=None):
    n = len(next(iter(probs)))
    v = np.zeros(2 ** n)
    for key, p in probs.items():
        v[str_to_idx(key)] = p
    if e01 is not None:
        v = assignment_matrix(e01, e10, n) @ v
    v = np.clip(v, 0, None)
    v = v / v.sum()
    return dict(zip((idx_to_str(i, n) for i in range(2 ** n)),
                    (int(c) for c in rng.multinomial(shots, v))))


def simulate_device(vendor, shots=4000, seed=20260921, n_boot=400,
                    sigma_true=(4, 3, 2, 1, 0), verbose=True):
    """POWER STUDY: what this design would return on a device with the quoted vendor
    fidelities at this shot count -- computed BEFORE any shots are bought.

    Ground truth is the model's own (w, r) at the vendor rates; the synthetic counts
    are that model's output distribution pushed through a readout channel and
    sampled multinomially; the recovered numbers then go through exactly the analysis
    chain the real data will use, including the layout detection, the assignment
    matrix built from the synthetic probe, and the bootstrap.

    What this buys is not a prediction of the device but a prediction of the ERROR
    BARS, which is the only part of the plan the budget can actually settle: it says
    whether the shots on offer resolve r well enough to price the over-read, and it
    says so in the same units the artifact will report."""
    f1, f2, fro = vendor
    e_ro = 1.0 - fro
    e01 = [e_ro] * NQ5
    e10 = [e_ro] * NQ5
    rng = np.random.default_rng(seed)
    truth = {}
    pg, pm = stage_circuits('probe')
    pres = []
    for g, m in zip(pg, pm):
        pres.append(_sample(_to_observed(exact_probs(g), sigma_true), shots, rng,
                            e01, e10))
    sigma, probe_out = analyze_probe(pres, pm, shots, verbose=False)
    assert tuple(sigma) == tuple(sigma_true), (sigma, sigma_true)
    tg, tm = stage_circuits('tomo')
    tres = []
    for g, m in zip(tg, tm):
        # `g` already contains prep(psi), so psi_vec must stay at |00000>: passing
        # prep_vec() here as well applies the prep twice, which silently turns |+>
        # into |0> and every ensemble average into nonsense.  The self-tests caught
        # exactly this in test (e); the same trap is guarded here by construction.
        probs = model_output_probs(g, 1.0 - f1, 1.0 - f2)
        tres.append(_sample(_to_observed(probs, sigma_true), shots, rng, e01, e10))
        if m.get('kind') != 'tomo':
            continue          # the job's own calibration circuits, no truth to record
        d = model_mixedness(g, p1=1.0 - f1, p2=1.0 - f2)
        truth['%s|%s|r%d' % (m['psi'], m['basis'], m['rounds'])] = dict(
            w=d['codespace_weight'], r=d['radius'],
            sigma_cs=d['bloch'][2], sigma_z_uncond=d['sigma_z_uncond'])
    # Use the calibration the tomo job CARRIES, not the probe's, so the power study
    # exercises the same path analyze_stage takes on real counts (QCLOUD_SETUP.md 8.2:
    # a submission has one block, so a job must not assume the canary's block is
    # still its own).
    zs = [r for r, m in zip(tres, tm)
          if m.get('kind') == 'cal' and m.get('cal') == 'zero']
    os_ = [r for r, m in zip(tres, tm)
           if m.get('kind') == 'cal' and m.get('cal') == 'one']
    assert len(zs) == 2 and len(os_) == 1, (len(zs), len(os_))
    tomo_cal = assignment_from_cal(zs[0], os_[0], NQ5)
    out, lines = analyze_tomo(tres, tm, sigma, tomo_cal, shots,
                              n_boot=n_boot, seed=seed, vendor=vendor, verbose=False)
    if verbose:
        print('[simulate] device model: f1q=%.4f f2q=%.4f fread=%.4f, %d shots per '
              'circuit, layout %s recovered exactly' % (f1, f2, fro, shots,
                                                        list(sigma)), flush=True)
        for ln in lines:
            print(ln, flush=True)
        print('[simulate] ground truth vs recovered (mitigated):')
        for psi in TOMO_STATES:
            t = truth['%s|Z|r1' % psi]
            rec = out['per_input'][psi]
            b = rec['bootstrap']['mitigated']
            print('   |%s>_L  w: true %.4f  recovered %.4f   r: true %.4f  '
                  'recovered %.4f [%.4f, %.4f]  raw %.4f'
                  % (psi, t['w'], rec['w']['mitigated'], t['r'],
                     rec['radius_mitigated'], b['radius_ci95'][0],
                     b['radius_ci95'][1], rec['radius_raw']), flush=True)
        d0 = out['per_input']['0']['bootstrap']['mitigated']
        half = 0.5 * (d0['radius_ci95'][1] - d0['radius_ci95'][0])
        print('[simulate] resolution: 95%% CI half-width on r = %.4f at %d shots; '
              '%d shots would be needed for +-0.02'
              % (half, shots,
                 int(shots * (half / 0.02) ** 2) if half > 0 else shots), flush=True)
        L = out['linear_ensemble']['mitigated']
        print('[simulate] M2 achievable: |L_Z - L_X| bounded by +-%.4f (95%%) while '
              'the terms span %.3f, i.e. %.1f%% of the span'
              % (1.96 * L['stderr'], L['term_span'],
                 100 * L['ci_over_span']), flush=True)
    return dict(vendor=list(vendor), shots=shots, seed=seed, truth=truth,
                tomo=out, probe=probe_out, sigma=list(sigma))


def seed_scan(vendor, shots=8000, seeds=(20260921, 11, 202, 3003, 40004, 777777),
              verbose=True):
    """Repeat the power study over several sampling seeds and report the DISTRIBUTION
    of the two test statistics, not one draw of them.

    This exists because M2's null hypothesis is "L_Z - L_X = 0", and a single draw
    lands beyond 2 sigma about 5% of the time.  Without the scan, one unlucky draw
    would look like a systematic violation of an identity that selftest (e) proves
    to 5e-17; with it, the reported z-scores can be seen to have mean ~0 and rms ~1,
    which is what "consistent with exact linearity, limited by shots" means.  The
    same scan gives the seed-to-seed scatter of r, an independent check that the
    bootstrap CI is the right width."""
    zs, rs, ws = [], [], []
    for sd in seeds:
        s = simulate_device(vendor, shots=shots, seed=sd, n_boot=1, verbose=False)
        L = s['tomo']['linear_ensemble']['mitigated']
        zs.append(L['diff'] / L['stderr'] if L['stderr'] > 0 else float('nan'))
        rs.append(s['tomo']['per_input']['0']['radius_mitigated'])
        ws.append(s['tomo']['per_input']['0']['w']['mitigated'])
        if verbose:
            print('  seed %-9d L_Z=%+.5f L_X=%+.5f diff=%+.5f +-%.5f = %+5.2f '
                  'sigma   r(|0>)=%.4f  w=%.4f'
                  % (sd, L['L_Z'], L['L_X'], L['diff'], L['stderr'], zs[-1],
                     rs[-1], ws[-1]), flush=True)
    zs = np.array(zs, dtype=float)
    out = dict(vendor=list(vendor), shots=shots, seeds=list(seeds),
               m2_zscores=[float(z) for z in zs],
               m2_z_mean=float(np.mean(zs)), m2_z_rms=float(np.sqrt(np.mean(zs ** 2))),
               m2_z_max_abs=float(np.max(np.abs(zs))),
               r_scatter_ddof1=float(np.std(rs, ddof=1)),
               r_values=[float(x) for x in rs],
               w_scatter_ddof1=float(np.std(ws, ddof=1)),
               interpretation=(
                   'mean ~0 and rms ~1 means the residual L_Z - L_X is sampling '
                   'noise on an identity that holds exactly, not a systematic; the '
                   'r scatter is an independent estimate of the same error the '
                   'bootstrap reports, so the two agreeing is what licenses the '
                   'quoted CI'))
    if verbose:
        print('[seed-scan] M2 z-score over %d seeds: mean %+.2f  rms %.2f  '
              'max|z| %.2f   (pure sampling noise => mean ~0, rms ~1)'
              % (len(seeds), out['m2_z_mean'], out['m2_z_rms'],
                 out['m2_z_max_abs']), flush=True)
        print('[seed-scan] r(|0>) scatter %.4f, w scatter %.4f over the same seeds'
              % (out['r_scatter_ddof1'], out['w_scatter_ddof1']), flush=True)
    return out


def make_figure(art, tag):
    """Four panels: the measured radius against the model-implied one, the code-space
    weight at both depths, the M2 ensemble equality, and the branch-summed point."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(ROOT, 'figures'), exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    tomo = art.get('tomo')
    if tomo:
        pi = tomo['per_input']
        xs = np.arange(len(TOMO_STATES))
        a = ax[0][0]
        for i, var in enumerate(('raw', 'mitigated')):
            vals = [pi[p]['radius_' + var] for p in TOMO_STATES]
            ci = [pi[p]['bootstrap'][var]['radius_ci95'] for p in TOMO_STATES]
            err = np.array([[v - lo for v, (lo, _h) in zip(vals, ci)],
                            [hi - v for v, (_lo, hi) in zip(vals, ci)]])
            a.errorbar(xs + 0.18 * (i * 2 - 1), vals, yerr=err, fmt='os'[i],
                       capsize=3, label='r %s' % var)
        try:
            mb = json.load(open(os.path.join(ROOT, 'mixed_input_boundary.json')))
            a.axhline(mb['device_prep']['radius_max'], ls='--', color='gray',
                      label='model-implied r, 36 gates (ED 10d)')
        except Exception:
            pass
        if tomo['model'].get('rounds1', {}).get('p_instr_fitted') is not None:
            a.axhline(tomo['model']['rounds1']['r_predicted'], ls=':', color='crimson',
                      label='model at the fitted rate, matched depth')
        a.set_xticks(xs)
        a.set_xticklabels(['|%s>L' % p for p in TOMO_STATES])
        a.set_ylim(0, 1.05)
        a.set_ylabel('Bloch radius r')
        a.legend(fontsize=7)
        a.grid(alpha=0.3)
        a.set_title('M1 measured device mixedness')
        b = ax[0][1]
        b.bar(xs - 0.18, [pi[p]['w']['mitigated'] for p in TOMO_STATES], 0.34,
              label='w, depth 1 (78 instr)')
        ds = tomo['depth_scan']
        b.bar(xs[:2] + 0.18, [ds['0|Z']['w_r2'], ds['+|X']['w_r2']], 0.34,
              label='w, depth 2 (150 instr)')
        b.set_xticks(xs)
        b.set_xticklabels(['|%s>L' % p for p in TOMO_STATES])
        b.set_ylabel('code-space weight w')
        b.legend(fontsize=7)
        b.grid(alpha=0.3, axis='y')
        b.set_title('M5 code-space weight vs depth')
        c = ax[1][0]
        for i, var in enumerate(('raw', 'mitigated')):
            L = tomo['linear_ensemble'][var]
            c.bar(np.arange(4) + 0.18 * (i * 2 - 1),
                  [L['terms'][p] for p in TOMO_STATES], 0.34, label='terms %s' % var)
        L = tomo['linear_ensemble']['mitigated']
        c.axhline(L['L_Z'], color='navy', ls='-', label='L_Z = %.4f' % L['L_Z'])
        c.axhline(L['L_X'], color='crimson', ls='--', label='L_X = %.4f' % L['L_X'])
        c.set_xticks(np.arange(4))
        c.set_xticklabels(['<Z_L> |%s>' % p for p in TOMO_STATES])
        c.set_ylabel('unconditional <Z_L>')
        c.legend(fontsize=7)
        c.grid(alpha=0.3, axis='y')
        c.set_title('M2 two pure ensembles of I/2: L_Z must equal L_X')
    d = ax[1][1]
    bench = (art.get('bench') or {}).get('per_state')
    if bench:
        ks = sorted(bench)
        d.bar(np.arange(len(ks)) - 0.18, [bench[k]['F_ideal_same_frame'] for k in ks],
              0.34, label='ideal protocol (same frame)')
        err = np.array([[bench[k]['F_hw'] - bench[k]['F_hw_wilson95'][0] for k in ks],
                        [bench[k]['F_hw_wilson95'][1] - bench[k]['F_hw'] for k in ks]])
        d.bar(np.arange(len(ks)) + 0.18, [bench[k]['F_hw'] for k in ks], 0.34,
              yerr=err, capsize=3, label='hardware, 16 branches summed')
        d.set_xticks(np.arange(len(ks)))
        d.set_xticklabels(['|%s> p=0' % k for k in ks])
        d.set_ylabel('recovery fidelity F')
        d.legend(fontsize=7)
        d.grid(alpha=0.3, axis='y')
        d.set_title('full-branch benchmark point')
    else:
        d.text(0.5, 0.5, 'bench stage not run', ha='center', va='center',
               transform=d.transAxes)
        d.set_axis_off()
    fig.tight_layout()
    path = os.path.join(ROOT, 'figures', 'fig_hw_mixedness_%s.png' % tag)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print('[out] %s' % path, flush=True)
    return path


# ---------------------------------------------------------------------
# 11) stage plumbing shared by --mode cloud and --mode analyze, and the CLI
# ---------------------------------------------------------------------
def analyze_stage(stage, results, meta, shots_requested, art, phi=None,
                  vendor=None, n_boot=2000, verbose=True):
    """Push one stage's counts through the analysis and merge the result into `art`.

    The tomo stage depends on the probe stage for the bit order and the assignment
    matrix.  If no probe is available the layout is recovered from the tomography
    circuits themselves -- which determines the logical position and leaves the four
    code-space positions tied, provably harmlessly (selftest (f)) -- and mitigation
    is skipped, with the fallback recorded in the artifact rather than hidden."""
    if stage == 'probe':
        sigma, out = analyze_probe(results, meta, shots_requested, verbose=verbose)
        art['probe'] = dict(analysis=out, shots_requested=shots_requested,
                            n_circuits=len(results))
        art.setdefault('meta', {})['layout_sigma'] = list(sigma)
        return out
    if stage in ('tomo', 'ladder'):
        pr = art.get('probe', {}).get('analysis')
        # the bit order needs the fingerprint circuits, which only the probe stage
        # submits; the assignment matrix does not, and the tomo stage carries its
        # own three calibration circuits so that it does not have to assume the
        # probe's block is still this job's block.
        own = [m.get('cal') for m in meta if m.get('kind') == 'cal']
        own_cal = None
        if own.count('zero') >= 2 and own.count('one') >= 1:
            zs = [r for r, m in zip(results, meta)
                  if m.get('kind') == 'cal' and m.get('cal') == 'zero']
            os_ = [r for r, m in zip(results, meta)
                   if m.get('kind') == 'cal' and m.get('cal') == 'one']
            own_cal = assignment_from_cal(zs[0], os_[0], NQ5)
        if pr:
            sigma = tuple(pr['layout_sigma'])
            lay_src = 'probe stage'
        else:
            tm = [m for m in meta if m.get('kind') == 'tomo']
            tr = [r for r, m in zip(results, meta) if m.get('kind') == 'tomo']
            ideals = [exact_probs(build_tomo(m['psi'], m['basis'], m['rounds']))
                      for m in tm]
            sigma, ev = detect_layout_ml(tr, ideals, verbose=verbose)
            lay_src = ('the tomography circuits alone (the four code-space positions '
                       'are tied and readout mitigation is applied per observed '
                       'position, which selftest (f) proves is harmless)')
            art.setdefault('meta', {})['layout_evidence'] = ev
        if own_cal is not None:
            cal, source = own_cal, 'this job\'s own calibration circuits'
        elif pr:
            cal, source = pr['assignment'], 'probe stage'
        else:
            cal = dict(e01=[0.0] * NQ5, e10=[0.0] * NQ5, n_zero=0, n_one=0)
            source = ('none available: readout mitigation is skipped, so every '
                      'mitigated number equals its raw value')
        if pr and own_cal is not None:
            d = max(abs(a - b) for a, b in zip(pr['assignment']['e01'], own_cal['e01']))
            source += ' (max |e01 - probe e01| = %.4f)' % d
        source += '; layout from ' + lay_src
        out, lines = analyze_tomo(results, meta, sigma, cal, shots_requested,
                                  n_boot=n_boot, vendor=vendor, verbose=False)
        out['calibration_source'] = source
        art['tomo'] = out
        art.setdefault('meta', {})['layout_sigma'] = list(sigma)
        if verbose:
            for ln in lines:
                print(ln, flush=True)
        return out
    if stage == 'bench':
        assert phi is not None
        out = analyze_bench(results, meta, phi, shots_requested, verbose=verbose)
        art['bench'] = out
        return out
    raise ValueError(stage)


def _load_dump(stage):
    path = os.path.join(ROOT, STAGE_RAW[stage])
    with open(path) as f:
        raw = json.load(f)
    return [{str(k): int(v) for k, v in d.items()} for d in raw]


def print_plan(phi, budget_s=300.0, vendor=None):
    """Print the executable run plan of QCLOUD_SETUP.md 8, with the circuit counts
    and instruction totals computed from the builders rather than typed in, so the
    table cannot drift away from the code it describes.  Touches nothing."""
    ms_per_shot = 14.0        # the pessimistic model: 1800 s / 128 000 shots
    rows, tot_shots, tot_cost = [], 0, 0.0
    print('=' * 78)
    print('RUN PLAN -- %s, one submission per job, one physical block per job'
          % 'OriginQ Wukong')
    print('Cost model is UNKNOWN and the two candidates differ ~50x (8.1):')
    print('  pessimistic  metered per shot   -> 300 s buys ~%d shots'
          % int(budget_s * 1000 / ms_per_shot))
    print('  optimistic   metered per job wall clock (queue included)')
    print('               -> shots are nearly free, the JOB COUNT is the budget')
    print('  evidence for neither is in the artifacts; Job 1 exists to decide.')
    print('=' * 78)
    for stage in ('probe', 'tomo', 'bench', 'ladder'):
        gates, meta = stage_circuits(stage, phi if stage == 'bench' else None)
        instr = [len(g) for g in gates]
        cnot = [sum(1 for x in g if x[0] == 'CNOT') for g in gates]
        shots = STAGE_SHOTS[stage]
        n_shots = shots * len(gates)
        cost = n_shots * ms_per_shot / 1000.0
        tot_shots += n_shots
        tot_cost += cost
        depths = sorted({m['rounds'] for m in meta if 'rounds' in m})
        rows.append((STAGE_JOB[stage], stage, STAGE_NQ[stage], len(gates),
                     min(instr), max(instr), sum(cnot), shots, n_shots, cost,
                     depths))
    hdr = ('job stage   nq  circ  instr/circuit   CNOT   shots/circ  total shots  '
           'pessimistic cost  depths')
    print(hdr)
    print('-' * len(hdr))
    for (job, stage, nq, nc, imin, imax, ct, sh, tot, cost, depths) in rows:
        print('%3d  %-7s %2d  %4d  %5d-%-5d %7d %10d %12d %11.0f s   %s'
              % (job, stage, nq, nc, imin, imax, ct, sh, tot, cost,
                 depths or '-'))
    print('-' * len(hdr))
    print('    %-7s      %4d                            %12d %11.0f s'
          % ('TOTAL', sum(r[3] for r in rows), tot_shots, tot_cost))
    print()
    print('THE SHOTS / COST / PRECISION TRADE-OFF FOR JOB 2 (19 circuits).')
    print('This is the decision the abort rules actually turn on, so it is printed')
    print('rather than left in a document.  "sigma(r<1)" assumes the vendor-')
    print('predicted r~0.86 and CI = 0.028 x sqrt(8000/S), the scaling measured by')
    print('the power study; the pessimistic cost is S x 19 x 14 ms.')
    print('  shots/circ  total shots  pessimistic cost  CI half-width  sigma(r<1)')
    for s_ in (8000, 4000, 2000, 1000, 500):
        n_ = s_ * 19
        ci = 0.028 * math.sqrt(8000.0 / s_)
        print('  %10d  %11d  %13.0f s  %13.3f  %10.1f%s'
              % (s_, n_, n_ * ms_per_shot / 1000.0, ci, 0.14 / ci,
                 '   <- whole remaining budget' if s_ == 1000 else ''))
    print('  => under the PESSIMISTIC model Job 2 at 1000 shots (266 s) IS the')
    print('     entire remaining budget: Jobs 3 and 4 are then impossible, and the')
    print('     right choice is to spend it on Job 2 (new science) rather than on')
    print('     Job 3 (a better-statistics repeat of the 2026-09 feasibility run).')
    print('  => the built-in halving ladder 8000->4000->2000->1000 is itself the')
    print('     MEASUREMENT of the cost model: a refusal is returned at submission')
    print('     time and costs nothing (2026-09: batch1 ran, batch2 was refused),')
    print('     so each refusal narrows the affordable shot count for free.')
    print('     ASSUMPTION BEING RELIED ON: the vendor refuses a job whose cost')
    print('     exceeds the remaining budget, rather than accepting it and letting')
    print('     it overrun.  If that is wrong, starting at 8000 could bill ~2100 s')
    print('     against a 300 s allocation -- so start Job 2 at 1000 unless the')
    print('     user has confirmed the quota is hard-limited at submission.')
    print()
    print('ABORT RULES (8.3) -- w1 = Job 1 wall clock, measured, not assumed:')
    print('  Job 1 refused ("QPU time is insufficient") -> budget already gone, STOP')
    print('  w1 <= 60 s   -> Job 2 at 8000 shots (halving ladder on refusal),')
    print('                  then Job 3 at 2000, then Job 4 at 4000')
    print('  60 < w1 <= 180 s -> Job 2 at 3000 shots; Job 3 at 1000 only if')
    print('                  w1+w2 <= 240 s; Job 4 dropped')
    print('  w1 > 180 s   -> Job 2 at 1500 shots, and DROP its depth-2 grid and')
    print('                  its own calibration circuits (12 d=1 circuits only);')
    print('                  Jobs 3 and 4 dropped')
    print('  a refusal costs nothing, so shots start optimistic and walk down')
    print()
    print('WHAT EACH JOB BUYS ON ITS OWN (8.4):')
    print('  Job 1  5x5 readout assignment matrix on a second chip (fills the')
    print('         ChipInfo 401 gap), unique bit order, the platform\'s ~8% shots')
    print('         shortfall, and the cost model.  No mixedness claim.')
    print('  Job 2  M1 measured w/r/purity (r CI half-width 0.028 at 8000 shots,')
    print('         so r<1 is a ~5 sigma statement), M2 the linear-fraction')
    print('         ensemble equality to +-0.023, M3 the isotropic surrogate at')
    print('         the hardware depth, M5 two null controls.')
    print('  Job 3  upgrades the paper\'s "4 circuits x 400 shots feasibility" to a')
    print('         16-branch protocol benchmark on a second chip, plus the')
    print('         mid/late A/B that re-tests the documented anomaly.')
    print('  Job 4  rounds=%s -> a measured radius-vs-depth curve, i.e. a per-round'
          % (LADDER_ROUNDS,))
    print('         logical error rate to compare against the threshold behaviour.')
    print()
    print('NOT PURSUED AT ANY BUDGET: the over-read delta^T C delta (~5e7 shots)')
    print('and any significance claim for "4.4% over identity" (~2 sigma at best).')
    if vendor:
        vp = vendor_prediction(*vendor)
        print()
        print('what the quoted vendor fidelities predict for the depth-1 circuit:')
        for k in ('two_qubit_survival', 'codespace_weight', 'radius',
                  'radius_as_measured'):
            print('  %-24s %.6f' % (k, vp[k]))
    print('=' * 78)
    return dict(rows=[dict(zip(('job', 'stage', 'nq', 'circuits', 'instr_min',
                                'instr_max', 'cnot_total', 'shots_per_circuit',
                                'shots_total', 'pessimistic_cost_s', 'depths'), r))
                      for r in rows],
                shots_total=tot_shots, pessimistic_cost_s=tot_cost,
                budget_s=budget_s, ms_per_shot_assumed=ms_per_shot)


def main(argv=None):
    # declared first: LADDER_ROUNDS is read by the --rounds help string below and
    # rebound from it, and Python requires the global declaration to precede any
    # use of the name in the function body
    global LADDER_ROUNDS
    ap = argparse.ArgumentParser(
        description='measured device mixedness for the mixed-input boundary',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', default='verify',
                    choices=['verify', 'sample', 'selftest', 'simulate', 'preflight',
                             'plan', 'cloud', 'analyze'])
    ap.add_argument('--stage', default='all',
                    choices=['probe', 'tomo', 'bench', 'ladder', 'all'])
    ap.add_argument('--rounds', default=None,
                    help='comma-separated extra depths for the ladder stage '
                         '(default %s; 8 costs 581 instructions/288 CNOT per circuit)'
                         % (LADDER_ROUNDS,))
    ap.add_argument('--chip-id', default='WK_C180_2')
    ap.add_argument('--chip-ids', default=None,
                    help='comma-separated chips for --mode preflight '
                         '(default: --chip-id)')
    ap.add_argument('--token', default=os.environ.get('ORIGINQ_TOKEN', ''))
    ap.add_argument('--shots', type=int, default=None,
                    help='override the per-stage default %s' % STAGE_SHOTS)
    ap.add_argument('--block', default=None,
                    help='comma-separated physical qubit block (default: the '
                         'top-ranked best_qubit_blocks(n) on the chip, free query)')
    ap.add_argument('--chunk', type=int, default=None,
                    help='circuits per job (default: the whole stage in ONE job)')
    ap.add_argument('--job-timeout', type=int, default=21600)
    ap.add_argument('--no-options', action='store_true')
    ap.add_argument('--n-boot', type=int, default=2000)
    ap.add_argument('--seeds', type=int, default=1,
                    help='for --mode simulate: repeat the power study over this '
                         'many sampling seeds and report the distribution of the '
                         'M2 z-score and of r, so one unlucky draw is not mistaken '
                         'for a systematic')
    ap.add_argument('--noise', default='depolarizing')
    ap.add_argument('--vendor', default=None,
                    help='f1q,f2q,fread -- record what these aggregate vendor '
                         'fidelities predict for this circuit through the same model')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--json', default=OUT_JSON)
    ap.add_argument('--no-figure', action='store_true')
    ap.add_argument('--figure', action='store_true',
                    help='for --mode simulate only: also write the figure. Off by '
                         'default so a power-study plot cannot be mistaken for '
                         'hardware data sitting in the same figures/ directory')
    a = ap.parse_args(argv)

    setup()
    vendor = tuple(float(x) for x in a.vendor.split(',')) if a.vendor else None
    if a.rounds:
        LADDER_ROUNDS = tuple(int(x) for x in a.rounds.split(',') if x.strip())
    stages = (['probe', 'tomo', 'bench', 'ladder'] if a.stage == 'all'
              else [a.stage])

    if a.mode == 'selftest':
        st = selftests()
        print('[selftest] PASS %s' % json.dumps(st, default=str), flush=True)
        return 0

    if a.mode == 'verify':
        st = selftests()
        print('\n=== exact ideal reference of every submitted circuit ===')
        for stage in stages:
            phi = q.load_angles(a.noise) if stage == 'bench' else None
            gates, meta = stage_circuits(stage, phi)
            inv = {}
            for g in gates:
                key = (len(g), sum(1 for x in g if x[0] == 'CNOT'),
                       sum(1 for x in g if x[0] == 'MEAS'))
                inv[key] = inv.get(key, 0) + 1
            print('  stage %-6s %2d circuits, (instructions, CNOT, MEAS) '
                  'histogram %s' % (stage, len(gates), dict(inv)))
            if stage in ('tomo', 'ladder'):
                for g, m in zip(gates, meta):
                    if m.get('kind') != 'tomo':
                        print('    cal|%s rep%d  ideal: a product of |0> (or |1> '
                              'on every qubit), used only for the assignment matrix'
                              % (m.get('cal'), m.get('rep')))
                        continue
                    w, nvec = ideal_bloch_of(g)
                    print('    %s|%s|r%d  ideal w=%.12f  <Z>rot=%+.12f (want %+.0f)'
                          % (m['psi'], m['basis'], m['rounds'], w, nvec[2],
                             IDEAL_BLOCH[m['psi']][BASIS_AXIS[m['basis']]]))
        print('[verify] self-tests %s' % json.dumps(st, default=str), flush=True)
        return 0

    if a.mode == 'sample':
        if not q.HAS_PYQPANDA3:
            print('ERROR: pyqpanda3 required: %s' % q._PYQPANDA_IMPORT_ERR)
            return 3
        dry_run(shots=a.shots or 4000, vendor=vendor)
        return 0

    if a.mode == 'simulate':
        if vendor is None:
            print('ERROR: --mode simulate needs --vendor f1q,f2q,fread')
            return 2
        sim = simulate_device(vendor, shots=a.shots or STAGE_SHOTS['tomo'],
                              n_boot=a.n_boot)
        art = load_artifact(a.json)
        art['power_study'] = sim
        if a.seeds > 1:
            art['power_study']['seed_scan'] = seed_scan(
                vendor, shots=a.shots or STAGE_SHOTS['tomo'],
                seeds=tuple(SEED_BANK[:a.seeds]))
        art.setdefault('meta', {}).update(dict(
            task='power study for the measured-device-mixedness design',
            mode=a.mode, vendor=vendor,
            shots=a.shots or STAGE_SHOTS['tomo'], spends_qpu_time=False))
        if a.figure:
            # off by default: a power-study figure sitting in figures/ beside the
            # real hardware ones is exactly the kind of thing that gets mistaken
            # for data in a manuscript, so it has to be asked for explicitly
            make_figure(dict(tomo=sim['tomo']),
                        a.tag or 'power_study_%s' % a.chip_id)
        save_artifact(art, a.json)
        return 0

    if a.mode == 'preflight':
        chips = ([c.strip() for c in a.chip_ids.split(',') if c.strip()]
                 if a.chip_ids else [a.chip_id])
        out = preflight(chips, token=a.token or None)
        if vendor:
            out['vendor_prediction'] = vendor_prediction(*vendor)
            print('[preflight] what the quoted vendor fidelities predict: %s'
                  % json.dumps(out['vendor_prediction']), flush=True)
        path = os.path.join(ROOT, 'hw_mixedness_preflight.json')
        with open(path, 'w') as f:
            json.dump(out, f, indent=1, default=str, ensure_ascii=False)
        print('[out] %s' % path, flush=True)
        return 0

    # ---- cloud / analyze ----
    phi = q.load_angles(a.noise)
    q.PHI_GLOBAL = phi

    if a.mode == 'plan':
        plan = print_plan(phi, vendor=vendor)
        art0 = load_artifact(a.json)
        art0.setdefault('meta', {})['plan'] = plan
        save_artifact(art0, a.json)
        return 0

    art = load_artifact(a.json)
    art.setdefault('meta', {}).update(dict(
        task='measured device mixedness for the mixed-input boundary',
        chip_id=a.chip_id, mode=a.mode, stages=stages, vendor=vendor,
        angles=(os.environ.get('VSCR_ANGLES_FILE')
                or 'vscr_angles_%s.npz'
                % ('dep' if a.noise == 'depolarizing' else a.noise)),
        python=sys.version.split()[0], numpy=np.__version__))

    if a.mode == 'analyze':
        loaded = {}
        for stage in stages:
            path = os.path.join(ROOT, STAGE_RAW[stage])
            if not os.path.exists(path):
                print('[analyze] %s: no dump at %s -- skipped (that job was '
                      'dropped by the abort rules, or has not run yet)'
                      % (stage, STAGE_RAW[stage]), flush=True)
                continue
            results = _load_dump(stage)
            _g, meta = stage_circuits(stage, phi if stage == 'bench' else None)
            assert len(results) == len(meta), (len(results), len(meta), stage)
            print('[analyze] %s: %d circuits from %s'
                  % (stage, len(results), STAGE_RAW[stage]), flush=True)
            loaded[stage] = (results, meta)
        # the ladder job is the SAME estimator at deeper rounds, so its counts are
        # appended to the tomography's and analysed once; analysed alone it would
        # have no depth-1 circuit and analyze_tomo would (correctly) refuse it
        if 'tomo' in loaded and 'ladder' in loaded:
            (rt, mt), (rl, ml) = loaded['tomo'], loaded['ladder']
            loaded['tomo'] = (rt + rl, mt + ml)
            print('[analyze] ladder merged into tomo: %d + %d circuits, depths %s'
                  % (len(rt), len(rl),
                     sorted({m['rounds'] for m in mt + ml})), flush=True)
            loaded.pop('ladder')
        for stage in [s for s in stages if s in loaded]:
            results, meta = loaded[stage]
            # read back what was ACTUALLY requested for this stage rather than
            # assuming the default: the abort rules mean the tomo job may have run
            # at 8000, 3000 or 1500 shots, and the ladder at something else again
            sr = a.shots or art.get('runs', {}).get(stage, {}).get(
                'shots_requested') or STAGE_SHOTS[stage]
            analyze_stage(stage, results, meta, sr, art,
                          phi=phi, vendor=vendor, n_boot=a.n_boot)
    else:
        if a.stage == 'all':
            print('ERROR: --mode cloud refuses --stage all.  Each job is one queue '
                  'wait and the abort rules of QCLOUD_SETUP.md 8.3 are a HUMAN '
                  'decision taken from the measured wall clock of the previous job; '
                  'running all four stages unattended could spend the entire 300 s '
                  'budget on the first one.  Submit them one at a time:\n'
                  '  bash launch_mixedness.sh probe   # then read wall_s\n'
                  '  bash launch_mixedness.sh tomo\n'
                  '  bash launch_mixedness.sh bench\n'
                  '  bash launch_mixedness.sh ladder')
            return 2
        tok = a.token
        if not tok:
            import hw_error_budget as heb
            tok = heb.resolve_token()
        if not q.HAS_PYQPANDA3:
            print('ERROR: pyqpanda3 required: %s' % q._PYQPANDA_IMPORT_ERR)
            return 3
        block = ([int(x) for x in a.block.split(',') if x.strip()]
                 if a.block else None)
        for stage in stages:
            shots = a.shots or STAGE_SHOTS[stage]
            attempts, run = [], None
            while True:
                try:
                    run = run_stage(stage, tok, a.chip_id, shots=shots, block=block,
                                    chunk=a.chunk, job_timeout=a.job_timeout,
                                    use_options=not a.no_options, phi=phi)
                    break
                except Exception as exc:
                    attempts.append(dict(stage=stage, shots=shots,
                                         error=repr(exc)[:400]))
                    msg = str(exc).lower()
                    if (not any(k in msg for k in QPU_REFUSAL_MARKERS)
                            or shots <= SHOTS_FLOOR):
                        art.setdefault('refused_attempts', []).extend(attempts)
                        save_artifact(art, a.json)
                        raise
                    nxt = max(SHOTS_FLOOR, shots // 2)
                    print('[cloud] %s refused at %d shots (%s); retrying at %d'
                          % (stage, shots, repr(exc)[:160], nxt), flush=True)
                    shots = nxt
            run['attempts'] = attempts
            art.setdefault('runs', {})[stage] = stage_record(run)
            analyze_stage(stage, run['results'], run['meta'],
                          run['shots_requested'], art, phi=phi, vendor=vendor,
                          n_boot=a.n_boot)
            # save after EVERY stage: a quota refusal on a later one then costs
            # nothing that was already measured
            save_artifact(art, a.json)
            block = None

    tag = a.tag or '%s_%s' % (a.chip_id, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if not a.no_figure and 'tomo' in art:
        make_figure(art, tag)
    save_artifact(art, a.json)
    return 0


if __name__ == '__main__':
    sys.exit(main())

