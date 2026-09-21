"""make_ed.py — build paper/extended_data.tex (ED tables) from the analysis
artifacts: ideal per-frame references, per-branch conditional fidelities,
hardware feasibility statistics, and bit-layout evidence."""
import json, os, sys
import math
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, '..'))
NUM = json.load(open(os.path.join(ROOT, '..', 'paper_numbers.json')))
HW = json.load(open(os.path.join(ROOT, '..', 'hw_feasibility_numbers.json')))
RES = os.path.join(ROOT, '..', 'vscr_paper_results.npz')

lines = []
A = lines.append
A(r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{amsmath,amssymb,booktabs,longtable,array}
\usepackage{graphicx}
\graphicspath{{figures/}}
\begin{document}
\section*{Extended Data Tables --- VSCR $[\![5,1,3]\!]$ manuscript}""")

# ---------------- ED Table 1: ideal per-frame references ----------------
def sci(x, digits=2):
    """Scientific notation, already wrapped in math mode.

    `\\times` is a math-mode symbol, so emitting it bare makes every table cell
    that uses this helper a LaTeX error ("Missing $ inserted").  It was emitted
    bare in 411 places across the ED tables and never caught, because no TeX
    engine exists on this host and extended_data.tex has therefore never been
    compiled.  Wrapping here fixes all of them at once; `audit_numbers.py` now
    asserts the property directly, by stripping every $...$ span from the
    generated file and requiring no `\\times` to survive."""
    mnt, ex = f'{x:.{digits}e}'.split('e')
    return f'${mnt}\\times 10^{{{int(ex)}}}$'


hw = NUM['hardware_frame_numbers']
A(r"""
\subsection*{ED Table 1: exact ideal reference of the hardware benchmark}
Zero-noise per-frame circuit fidelity $F=\sum_s F_s$ (dense statevector,
late-measure branch circuits compiled from the trained depolarizing model)
for logical states $\ket{0},\ket{+}$; every single-Pauli frame is corrected
exactly, so the $p$-dependent ideal benchmark value follows from Pauli-frame
averaging, $F(p)=1-O(p^2)$. Worst circuit-vs-density-matrix deviation
%s. Frames are single-qubit Pauli injections.""" %
  sci(hw['worst_circuit_vs_simulator']))
A(r'\begin{longtable}{llcc}')
A(r'\toprule frame & correction $C_s$ & $F(\ket{0})$ & $F(\ket{+})$ \\ \midrule')
import re as _re
for i, fr in enumerate(hw['frames']):
    f0 = hw['per_frame'][f'0|{fr}']['F_circuit']
    fp = hw['per_frame'][f'+|{fr}']['F_circuit']
    A(f'{fr} & ${i}$ & {f0:.6f} & {fp:.6f} \\\\')
A(r'\bottomrule\end{longtable}')

# ---------------- ED Table 2: per-branch conditional fidelities ----------
if os.path.exists(RES):
    d = np.load(RES, allow_pickle=True)
    A(r"""
\subsection*{ED Table 2: per-branch conditional fidelity $\mathrm{cf}_s$}
Depolarizing channel; $p=0.05$ and $p=0.15$; perfect-code decoder vs
cold-started v1 model vs warm-started paper model.""")
    A(r'\begin{longtable}{lccc|ccc}')
    A(r'\toprule & \multicolumn{3}{c}{$p=0.05$} & \multicolumn{3}{c}{$p=0.15$} \\')
    A(r'branch $s$ ($C_s$) & dec & cold & warm & dec & cold & warm \\ \midrule')
    import ssvr_qec as m  # noqa  (only for SYND_TABLE labels)
    for s in range(16):
        lab = m.SYND_TABLE[m.SYND_BITS[s]]
        A(f'{s} ({lab}) & {d["cf_dec_05"][s]:.4f} & {d["cf_cold_05"][s]:.4f} & '
          f'{d["cf_warm_05"][s]:.4f} & {d["cf_dec_15"][s]:.4f} & '
          f'{d["cf_cold_15"][s]:.4f} & {d["cf_warm_15"][s]:.4f} \\\\')
    A(r'\bottomrule\end{longtable}')

# ---------------- ED Table 3: hardware feasibility statistics ------------
A(r"""
\subsection*{ED Table 3: WK\_C180 feasibility statistics}
$400$ shots requested per circuit; $n$ usable; $P_{\rm exp}$ probability of
the expected bitstring; sel/hit post-selected counts; $F_s$ with Wilson
95\% interval; TV = total-variation distance to the exact ideal joint
distribution.""")
A(r'\begin{longtable}{lclcccccc}')
A(r'\toprule circuit & scheme & $s$ & $n$ & $P_{\rm exp}^{\rm hw}$ & '
  r'$P_{\rm exp}^{\rm ideal}$ & sel & hit & $F_s$ [95\% CI] & TV \\ \midrule')
for name in 'ABCD':
    c = HW['circuits'][name]
    st = c['branch_stats']
    ci = f"[{st['F_s_lo95']:.2f},{st['F_s_hi95']:.2f}]" if st['n_sel'] else '--'
    fs = f"{st['F_s']:.2f}" if st['n_sel'] else '--'
    A(f"{name} & {c['kind']} & {c['s']} & {st['n_total']} & "
      f"{c['P_expected_key_hw']:.3f} & {c['P_expected_key_ideal']:.3f} & "
      f"{st['n_sel']} & {st['n_hit']} & {fs} {ci} & "
      f"{c['tv_distance_hw_ideal']:.3f} \\\\")
A(r'\bottomrule\end{longtable}')

# ---- ED Table 3b: the end-to-end shot budget --------------------------------
# ED Table 3 gives the per-circuit COUNTS.  It does not give what a claim costs,
# which is the thing a reviewer has to be able to check: how many shots were
# executed to produce the 55 the headline fidelity rests on, and how many would be
# needed to make that headline precise.  Post-selection is a diagnostic here, not a
# protocol, so its discard ratio is reported rather than absorbed silently into an
# error bar.
def _wilson(k, n, z=1.96):
    """Wilson score interval, reproducing branch_stats' F_s_lo95/F_s_hi95 exactly.

    Verified bit-for-bit against the artifact for both circuits that define an
    interval (44/55 and 9/25), so the sample sizes derived here are consistent with
    the intervals the paper already prints rather than with some other normal
    approximation.
    """
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return c - h, c + h


def _n_for_half(p, target, z=1.96):
    """Smallest n whose Wilson interval at observed rate p is no wider than
    +/-`target`.  Bisected on the true Wilson half-width rather than on the
    textbook normal approximation, which differs by a few percent at these n."""
    lo, hi = 1, 10 ** 9
    while lo < hi:
        mid = (lo + hi) // 2
        a, b = _wilson(round(p * mid), mid, z)
        if (b - a) / 2.0 <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def _grp(x):
    """Thousands grouping with a LaTeX thin-space, not a comma.

    A comma renders as punctuation with a following space, so `44,691` reads as two
    numbers in a table cell.  Both this file and main.tex go through the same
    rendering so that audit_numbers.py can re-derive one string and match it in
    both places.
    """
    return '{:,}'.format(int(x)).replace(',', r'\,')


_C = HW['circuits']
_REQ = HW['shots_requested']
_tot_req = _REQ * len(_C)
_tot_use = sum(_C[n]['shots_used'] for n in 'ABCD')
_tot_sel = sum(_C[n]['branch_stats']['n_sel'] for n in 'ABCD')
_tot_hit = sum(_C[n]['branch_stats']['n_hit'] for n in 'ABCD')
_D = _C['D']['branch_stats']
_Ci = _C['C']['branch_stats']
_B = _C['B']['branch_stats']
_yD = _D['n_sel'] / float(_REQ)
_yC = _Ci['n_sel'] / float(_REQ)
_hwD = (_wilson(_D['n_hit'], _D['n_sel'])[1] - _wilson(_D['n_hit'], _D['n_sel'])[0]) / 2.0
A(r"""
\paragraph*{ED Table 3b: what the feasibility run cost, end to end.}
Table 3 reports counts; this reports the accounting behind them.  ``Executed'' is
what was requested from the device, ``usable'' what came back parseable under the
established layout, ``sel'' the subset post-selected on the \emph{correctly read}
syndrome, and ``hit'' the subset of those whose data register reproduces the decoded
logical state.  Every yield is taken against \emph{executed}, so the last column is
the true end-to-end efficiency of turning one device shot into one usable
post-selected logical outcome: $%s$ shots bought $%d$ of them, $%.2f\%%$.

Two entries are easy to misread and are stated explicitly.  Circuit A has
$\mathrm{sel}=0$, so its conditional fidelity is \emph{undefined}, not zero; circuit
B, with $%d$ selected shots and no hits, is the one that measures $F_s=0.00$ with an
interval.  The two together are the ``$0$ hits in $%d$ shots'' statement in the main
text.  And the headline $F_{12}=0.80$ rests on $%d$ of the $%d$ shots executed on
circuit D, a Wilson half-width of $\pm%.3f$: a feasibility signal, not a precision
measurement."""
  % (_grp(_tot_req), _tot_hit, 100.0 * _tot_hit / _tot_req, _B['n_sel'],
     _C['A']['shots_used'] + _B['n_total'], _D['n_sel'], _REQ, _hwD))
A(r'\begin{longtable}{lclccccccc}')
A(r'\toprule circuit & scheme & $s$ & executed & usable & sel & hit & '
  r'usable/exec & sel/exec & hit/exec \\ \midrule')
for _nm in 'ABCD':
    _c = _C[_nm]
    _st = _c['branch_stats']
    A('%s & %s & %d & %d & %d & %d & %d & %.1f\\%% & %.1f\\%% & %.2f\\%% \\\\'
      % (_nm, _c['kind'], _c['s'], _REQ, _c['shots_used'], _st['n_sel'],
         _st['n_hit'], 100.0 * _c['shots_used'] / _REQ,
         100.0 * _st['n_sel'] / _REQ, 100.0 * _st['n_hit'] / _REQ))
A('total & --- & --- & %s & %s & %d & %d & %.1f\\%% & %.1f\\%% & %.2f\\%% \\\\'
  % (_grp(_tot_req), _grp(_tot_use), _tot_sel, _tot_hit,
     100.0 * _tot_use / _tot_req,
     100.0 * _tot_sel / _tot_req, 100.0 * _tot_hit / _tot_req))
A(r'\bottomrule\end{longtable}')

# The costing of precision, from the MEASURED selection yields of the two
# late-measure circuits.  Both are priced: the injected branch and the identity
# branch differ by a factor 2.2 in selection yield, so quoting only the cheaper one
# would understate what the benchmark actually costs.
_rows = []
for _tgt in (0.05, 0.02, 0.01):
    _nD = _n_for_half(_D['F_s'], _tgt)
    _nC = _n_for_half(_Ci['F_s'], _tgt)
    _rows.append((_tgt, _nD, math.ceil(_nD / _yD), _nC, math.ceil(_nC / _yC)))
A(r"""
\vspace{4pt}\noindent\emph{What precision would cost.}  Holding each circuit at its
observed rate and pricing it at its own \emph{measured} selection yield (injected
branch $%.4f$, identity branch $%.4f$, a factor $%.1f$ apart):

\noindent\begin{tabular}{lcccc}
\toprule target half-width & sel (D) & executed (D) & sel (C) & executed (C) \\
\midrule""" % (_yD, _yC, _yD / _yC))
for _tgt, _nD, _eD, _nC, _eC in _rows:
    A('$\\pm%.2f$ & %d & %s & %d & %s \\\\'
      % (_tgt, _nD, _grp(_eD), _nC, _grp(_eC)))
A(r'\bottomrule\end{tabular}')
_nselD = int(round(1000 * _yD))
_nselC = int(round(1000 * _yC))
_hw1kD = (_wilson(round(_D['F_s'] * _nselD), _nselD)[1]
          - _wilson(round(_D['F_s'] * _nselD), _nselD)[0]) / 2.0
_hw1kC = (_wilson(round(_Ci['F_s'] * _nselC), _nselC)[1]
          - _wilson(round(_Ci['F_s'] * _nselC), _nselC)[0]) / 2.0
A(r"""
\vspace{4pt}\noindent The benchmark designed in Methods is $2$ logical states
$\times$ $2$ error rates $\times$ $2$ Pauli frames $\times$ $16$ branches at $10^3$
shots: $%d$ circuits and $%s$ executed shots in total.  At the measured yields that
buys $\pm%.3f$ per injected branch and $\pm%.3f$ per identity-like one---and the
\emph{entire} benchmark budget is smaller than what a single identity-like branch
needs for $\pm0.01$ ($%s$ shots).  The design is therefore a survey of $16$ branches
at few-percent-to-ten-percent precision, not a precision measurement of any one of
them, and on this device the shot budget, not the gate count, is what bounds it.
None of this is an argument against running it; it is the statement of what the
result would mean if it were run."""
  % (2 * 2 * 2 * 16, _grp(2 * 2 * 2 * 16 * 1000), _hw1kD, _hw1kC,
     _grp(_rows[-1][4])))

# ---- ED Table 3c: where the syndrome weight actually sits -------------------
# main.tex's second and third findings are statements about the SHAPE of the
# hardware syndrome distribution, not only about how much of it survives
# post-selection, and one of them is a mode claim ("the s=8 relaxation satellite
# is the mode; the correctly read s=12 is second").  A mode claim is exactly the
# kind that a reader can only check against a bar chart, and exactly the kind that
# goes silently wrong when the prose is written from memory of the chart: the first
# draft of that sentence said the distribution "peaks at s=12", which the artifact
# contradicts by a factor 1.75.  Tabulated so the claim is checkable arithmetic.
_SUM = HW['summary']


def _syn(c):
    """Shape of one circuit's read-syndrome distribution, from the artifact.

    `excess` is P(injected s) against the 1/16 that a distribution blind to the
    code would put there.  `rank` counts strictly heavier outcomes, so a tie never
    promotes a syndrome.  `mode` breaks ties towards the smaller index, which is
    irrelevant here (all four modes are unique) but keeps it deterministic.
    """
    d = {int(k): float(v) for k, v in c['syndrome_dist_hw'].items()}
    s = c['s']
    n = c['shots_used']
    sd = math.sqrt((1.0 / 16) * (15.0 / 16) / n)   # binomial sd of one cell
    return {'d': d, 'sum': sum(d.values()),
            'mode': max(sorted(d), key=lambda k: d[k]),
            'p_mode': d[max(sorted(d), key=lambda k: d[k])],
            'p_s': d[s], 'rank': 1 + sum(1 for v in d.values() if v > d[s] + 1e-12),
            'excess': 16.0 * d[s], 'z_uniform': (d[s] - 1.0 / 16) / sd,
            'tv_uniform': 0.5 * sum(abs(v - 1.0 / 16) for v in d.values()),
            'tv_ideal': c['tv_distance_hw_ideal']}


_S = {n: _syn(_C[n]) for n in 'ABCD'}
A(r"""
\paragraph*{ED Table 3c: where the syndrome weight actually sits.}
Table 3 counts post-selected shots; this gives the distribution they were drawn
from.  ``mode'' is the most likely read syndrome, ``rank'' the position of the
\emph{correctly read} syndrome within it, ``excess'' that syndrome's probability
against the $1/16=0.0625$ a distribution blind to the code would place there, and
$\sigma$ the same excess in binomial standard deviations at that circuit's shot
count.  Four entries carry the main text's findings.  On the injected late-measure
circuit D the $a_1$-relaxation satellite $s=8$ is the mode and the correct $s=12$
is second, out-occurred by a factor $%.2f$---yet that second place is still a
$%.1f\sigma$ excess over code-blind, which is what makes the post-selected
$F_{12}=0.80$ a syndrome-conditioned result rather than a selection artefact.  On
the identity circuit C the correct syndrome shows $%.1f\sigma$: no detectable
excess at all.  The mid-measure injected circuit B keeps a $%.1f\sigma$ syndrome
excess while producing zero data hits, so what fails under mid-circuit measurement
is the \emph{data} register, not the syndrome readout.  And the mid-measure
identity circuit A never returns the correct all-zero syndrome ($0$ of $%d$ shots,
a $%.1f\sigma$ deficit).  TV uniform is the total-variation distance to the flat
distribution, TV ideal the distance to the exact ideal joint distribution already
reported in Table 3."""
  % (_SUM['late_D_relaxation_satellite_s8_over_s12'], _S['D']['z_uniform'],
     _S['C']['z_uniform'], _S['B']['z_uniform'], _C['A']['shots_used'],
     abs(_S['A']['z_uniform'])))
A(r'\begin{longtable}{lclcccccccc}')
A(r'\toprule circuit & scheme & injected $s$ & mode & $P(\mathrm{mode})$ & '
  r'$P(s)$ & rank of $s$ & excess & $\sigma$ & TV uniform & TV ideal \\ \midrule')
for _nm in 'ABCD':
    _s = _S[_nm]
    A('%s & %s & %d & %d & %.3f & %.3f & %d & %.2f$\\times$ & %+.1f & %.3f & '
      '%.3f \\\\'
      % (_nm, _C[_nm]['kind'], _C[_nm]['s'], _s['mode'], _s['p_mode'], _s['p_s'],
         _s['rank'], _s['excess'], _s['z_uniform'], _s['tv_uniform'],
         _s['tv_ideal']))
A(r'\bottomrule\end{longtable}')

# ---------------- ED Table 4: layout evidence ---------------------------
A(r"""
\subsection*{ED Table 4: bitstring-layout hypothesis evidence}
Branch-consistency scores over the four feasibility circuits (pre-fix
scoring) and data-hit evidence that established the final layout
(raw key $=[a_3a_2a_1a_0][d_4d_3d_2d_1d_0]$, i.e.\ reverse(key)$[5{:}9]$ =
syndrome). Circuit D ($s=12$) discriminates: only the reversed-ancilla
hypothesis yields data hits.""")
A(r'\begin{tabular}{lcc}')
A(r'\toprule hypothesis & consistency score & data hits (D) \\ \midrule')
A(r'anc@key[0:4], fwd  & 46  & 0 \\')
A(r'anc@key[5:9], fwd  & 192 & 0 \\')
A(r'anc@key[0:4], rev  & 179 & 0 \\')
A(r'anc@key[5:9], rev  & 124 & 44 \\')
A(r'\bottomrule\end{tabular}')
A(r"""
\vspace{6pt}\noindent\emph{Note.} The consistency score alone favours a
wrong hypothesis (it counts selected shots, not decoded-data hits); the
data-hit column is the decisive evidence and motivated the corrected
\texttt{detect\_layout} scoring (lexicographic: data hits, then selected).""")

# ---------------- ED Tables 5-7: same-family baselines & ablations -------
if 'abl' in NUM:
    abl = NUM['abl']
    npar = abl['n_params']
    A(r"""
\subsection*{ED Table 5: method landscape}
Taxonomy of all recovery methods benchmarked in this work. VSCR is the only
method that combines a learned/variational \emph{quantum unitary} recovery
with projective syndrome conditioning; the ablations (VQR-ind, global
unitary) remove one ingredient at a time, and the SDP row is the
information-theoretic ceiling over all CPTP recoveries given the projective
readout (no circuit; not executable).""")
    A(r'\begin{longtable}{lccccc}')
    A(r'\toprule method & projection $P_s$ & unitary recovery & learned &'
      r' labels needed & trainable params \\ \midrule')
    A(r'Raw (no recovery) & -- & -- & -- & -- & 0 \\')
    A(r'Perfect-code decoder & \checkmark & Pauli $C_s$ (fixed) & -- & -- & 0 \\')
    A(r'ZNE / VD / LinDR-phys (oracles) & -- & -- & LinDR fit & oracle access & '
      r'$1024^2$ (LinDR) \\')
    A(r'Global variational unitary (QVECTOR-style) & -- & \checkmark & '
      r'\checkmark & -- & %d \\' % npar['global (no proj.)'])
    A(r'VQR-ind (per-syndrome table, no sharing) & \checkmark & \checkmark & '
      r'\checkmark & -- & %d \\' % npar['VQR-ind table'])
    A(r'VSCR warm (hypernetwork, ours) & \checkmark & \checkmark & '
      r'\checkmark & -- & %d \\' % npar['VSCR hypernet'])
    A(r'SDP optimal CPTP recovery (ceiling) & \checkmark & general CPTP & '
      r'-- (solved) & channel model & --- \\')
    A(r'\bottomrule\end{longtable}')

    ind = abl['ind']; glob = abl['global']
    warm_ref = {'depolarizing': NUM['dep_p010']['VSCR warm (ours)'],
                'amplitude_damping': NUM['ad_p010']['VSCR warm (ours)'],
                'mixed': NUM['mx_p010']['VSCR warm (ours)'],
                'coherent': NUM['coh_p010']['VSCR warm (ours)']}
    dec_ref = {'depolarizing': NUM['dep_p010']['Perfect-code decoder'],
               'amplitude_damping': NUM['ad_p010']['Perfect-code decoder'],
               'mixed': NUM['mx_p010']['Perfect-code decoder'],
               'coherent': NUM['coh_p010']['Perfect-code decoder']}
    A(r"""
\subsection*{ED Table 6: same-family baselines and architecture ablations}
All rows use the identical unsupervised loss, ansatz, optimizer and
curriculum budget (VQR-ind: one seed; VSCR: best of two seeds as in the main
text). $\bar F$ at $p=0.10$ ($\varepsilon=0.10$ for coherent) over $200$
fresh Haar-random logical states; ``global'' = single variational unitary
without syndrome projection; $\Delta_{\rm ind}$ = max $|\bar F$ difference
between VQR-ind and warm VSCR over the full $p$-grid. Low-data block:
depolarizing $p=0.10$ fidelity after training on a fixed pool of $K$ logical
states (warm/cold start; $400/600$ epochs).""")
    A(r'\begin{longtable}{lccccc}')
    A(r'\toprule channel & decoder & VSCR warm & VQR-ind & $\Delta_{\rm ind}$'
      r' & global (no proj.) \\ \midrule')
    for noise in ['depolarizing', 'amplitude_damping', 'mixed', 'coherent']:
        i10 = 5 if len(ind[noise]['F']) == 10 else 1
        A(f"{noise.replace('_', ' ')} & {dec_ref[noise]:.4f} & "
          f"{warm_ref[noise]:.4f} & {ind[noise]['F'][i10]:.4f} & "
          f"{sci(ind[noise]['max_abs_diff_vs_warm'])} & "
          f"{glob[noise]['F'][i10]:.4f} \\\\")
    A(r'\bottomrule\end{longtable}')
    ld = abl['lowdata']
    A(r'\begin{longtable}{lcccc}')
    A(r'\toprule low-data ($p=0.10$) & $K=2$ & $K=4$ & $K=8$ & $K=16$ \\ '
      r'\midrule')
    rows = [('warm hypernet $\\bar F$', 'warm_hyper_F'),
            ('warm VQR-ind $\\bar F$', 'warm_ind_F'),
            ('cold hypernet $\\bar F$', 'cold_hyper_F'),
            ('cold VQR-ind $\\bar F$', 'cold_ind_F'),
            ('cold hypernet min cf$_s$', 'cold_hyper_mincf'),
            ('cold VQR-ind min cf$_s$', 'cold_ind_mincf')]
    for lab, key in rows:
        vals = ' & '.join(f'{v:.4f}' for v in ld[key])
        A(f'{lab} & {vals} \\\\')
    A(r'\bottomrule\end{longtable}')

    pl_dep = abl['pauli_lookup']['dep_0.15']
    pl_coh = abl['pauli_lookup']['coh_0.15']
    coh_det = abl.get('coh_pauli_detail', {})
    A(r"""
\subsection*{ED Table 7: SDP optimal-CPTP-recovery ceiling}
Per channel and strength: $\bar F^{\rm CPTP}$ = optimal average fidelity
over ALL trace-preserving recoveries (any Kraus structure) acting after the
projective syndrome readout, from the exact $4\times4$-Choi semidefinite
program per branch; $\bar F^{\rm dec}$ = perfect-code decoder (200-state
MC); $\Delta\bar F$ = ceiling $-$ decoder; $g_{\max}$ = largest per-branch
gap $F_s^{\max}-\mathrm{cf}_s^{\rm dec}$ over branches with $P(s)>10^{-3}$;
``top branch'' = largest gap over ALL branches with $P(s)>10^{-9}$ (weight
in parentheses): the non-Pauli headroom of the amplitude-damping and
coherent channels is concentrated in low-weight branches.
Pauli-restricted exhaustive search (4 logical classes per branch): for
depolarizing $p=0.15$ it reproduces the decoder table on every branch (max
gain of other classes $%.1e$; brute-force cross-check over random Pauli
strings $%.1e$); for coherent $\varepsilon=0.15$, %d branches prefer a
different logical class, all of weight $\le10^{-3}$ (largest
conditional-fidelity gain $+%.2f$ on branch $s=%d$, weight $%.1e$).""" % (
        max(pl_dep['gain_over_I']), pl_dep['max_bruteforce_dev'],
        coh_det.get('n_deviating', 0), coh_det.get('gain', 0.0),
        coh_det.get('branch', -1), coh_det.get('weight', 0.0)))
    A(r'\begin{longtable}{lccccc}')
    A(r'\toprule channel / strength & $\bar F^{\rm CPTP}$ & '
      r'$\bar F^{\rm dec}$ & $\Delta\bar F$ & $g_{\max}$ & top branch gap '
      r'(weight) \\ \midrule')
    for key, r in abl['sdp'].items():
        import numpy as _np
        Fs = _np.array(r['F_s_max']); ps = _np.array(r['p_s'])
        cf = _np.array(r.get('cf_dec_branch', [0.0] * 16))
        gaps = _np.where(ps > 1e-9, Fs - cf, -9.0)
        j = int(_np.argmax(gaps))
        A(f"{key.replace('_', ' ')} & {r['F_cptp']:.6f} & {r['F_dec']:.6f} & "
          f"{r['F_gap_bar']:+.2e} & {r['max_branch_gap']:.2e} & "
          f"{gaps[j]:+.3f} ({ps[j]:.1e}) \\\\")
    A(r'\bottomrule\end{longtable}')

# ---------------- ED Table 8: decoder saddle, refinement and floor ----------
# Data from the Fix A/B stage that `vscr_paper.train_warm_best` now runs: the
# separable per-syndrome refinement plus the decoder floor.
_ref = {}
for _n, _infos in list(NUM.get('infos', {}).items()):
    _r = [x for x in _infos if x.get('stage') == 'refine']
    if _r:
        _ref[_n] = _r[0]
_r = [x for x in NUM.get('infos_coh', []) if x.get('stage') == 'refine']
if _r:
    _ref['coherent'] = _r[0]
if _ref:
    A(r"""
\subsection*{ED Table 8: the decoder saddle, the separable refinement and the floor}
Per channel, over the $p$-average of that channel's training curriculum.
$\bar F^{\rm dec}$ = exact Pauli-decoder objective; $\bar F^{\rm grad}$ = what
multi-seed gradient training alone returns; $\bar F^{\rm ref}$ = after the
separable per-syndrome refinement; $\bar F^{\rm cert}$ = certified ceiling of the
per-syndrome-unitary family; ``used'' = which candidate the decoder floor
selected. Headroom is $\bar F^{\rm cert}-\bar F^{\rm dec}$, i.e.\ exactly the
eigenvalue gap above the saddle, and ``captured'' is the fraction of it the
refinement recovers. $\Delta\phi$ and spread are
$\max_s\lVert\phi_s-\phi^{\rm dec}_s\rVert_\infty$ and
$\max_{s,s'}\lVert\Delta\phi_s-\Delta\phi_{s'}\rVert_\infty$ for the table
actually used; a spread of $\sim10^{-16}$ means the decoder was retained, so no
learned deviation was claimed. Note that $\bar F^{\rm grad}$ falls \emph{below}
$\bar F^{\rm dec}$ on depolarizing and mixed noise: with an identically vanishing
gradient, Adam normalizes round-off and random-walks, which is precisely what the
floor exists to prevent.""")
    A(r'\begin{longtable}{lccccccc}')
    A(r'\toprule channel & $\bar F^{\rm dec}$ & $\bar F^{\rm grad}$ & '
      r'$\bar F^{\rm ref}$ & $\bar F^{\rm cert}$ & headroom & captured & '
      r'used \\ \midrule')
    for _n in ('depolarizing', 'amplitude_damping', 'mixed', 'coherent'):
        if _n not in _ref:
            continue
        r = _ref[_n]
        _head = r['F_cert'] - r['F_decoder']
        # When the certified headroom is zero the capture fraction is 0/0 and any
        # number printed for it is meaningless -- the honest entry is "n/a".
        if r.get('headroom_is_zero') or abs(_head) <= 1e-12:
            _cap = r'n/a ($0$)'
        else:
            _cap = ('%.4f\\%%'
                    % (100 * r.get('capture_vs_decoder', r['capture_frac'])))
        A(f"{_n.replace('_', ' ')} & {r['F_decoder']:.9f} & "
          f"{r['F_grad']:.9f} & {r['F_refined']:.9f} & {r['F_cert']:.9f} & "
          f"{_head:+.2e} & {_cap} & "
          f"{r['picked']} \\\\")
    A(r'\bottomrule\end{longtable}')
    A(r'\begin{longtable}{lcccc}')
    A(r'\toprule channel & blocks improved & $\Delta\phi$ & spread & '
      r'objective $==$ production to \\ \midrule')
    for _n in ('depolarizing', 'amplitude_damping', 'mixed', 'coherent'):
        if _n not in _ref:
            continue
        r = _ref[_n]
        A(f"{_n.replace('_', ' ')} & {r['n_improved']}/16 & "
          f"{r['branch_deviation']:.2e} & {r['syndrome_spread']:.2e} & "
          f"{r['prod_err']:.1e} \\\\")
    A(r'\bottomrule\end{longtable}')

# ---------------- ED Table 9: physical-projection and Petz audits -----------
_zne = NUM.get('zne_unphysical_overshoot', {})
_zc = NUM.get('coh_zne_unphysical_overshoot', {})
if _zne or _zc:
    A(r"""
\subsection*{ED Table 9: unphysical ZNE overshoot audit, and the Petz baseline}
Left: the RAW (unprojected) Richardson-ZNE estimator
$\langle\psi|3\rho(p)-3\rho(2p)+\rho(3p)|\psi\rangle$ over $60$ Haar-random
logical states. It is an unbounded linear functional, so nothing constrains it to
$[0,1]$; on the coherent channel it exceeds $1$ for every state at every strength,
reaching $1.1306$ at $\varepsilon=0.30$, whereas on the three incoherent channels
it never does. That asymmetry is why the artefact survived unnoticed, and it is
why all reported ZNE numbers are the physically projected estimator and every
fidelity axis is capped at $1$. Right: the noise-adapted Petz recovery channel at
$p=0.10$, a genuine CPTP recovery given the exact noise model.""")
    A(r'\begin{longtable}{lccc|cc}')
    A(r'\toprule channel & $p$ & $\max F_{\rm raw}$ & $n(F>1)$ & '
      r'Petz $\bar F$ @ $p{=}0.10$ & decoder $\bar F$ @ $p{=}0.10$ \\ \midrule')
    _petz = {'depolarizing': (NUM['dep_p010'].get('Petz recovery (2024)'),
                              NUM['dep_p010']['Perfect-code decoder']),
             'amplitude_damping': (NUM['ad_p010'].get('Petz recovery (2024)'),
                                   NUM['ad_p010']['Perfect-code decoder']),
             'mixed': (NUM['mx_p010'].get('Petz recovery (2024)'),
                       NUM['mx_p010']['Perfect-code decoder']),
             'coherent': (NUM.get('coh_p010', {}).get('Petz recovery (2024)'),
                          NUM.get('coh_p010', {}).get('Perfect-code decoder'))}
    _all = [(c, pp) for c, oo in _zne.items() for pp in oo.items()]
    _all += [('coherent', pp) for pp in _zc.items()]
    _seen = set()
    for _c, (_p, _v) in _all:
        if _c in _seen:
            continue
        _seen.add(_c)
        _mx = max(v['max_F_raw_zne'] for v in
                  (_zne.get(_c, {}) if _c != 'coherent' else _zc).values())
        _no = sum(v['n_over_1'] for v in
                  (_zne.get(_c, {}) if _c != 'coherent' else _zc).values())
        _nt = sum(v['n_test'] for v in
                  (_zne.get(_c, {}) if _c != 'coherent' else _zc).values())
        _pz, _dc = _petz.get(_c, (None, None))
        A(f"{_c.replace('_', ' ')} & $\\le0.30$ & {_mx:.8f} & "
          f"{_no}/{_nt} & "
          f"{('%.6f' % _pz) if _pz is not None else '---'} & "
          f"{('%.6f' % _dc) if _dc is not None else '---'} \\\\")
    A(r'\bottomrule\end{longtable}')

# --------- ED Tables 10a/10b: decoder stationarity, proved then swept ---------
SB = os.path.join(ROOT, '..', 'stationarity_boundary.json')
_sb = json.load(open(SB)) if os.path.exists(SB) else {}
_s = _sb.get('summary') or {}
_rows = _sb.get('rows') or []
_cert = _sb.get('certificate') or []
if _cert:
    A(r"""
\subsection*{ED Table 10a: the analytic nine-sector stationarity certificate}
The sweep in ED Table~10b is a sample of $90$ pairs; this is a proof covering
every ensemble. Because $\tr[M\rho_{\mathbf n}]=a+\mathbf b\!\cdot\!\mathbf n$
for $\rho_{\mathbf n}=(I+\mathbf n\!\cdot\!\boldsymbol\sigma)/2$, the objective
of \emph{any} ensemble $\{(w_i,\mathbf n_i)\}$ is the exact quadratic
$\sum_i w_i[c_0+\mathbf c_1\!\cdot\!\mathbf n_i+\mathbf n_i^{\mathsf T}C\,
\mathbf n_i]$, and its nine coefficients depend on $\phi$ \emph{alone}. Writing
$\nabla C=\nabla C_0+\tfrac13(\tr\nabla C)I$ and using $|\mathbf n|=1$ splits
$\nabla_\phi\bar F_{\rm ens}$ into three terms that are spherical harmonics of
degree $\ell=0,1,2$; harmonics of distinct degree are orthogonal on $S^2$, so a
non-negative-weighted sum of them can vanish for every choice of points and
weights only if each sector vanishes separately. The columns below are precisely
those sector gradients at $\phi^{\rm dec}$, maximised over all $960$ ansatz
angles and over the nine channels. Their vanishing is necessary \emph{and
sufficient} for stationarity under every input state distribution, with no
assumption about the noise channel. ``FD coeff'' is a central difference of the
nine coefficient functions themselves rather than of the assembled objective, so
it cross-checks autograd on a different code path. The $\langle Z_L\rangle$
column is the maximum over the four sectors of the hardware-style observable
loss, which is only \emph{degree one} in $\mathbf n$ and therefore has no
$\ell=2$ sector and no a priori reason to be stationary. Controls are the same
measurement at a random angle table; the weakest over all channels is %s for the
fidelity sectors and %s for $\langle Z_L\rangle$, thirteen to fourteen orders of
magnitude above the zeros, so the certificate is not reporting a dead gradient.
Totals: $\max|S_0|=$\,%s, $\max|S_1|=$\,%s, $\max|S_2|=$\,%s,
FD\,$=$\,%s, $\max|\nabla z|=$\,%s.""" % (
        sci(_s['cert_min_control_sector']), sci(_s['cert_Z_min_control_sector']),
        sci(_s['cert_max_S0']), sci(_s['cert_max_S1']), sci(_s['cert_max_S2']),
        sci(_s['cert_fd_coeff_max']), sci(_s['cert_Z_min_sector'])))
    A(r'\begin{longtable}{llccccc}')
    A(r'\toprule channel & class & $\max\lvert S_0\rvert$ & '
      r'$\max\lvert S_1\rvert$ & $\max\lvert S_2\rvert$ & FD coeff & '
      r'$\max\lvert\nabla z\rvert$ \\')
    A(r' & & ($\ell{=}0$) & ($\ell{=}1$) & ($\ell{=}2$) & '
      r'& ($\langle Z_L\rangle$) \\ \midrule')
    for _r in _cert:
        A(f"{_r['channel'].replace('_', ' ')} & {_r['channel_class']} & "
          f"{sci(_r['max_S0'])} & {sci(_r['max_S1'])} & {sci(_r['max_S2'])} & "
          f"{sci(_r['fd_coeff_max'])} & {sci(_r['max_Z_sector'])} \\\\")
    A(r'\bottomrule\end{longtable}')

if _rows:
    # The caption must quote the same two-estimator minimum that the table column
    # reports.  summary['min_control_grad'] is the AUTOGRAD-only minimum and is
    # slightly larger, which made the caption disagree with its own table.
    _ctl_min = min(min(_r['control_grad_autograd'], _r['control_grad_fd'])
                   for _r in _rows)
    _grad_max = max(max(_r['grad_autograd'], _r['grad_fd']) for _r in _rows)
    A(r"""
\subsection*{ED Table 10b: the direct sweep, as a corollary of Table 10a}
The direct measurement that ED Table~10a now subsumes, retained because it probes
the assembled objective rather than its nine coefficients, and because it records
the landscape classification per channel. Is $\phi^{\rm dec}$ stationary only
because the Haar ensemble twirls the logical state? No. Each of the nine channels
is scored by ten functionals: the exact Haar
average, three complex two-designs (which agree with the Haar closed form to
machine precision precisely because they are two-designs), five ensembles that
are \emph{not} two-designs, and two hardware-style benchmarks that replace the
state fidelity by the recovered logical $\langle Z_L\rangle$. Columns report the
largest gradient norm at $\phi^{\rm dec}$ within each functional group, and the
smallest control gradient---the same autograd and finite-difference machinery
evaluated at a random angle table, which must be large for a zero to mean
anything. The central difference at $\phi^{\rm dec}$ is \emph{bitwise} zero in
every one of the $90$ pairs. Totals: %d two-design and %d non-two-design pairs,
all stationary; max gradient %s, min control %s.""" % (
        _s['n_2designs'], _s['n_non_designs'],
        sci(_grad_max), sci(_ctl_min)))
    A(r'\begin{longtable}{llcccccc}')
    A(r'\toprule channel & class & headroom & landscape & '
      r'\multicolumn{1}{c}{max grad} & \multicolumn{1}{c}{max grad} & '
      r'\multicolumn{1}{c}{max grad} & min control \\')
    A(r' & & & & 2-design & non-2-design & $\langle Z_L\rangle$ & grad \\ '
      r'\midrule')
    _seen = []
    for _r in _rows:
        if _r['channel'] not in [c for c, _ in _seen]:
            _seen.append((_r['channel'], _r))
    for _c, _r0 in _seen:
        _g = [_r for _r in _rows if _r['channel'] == _c]
        _by = {}
        for _r in _g:
            _k = ('observable_Z' if _r['loss'] == 'observable_Z'
                  else ('2design' if _r['ensemble_class'] == '2-design'
                        else 'non2design'))
            _by[_k] = max(_by.get(_k, 0.0), _r['grad_autograd'], _r['grad_fd'])
        _ctl = min(min(_r['control_grad_autograd'], _r['control_grad_fd'])
                   for _r in _g)
        A(f"{_c.replace('_', ' ')} & {_r0['channel_class']} & "
          f"{sci(_r0['headroom'])} & {_r0['landscape']} & "
          f"{sci(_by['2design'])} & {sci(_by['non2design'])} & "
          f"{sci(_by['observable_Z'])} & {sci(_ctl)} \\\\")
    A(r'\bottomrule\end{longtable}')

# --------- ED Table 11: exact code-size scaling and the readout law -----------
SC = os.path.join(ROOT, '..', 'scaling_results.json')
if os.path.exists(SC):
    _sc = json.load(open(SC))
    A(r"""
\subsection*{ED Table 11: exact code-size scaling, the Petz comparator, and the
readout law}
Every entry is exact rather than extrapolated: the reduced-branch formalism
expresses the decoder fidelity, the per-branch-unitary ceiling (the optimum of
the family VSCR occupies), the optimal-CPTP ceiling and the noise-adapted Petz
fidelity as functions of $2\times2$ blocks alone, so $[\![7,1,3]\!]$ and
$[\![9,1,3]\!]$ cost the same $4\times4$ Rayleigh quotient and Choi program per
branch as $[\![5,1,3]\!]$. The $n=5$ rows reproduce the audited numbers of ED
Tables 7 and 9. $\eta^\ast$ is the per-ancilla readout error at which the
\emph{absolute} headroom drops below $10^{-3}$, the resolution scale of a
benchmark; ``---'' means it already starts below that scale. The mixed channel
needs a dense $8^n$ Kraus set and is evaluated at $n=5$ only.
$\bar F^{\rm CPTP}$ is maximised over trace-preserving \emph{channels}, with
trace preservation imposed as an equality rather than as $\preceq I/2$; the
inequality form let the solver stop in the interior of the sub-trace-preserving
set and under-reported the $[\![9,1,3]\!]$ amplitude-damping rows by $3.7\times$
and $16\times$. ED Table~12 resolves that same ceiling by Kraus rank, i.e.\ by
how many ancillas a gate-level instrument needs in order to reach it.""")
    A(r'\begin{longtable}{llccccccc}')
    A(r'\toprule code & channel & $p$ & $\bar F^{\rm dec}$ & '
      r'$\bar F^{\rm unit}$ & $\bar F^{\rm CPTP}$ & headroom (unit.) & Petz & '
      r'$\eta^\ast$ \\ \midrule')
    for _name, _cd in _sc['codes'].items():
        for _key, _p in _cd['points'].items():
            _ch, _pv = _key.rsplit('_', 1)
            _eta = _p['eta_headroom_below_resolvable']
            A(f"$[\\![{_name}]\\!]$ & "
              f"{_ch.replace('_', ' ')} & {_pv} & "
              f"{_p['F_dec']:.9f} & {_p['F_unit']:.9f} & "
              f"{_p['F_cptp']:.9f} & {sci(_p['headroom_unit'])} & "
              f"{_p['F_petz']:.9f} & "
              f"{('---' if _eta is None else '%.4f' % _eta)} \\\\")
        A(r'\midrule')
    A(r'\bottomrule\end{longtable}')

    A(r"""
\paragraph*{ED Table 11b: verification of the readout suppression law.}
A per-ancilla readout error $\eta$ maps the true syndrome $s$ to an observed
$\tilde s$, and the instrument then applies $R_{\tilde s}$. For a Pauli recovery
every cross-branch block $V^\dagger C_{\tilde s}W_s$ with $\tilde s\neq s$
vanishes identically, so the $2^{(n-k)}\times2^{(n-k)}$ readout-error sum
collapses onto its diagonal and the fidelity obeys the exact law
$\bar F(\eta)=(1-\eta)^{n-k}\bar F(0)$. The \emph{relative} advantage of VSCR over
the decoder is therefore preserved unchanged by readout noise, while the
\emph{absolute} headroom shrinks by $(1-\eta)^{n-k}$---which is what $\eta^\ast$
above quantifies. The first two columns are the exact block norms measured on
each code (they are the reason the law holds); the last two compare the law
against a direct evaluation of the full cross-branch sum at $\eta=0.01$ and
$\eta=0.05$, reporting only the $2(n-k)$ pairs that are nonzero out of
$2^{2(n-k)}$.""")
    A(r'\begin{longtable}{lccccc}')
    A(r'\toprule code & $n{-}k$ & syndromes & '
      r'$\max_{\tilde s\neq s}\lvert V^\dagger C_{\tilde s}W_s\rvert$ & '
      r'$\lVert V^\dagger C_sW_s-I\rVert$ & '
      r'max $\lvert$exact$-$law$\rvert$ \\ \midrule')
    for _name, _cd in _sc['codes'].items():
        _dev = max([v['dev'] for _p in _cd['points'].values()
                    for v in _p['eta_exact_check'].values()] or [0.0])
        _pn = max([v['pairs_nonzero'] for _p in _cd['points'].values()
                   for v in _p['eta_exact_check'].values()] or [0])
        _pt = max([v['pairs_total'] for _p in _cd['points'].values()
                   for v in _p['eta_exact_check'].values()] or [0])
        A(f"$[\\![{_name}]\\!]$ & ${_cd['n'] - 1}$ & ${_cd['nsyn']}$ & "
          f"{sci(_cd['readout_cross_offdiag'])} & "
          f"{sci(_cd['readout_cross_diag_dev'])} & {sci(_dev)} "
          f"(${_pn}/{_pt}$ pairs) \\\\")
    A(r'\bottomrule\end{longtable}')

# --------- ED Table 12: the Kraus-rank ladder / one-ancilla instrument --------
AR = os.path.join(ROOT, '..', 'ancilla_recovery.json')
if os.path.exists(AR):
    _ar = json.load(open(AR))
    _lad = _ar['ladder']
    _st = _ar['selftests']
    # The rank of the Choi matrix equals the Kraus rank of the branch recovery,
    # which by Stinespring equals the minimal ancilla dimension that realises it
    # as a UNITARY followed by discarding the ancilla.  So this table is not a
    # relaxation hierarchy of bounds: it is a count of ancilla qubits.
    A(r"""
\subsection*{ED Table 12: the Kraus-rank ladder --- how many ancillas the
non-unitary headroom actually needs}
The optimal-CPTP ceiling of ED Table~11 is a $4\times4$ Choi program whose Choi
matrix $C=\sum_j\lvert B_j\rangle\!\rangle\langle\!\langle B_j\rvert$ may have
rank up to $4$.  Restricting the factor to $C=LL^\dagger$ with
$L\in\mathbb C^{4\times r}$ restricts $\mathrm{rank}(C)$ to $r$, and nothing else
in the program changes.  That restriction has an exact physical meaning: a CPTP
map on the two-dimensional logical space is realised by appending an ancilla of
dimension equal to its Kraus rank, applying a \emph{unitary}, and discarding the
ancilla.  The ladder is therefore a count of ancillas, not a hierarchy of
relaxations: $r{=}1$ is a unitary branch recovery (zero ancillas, the family
VSCR compiles to), $r{=}2$ is a one-ancilla gate-level instrument, and $r{=}4$ is
the full CPTP ceiling of ED Table~11.  ``non-unitary headroom'' is
$\bar F^{(4)}-\bar F^{(1)}$, the part of the ceiling no unitary branch recovery
reaches; ``one-ancilla share'' is the fraction of it that a single ancilla qubit
recovers, $(\bar F^{(2)}-\bar F^{(1)})/(\bar F^{(4)}-\bar F^{(1)})$, and is
reported as ``---'' when the denominator vanishes because the unitary family
already attains the ceiling.""" + r"""

The two end rungs are not new numbers: they are asserted to reproduce the audited
production ceilings of ED Table~11, to %s at $r{=}1$ against
\texttt{\_max\_unitary\_J} and to %s at $r{=}4$ against \texttt{\_sdp\_branch},
and the Choi objective is asserted to equal production
\texttt{cf\_unnormalised} to %s.  The rank-$2$ rung is computed twice, once by the
constrained Choi program and once by an unconstrained optimisation over
$U=\exp(iH)$ on ancilla$\,\otimes\,$logical that shares no variables, constraints
or objective expression with it; the two agree to %s.""" % (
        sci(_st['conventions']['worst_r1_vs_production'], 1),
        sci(_st['conventions']['worst_r4_vs_production'], 1),
        sci(_st['conventions']['worst_choi_vs_kraus'], 1),
        sci(_st['tp_stinespring']['direct'], 1)))
    A(r'\begin{longtable}{llcccccc}')
    A(r'\toprule code & channel & $p$ & $\bar F^{\rm dec}$ & '
      r'$\bar F^{(1)}$ unit. & $\bar F^{(2)}$ 1 anc. & $\bar F^{(4)}$ CPTP & '
      r'1-anc.\ share \\ \midrule')
    for _r in _lad:
        _frac = _r.get('one_ancilla_fraction_of_nonunitary_headroom')
        A(f"$[\\![{_r['code']}]\\!]$ & {_r['channel'].replace('_', ' ')} & "
          f"{_r['p']:g} & {_r['F_dec']:.9f} & {_r['F']['1']:.9f} & "
          f"{_r['F']['2']:.9f} & {_r['F']['4']:.9f} & "
          f"{('---' if _frac is None else '%.6f' % _frac)} \\\\")
    A(r'\bottomrule\end{longtable}')

    A(r"""
\paragraph*{ED Table 12b: the non-unitary headroom, resolved by ancilla count.}
Same runs, reported as headroom over the decoder rather than as fidelities, so
that the size of each rung's gain is visible rather than hidden in the ninth
decimal.  ``residual'' is $\bar F^{(4)}-\bar F^{(2)}$, what a second ancilla
qubit would still buy.  Entries of order $10^{-13}$, including the negative ones,
are the noise floor of two independent multi-start solves of the same optimum and
mean the ladder is \emph{flat} there: at $n=5$ and $n=7$ the unitary family already
attains the CPTP ceiling, so the ``share'' column is undefined ($0/0$) and shown as
``---'' rather than as $1$.  Only the two $[\![9,1,3]\!]$ amplitude-damping rows
have a non-unitary headroom large enough to divide by, and there the share is
$1.000000$.""" + r"""
""")
    A(r'\begin{longtable}{llcccc}')
    A(r'\toprule code & channel:$p$ & non-unitary headroom & '
      r'one-ancilla gain & residual after 1 ancilla & share \\ \midrule')
    for _r in _lad:
        _frac = _r.get('one_ancilla_fraction_of_nonunitary_headroom')
        A(f"$[\\![{_r['code']}]\\!]$ & "
          f"{_r['channel'].replace('_', ' ')}:{_r['p']:g} & "
          f"{sci(_r['nonunitary_headroom'])} & {sci(_r['one_ancilla_gain'])} & "
          f"{sci(_r['one_ancilla_gap_to_cptp'])} & "
          f"{('---' if _frac is None else '%.6f' % _frac)} \\\\")
    A(r'\bottomrule\end{longtable}')

    # ---- 12c: the verification summary behind every emitted witness ----------
    def _cx(z):
        return '%+.6f%+.6f\,i' % (z.real, z.imag)

    _wit = [(r, w) for r in _lad for w in r['witness_branches']]
    if _wit:
        A(r"""
\paragraph*{ED Table 12c: the one-ancilla instrument, verified four ways.}
Each row is the branch on which going from rank $1$ to rank $2$ buys the most, at
the stated code$\times$channel$\times$strength.  ``gain'' is that branch's
contribution to $\bar F^{(2)}-\bar F^{(1)}$.  The four columns after it are the
independent checks described in Methods: the fidelity of the repaired Kraus pair
scored by the production Haar estimator, the fidelity of the unconstrained
$\exp(iH)$ circuit optimisation, the trace-preservation defect
$\max|\sum_iB_i^\dagger B_i-I|$, and the unitarity defect of the explicit
$4\times4$ Stinespring dilation.  The last column is
$\max|\mathrm{Tr}_a[U(\rho\otimes\ket0\bra0)U^\dagger]-\sum_iB_i\rho
B_i^\dagger|$ over random $\rho$.  Witnesses are taken where the rank-$2$ gain is
\emph{largest}, not where the numbers look best.""" + "\n")
        A(r'\begin{longtable}{llccccccc}')
        A(r'\toprule code & channel:$p$ & $s$ & $p_s$ & gain & '
          r'$\mathrm{cf}$ Choi & $\mathrm{cf}$ circuit & '
          r'$|\Delta|$ & TP / unit.\ / map defects \\ \midrule')
        for _r, _w in _wit:
            A(f"$[\\![{_r['code']}]\\!]$ & "
              f"{_r['channel'].replace('_', ' ')}:{_r['p']:g} & "
              f"{_w['syndrome']} & ${_w['p_s']:.3e}$ & "
              f"{sci(_w['gain_rank2_over_rank1'] or 0.0)} & "
              f"{_w['cf_choi']:.10f} & {_w['cf_direct']:.10f} & "
              f"{sci(abs(_w['cf_choi'] - _w['cf_direct']))} & "
              f"{sci(_w['tp_defect_repaired'], 1)} / "
              f"{sci(_w['unitarity_defect'], 1)} / "
              f"{sci(_w['map_defect'], 1)} \\\\")
        A(r'\bottomrule\end{longtable}')

    # ---- 12d: the explicit instrument at the point the paper quotes ----------
    _key = [r for r in _lad if r['n'] == 9 and r['channel'] == 'amplitude_damping'
            and abs(r['p'] - 0.10) < 1e-12 and r['witness_branches']]
    if _key:
        _w = _key[0]['witness_branches'][0]
        _B = [[[complex(z[0], z[1]) for z in row] for row in M2]
              for M2 in _w['kraus']]
        _U = [[complex(z[0], z[1]) for z in row] for row in _w['stinespring_U']]
        A(r"""
\paragraph*{ED Table 12d: the instrument itself, at $[\![9,1,3]\!]$ amplitude
damping $p=0.10$.}
This is not a bound but the recovery: the trace-preserving Kraus pair of the
highest-gain branch (syndrome $s=%d$, weight $p_s=%.4e$), and the $4\times4$
unitary $U$ on ancilla$\,\otimes\,$logical whose ancilla column zero reproduces
it, $B_i=\bra i_aU\ket0_a$.  Ordering is ancilla-then-logical, flat index
$a\cdot2+l$.  Executing the branch means: append one ancilla in $\ket0$, apply
$U$, discard the ancilla.  No mid-circuit measurement, no post-selection, no
feed-forward.  $\sum_iB_i^\dagger B_i=I$ to %s and $U^\dagger U=I$ to %s.""" % (
            _w['syndrome'], _w['p_s'],
            sci(_w['tp_defect_repaired'], 1),
            sci(_w['unitarity_defect'], 1)) + "\n")
        A(r'\begin{center}\begin{tabular}{cc}')
        A(r'\toprule $B_0$ & $B_1$ \\ \midrule')
        _cells = [r'$\begin{pmatrix}%s & %s\\ %s & %s\end{pmatrix}$'
                  % (_cx(M[0][0]), _cx(M[0][1]), _cx(M[1][0]), _cx(M[1][1]))
                  for M in _B]
        A(_cells[0] + ' & ' + _cells[1] + r' \\')
        A(r'\bottomrule\end{tabular}\end{center}')
        # `$U=$ \begin{pmatrix}` closed math before the matrix, leaving pmatrix in
        # TEXT mode (a hard LaTeX error) plus a dangling `$` that never closed --
        # and because no TeX engine exists on this host the file has never been
        # compiled, so it survived.  The math span must cover the whole matrix, as
        # the $B_0$/$B_1$ cells above already do.  audit_numbers.py now asserts both
        # that the file's unescaped `$` count is even and that every `\begin{pmatrix}`
        # sits inside a math span.
        A(r'\begin{center}$U='
          r'\begin{pmatrix}'
          + ' & '.join('%s' % _cx(z) for z in _U[0]) + r'\\ '
          + ' & '.join('%s' % _cx(z) for z in _U[1]) + r'\\ '
          + ' & '.join('%s' % _cx(z) for z in _U[2]) + r'\\ '
          + ' & '.join('%s' % _cx(z) for z in _U[3])
          + r'\end{pmatrix}$\end{center}')

# ----- ED Table 13: the seed spread behind every "best of two" number ----------
MS = os.path.join(ROOT, '..', 'multiseed_results.json')
if os.path.exists(MS):
    _ms = json.load(open(MS))
    _msd = _ms['diagnostics']
    _msa = _ms['aggregate']
    _rep = _ms['reproduction']
    _gc = _ms.get('gpu_check') or {}
    A(r"""
\subsection*{ED Table 13: seed-to-seed statistics of the trained pipeline}
Every trained number in the main text is the better of the two seeds
$\{1234,2024\}$, selected by the label-free worst-branch conditional fidelity.
This table reports what that selection costs.  Each row is $S=%d$ seeds run
through the \emph{unmodified} production path---\texttt{train\_warm\_best} with a
single seed, and \texttt{train\_ind} followed by the identical
\texttt{refine\_with\_floor}---so a per-seed value is exactly the value
production would have produced for that seed.  Every fidelity listed is a
deterministic quadrature, hence the standard deviation below is seed variability
with no Monte-Carlo term folded in; it is the sample standard deviation
(ddof $=1$).  The reproduction gate is why the other seeds can be trusted: %d
fields of the two production seeds return bit-for-bit against
\texttt{paper\_numbers.json} after distribution over pinned cores in fresh
interpreters (%d failures).""" % (
        _msa[sorted(_msa)[0]]['n_seeds'], _rep['n_fields_compared'],
        _rep['n_failures']))
    A(r'\begin{longtable}{llccccccc}')
    A(r'\toprule protocol & channel & mean $\bar F_{\rm used}$ & std & sem & '
      r'best of two & bias ($\sigma$) & headroom & headroom/std \\ \midrule')
    for _k in sorted(_msd):
        _d = _msd[_k]
        _proto, _ch = _k.split('/')
        _bias = ('n/a' if _d['selection_bias_sigma'] is None
                 else '%+.2f' % _d['selection_bias_sigma'])
        _snr = _d['signal_to_seed_noise']
        _snrs = ('n/a' if _snr is None
                 else (r'$\infty$' if _snr == float('inf') else sci(_snr, 2)))
        A(f"{_proto} & {_ch.replace('_', ' ')} & "
          f"{_d['F_used_mean']:.12f} & {sci(_d['F_used_std'])} & "
          f"{sci(_d['F_used_sem'])} & {_d['F_used_best_of_two']:.12f} & "
          f"${_bias}\\sigma$ & {sci(_d['headroom_mean'])} & {_snrs} \\\\")
    A(r'\bottomrule\end{longtable}')

    A(r"""
\paragraph*{ED Table 13b: reading the two ``n/a'' columns.}
Both are undefined rather than zero, and for one underlying reason.  On
depolarizing and mixed noise the decoder \emph{is} the optimum of the
per-syndrome-unitary family, so the certified headroom is signed round-off
(flagged \texttt{headroom\_is\_zero}) and every seed converges to the \emph{same}
angle table, which makes the standard deviation round-off as well.  A ratio of two
round-off quantities is $0/0$: printing it as ``$-149\sigma$'' or as a large
signal-to-noise would manufacture a trend out of floating-point noise, so the
driver returns \texttt{None} and the audit asserts that it does.  Where the
headroom is real the ratio is enormous---%s on amplitude damping and %s on
coherent noise for the warm architecture---which is the quantitative answer to
``could this have been a lucky seed?''.  The selection bias is small and \emph{not}
systematically positive: $%+.2f\sigma$ on amplitude damping and $%+.2f\sigma$ on
coherent noise, because the selection ranks on the worst-branch conditional
fidelity rather than on $\bar F$.  The seed standard deviation on amplitude damping
is %s of the $1.1\times10^{-4}$ Monte-Carlo standard error that the exact
quadrature was introduced to avoid, so the seed lottery sits five orders of
magnitude below an estimator noise the paper already refuses to tolerate.""" % (
        sci(_msd['warm/amplitude_damping']['signal_to_seed_noise'], 2),
        sci(_msd['warm/coherent']['signal_to_seed_noise'], 2),
        _msd['warm/amplitude_damping']['selection_bias_sigma'],
        _msd['warm/coherent']['selection_bias_sigma'],
        sci(_msd['warm/amplitude_damping']['seed_std_vs_mc_sem'], 2)))
    if _gc.get('available'):
        _hr = _msd['warm/amplitude_damping']['headroom_mean']
        A(r"""
\paragraph*{ED Table 13c: why this sweep ran on pinned CPU cores, measured.}
The same curriculum executed on CPU and on CUDA (%s, torch %s) reaches the same
objective to %s, a factor $%s$ below the amplitude-damping headroom, which rules
out the seed spread being an artifact of one host's BLAS association order.  The
angle tables differ by %s while the objective differs by %s: a single fidelity is
realised by a whole manifold of tables, so only $\bar F$ is a meaningful coordinate
to compare across devices.  Timing one forward--backward pass of the exact
objective at increasing batch shows the GPU $%.2f\times$ \emph{slower} at one seed
and reaching parity only at four; it does amortise batching ($%.1f\times$ from
batch 16 to 1024, against $%.1f\times$ on the CPU), but sixteen seeds batched would
merely match eight-way process parallelism, and buying that would mean
reimplementing the audited hypernetwork parameterisation in batched
form---forfeiting the bit-exact reproduction this table rests on.""" % (
            _gc['device'], _gc['torch'], sci(_gc['dF_cpu_vs_cuda'], 2),
            '%.0f' % (_hr / max(_gc['dF_cpu_vs_cuda'], 1e-300)),
            sci(_gc['dphi_cpu_vs_cuda'], 2), sci(_gc['dF_cpu_vs_cuda'], 2),
            1.0 / _gc['cuda_speedup_at_one_seed'],
            _gc['cuda_scaling_1024_over_16'],
            _gc['cpu_scaling_1024_over_16']))
        A(r'\begin{center}\begin{tabular}{lcccc}')
        A(r'\toprule batch $B$ & seeds & CPU (ms) & CUDA (ms) & CUDA/CPU '
          r'\\ \midrule')
        for _b in sorted(_gc['batched_step'], key=int):
            _r = _gc['batched_step'][_b]
            A(f"${_b}$ & ${_r['seeds']}$ & {_r['cpu_ms']:.3f} & "
              f"{_r['cuda_ms']:.3f} & {_r['cuda_over_cpu']:.2f} \\\\")
        A(r'\bottomrule\end{tabular}\end{center}')

# ----- ED Table 14: multi-round logical storage -------------------------------
SR = os.path.join(ROOT, '..', 'storage_rounds.json')
if os.path.exists(SR):
    _sr = json.load(open(SR))
    _srs = _sr['summary']
    _srr = _sr['records']
    _srf = _sr['full_space'] or []
    _srb = _sr.get('device_benchmark') or {}
    _srv = _sr.get('validation') or {}
    _Rlast = _sr['config']['rounds'][-1]
    A(r"""
\subsection*{ED Table 14: multi-round logical storage}
One round of noise, syndrome measurement and syndrome-conditioned recovery is the
instrument $\mathcal{E}(\rho)=\sum_sR_sP_s\mathcal{N}(\rho)P_sR_s^\dagger$; a
memory of $R$ rounds is $\mathcal{E}^R$.  Because $R_s$ maps
$\operatorname{im}P_s$ into the code space, the map closes on the $2\times2$
logical space,
$\Lambda(\sigma)=\sum_sG_s(\sum_kA_k^{(s)}\sigma A_k^{(s)\dagger})G_s^\dagger$,
so $\Lambda$ is a $4\times4$ superoperator and $R$ rounds cost a single
eigendecomposition of it.  Since $\Lambda$ is linear in $\ket\psi\bra\psi$,
$F(R)$ is degree $2$ in the Bloch vector for every $R$ and the $3\times7$ Haar
rule used throughout the paper stays exact at all $R$: no curve below carries a
Monte-Carlo error.  At $R=1$ the map reproduces \texttt{exact\_F} and
\texttt{opt\_unitary\_ceiling[F\_dec]} to $10^{-12}$, and %d comparisons
against the published \texttt{*\_p010} tables pass at their own sampling scale
(%d failures).  ``survival'' is
$\mathrm{adv}(R{=}%d)/\mathrm{adv}(R{=}1)$: above $1$ the single-round headroom
\emph{compounds} into memory time, below $1$ it washes out.""" % (
        _srv.get('n_comparisons', 0), _srv.get('n_failures', 0), _Rlast))

    A(r"""
\paragraph*{ED Table 14a: the advantage compounds (ideal readout).}
``n/a'' in the survival column means the single-round advantage is at round-off
($\le10^{-9}$), i.e.\ the two recoveries are the \emph{same} recovery on that
channel because the decoder already is the family optimum; the ratio would be
$0/0$.  $R_{1/2}$ is the round count at which $\bar F$ falls halfway to the fixed
point $F_\infty$, which is $1/2$ throughout because the corrected logical channel
is unital; ``$>$'' means it exceeded the $10^{6}$-round search cap, so the gain is
a lower bound.""")
    A(r'\begin{longtable}{llccccccc}')
    A(r'\toprule channel & $p$ & recov. & adv@$R{=}1$ & adv@$R{=}%d$ & survival '
      r'& $R_{1/2}$ dec. & $R_{1/2}$ alt. & gain \\ \midrule' % _Rlast)
    for _r in _srs:
        if _r['code'] != '5,1,3' or _r['eta'] != 0.0:
            continue
        if _r['recovery'] not in ('warm', 'ceiling'):
            continue
        # `sci()` already wraps its output in $...$, so anything passed through it
        # must not be wrapped again; the same applies to the gain column, where a
        # capped value is rendered as a lower bound.  Double wrapping produces
        # `$$`, which LaTeX reads as display math and the audit rejects.
        _g = _r['R_half_gain']
        if _g is None:
            _gtxt = ('>%.0f' % (1e6 / _r['R_half_dec'])
                     if _r.get('R_half_capped') and _r['R_half_dec'] else None)
        else:
            _gtxt = '%.2f' % _g
        _gt = 'n/a' if _gtxt is None else '$%s$' % _gtxt
        _alt = _r['R_half_alt']
        _altt = '$>10^{6}$' if _alt is None else '$%d$' % int(_alt)
        _sv = _r['survival_ratio']
        A(f"{_r['channel'].replace('_', ' ')} & ${_r['p']:.2f}$ & "
          f"{_r['recovery']} & {sci(_r['advantage_R1'])} & "
          f"{sci(_r['advantage_R%d' % _Rlast])} & "
          f"{('n/a' if _sv is None else '$%.2f$' % _sv)} & "
          f"${int(_r['R_half_dec'])}$ & {_altt} & {_gt} \\\\")
    A(r'\bottomrule\end{longtable}')

    A(r"""
\paragraph*{ED Table 14b: readout error erodes the compounding.}
A Pauli recovery has identically vanishing cross-branch blocks
$V^\dagger R_{\tilde s}W_s$, so its fidelity obeys
$F(\eta,R)=(1-\eta)^{(n-k)R}F(0,R)$ \emph{exactly}, verified to $10^{-13}$ out to
$R=10$.  The learned recovery has cross-branch blocks of order $10^{-3}$ and does
violate that law---at the $10^{-11}$ level, so it is unprotected in principle and
protected to eight decimal places in practice.  What readout error does destroy is
the \emph{accumulation}: the survival factor falls through $1$ at $\eta=0.02$, i.e.
a few-percent mid-circuit readout error preserves the single-round advantage while
abolishing the multi-round one.""")
    A(r'\begin{longtable}{llcccc}')
    A(r'\toprule channel & $p$ & $\eta$ & adv@$R{=}1$ & adv@$R{=}%d$ & survival '
      r'\\ \midrule' % _Rlast)
    for _row in _srs:
        if _row['code'] != '5,1,3' or _row['recovery'] != 'warm':
            continue
        if _row['channel'] not in ('amplitude_damping', 'coherent'):
            continue
        A(f"{_row['channel'].replace('_', ' ')} & ${_row['p']:.2f}$ & "
          f"${_row['eta']:.3f}$ & {sci(_row['advantage_R1'])} & "
          f"{sci(_row['advantage_R%d' % _Rlast])} & "
          f"{('n/a' if _row['survival_ratio'] is None else '%.2f' % _row['survival_ratio'])}"
          r' \\')
    A(r'\bottomrule\end{longtable}')



    A(r"""
\paragraph*{ED Table 14c: how the decoder's memory scales with the code.}
Decoder-only, ideal readout: $n=7$ and $n=9$ have no trained table, but the
minimum-weight lookup decoder is constructible from the stabilizers alone, so the
baseline memory extends to all three codes while the learned curves do not.  These
are the same reduced blocks that ED Table~11 certifies, iterated.  Note that
$R_{1/2}$ is \emph{not} monotone in $n$: $[\![9,1,3]\!]$ stores longest under
amplitude damping despite a lower single-round fidelity, because its asymptotic
decay rate is set by how fast weight-two errors accumulate across three GHZ
blocks rather than by the per-round correction quality.  Fixed $p$ across three
different distance-$3$ codes is therefore not a scaling parameter, exactly as in
ED Table~11.""")
    A(r'\begin{longtable}{llcccccc}')
    A(r'\toprule code & channel & $p$ & $\bar F(R{=}1)$ & $\bar F(R{=}12)$ & '
      r'$\bar F(R{=}%d)$ & $R_{1/2}$ & $F_\infty$ \\ \midrule' % _Rlast)
    for _rec in _srr:
        if _rec['recovery'] != 'decoder' or _rec['eta'] != 0.0:
            continue
        _rd = _rec['rounds']
        A(f"$[\\![{_rec['code']}]\\!]$ & {_rec['channel'].replace('_', ' ')} & "
          f"${_rec['p']:.2f}$ & ${_rec['F'][_rd.index(1)]:.6f}$ & "
          f"${_rec['F'][_rd.index(12)]:.6f}$ & ${_rec['F'][-1]:.6f}$ & "
          f"{('$>10^{6}$' if _rec.get('R_half') is None else '$%d$' % _rec['R_half'])}"
          f" & ${_rec['F_inf']:.6f}$ \\\\")
    A(r'\bottomrule\end{longtable}')

    if _srf:
        _wd = max(r['max_abs_full_vs_reduced'] for r in _srf
                  if r['recovery'] == 'decoder')
        _ww = max(r['max_abs_full_vs_reduced'] for r in _srf
                  if r['recovery'] != 'decoder')
        _lw = max(r['leakage_max'] for r in _srf)
        A(r"""
\paragraph*{ED Table 14d: the reduced map against the full density matrix.}
The reduced map computes $(V^\dagger\mathcal{E}V)^R$ while the full simulation
computes $V^\dagger\mathcal{E}^RV$; they coincide exactly when every $R_s$ maps
$\operatorname{im}P_s$ isometrically into the code space, and differ by the
leakage otherwise.  Over %d configurations the worst disagreement is %s for the
Pauli decoder (round-off: the reduction is exact there) and %s for the learned
table, whose unitarity deviation is of order $10^{-6}$.  The leaked fraction
$1-\operatorname{Tr}[P\rho]/\operatorname{Tr}\rho$ peaks at %s and, decisively,
\emph{saturates} with $R$ instead of compounding---each round re-projects onto the
syndrome subspaces, so what the table leaks is set by its own unitarity deviation
and does not accumulate.  That is what licenses the cheap map for long memories.
The per-branch cost is held at $O(\dim^2)$ rather than $O(\dim^3)$ through
$R_sP_s\rho P_sR_s^\dagger=(R_sW_s)(W_s^\dagger\rho W_s)(R_sW_s)^\dagger$, which is
what makes $[\![9,1,3]\!]$ affordable at all.""" % (
            len(_srf), sci(_wd, 2), sci(_ww, 2), sci(_lw, 2)))
        A(r'\begin{longtable}{lllcclccc}')
        A(r'\toprule code & channel & $p$ & recov. & device & '
          r'max$\lvert$full$-$red.$\rvert$ & leakage & unitarity dev. & s '
          r'\\ \midrule')
        for _rec in _srf:
            A(f"$[\\![{_rec['code']}]\\!]$ & "
              f"{_rec['channel'].replace('_', ' ')} & ${_rec['p']:.2f}$ & "
              f"{_rec['recovery']} & {_rec['device']} & "
              f"{sci(_rec['max_abs_full_vs_reduced'])} & "
              f"{sci(_rec['leakage_max'])} & {sci(_rec['unitarity_dev'])} & "
              f"{_rec['wall_s']:.1f} \\\\")
        A(r'\bottomrule\end{longtable}')
    if _srb.get('points'):
        _n9 = [q for q in _srb['points'] if q['code'] == '9,1,3']
        _n5 = [q for q in _srb['points'] if q['code'] == '5,1,3']
        _s9 = _n9[0]['cuda_speedup'] if _n9 and 'cuda_speedup' in _n9[0] else None
        _s5 = _n5[0]['cuda_speedup'] if _n5 and 'cuda_speedup' in _n5[0] else None
        A(r"""
\paragraph*{ED Table 14e: device choice for the full-space path, measured.}
Both ends of the crossover, so the policy is not quoted from whichever end
flatters the GPU.  At $[\![5,1,3]\!]$ ($\dim=32$) the full-space path is
launch-bound and CUDA is %.2f$\times$ \emph{slower} than a pinned core; at
$[\![9,1,3]\!]$ ($\dim=512$) the branch einsum is real arithmetic and CUDA is
%.2f$\times$ faster.  The curves agree to %s and %s respectively, so this is a
throughput choice and not a numerical one.  Hence the policy in
\texttt{storage\_rounds.device\_for}: CPU below $\dim=128$, CUDA at or above it.
On this host \texttt{nvidia-smi} fails with a driver/library version mismatch
(kernel module $595.84$ against a $595.91.07$ userspace) while the CUDA runtime
works normally, so a broken NVML is not evidence that the GPU is unusable---only
that \texttt{nvidia-smi} is.""" % (
            1.0 / _s5 if _s5 else float('nan'), _s9 or float('nan'),
            sci(_n5[0]['max_abs_curve_diff'], 2) if _n5 else 'n/a',
            sci(_n9[0]['max_abs_curve_diff'], 2) if _n9 else 'n/a'))
        A(r'\begin{longtable}{llcccc}')
        A(r'\toprule code & recovery & $\dim$ & CPU (s) & CUDA (s) & CUDA '
          r'speedup \\ \midrule')
        for _q in _srb['points']:
            _sp = _q.get('cuda_speedup')
            A(f"$[\\![{_q['code']}]\\!]$ & {_q['recovery']} & ${_q['dim']}$ & "
              f"{_q['cpu']['seconds']:.2f} & "
              + (f"{_q['cuda']['seconds']:.2f} & " if 'cuda' in _q else 'n/a & ')
              + ('n/a' if _sp is None else '%.2f$\\times$' % _sp) + r' \\')
        A(r'\bottomrule\end{longtable}')

    # ---- ED Table 14f: recovery-gate noise --------------------------------
    # Every table above composes EXACT unitaries R_s.  This one composes each
    # recovery with a global depolarizing layer of strength lam and bisects the
    # thresholds, so that the multi-round claim is quoted at a recovery noise
    # level nobody has to take on faith.
    _rn = _sr.get('recovery_noise') or []
    if _rn:
        _live = [r for r in _rn if r['lam_star'] is not None]
        _RL = _rn[0]['R_last']
        _prod = [r['lam_half_times_R'] for r in _live
                 if r['lam_half_times_R'] is not None]
        _uni = max(abs(r['lam_grid'][i]['advantage'] / r['advantage_lam0']
                       - (_live[0]['lam_grid'][i]['advantage']
                          / _live[0]['advantage_lam0']))
                   for r in _live
                   for i, g in enumerate(r['lam_grid'])
                   if g['lam'] <= r['exp_law_lam_max'])
        A(r"""
\paragraph*{ED Table 14f: recovery-gate noise, and where the advantage stops surviving it.}
Every table above composes \emph{exact} unitaries $R_s$, which no hardware
delivers.  Here each recovery is composed with a global depolarizing layer of
strength $\lambda$ and the round map becomes a $5\times5$ \emph{affine} map on the
$4$ logical components plus one scalar carrying the out-of-code weight the
recovery fails to return.  Five dimensions are necessary rather than convenient:
the $4\times4$ restriction of the same physics discards that escaped weight and
therefore \emph{under-reports} $F$, by $0.37$ at $R=12$ on coherent/warm at
$\lambda=0.2$.  It is invisible at $R=1$, where nothing has escaped yet, and grows
with $R$---which is why a single-round check cannot catch it and why
\texttt{storage\_rounds.py --selftest} locks both halves of that fact.  The
reduction's only approximation is $q_\mathrm{esc}$, reported per row and checked
against the exact $\dim\times\dim$ evolution in the same row rather than assumed
small; on the decoder rows it is round-off because a Pauli recovery returns every
branch exactly to the code space.

Two thresholds are bisected to $10^{-12}$ in $\lambda$, both at \emph{equal}
$\lambda$ for the learned table and the decoder: $\lambda^*$, where the
$R=%d$ advantage \emph{changes sign}, and $\lambda_{1/2}$, where its
\emph{magnitude} has halved.  The second is the binding one---the tightest ratio
here is $\lambda^*/\lambda_{1/2}=%.1f$---so quoting $\lambda^*$ alone would
oversell the claim.  On depolarizing and mixed noise no threshold exists: the
decoder is already the family optimum, the advantage is $0$ at $\lambda=0$, and the
scan reports ``degenerate'' rather than manufacturing a number out of round-off.

$\lambda$ is the strength of \emph{one recovery layer}, not a per-gate error rate,
so $\epsilon^*/n=\lambda^*/n$ is the depth-$1$ reading and is optimistic by the
recovery circuit depth $d$ (roughly $\lambda\sim d\,n\,\epsilon$).  A universality
result falls out of the grid: the \emph{normalised} advantage
$\mathrm{adv}(\lambda)/\mathrm{adv}(0)$ is the same channel-blind curve on all %d
non-degenerate rows (spread %s over $\lambda\le%.2f$) and tracks
$e^{-\lambda R}$, so $\lambda_{1/2}R\to\ln2$: measured %s--%s against %s.
$\lambda^*$ is \emph{not} universal, spanning %.3f--%.3f, because the sign change
is set by channel-specific subleading structure rather than by the decay rate."""
          % (_RL, min(r['lam_star'] / r['lam_half'] for r in _live),
             len(_live), sci(_uni, 1), _live[0]['exp_law_lam_max'],
             '$%.4f$' % min(_prod), '$%.4f$' % max(_prod),
             '$%.6f$' % math.log(2.0),
             min(r['lam_star'] for r in _live),
             max(r['lam_star'] for r in _live)))
        A(r'\begin{longtable}{llccccccc}')
        A(r'\toprule channel & $p$ & recov. & $\lambda^*$ & $\epsilon^*/n$ & '
          r'$\lambda_{1/2}$ & $\lambda_{1/2}R$ & $q_\mathrm{esc}$ & '
          r'max$\lvert$affine$-$full$\rvert$ \\ \midrule')
        for _r in _rn:
            _fs = _r['full_space_check']
            A(f"{_r['channel'].replace('_', ' ')} & ${_r['p']:.2f}$ & "
              f"{_r['recovery']} & "
              + ('$%.6f$' % _r['lam_star'] if _r['lam_star'] is not None
                 else 'degenerate') + ' & '
              + ('$%.4f$' % _r['eps_per_qubit_star']
                 if _r['eps_per_qubit_star'] is not None else '---') + ' & '
              + ('$%.6f$' % _r['lam_half'] if _r['lam_half'] is not None
                 else '---') + ' & '
              + ('$%.4f$' % _r['lam_half_times_R']
                 if _r['lam_half_times_R'] is not None else '---') + ' & '
              + f"{sci(_r['q_escape_max'], 1)} & "
              + f"{sci(_fs['max_abs_affine_vs_full'], 1)} \\\\")
        A(r'\bottomrule\end{longtable}')

A(r'\end{document}')

open(os.path.join(ROOT, 'extended_data.tex'), 'w').write('\n'.join(lines))
print('wrote paper/extended_data.tex,', len(lines), 'lines')
