"""make_ed.py — build paper/extended_data.tex (ED tables) from the analysis
artifacts: ideal per-frame references, per-branch conditional fidelities,
hardware feasibility statistics, and bit-layout evidence."""
import json, os, sys
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
    mnt, ex = f'{x:.{digits}e}'.split('e')
    return f'{mnt}\\times 10^{{{int(ex)}}}'


hw = NUM['hardware_frame_numbers']
A(r"""
\subsection*{ED Table 1: exact ideal reference of the hardware benchmark}
Zero-noise per-frame circuit fidelity $F=\sum_s F_s$ (dense statevector,
late-measure branch circuits compiled from the trained depolarizing model)
for logical states $\ket{0},\ket{+}$; every single-Pauli frame is corrected
exactly, so the $p$-dependent ideal benchmark value follows from Pauli-frame
averaging, $F(p)=1-O(p^2)$. Worst circuit-vs-density-matrix deviation
$%s$. Frames are single-qubit Pauli injections.""" %
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

A(r'\end{document}')
open(os.path.join(ROOT, 'extended_data.tex'), 'w').write('\n'.join(lines))
print('wrote paper/extended_data.tex,', len(lines), 'lines')
