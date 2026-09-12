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
\usepackage{amsmath,booktabs,longtable,array}
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

A(r'\end{document}')
open(os.path.join(ROOT, 'extended_data.tex'), 'w').write('\n'.join(lines))
print('wrote paper/extended_data.tex,', len(lines), 'lines')
