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
Totals: $\max|S_0|=%s$, $\max|S_1|=%s$, $\max|S_2|=%s$, FD $=%s$,
$\max|\nabla z|=%s$.""" % (
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
needs a dense $8^n$ Kraus set and is evaluated at $n=5$ only.""")
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

A(r'\end{document}')
open(os.path.join(ROOT, 'extended_data.tex'), 'w').write('\n'.join(lines))
print('wrote paper/extended_data.tex,', len(lines), 'lines')
