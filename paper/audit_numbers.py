#!/usr/bin/env python3
"""Consistency audit of every number, figure and cross-reference in the paper.

This is the machine-checkable counterpart of `run_selftests.py`: where the self
tests check that the *code* computes what it claims, this script checks that the
*artifacts* are mutually consistent and physically meaningful. It reads only
files on disk and recomputes nothing, so it runs in well under a second and can
be used as a pre-commit / pre-submission gate.

It asserts, over `paper_numbers.json`, `paper/main.tex`,
`paper/extended_data.tex`, `paper/refs.bib` and `paper/figures/`:

  1. every reported fidelity is a probability in [0, 1] (no unphysical > 1);
  2. the warm start never loses to the perfect-code decoder, at any p, on any
     of the four channels (the decoder floor of `refine_with_floor`);
  3. the Fix A/B refinement bookkeeping is coherent: `F_used` really is
     `max(F_decoder, F_refined)`, the rigorous bound `F_bound >= F_cert`,
     `capture_frac` in [0, 1], `cert_valid`, and the certified non-Pauli
     headroom vanishes exactly on depolarizing/mixed while being real on
     amplitude damping (90% captured) and coherent (99.9996% captured);
  4. the SDP ceiling obeys weak duality at all 8 operating points, i.e. it
     bounds both the best unitary and the exact decoder, and reaches 1 on a
     channel where a known unitary is exactly correctable;
  5. no Fig. 4(b) "gap to ceiling" bar is negative or exceeds its own bound
     (the bug that the ceiling fix corrected);
  6. the ablation is fair: the independent per-syndrome table went through the
     same refinement, and matches the warm hypernetwork to <= 1e-8 on the full
     p-grid, while the cold-start low-data sweep is non-monotone in K;
  7. the raw-ZNE overshoot audit shows 0 unphysical values on incoherent
     channels (and reports the coherent ones, which motivate ZNE-phys);
  8. every LaTeX \\cite and \\ref resolves, no \\label is orphaned, and the
     duplicate `fig:coherent` label stays removed;
  9. the ED tables through 11b are present with balanced tabulars, and
     `extended_data.tex` is environment-balanced (Table 12 is locked in 9c/9d and
     Tables 13-14 in section 14b);
 10. the key literature is in the bibliography;
 11. every \\includegraphics target exists on disk.

  12. no math-mode symbol survives outside a $...$ span in the generated ED
      file, which is the property a missing TeX engine would otherwise hide;
  13. the T3 multi-seed artifact: the two production seeds reproduce
      `paper_numbers.json` bit-for-bit, no seed breaches the decoder floor or the
      certified ceiling, the certified headroom is seed-independent, the headroom
      dwarfs the seed spread wherever there is headroom, the device decision is a
      measurement on file, and the two degenerate ratios are reported undefined
      rather than as 0/0 dressed up as large numbers;
  14. the T4 storage artifact: the R=1 anchor against the published `*_p010`
      tables, F(0)=1 and monotone F(R) in every record, the reduced map validated
      against the full density matrix with leakage that saturates rather than
      compounds, the advantage compounding where there is one and being identical
      where the decoder is already optimal, readout error eroding it
      monotonically, the decoder's cross-branch blocks vanishing (the premise of
      the exact readout law) while the learned recovery's do not, and a device
      policy that matches the CPU/CUDA measurement used to justify it.
  15. the hardware feasibility artifact: the counts nest, the reported Wilson
      intervals are the z=1.96 ones recomputed here, the summary block agrees with
      the per-circuit records, the syndrome mode really is the relaxation satellite
      the prose says it is, and every shot-count, yield and precision-cost mantissa
      in main.tex and ED Tables 3b/3c is re-derived from the artifact -- these being
      the only numbers in the paper that no script in the repository can regenerate.
  16. the effective hardware error budget: no QPU time was spent deriving it, the
      vendor-calibration record states what was refused and when, the gate
      inventory adds up three ways and matches the architecture the paper
      describes, and every per-ancilla error rate, independence ratio, asymmetry
      and loss share is re-derived a second time from the raw counts rather than
      trusted from the script that first computed it.

Exit status is 0 iff all checks pass, so it composes with `&&` in CI.
"""
import json
import math
import os
import re

# Anchor to the repository root from this file's location, so the audit works
# from any checkout and any working directory.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
BS = chr(92)          # backslash, kept out of f-strings
ok = True


def chk(cond, msg):
    global ok
    print(('  OK   ' if cond else '  FAIL ') + msg)
    ok = ok and bool(cond)


print('=== FINAL CONSISTENCY AUDIT ===')
for f in ('ssvr_qec.py', 'vscr_paper.py', 'vscr_paper_abl.py', 'vscr_paper_coh.py',
          'vscr_general.py', 'scaling_analysis.py', 'stationarity_boundary.py',
          'run_selftests.py', 'diag_check.py', 'diag_optunit.py',
          'ancilla_recovery.py', 'multiseed_stats.py', 'storage_rounds.py',
          'hw_verify_analysis.py', 'hw_error_budget.py',
          'paper/make_ed.py', 'paper/fill_numbers.py',
          'paper/main.tex', 'paper/extended_data.tex', 'paper/refs.bib',
          'paper/README.md',
          'scaling_results.json', 'stationarity_boundary.json',
          'ancilla_recovery.json', 'multiseed_results.json',
          'storage_rounds.json', 'hw_feasibility_numbers.json',
          'hw_error_budget.json'):
    chk(os.path.getsize(f) > 0, f + ' present')

P = json.load(open('paper_numbers.json'))

# ---- 1. every reported fidelity is a physical probability -------------------
CURVES = ('dep_full_curves', 'ad_full_curves', 'mx_full_curves', 'coh_full_curves')
P010 = ('dep_p010', 'ad_p010', 'mx_p010', 'coh_p010')
vals, over = [], []
for k in CURVES:
    for meth, vs in P[k].items():
        for i, v in enumerate(vs):
            vals.append(v)
            if v > 1 + 1e-9 or v < -1e-9:
                over.append((k, meth, i, v))
for k in P010:
    for meth, v in P[k].items():
        vals.append(v)
        if v > 1 + 1e-9 or v < -1e-9:
            over.append((k, meth, None, v))
chk(len(vals) > 200, '%d reported fidelity values collected' % len(vals))
chk(not over, 'all reported fidelities in [0, 1] (max=%.16f, viol=%s)'
    % (max(vals), over[:4]))

# ---- 2. the warm start never loses to the decoder --------------------------
DEC, WARM = 'Perfect-code decoder', 'VSCR warm (ours)'
for k in CURVES:
    d, w = P[k][DEC], P[k][WARM]
    bad = [(i, a, b) for i, (a, b) in enumerate(zip(d, w)) if b < a - 1e-12]
    chk(not bad, '%s: F_warm >= F_decoder at all %d p (worst=%s)'
        % (k, len(d), bad[:2]))
for k in P010:
    d, w = P[k][DEC], P[k][WARM]
    chk(w >= d - 1e-12, '%s: F_warm %.12f >= F_dec %.12f (%+.3e)' % (k, w, d, w - d))

# ---- 3. Fix A/B refinement bookkeeping -------------------------------------
RK = ('F_decoder', 'F_grad', 'F_refined', 'F_used', 'F_cert', 'F_bound',
      'headroom', 'capture_frac', 'cert_valid', 'prod_err', 'picked')
refines = [('infos.' + ch, P['infos'][ch][-1]) for ch in P['infos']]
refines.append(('infos_coh', P['infos_coh'][-1]))
chk(len(refines) == 4, 'found 4 curriculum refinement records')
for tag, r in refines:
    chk(all(k in r for k in RK), tag + ': all refine fields present')
    chk(r['stage'] == 'refine', tag + ": stage == 'refine'")
    chk(r['cert_valid'] is True, tag + ': cert_valid is True')
    chk(r['headroom'] >= -1e-12, tag + ': headroom %.3e >= 0' % r['headroom'])
    chk(0.0 <= r['capture_frac'] <= 1.0 + 1e-9,
        tag + ': capture_frac %.6f in [0,1]' % r['capture_frac'])
    chk(abs(r['F_used'] - max(r['F_decoder'], r['F_refined'])) < 1e-12,
        tag + ': F_used == max(F_decoder, F_refined) [floor engaged]')
    chk(r['F_used'] >= r['F_decoder'] - 1e-12,
        tag + ': F_used >= F_decoder (%+.3e)' % (r['F_used'] - r['F_decoder']))
    chk(r['F_bound'] >= r['F_cert'] - 1e-9,
        tag + ': F_bound %.12f >= F_cert %.12f' % (r['F_bound'], r['F_cert']))
    chk(r['F_cert'] <= 1 + 1e-9, tag + ': F_cert <= 1')
    chk(r['prod_err'] < 1e-12, tag + ': prod_err %.2e < 1e-12' % r['prod_err'])
    chk(r['picked'] in ('decoder', 'refined'), tag + ': picked = ' + r['picked'])

chk(P['infos']['depolarizing'][-1]['headroom'] < 1e-6,
    'depolarizing headroom ~0 (decoder already optimal there)')
chk(P['infos']['mixed'][-1]['headroom'] < 1e-6,
    'mixed headroom ~0 (decoder already optimal there)')
chk(P['infos']['amplitude_damping'][-1]['headroom'] > 1e-4,
    'amplitude-damping headroom %.3e > 1e-4'
    % P['infos']['amplitude_damping'][-1]['headroom'])
chk(0.89 < P['infos']['amplitude_damping'][-1]['capture_frac'] < 0.91,
    'amplitude-damping capture_frac %.6f ~ 0.90'
    % P['infos']['amplitude_damping'][-1]['capture_frac'])
chk(P['infos_coh'][-1]['capture_frac'] > 0.9999,
    'coherent capture_frac %.6f > 0.9999' % P['infos_coh'][-1]['capture_frac'])

# ---- 4. SDP ceiling obeys weak duality: it bounds every feasible point ------
sd = P['abl']['sdp']
chk(len(sd) == 8, 'abl.sdp has 8 operating points (%s)' % sorted(sd))
for key in sorted(sd):
    d = sd[key]
    c, u, dx = d['F_cptp'], d['F_unit'], d['F_dec_exact']
    chk(c >= u - 1e-9, 'sdp[%s]: F_cptp %.9f >= F_unit %.9f' % (key, c, u))
    chk(c >= dx - 1e-9, 'sdp[%s]: F_cptp %.9f >= F_dec_exact %.9f' % (key, c, dx))
    chk(c <= 1 + 1e-9, 'sdp[%s]: F_cptp %.9f <= 1' % (key, c))
chk(sd['coh_0.15']['F_cptp'] > 0.999999,
    'sdp[coh_0.15]: ceiling == 1 (a known unitary is exactly correctable)')
chk(abs(sd['ad_0.1']['F_cptp'] - sd['ad_0.1']['F_unit']) < 1e-9,
    'sdp[ad_0.1]: F_cptp == F_unit == %.9f' % sd['ad_0.1']['F_cptp'])
chk(sd['ad_0.1']['F_dec_exact'] < sd['ad_0.1']['F_unit'],
    'sdp[ad_0.1]: decoder %.9f strictly below ceiling %.9f'
    % (sd['ad_0.1']['F_dec_exact'], sd['ad_0.1']['F_cptp']))

# ---- 5. Fig.4(b) gap-to-ceiling bars are non-negative ----------------------
g = P['abl']['gap_to_ceiling']
chk(set(g) == {'depolarizing', 'amplitude_damping', 'mixed', 'coherent'},
    'gap_to_ceiling covers all 4 channels')
bad = [(c, k, v) for c, d in g.items() for k, v in d.items()
       if k != 'F_cptp' and isinstance(v, float) and v < -1e-12]
chk(not bad, 'no negative gap-to-ceiling (a bar above its own bound): %s' % bad)
for c in sorted(g):
    d = g[c]
    for k, v in d.items():
        if k != 'F_cptp' and isinstance(v, float):
            chk(v <= d['F_cptp'] + 1e-9,
                'gap[%s][%s] = %.3e consistent with F_cptp = %.9f'
                % (c, k, v, d['F_cptp']))

# ---- 6. the ablation is fair: the independent table is refined too ---------
ind = P['abl']['ind']
chk(set(ind) == {'depolarizing', 'amplitude_damping', 'mixed', 'coherent'},
    'abl.ind covers all 4 channels')
worst = 0.0
for ch in sorted(ind):
    d = ind[ch]
    rf = [i for i in d['infos'] if i.get('stage') == 'refine']
    chk(len(rf) == 1, 'abl.ind[%s]: independent table WAS refined (Fix A/B)' % ch)
    if rf:
        chk(-1e-12 <= rf[0]['capture_frac'] <= 1.0000001,
            'abl.ind[%s]: capture_frac %.6f valid' % (ch, rf[0]['capture_frac']))
    worst = max(worst, d['max_abs_diff_vs_warm'])
chk(worst <= 1e-8,
    'independent table == warm hypernetwork to %.3e <= 1e-8 on the full p-grid'
    % worst)

ld = P['abl']['lowdata']
chk(ld['K'] == [2, 4, 8, 16], 'abl.lowdata swept K = %s' % ld['K'])
wd = max(abs(a - b) for a, b in zip(ld['warm_hyper_F'], ld['warm_ind_F']))
chk(wd < 1e-8, 'warm low-data: hyper vs independent indistinguishable (%.2e)' % wd)
chk(min(ld['warm_hyper_mincf']) > 0.87,
    'warm low-data: min cf_s stays above 0.87 at every K')
cd = [b - a for a, b in zip(ld['cold_hyper_F'], ld['cold_ind_F'])]
chk(min(cd) < 0 < max(cd),
    'cold low-data: ordering is NOT monotone in K (ind-hyper spans %+.2e..%+.2e)'
    % (min(cd), max(cd)))

# ---- 7. raw-ZNE overshoot audit (why the physical ZNE variant is needed) ----
z_in = P['zne_unphysical_overshoot']
chk(set(z_in) == {'depolarizing', 'amplitude_damping', 'mixed'},
    'incoherent raw-ZNE overshoot audit covers 3 channels')
tot_in = sum(v for d in z_in.values() for v in d.values()
             if isinstance(v, (int, float)))
chk(tot_in == 0,
    'incoherent channels: 0 unphysical raw-ZNE values > 1 (total=%s)' % tot_in)
z_coh = P['coh_zne_unphysical_overshoot']
chk(len(z_coh) == 10, 'coherent raw-ZNE overshoot audit covers %d p values'
    % len(z_coh))
print('    coh_zne_unphysical_overshoot[0.3] = ' + json.dumps(z_coh['0.3']))

# ---- 8. every LaTeX cross-reference resolves; no float is orphaned ---------
t = open('paper/main.tex').read()
r = open('paper/refs.bib').read()
chk(t.count(BS + 'begin{') == t.count(BS + 'end{'),
    'main.tex environment balance (%d begin / %d end)'
    % (t.count(BS + 'begin{'), t.count(BS + 'end{')))

bib = set(re.findall(r'(?m)^@[A-Za-z]+\{([^,]+),', r))
labels = set(re.findall(re.escape(BS) + r'label\{([^}]*)\}', t))
cites = set()
for grp in re.findall(re.escape(BS) + r'cite[a-z]*\{([^}]*)\}', t):
    cites.update(k.strip() for k in grp.split(',') if k.strip())
refs = set(re.findall(
    re.escape(BS) + r'(?:ref|eqref|cref|Cref|autoref)\{([^}]*)\}', t))

chk(len(bib) >= 40, 'refs.bib has %d entries' % len(bib))
chk(not (cites - bib), 'no undefined cite keys (%s)' % sorted(cites - bib))
chk(not (refs - labels), 'no undefined ref targets (%s)' % sorted(refs - labels))
chk(not (labels - refs),
    'no orphaned label -- every float/equation is referenced (%s)'
    % sorted(labels - refs))
for lab in ('fig:benchmark', 'fig:schematic', 'fig:training',
            'fig:ablation', 'fig:hardware', 'eq:F', 'eq:petz', 'eq:warm'):
    chk(lab in refs, 'main.tex references ' + lab)
chk(len(cites) >= 40, 'main.tex cites %d distinct keys' % len(cites))
chk('fig:coherent' not in labels,
    'stale duplicate fig:coherent label is gone (coherent is a panel of '
    'fig:benchmark, so a second label would print the same number)')
chk('fig_coherent.pdf' in t, 'the coherent panel graphic is still included')
chk(t.count(BS + 'label{fig:benchmark}') == 1,
    'fig:benchmark is labelled exactly once')

# ---- 9. Extended Data: all nine tables present and well formed --------------
e = open('paper/extended_data.tex').read()
chk(e.count(BS + 'begin{') == e.count(BS + 'end{'),
    'extended_data.tex environment balance (%d begin / %d end)'
    % (e.count(BS + 'begin{'), e.count(BS + 'end{')))
chk(e.count(BS + 'begin{tabular}') == e.count(BS + 'end{tabular}'),
    'extended_data.tex tabular balance (%d/%d)'
    % (e.count(BS + 'begin{tabular}'), e.count(BS + 'end{tabular}')))


def _math_spans(src):
    """Return the [open, close) index pairs of every inline-math span in `src`.

    Only unescaped `$` delimiters count, so a literal `\\$` in prose cannot shift
    the pairing.  The second return value is the index of an opener that never
    closed (-1 when the file ends outside math mode).
    """
    idx = [i for i, c in enumerate(src)
           if c == '$' and (i == 0 or src[i - 1] != '\\')]
    spans, start = [], -1
    for i in idx:
        if start < 0:
            start = i
        else:
            spans.append((start, i))
            start = -1
    return spans, (start if start >= 0 else -1), len(idx)


# No TeX engine exists on this host, so `extended_data.tex` has never been compiled
# and a malformed math delimiter cannot be caught by building the PDF.  That is not
# hypothetical: ED Table 12 emitted `\begin{center}$U=$ \begin{pmatrix}...`, which
# closes math BEFORE the matrix -- putting pmatrix in text mode (a hard LaTeX error)
# and leaving a dangling `$` that unbalances every span after it.  It survived for
# exactly the reason the 411 bare `\times` cells did.  These three checks are the
# cheap structural substitute for a compile: balanced delimiters, no unclosed span,
# and no matrix environment sitting outside math.
for _nm, _src in (('extended_data.tex', e), ('main.tex', t)):
    _sp, _open, _n = _math_spans(_src)
    chk(_n % 2 == 0, '%s has an even number of unescaped $ delimiters (%d)'
        % (_nm, _n))
    chk(_open < 0, '%s ends outside math mode (no unclosed $ span)' % _nm)
    for _env in ('pmatrix', 'bmatrix', 'vmatrix', 'matrix', 'cases', 'aligned'):
        _beg = BS + 'begin{' + _env + '}'
        _out = [i for i in range(len(_src))
                if _src.startswith(_beg, i)
                and not any(a <= i < b for a, b in _sp)]
        chk(not _out, '%s keeps every %s inside a math span (%d stray)'
            % (_nm, _beg, len(_out)))
for n in range(1, 12):
    if n == 10:
        # Table 10 was split: 10a is the analytic nine-sector certificate and 10b
        # the direct 90-pair sweep that 10a subsumes.
        chk('ED Table 10a:' in e and 'ED Table 10b:' in e,
            'ED Tables 10a and 10b present (subsection headers)')
        continue
    chk(('ED Table %d:' % n) in e, 'ED Table %d present (subsection header)' % n)
chk('ED Table 11b:' in e, 'ED Table 11b present (readout-law verification)')
chk('n/a ($0$)' in e, 'ED Table 8 shows n/a for zero-headroom channels')
for n in (7, 8, 11):
    chk(t.count('ED Table~%d' % n) >= 1,
        'main.tex points the reader at ED Table~%d' % n)
chk('ED Table~10a' in t and 'ED Table~10b' in t,
    'main.tex points the reader at both stationarity tables 10a and 10b')

# ---- 9b. the two reviewer-response tables carry their load-bearing numbers ---
# ED Table 10a is the analytic nine-sector certificate and ED Table 10b the direct
# 90-pair sweep it subsumes.  Audit the artifacts behind both rather than the
# rendered digits.
_sb = json.load(open('stationarity_boundary.json'))
_sbs = _sb['summary']
_cert = _sb['certificate']

# 10a -- the certificate.  This is what turns the sweep from evidence into a
# proof: the objective is an exact quadratic in the input Bloch vector whose nine
# l=0,1,2 sector coefficients are ensemble-independent, and those sectors are
# orthogonal on S^2, so their gradients vanishing is necessary AND sufficient for
# stationarity under every input ensemble.
chk(len(_cert) == 9, 'certificate covers %d noise channels' % len(_cert))
chk(_sbs['cert_n_angles'] == 960,
    'every sector gradient is maximised over all %d ansatz angles'
    % _sbs['cert_n_angles'])
chk(_sb['stationary_proved_for_every_ensemble'] is True,
    'the nine sector gradients vanish, so phi^dec is PROVED stationary for every '
    'input ensemble rather than merely for the 90 sampled pairs')
chk(_sbs['cert_max_sector'] < 1e-14,
    'worst fidelity sector gradient is %s' % _sbs['cert_max_sector'])
chk(_sbs['cert_fd_coeff_max'] < 1e-8,
    'central differences of the nine coefficient functions agree with autograd: %s'
    % _sbs['cert_fd_coeff_max'])
chk(_sbs['cert_min_control_sector'] > 1e-4,
    'weakest certificate control is %s, so the zeros are measurable'
    % _sbs['cert_min_control_sector'])
chk(_sbs['cert_min_control_sector'] / _sbs['cert_max_sector'] > 1e10,
    'certificate control exceeds the sector gradient by >10 orders of magnitude')
chk(_sbs['cert_Z_min_sector'] < 1e-14,
    'the four degree-1 <Z_L> sectors also vanish: %s' % _sbs['cert_Z_min_sector'])
chk(_sbs['cert_Z_min_control_sector'] > 1e-4,
    'weakest <Z_L> control is %s' % _sbs['cert_Z_min_control_sector'])
for _r in _cert:
    chk(_r['max_sector'] < 1e-14 and _r['control_max_sector'] > 1e-4,
        'certificate per channel: %s (sector %s, control %s)'
        % (_r['channel'], _r['max_sector'], _r['control_max_sector']))
    chk(_r['max_Z_sector'] < 1e-14,
        'certificate per channel, <Z_L> sectors: %s (%s)'
        % (_r['channel'], _r['max_Z_sector']))

# 10b -- the sweep, retained as the numerical corollary of 10a
chk(_sbs['max_grad_at_decoder'] is not None,
    'the 90-pair sweep was actually run (artifact is not --certificate-only)')
chk(len(_sb['rows']) == 90,
    'stationarity sweep covers %d channel x functional pairs' % len(_sb['rows']))
chk(_sbs['n_stationary_2designs'] == _sbs['n_2designs'],
    'all %d two-design pairs stationary' % _sbs['n_2designs'])
chk(_sbs['n_stationary_non_designs'] == _sbs['n_non_designs'],
    'all %d NON-two-design pairs stationary (the twirl is not the reason)'
    % _sbs['n_non_designs'])
chk(_sbs['max_grad_at_decoder'] < 1e-14,
    'max gradient at phi^dec is %s' % _sbs['max_grad_at_decoder'])
chk(_sbs['min_control_grad'] > 1e-4,
    'min control gradient is %s, so the zeros are measurable'
    % _sbs['min_control_grad'])
chk(_sbs['min_control_grad'] / _sbs['max_grad_at_decoder'] > 1e10,
    'control exceeds decoder gradient by >10 orders of magnitude')
chk(_sb['stationary_on_all_tested'] is True, 'no boundary found in this class')
for _r in _sb['rows']:
    chk(_r['grad_fd'] == 0.0,
        'central difference is bitwise zero: %s / %s'
        % (_r['channel'], _r['ensemble']))
chk('ED Table 10a:' in e and 'ED Table 10b:' in e,
    'both stationarity tables are rendered in extended_data.tex')
chk('90' in e, 'ED Table 10b states the 90-pair total')
# The caption must quote the same two-estimator control minimum that the table's
# own "min control" column reports.  `\$?` in the patterns below is because sci()
# emits its value already wrapped in math mode.  summary['min_control_grad'] is autograd-only
# and slightly larger, so quoting it made the caption disagree with its table.
_ctl_min = min(min(_r['control_grad_autograd'], _r['control_grad_fd'])
               for _r in _sb['rows'])
_grad_max = max(max(_r['grad_autograd'], _r['grad_fd']) for _r in _sb['rows'])
_mctl = re.search(r'min control \$?(\d\.\d+)\\times 10\^\{(-?\d+)\}', e)
_mgrad = re.search(r'max gradient \$?(\d\.\d+)\\times 10\^\{(-?\d+)\}', e)
chk(_mctl is not None and _mgrad is not None,
    'ED Table 10b caption quotes both a max gradient and a min control')
if _mctl and _mgrad:
    _cap_ctl = float(_mctl.group(1)) * 10.0 ** int(_mctl.group(2))
    _cap_grad = float(_mgrad.group(1)) * 10.0 ** int(_mgrad.group(2))
    _tol_ctl = 0.005 * 10.0 ** int(_mctl.group(2))
    _tol_grad = 0.005 * 10.0 ** int(_mgrad.group(2))
    chk(abs(_cap_ctl - _ctl_min) <= _tol_ctl,
        'ED Table 10b caption min control %.3g equals the table column minimum '
        '%.3g' % (_cap_ctl, _ctl_min))
    chk(abs(_cap_grad - _grad_max) <= _tol_grad,
        'ED Table 10b caption max gradient %.3g equals the table maximum %.3g'
        % (_cap_grad, _grad_max))

# ED Table 11 is the scaling sweep: n=5 must reproduce the audited paper numbers,
# and every code size must satisfy the exact readout law.
_sca = json.load(open('scaling_results.json'))
_sdp = P['abl']['sdp']
_v = _sca['validation_n5']
chk(max(_v.values()) < 1e-11,
    'scaling n=5 reproduces paper_numbers.json to %s' % max(_v.values()))
# Petz references live in the per-channel benchmark blocks, not in abl.sdp.
_pz = {'depolarizing_0.1': P['dep_p010']['Petz recovery (2024)'],
       'amplitude_damping_0.1': P['ad_p010']['Petz recovery (2024)'],
       'mixed_0.1': P['mx_p010']['Petz recovery (2024)']}
for _ref, _k in (('dep_0.1', 'depolarizing_0.1'), ('ad_0.1', 'amplitude_damping_0.1'),
                 ('mixed_0.1', 'mixed_0.1')):
    _p5 = _sca['codes']['5,1,3']['points'][_k]
    _a = _sdp[_ref]
    chk(abs(_p5['F_dec'] - _a['F_dec_exact']) < 1e-11,
        'scaling F_dec[%s] == audited F_dec_exact' % _ref)
    chk(abs(_p5['F_unit'] - _a['F_unit']) < 1e-11,
        'scaling F_unit[%s] == audited F_unit' % _ref)
    chk(abs(_p5['F_cptp'] - _a['F_cptp']) < 1e-9,
        'scaling F_cptp[%s] == audited F_cptp' % _ref)
    # The audited Petz baseline in paper_numbers.json is a 400-state Monte-Carlo
    # estimate from ssvr_qec.petz_recovery_fidelity; the scaling value is the exact
    # Haar average. scaling_analysis._selftest_petz ties them together with the
    # tolerance 5*sem + 1e-4, so the audit uses the same 1e-4 floor rather than
    # demanding agreement to machine precision between an estimator and an exact
    # quantity.
    chk(abs(_p5['F_petz'] - _pz[_k]) < 1e-4,
        'scaling Petz[%s] == audited MC Petz baseline within its error '
        '(exact %.9f vs %.9f, d %.1e)'
        % (_ref, _p5['F_petz'], _pz[_k], abs(_p5['F_petz'] - _pz[_k])))
for _nm, _cd in _sca['codes'].items():
    chk(_cd['readout_cross_offdiag'] < 1e-13,
        '[[%s]] cross-branch readout block vanishes (%s)'
        % (_nm, _cd['readout_cross_offdiag']))
    chk(_cd['readout_cross_diag_dev'] < 1e-13,
        '[[%s]] decoder block is the identity (%s)'
        % (_nm, _cd['readout_cross_diag_dev']))
    for _k, _p in _cd['points'].items():
        for _eta, _c in _p['eta_exact_check'].items():
            chk(_c['dev'] < 1e-11,
                '[[%s]] %s eta=%s obeys (1-eta)^(n-k) exactly (dev %s)'
                % (_nm, _k, _eta, _c['dev']))
        # the Petz fidelity must actually depend on the noise strength; a
        # repeated value here is how a threading/caching bug first showed up.
        # 'coherent' is excluded because Petz inverts a known unitary exactly, so
        # its fidelity is 1.0 at every strength by construction -- asserted
        # separately below rather than mistaken for a stale cache entry.
    for _ch in ('amplitude_damping',):
        _ks = sorted(k for k in _cd['points'] if k.startswith(_ch))
        if len(_ks) == 2:
            _a, _b = (_cd['points'][k]['F_petz'] for k in _ks)
            chk(abs(_a - _b) > 1e-9,
                '[[%s]] %s Petz is p-dependent (%.9f vs %.9f)' % (_nm, _ch, _a, _b))
    for _k, _p in _cd['points'].items():
        if _k.startswith('coherent'):
            chk(abs(_p['F_petz'] - 1.0) < 1e-12,
                '[[%s]] %s Petz inverts the known unitary exactly (%.12f)'
                % (_nm, _k, _p['F_petz']))
        if _k.startswith('depolarizing'):
            chk(_p['F_petz'] < _p['F_dec'] - 1e-3,
                '[[%s]] %s Petz trails the syndrome-resolved decoder '
                '(%.6f vs %.6f)' % (_nm, _k, _p['F_petz'], _p['F_dec']))
        if _k.startswith('depolarizing') or _k.startswith('mixed'):
            chk(_p['headroom_cptp'] < 1e-12,
                '[[%s]] %s Pauli-type noise: decoder is CPTP-optimal (%s)'
                % (_nm, _k, _p['headroom_cptp']))
        if _k.startswith('coherent'):
            chk(abs(_p['F_unit'] - 1.0) < 1e-12,
                '[[%s]] %s a perfect branch-unitary recovery exists (%.12f)'
                % (_nm, _k, _p['F_unit']))

# ---- 9c. the Kraus-rank ladder: how many ancillas the headroom needs --------
# ED Table 12 turns the optimal-CPTP ceiling into a COUNT OF ANCILLAS: rank(C)=r
# is exactly the minimal Stinespring ancilla dimension of the branch recovery.
# The load-bearing checks are that the two END rungs reproduce the audited
# production ceilings (so the ladder is not a new unvalidated quantity), that the
# ladder is monotone in rank, that the returned rank-2 instrument is exactly trace
# preserving with an exactly unitary dilation, and that an INDEPENDENT
# unconstrained circuit optimisation lands on the same value.
_ar = json.load(open('ancilla_recovery.json'))
_arc = _ar['selftests']['conventions']
_art = _ar['selftests']['tp_stinespring']
_arctl = _ar['selftests']['controls']
chk(_arc['worst_r1_vs_production'] < 1e-9,
    'ladder rank 1 reproduces the production unitary ceiling to %s'
    % _arc['worst_r1_vs_production'])
chk(_arc['worst_r4_vs_production'] < 1e-9,
    'ladder rank 4 reproduces the production CPTP ceiling to %s'
    % _arc['worst_r4_vs_production'])
chk(_arc['worst_r4_equality_vs_inequality'] < 1e-9,
    'the equality-TP optimum matches the inequality-TP one at r=4 (%s), so the '
    'ceiling is over channels, not sub-channels'
    % _arc['worst_r4_equality_vs_inequality'])
chk(_arc['worst_choi_vs_kraus'] < 1e-11,
    'the multi-Kraus Choi objective equals production cf_unnormalised to %s'
    % _arc['worst_choi_vs_kraus'])
chk(_arc['worst_monotonicity'] < 1e-9,
    'the ladder is monotone in rank; worst violation %s'
    % _arc['worst_monotonicity'])
chk(_art['tp_rep'] < 1e-12,
    'every witness Kraus pair is exactly trace preserving (%s)' % _art['tp_rep'])
chk(_art['unit'] < 1e-12,
    'every Stinespring dilation is exactly unitary (%s)' % _art['unit'])
chk(_art['map_'] < 1e-12,
    'Tr_a[U(rho x |0><0|)U^dag] == sum_i B_i rho B_i^dag (%s)' % _art['map_'])
chk(_art['direct'] < 1e-9,
    'the unconstrained exp(iH) circuit route matches the constrained Choi '
    'rank-2 optimum (%s)' % _art['direct'])
chk(_art['direct_unit'] < 1e-12 and _art['direct_tp'] < 1e-12,
    'the circuit route is itself exactly unitary (%s) and trace preserving (%s)'
    % (_art['direct_unit'], _art['direct_tp']))
for _ch, _c in _arctl.items():
    chk(_c['ladder_spread'] < 1e-9,
        'control channel %s gives a FLAT ladder across all rungs (spread %s)'
        % (_ch, _c['ladder_spread']))

# cross-artifact: the ladder's end rungs ARE the scaling sweep's F_unit / F_cptp
_lad = {(r['code'], r['channel'], r['p']): r for r in _ar['ladder']}
_nlocked = 0
for _nm, _cd in _sca['codes'].items():
    for _k, _p in _cd['points'].items():
        _ch, _pv = _k.rsplit('_', 1)
        _r = _lad.get((_nm, _ch, float(_pv)))
        if _r is None:
            continue
        _nlocked += 1
        chk(abs(_r['F']['1'] - _p['F_unit']) < 1e-9,
            '[[%s]] %s: ladder rank 1 == scaling F_unit (%.10f vs %.10f)'
            % (_nm, _k, _r['F']['1'], _p['F_unit']))
        chk(abs(_r['F']['4'] - _p['F_cptp']) < 1e-8,
            '[[%s]] %s: ladder rank 4 == scaling F_cptp (%.10f vs %.10f)'
            % (_nm, _k, _r['F']['4'], _p['F_cptp']))
        chk(abs(_r['F_dec'] - _p['F_dec']) < 1e-11,
            '[[%s]] %s: ladder F_dec == scaling F_dec' % (_nm, _k))
        chk(_r['F']['2'] <= _r['F']['4'] + 1e-9,
            '[[%s]] %s: one ancilla never exceeds the full CPTP ceiling '
            '(weak duality)' % (_nm, _k))
chk(_nlocked >= 8, 'ladder and scaling sweep overlap on %d points' % _nlocked)

# the headline: at n=9 amplitude damping the ladder is FLAT above rank 2, so ONE
# ancilla qubit captures the entire non-unitary headroom
for _pv in (0.05, 0.10):
    _r = _lad[('9,1,3', 'amplitude_damping', _pv)]
    _frac = _r['one_ancilla_fraction_of_nonunitary_headroom']
    chk(_frac is not None and abs(_frac - 1.0) < 1e-6,
        '[[9,1,3]] amplitude damping p=%g: one ancilla recovers %.8f of the '
        'non-unitary headroom' % (_pv, _frac))
    chk(abs(_r['one_ancilla_gap_to_cptp']) < 1e-12,
        '[[9,1,3]] amplitude damping p=%g: rank 2 == rank 4 to %.2e'
        % (_pv, _r['one_ancilla_gap_to_cptp']))
    chk(_r['headroom']['1'] < 1e-9,
        '[[9,1,3]] amplitude damping p=%g: the unitary rung IS the decoder '
        '(headroom %.2e)' % (_pv, _r['headroom']['1']))
    chk(_r['headroom']['2'] > 1e-5,
        '[[9,1,3]] amplitude damping p=%g: one ancilla gains %.4e'
        % (_pv, _r['headroom']['2']))

# ---- 9d. main.tex and extended_data.tex quote the ladder artifact -----------
# Each number below is DERIVED from ancilla_recovery.json and rendered with the
# same helper, so prose that drifts away from the artifact fails the audit rather
# than silently disagreeing with it.


def _tex_sci(x, digits=2):
    """Render |x| as the LaTeX mantissa/exponent form the prose uses."""
    mnt, ex = ('%.*e' % (digits, abs(float(x)))).split('e')
    return r'%s\times10^{%d}' % (mnt, int(ex))


def _chk_bound(name, value, exp, phrase):
    """Assert an artifact defect is below a power of ten AND that main.tex says so.

    The defects below are floating-point round-off, so they move in the second
    significant digit between runs and a mantissa quoted in prose would churn on
    every regeneration.  A conservative power of ten is both what the number means
    ("machine precision") and stable under reruns.  `phrase` is matched against the
    whitespace-normalised prose so that line wrapping cannot break the lock."""
    _tfn = ' '.join(t.split())
    chk(value < 10.0 ** exp,
        '%s is %.2e, below the 10^{%d} bound the prose quotes'
        % (name, value, exp))
    chk(phrase in _tfn,
        'main.tex states the %s bound as %r' % (name, phrase))


for _pv in (0.05, 0.10):
    _r = _lad[('9,1,3', 'amplitude_damping', _pv)]
    _q = '+' + _tex_sci(_r['headroom']['4'])
    chk(_q in t,
        'main.tex quotes the n=9 amplitude-damping p=%g CPTP headroom as %s '
        '(artifact %.4e)' % (_pv, _q, _r['headroom']['4']))
    chk(_tex_sci(_r['one_ancilla_gap_to_cptp'], 1) in t,
        'main.tex quotes the rank-2 vs rank-4 flatness at p=%g as %s (artifact '
        '%.2e)' % (_pv, _tex_sci(_r['one_ancilla_gap_to_cptp'], 1),
                   _r['one_ancilla_gap_to_cptp']))
_spread = max(_c['ladder_spread'] for _c in _arctl.values())
_chk_bound('control-channel ladder spread', _spread, -13,
           'spread below $10^{-13}$')
# Both the results and the methods paragraph quote this bound.  Requiring two
# occurrences is what stops one of them from keeping a stale mantissa from an
# earlier run while the other is updated -- which is exactly how an 8.8e-14
# survived here after the artifact moved to 9.3e-14.
_nspread = ' '.join(t.split()).count('spread below $10^{-13}$')
chk(_nspread >= 2,
    'the results and methods paragraphs quote the SAME control-spread bound '
    '(%d occurrences)' % _nspread)
_wits = [w for _r in _ar['ladder'] for w in _r['witness_branches']]
chk(len(_wits) > 0, 'the ladder emitted %d hardware witnesses' % len(_wits))
_tp = max(w['tp_defect_repaired'] for w in _wits)
_un = max(w['unitarity_defect'] for w in _wits)
_mp = max(w['map_defect'] for w in _wits)
_dfc = max(abs(w['cf_choi'] - w['cf_direct']) for w in _wits)
chk(_tp < 1e-12, 'worst witness TP defect over all %d witnesses: %.2e'
    % (len(_wits), _tp))
chk(_un < 1e-12, 'worst witness dilation unitarity defect: %.2e' % _un)
chk(_mp < 1e-12, 'worst witness Tr_a circuit-vs-Kraus defect: %.2e' % _mp)
chk(_dfc < 1e-9, 'worst witness Choi-vs-circuit conditional-fidelity gap: %.2e'
    % _dfc)
_chk_bound('witness trace preservation', _tp, -14,
           'unitary, to below $10^{-14}$')
_chk_bound('witness dilation unitarity', _un, -14,
           'unitary, to below $10^{-14}$')
_chk_bound('Choi-vs-circuit agreement', _dfc, -10,
           'conditional fidelity to below $10^{-10}$')
# the third route: a brute-force Haar quadrature of the physical estimator, which
# uses neither the Choi matrix nor the degree-2 moment identity.  On branches where
# the estimator is exactly psi-independent (coherent: the optimal recovery is the
# inverse unitary; depolarizing: a Pauli twirl) the quadrature has zero variance and
# is a deterministic identity check, so those are measured relatively rather than in
# sigmas -- a naive sigma test would report a fifteen-digit agreement as a failure.
_q = _ar['selftests']['quadrature']
chk(_q['n_states'] >= 20000,
    'the quadrature used %d random pure states per branch' % _q['n_states'])
chk(_q['worst_z'] < 5.0,
    'quadrature agrees with the closed form to %.2f sigma on the %d branches '
    'that have sampling noise' % (_q['worst_z'],
                                  _q['n_branches'] - _q['n_psi_independent']))
chk(_q['worst_rel'] < 1e-11,
    'on the %d psi-independent branches the quadrature is a deterministic '
    'identity check; worst relative deviation %.2e'
    % (_q['n_psi_independent'], _q['worst_rel']))
chk(_q['n_psi_independent'] > 0 and
    _q['n_branches'] - _q['n_psi_independent'] > 0,
    'the quadrature covers both regimes: %d psi-independent and %d with sampling '
    'noise' % (_q['n_psi_independent'], _q['n_branches'] - _q['n_psi_independent']))
_tf = ' '.join(t.split())          # whitespace-normalised prose, for line-wrap safety
chk(('%.2f' % _q['worst_z']) + r'\sigma' in t,
    'main.tex quotes the quadrature agreement as %s sigma (artifact %.4f)'
    % ('%.2f' % _q['worst_z'], _q['worst_z']))
_chk_bound('psi-independent quadrature', _q['worst_rel'], -14,
           'to below $10^{-14}$ relative')
chk('$%d$ witnesses' % len(_wits) in _tf,
    'main.tex states the witness count %d that the artifact actually emits'
    % len(_wits))

# the strongest form of the hardware claim: the instrument exists on the PHYSICAL
# 2^n-dimensional register, not only on the two-dimensional logical space.  The
# (n+1)-qubit unitary Rt_s is built explicitly and four defects must vanish -- that
# sum_i T_i^dag T_i is the syndrome projector, that the two isometry families are
# orthonormal (which is exactly trace preservation), that Rt_s is unitary, and that
# it reproduces the Kraus operators through the paper's own reduced-block map
# (bra i|_a ox V^dagger) Rt_s (|0>_a ox W_s) = B_i.
_pd = _ar['selftests']['physical_dilation']
for _k, _tol in (('isometry_to_projector', 1e-10), ('column_orthonormality', 1e-10),
                 ('unitarity', 1e-10), ('reduced_block', 1e-10)):
    chk(_pd[_k] < _tol,
        'physical (n+1)-qubit dilation, %s defect %.2e' % (_k, _pd[_k]))
chk(_pd['n_branches'] > 0 and _pd['dim'] == 32,
    'the physical dilation was checked on %d branches of a dim=%d register'
    % (_pd['n_branches'], _pd['dim']))
for _k in ('isometry_to_projector', 'column_orthonormality', 'unitarity',
           'reduced_block'):
    _chk_bound('physical dilation ' + _k, _pd[_k], -14,
               'at machine precision on a $2^{n+1}$-dimensional register')
chk('Stinespring' in t and 'physical register' in t,
    'main.tex states that the dilation is a circuit on the physical register')

chk('ED Table 12:' in e and 'ED Table 12b:' in e,
    'both Kraus-rank ladder tables are rendered in extended_data.tex')
chk('ED Table 12c:' in e and 'ED Table 12d:' in e,
    'the witness verification table and the explicit instrument are rendered')
chk('ED Table~12' in t, 'main.tex points the reader at ED Table 12')
chk('gate-level instrument cannot express' not in t,
    'the falsified claim that a gate-level instrument cannot express the '
    'residual headroom has been removed from main.tex')
chk('Stinespring' in t, 'main.tex names the Stinespring dilation')

# ---- 10. key literature is present in the bibliography ---------------------
for k in ('petz_map', 'biswas_petz', 'beny_optimal', 'huggins_vd', 'vikstal_vd',
          'temme_zne', 'endo_zne', 'kandala_zne_hw', 'czarnik_cdr',
          'torlai_nn', 'meinerz_nn', 'tian_transformer', 'laflamme_5q',
          'shor_code', 'steane_code', 'gottesman_hamming'):
    chk(k in bib, 'refs.bib has ' + k)

# ---- 11. every graphic the LaTeX includes exists on disk -------------------
incs = sorted(set(re.findall(r'includegraphics\[[^\]]*\]\{([^}]*)\}', t)))
chk(len(incs) >= 6, 'main.tex includes %d graphics' % len(incs))
for f in incs:
    chk(os.path.exists('paper/figures/' + f) or os.path.exists('paper/' + f),
        'included graphic resolves on disk: ' + f)
for f in ('fig_sim_benchmark', 'fig_sim_ler', 'fig_training', 'fig_branch_cf',
          'fig_coherent', 'fig_ablation', 'fig_hw_feasibility', 'fig_schematic'):
    chk(os.path.exists('paper/figures/' + f + '.pdf'), 'figures/' + f + '.pdf')

# ---- 12. no math-mode symbol outside math mode in the generated ED file ------
# `\times` is a math-mode symbol.  make_ed.py's sci() helper used to emit it bare,
# which turned every one of the 411 table cells that used it into a LaTeX error
# ("Missing $ inserted").  No TeX engine exists on this host, so extended_data.tex
# has never been compiled and the bug was invisible.  Rather than depend on a
# compiler, this check verifies the property directly: strip every $...$ span from
# each line and require that no \times survives.  The file is machine-generated and
# uses only $...$ for inline math, so the line-based strip is exact here.
_bad = []
for _i, _ln in enumerate(e.splitlines(), 1):
    if BS + 'times' in re.sub(r'\$[^$]*\$', '', _ln):
        _bad.append(_i)
chk(not _bad,
    'extended_data.tex has no %s outside math mode (offending lines: %s)'
    % (BS + 'times', _bad[:10]))
chk(e.count(BS + 'times') > 100,
    'extended_data.tex renders %d scientific-notation values, all in math mode'
    % e.count(BS + 'times'))
chk('$$' not in e,
    'extended_data.tex has no $$ display math from double-wrapped sci() output')

print()
# ---- 13. T3: the seed spread behind every "best of two" number ---------------
# The main text reports the better of two seeds.  What makes that defensible is a
# measured spread, and what makes the spread trustworthy is that the two production
# seeds come back bit-for-bit from the new driver -- otherwise the extra fourteen
# seeds would be measuring a different pipeline.  Also locked: no seed may breach
# the decoder floor or the certified ceiling, the certified headroom must be
# seed-independent (it is a function of the channel and code alone), and the two
# degenerate ratios must be reported as undefined rather than as large numbers
# manufactured from round-off.
_ms = json.load(open('multiseed_results.json'))
_msd = _ms['diagnostics']
_rep = _ms['reproduction']
chk(_rep['bit_exact'] and _rep['n_failures'] == 0,
    'T3 reproduction gate: %d fields of seeds {1234,2024} are bit-exact against '
    'paper_numbers.json across %d configs'
    % (_rep['n_fields_compared'], _rep['n_configs_checked'] or 0))
chk(_rep['n_fields_compared'] >= 48,
    'T3 reproduction compares at least 48 fields (%d)'
    % _rep['n_fields_compared'])
chk(len(_msd) == 8,
    'T3 covers 4 channels x 2 protocols (%d groups)' % len(_msd))
for _k, _d in sorted(_msd.items()):
    chk(_d['n_seeds'] >= 16, '%s ran %d seeds' % (_k, _d['n_seeds']))
    chk(_d['n_below_decoder_floor'] == 0,
        '%s: no seed below the decoder floor' % _k)
    chk(_d['n_above_certified_ceiling'] == 0,
        '%s: no seed above the certified ceiling' % _k)
    chk(_d['headroom_is_seed_independent'],
        '%s: certified headroom is seed-independent (std %.1e)'
        % (_k, _d['headroom_std']))
    chk(_d['F_used_min'] >= _d['F_used_mean'] - 10 * max(_d['F_used_std'], 1e-18),
        '%s: no outlier seed below mean - 10 std' % _k)
# Channels with a REAL certified headroom: the effect must dwarf the seed lottery,
# and the reported number must not be a lucky draw.  Note that `ind` converges so
# tightly (std ~2e-16) that its sigma count is legitimately suppressed as a 0/0
# ratio, so the bias is asserted conditionally rather than assumed present.
for _k in ('warm/amplitude_damping', 'warm/coherent',
           'ind/amplitude_damping', 'ind/coherent'):
    _d = _msd[_k]
    chk(_d['signal_to_seed_noise'] >= 1e5,
        '%s: headroom sits %.3g sigma above the seed spread'
        % (_k, _d['signal_to_seed_noise']))
    _bs = _d['selection_bias_sigma']
    if _bs is None:
        chk(_d['std_at_machine_precision'],
            '%s: bias sigma suppressed only because std %.1e is at machine '
            'precision (all seeds converged to one optimum)'
            % (_k, _d['F_used_std']))
    else:
        chk(abs(_bs) <= 2.0,
            '%s: best-of-two selection bias is %+.2f sigma (<= 2)' % (_k, _bs))
    chk(_d['F_used_std'] < 1e-6,
        '%s: seed std %.2e is far below the headroom' % (_k, _d['F_used_std']))
# Channels where the decoder IS the optimum: both ratios are 0/0 and must be
# reported as undefined.  This is the check that stops "-149 sigma" reaching print.
for _k in ('warm/depolarizing', 'warm/mixed',
           'ind/depolarizing', 'ind/mixed'):
    _d = _msd[_k]
    chk(_d['headroom_is_zero'] and _d['signal_to_seed_noise'] is None,
        '%s: zero certified headroom -> SNR reported undefined, not as a ratio '
        'of round-off' % _k)
    chk(_d['std_at_machine_precision'] and _d['selection_bias_sigma'] is None,
        '%s: machine-precision seed std -> sigma count suppressed' % _k)
# The seed lottery must sit far below the estimator noise the paper refuses.
chk(_msd['warm/amplitude_damping']['seed_std_vs_mc_sem'] < 1e-3,
    'T3 seed std is %.2e of the 1.1e-4 Monte-Carlo SEM the quadrature avoids'
    % _msd['warm/amplitude_damping']['seed_std_vs_mc_sem'])
# The device decision must be a measurement on file, not an assertion.
_gc = _ms.get('gpu_check') or {}
chk(bool(_gc.get('available')),
    'T3 records the CUDA cross-check in multiseed_results.json')
if _gc.get('available'):
    chk(_gc['dF_cpu_vs_cuda'] < 1e-7,
        'T3 the same curriculum on CPU and CUDA agrees to %.2e'
        % _gc['dF_cpu_vs_cuda'])
    chk(_gc['verdict']['sweep_device'] == 'cpu'
        and _gc['verdict']['gpu_slower_at_one_seed'],
        'T3 device verdict recorded: the GPU is slower at one seed (%.2fx), so '
        'the sweep runs on pinned CPU cores'
        % _gc['cuda_speedup_at_one_seed'])
    chk(_gc['cuda_scaling_1024_over_16']
        < 0.5 * _gc['cpu_scaling_1024_over_16'],
        'T3 the GPU does amortise batching (%.1fx vs %.1fx from B=16 to 1024) -- '
        'the CPU choice is about reproduction, not ignorance of that'
        % (_gc['cuda_scaling_1024_over_16'], _gc['cpu_scaling_1024_over_16']))
    # The prose quotes "1.9x slower at one seed".  Wall-clock ratios jitter a few
    # percent between runs, so this is locked as a band rather than a mantissa --
    # the same reasoning `_chk_bound` uses for round-off defects.  The band is
    # tight enough that the claim would fail if the GPU ever became the faster
    # device for this workload, which is the only way the prose could become wrong.
    _slow = 1.0 / _gc['cuda_speedup_at_one_seed']
    chk(1.5 < _slow < 3.0,
        'T3 the GPU is %.2fx slower than a pinned core at one seed, consistent '
        'with the 1.9x the prose quotes' % _slow)
    # ED Table 13c renders its timings straight out of `gpu_check`, and until now
    # nothing in this audit ever compared the RENDERED table against the artifact
    # it was rendered from -- the ED file was only checked for the presence of its
    # subsection headings.  That gap is not hypothetical: multiseed_results.json was
    # regenerated and committed on 2026-09-18 without re-running make_ed.py, so the
    # committed extended_data.tex quoted 1.88x / 3.0x / 36.3x and a timing table
    # reading 5.405/10.863 against an artifact that said 1.86x / 2.9x / 33.3x and
    # 5.216/10.616 -- and all 705 checks still passed, because every one of them
    # looked at main.tex or at the artifact, never at the ED rendering of it.
    # main.tex quotes none of these numbers, so it was an Extended-Data-only
    # inconsistency; it is locked here so the table cannot drift from its source
    # again.  The format strings below are make_ed.py's own, character for character.
    _efn = ' '.join(e.split())
    for _q, _why in (
            ('GPU $%.2f\\times$' % (1.0 / _gc['cuda_speedup_at_one_seed']),
             'the one-seed GPU slowdown'),
            ('batching ($%.1f\\times$' % _gc['cuda_scaling_1024_over_16'],
             'the CUDA batch-scaling factor'),
            ('against $%.1f\\times$ on the CPU'
             % _gc['cpu_scaling_1024_over_16'],
             'the CPU batch-scaling factor')):
        chk(_q in _efn, 'ED Table 13c quotes %s as %s' % (_why, _q))
    for _b in sorted(_gc['batched_step'], key=int):
        _r = _gc['batched_step'][_b]
        _row = ('$%s$ & $%d$ & %.3f & %.3f & %.2f'
                % (_b, _r['seeds'], _r['cpu_ms'], _r['cuda_ms'],
                   _r['cuda_over_cpu']))
        chk(_row in _efn,
            'ED Table 13c batch-%s timing row matches multiseed_results.json (%s)'
            % (_b, _row))


# ---- 13c. main.tex quotes the seed statistics it is now entitled to ----------
# Every mantissa below is RE-DERIVED from the artifact with the same renderer the
# prose uses, so a regenerated sweep that moves a digit fails the audit instead of
# leaving stale prose behind -- which is exactly how the 8.8e-14 / 9.3e-14 split
# survived in section 9d until it was locked the same way.
_tfn13 = ' '.join(t.split())
_ad = _msd['warm/amplitude_damping']
_co = _msd['warm/coherent']
for _q, _why in (
        (_tex_sci(_ad['F_used_std'], 2), 'the amplitude-damping seed std'),
        (_tex_sci(_co['F_used_std'], 2), 'the coherent seed std'),
        (_tex_sci(_ad['headroom_mean'], 2), 'the amplitude-damping headroom'),
        (_tex_sci(_co['headroom_mean'], 2), 'the coherent headroom'),
        (_tex_sci(_ad['signal_to_seed_noise'], 1),
         'the amplitude-damping headroom-to-spread ratio in sigmas'),
        (_tex_sci(_co['signal_to_seed_noise'], 1),
         'the coherent headroom-to-spread ratio in sigmas'),
        ('%+.2f' % _ad['selection_bias_sigma'] + r'\sigma',
         'the amplitude-damping selection bias in sigmas'),
        ('%+.2f' % _co['selection_bias_sigma'] + r'\sigma',
         'the coherent selection bias in sigmas'),
        (_tex_sci(_gc['dF_cpu_vs_cuda'], 1) if _gc.get('available') else None,
         'the CPU-vs-CUDA objective agreement')):
    if _q is None:
        continue
    chk(_q in _tfn13, 'main.tex quotes %s as %s' % (_why, _q))
chk('%d fields of the two production seeds' % _rep['n_fields_compared']
    in _tfn13.replace('Forty-eight', '48'),
    'main.tex states the reproduction gate as %d bit-exact fields'
    % _rep['n_fields_compared'])
chk('best of two' in _tfn13 or 'better of two' in _tfn13,
    'main.tex still says plainly that the reported value is a maximum over two '
    'seeds, now that the spread is published beside it')

# ---- 14. T4: multi-round logical storage ------------------------------------
# A multi-round number is only a statement about the paper's quantity if it
# collapses onto the audited single-round number at R=1, so that anchor is locked
# first.  Then the physics claims the prose makes: the advantage compounds, the
# Pauli readout law is exact while the learned recovery only nearly obeys it, the
# reduced map matches the full density matrix, and leakage saturates.
_st = json.load(open('storage_rounds.json'))
_sv = _st['validation']
chk(_sv['passed'] and _sv['n_failures'] == 0,
    'T4 anchor: %d R=1 comparisons against paper_numbers.json[*_p010] pass at the '
    'Monte-Carlo tolerance %.0e' % (_sv['n_comparisons'], _sv['tolerance']))
_srs = _st['records']
chk(all(abs(r['F'][0] - 1.0) < 1e-14 for r in _srs),
    'T4 every curve starts at F(0)=1 (%d records)' % len(_srs))
chk(all(all(r['F'][i] >= r['F'][i + 1] - 1e-12
            for i in range(len(r['F']) - 1)) for r in _srs),
    'T4 F(R) is monotone non-increasing in every record: each round applies a '
    'noisy channel and no recovery can undo destroyed information')
chk(all(abs(r['branch_prob_sum'] - 1.0) < 1e-12 for r in _srs),
    'T4 branch probabilities sum to 1 in every record')
_srf = _st['full_space'] or []
chk(len(_srf) >= 20,
    'T4 the reduced map is validated against the full density matrix on %d '
    'configurations' % len(_srf))
_wd = max(r['max_abs_full_vs_reduced'] for r in _srf
          if r['recovery'] == 'decoder')
_wl = max(r['leakage_max'] for r in _srf)
chk(_wd < 1e-11,
    'T4 full-space == reduced to %.2e for the Pauli decoder, where the reduction '
    'is exact' % _wd)
_warm = [r for r in _srf if r['recovery'] != 'decoder']
chk(all(r['max_abs_full_vs_reduced'] < max(1e-6, 10 * r['unitarity_dev'])
        for r in _warm),
    'T4 full-space == reduced within the learned table\'s unitarity deviation')
chk(_wl < 1e-7,
    'T4 leaked fraction peaks at %.2e and saturates rather than compounding, '
    'which is what licenses the 2x2 reduction for long memories' % _wl)
chk(all(r['trace_dev_max'] < 1e-11 for r in _srf),
    'T4 the full-space round map is trace-preserving over 40 rounds')

# ---- 14b. T4 physics claims and the device policy ---------------------------
_srs_idx = {(r['code'], r['channel'], r['p'], r['recovery'], r['eta']): r
            for r in _srs}
_sus = {(s['code'], s['channel'], s['p'], s['recovery'], s['eta']): s
        for s in _st['summary']}


def _adv(code, ch, p, kind, eta, R):
    """advantage over the decoder at round index R, from the stored curves."""
    a = _srs_idx[(code, ch, p, kind, eta)]['F'][R]
    d = _srs_idx[(code, ch, p, 'decoder', eta)]['F'][R]
    return a - d


_RL = _st['config']['rounds'][-1]
# (1) the advantage COMPOUNDS where there is one to compound.
for _ch in ('amplitude_damping', 'coherent'):
    for _p in (0.05, 0.10):
        _s = _sus[('5,1,3', _ch, _p, 'warm', 0.0)]
        chk(_s['survival_ratio'] is not None and _s['survival_ratio'] > 5.0,
            'T4 %s p=%.2f: the warm advantage grows %.1fx from R=1 to R=%d'
            % (_ch, _p, _s['survival_ratio'], _RL))
        chk(_s['R_half_gain'] is None or _s['R_half_gain'] >= 1.0,
            'T4 %s p=%.2f: memory time R_1/2 does not shrink (%s)'
            % (_ch, _p, _s['R_half_gain']))
# (2) where the decoder is already optimal the two must be IDENTICAL at every R,
#     not merely close -- otherwise the multi-round map would be inventing an
#     advantage the single-round certification says does not exist.
for _ch in ('depolarizing', 'mixed'):
    for _p in (0.05, 0.10):
        _s = _sus[('5,1,3', _ch, _p, 'warm', 0.0)]
        chk(_s['advantage_degenerate'] and abs(_s['advantage_R1']) < 1e-15,
            'T4 %s p=%.2f: warm == decoder at R=1 to %.1e (decoder is optimal)'
            % (_ch, _p, _s['advantage_R1']))
        chk(abs(_s['advantage_R%d' % _RL]) < 1e-12,
            'T4 %s p=%.2f: warm == decoder at R=%d to %.1e'
            % (_ch, _p, _RL, _s['advantage_R%d' % _RL]))
        chk(_s['survival_ratio'] is None,
            'T4 %s p=%.2f: survival reported undefined, not as a 0/0 ratio'
            % (_ch, _p))
# (3) readout error monotonically erodes the compounding, and at a few percent
#     abolishes it -- the honest limit on the claim.
for _ch in ('amplitude_damping', 'coherent'):
    _p = 0.10
    _s0 = _sus[('5,1,3', _ch, _p, 'warm', 0.0)]['survival_ratio']
    _s1 = _sus[('5,1,3', _ch, _p, 'warm', 0.005)]['survival_ratio']
    _s2 = _sus[('5,1,3', _ch, _p, 'warm', 0.02)]['survival_ratio']
    chk(_s0 > _s1 > _s2,
        'T4 %s p=%.2f: survival falls monotonically with readout error '
        '(%.1f -> %.1f -> %.1f)' % (_ch, _p, _s0, _s1, _s2))
    chk(_s2 < 2.0,
        'T4 %s p=%.2f: at eta=0.02 the compounding is gone (survival %.2f)'
        % (_ch, _p, _s2))
# (4) the device policy must match the measurement it is justified by.
_srb = _st.get('device_benchmark') or {}
_pts = {q['code']: q for q in (_srb.get('points') or [])}
if _pts:
    chk('cuda_speedup' in _pts.get('9,1,3', {})
        and _pts['9,1,3']['cuda_speedup'] > 5.0,
        'T4 at [[9,1,3]] CUDA is %.2fx faster than a pinned core'
        % _pts.get('9,1,3', {}).get('cuda_speedup', float('nan')))
    chk('cuda_speedup' in _pts.get('5,1,3', {})
        and _pts['5,1,3']['cuda_speedup'] < 1.0,
        'T4 at [[5,1,3]] CUDA is %.2fx SLOWER, which is why the policy is per '
        'code size rather than global'
        % _pts.get('5,1,3', {}).get('cuda_speedup', float('nan')))
    chk(max(q['max_abs_curve_diff'] for q in _pts.values()) < 1e-12,
        'T4 CPU and CUDA curves agree to %.2e, so the device choice is about '
        'throughput and not numerics'
        % max(q['max_abs_curve_diff'] for q in _pts.values()))
# (5) prose locks, every mantissa re-derived from the artifact.
_tfn14 = ' '.join(t.split())
_w10 = _sus[('5,1,3', 'amplitude_damping', 0.10, 'warm', 0.0)]
_c10 = _sus[('5,1,3', 'coherent', 0.10, 'warm', 0.0)]
_wd = max(r['max_abs_full_vs_reduced'] for r in _srf
          if r['recovery'] == 'decoder')
_ww = max(r['max_abs_full_vs_reduced'] for r in _srf
          if r['recovery'] != 'decoder')
_cb_rec = _srs_idx[('5,1,3', 'amplitude_damping', 0.10, 'warm', 0.02)]
# The prose quotes the amplitude-damping p=0.10 case specifically, so derive that
# record rather than a maximum over all of them: a max would silently drift the
# moment any other channel's block norm grew, and the lock would then be checking
# a number the paper does not print.
_cb = _cb_rec['cross_block_norm']
chk(_cb > 1e-5,
    'T4 the learned non-Pauli recovery really has non-vanishing cross-branch '
    'blocks (%.2e), so it is genuinely unprotected by the readout law' % _cb)
chk(max(r.get('cross_block_norm', 0.0) for r in _srs
        if r['recovery'] == 'decoder') < 1e-13,
    'T4 the Pauli decoder cross-branch blocks vanish identically -- the premise '
    'of the exact suppression law F(eta,R)=(1-eta)^((n-k)R) F(0,R)')
for _q, _why in (
        (_tex_sci(_w10['advantage_R1'], 2), 'the R=1 amplitude-damping lead'),
        (_tex_sci(_w10['advantage_R%d' % _RL], 2),
         'the R=%d amplitude-damping lead' % _RL),
        (_tex_sci(_c10['advantage_R1'], 2), 'the R=1 coherent lead'),
        (_tex_sci(_c10['advantage_R%d' % _RL], 2),
         'the R=%d coherent lead' % _RL),
        (_tex_sci(_wd, 1), 'the decoder full-space/reduced agreement'),
        (_tex_sci(_ww, 1), 'the learned-table full-space/reduced agreement'),
        (_tex_sci(_wl, 1), 'the saturated leakage fraction'),
        (_tex_sci(_cb, 1), "the learned recovery's cross-branch block size"),
        ('%.1f' % _w10['survival_ratio'], 'the amplitude-damping survival factor'),
        ('%.1f' % _c10['survival_ratio'], 'the coherent survival factor'),
        ('%.1f' % _c10['R_half_gain'], 'the coherent memory-time gain'),
        ('%d' % _c10['R_half_dec'], 'the decoder coherent memory time in rounds'),
        ('%d' % _c10['R_half_alt'], 'the warm coherent memory time in rounds'),
        (_tex_sci(_pts['9,1,3']['max_abs_curve_diff'], 1) if _pts else None,
         'the CPU/CUDA curve agreement at [[9,1,3]]')):
    if _q is None:
        continue
    chk(_q in _tfn14, 'main.tex quotes %s as %s' % (_why, _q))
# (6) the ED tables exist and the main text points at them.
chk('ED Table 13:' in e, 'ED Table 13 present (multi-seed statistics)')
chk('ED Table 14:' in e, 'ED Table 14 present (multi-round storage)')
for _sub in ('13b', '13c', '14a', '14b', '14c', '14d', '14e'):
    chk('ED Table %s:' % _sub in e, 'ED Table %s present' % _sub)
for _n in (13, 14):
    chk(t.count('ED Table~%d' % _n) >= 1,
        'main.tex points the reader at ED Table~%d' % _n)


# ---- 14c. T4: recovery-gate noise (the realistic-recovery axis) -------------
# Every lock above assumes the recoveries R_s are EXACT unitaries, which no
# hardware delivers.  This section locks the threshold at which the multi-round
# advantage stops surviving a noisy recovery layer.  The sweep is affordable only
# because it runs on a reduced 5x5 affine map (4 logical components + 1 scalar for
# out-of-code escaped weight) rather than the dim x dim evolution, so that
# reduction's own approximation -- q_escape, the weight a learned recovery fails to
# bring back into the code space -- is locked against the exact full space in the
# same breath.  Otherwise these would be thresholds for a model nobody checked.
_rn = _st.get('recovery_noise') or []
chk(len(_rn) == 16,
    'T4 recovery-gate-noise scan covers all %d (channel, p, learned recovery) '
    'rows at n=5' % len(_rn))
_rn_idx = {(r['channel'], r['p'], r['recovery']): r for r in _rn}
_live = [r for r in _rn if r['lam_star'] is not None]
_dead = [r for r in _rn if r['lam_star'] is None]
chk(len(_live) == 8 and len(_dead) == 8,
    'T4 %d rows carry a located lam* (amplitude damping + coherent) and %d report '
    'None with a reason (depolarizing + mixed, where the decoder is already the '
    'family optimum)' % (len(_live), len(_dead)))
chk(all(r['threshold_status'] == 'advantage_degenerate_at_lam0'
        and r['lam_half'] is None and abs(r['advantage_lam0']) < 1e-15
        for r in _dead),
    'T4 every threshold-less row is threshold-less because the advantage is 0 at '
    'lam=0, not because a bisection failed to converge')
chk(all(r['full_space_check']['max_abs_affine_vs_full']
        < (1e-11 if r['recovery'] == 'decoder'
           else max(1e-8, 10 * r['full_space_check']['q_escape_at_lam']))
        for r in _rn),
    'T4 the 5x5 affine noisy-recovery map equals the exact full-space evolution on '
    'every row (worst %.2e), each at its own MEASURED q_escape scale rather than '
    'a shared constant'
    % max(r['full_space_check']['max_abs_affine_vs_full'] for r in _rn))
chk(max(r['q_escape_max'] for r in _rn) < 1e-4,
    'T4 q_escape, the reduced model\'s only approximation, peaks at %.2e over the '
    'whole lam scan' % max(r['q_escape_max'] for r in _rn))
chk(all(r['full_space_check']['trace_dev_max'] < 1e-9 for r in _rn),
    'T4 the noisy full-space evolution stays trace-preserving over %d rounds at '
    'every lam' % max(len(r['full_space_check']['rounds']) for r in _rn))
chk(all(all(seq[i] >= seq[i + 1] - 1e-12 for i in range(len(seq) - 1))
        for r in _rn for key in ('F_alt', 'F_dec')
        for seq in [[g[key] for g in r['lam_grid']]]),
    'T4 F(R=%d) falls monotonically in lam for the learned recovery AND the '
    'decoder: more recovery noise can never help' % _RL)
# The headline thresholds.  The manuscript quotes amplitude damping and coherent at
# p=0.10 with the warm recovery, so those two rows are locked SPECIFICALLY rather
# than through a maximum over all rows: a max would drift the moment another
# channel's table changed and would then be checking a number the paper never
# prints.  5e-4 in lam is far tighter than the 1e-7..2e-7 affine-vs-full residual
# these thresholds inherit, so the lock is on the physics, not on the last
# bisection digit.
_w_ad = _rn_idx[('amplitude_damping', 0.10, 'warm')]
_w_co = _rn_idx[('coherent', 0.10, 'warm')]
chk(abs(_w_ad['lam_star'] - 0.2497) < 5e-4,
    'T4 amplitude damping p=0.10: the warm advantage over the decoder changes sign '
    'at lam* = %.6f' % _w_ad['lam_star'])
chk(abs(_w_co['lam_star'] - 0.1746) < 5e-4,
    'T4 coherent p=0.10: lam* = %.6f, a factor %.2f tighter than amplitude damping'
    % (_w_co['lam_star'], _w_ad['lam_star'] / _w_co['lam_star']))
chk(abs(_w_ad['eps_per_qubit_star'] - _w_ad['lam_star'] / 5) < 1e-15
    and abs(_w_ad['eps_per_qubit_star'] - 0.0499) < 1e-3
    and abs(_w_co['eps_per_qubit_star'] - 0.0349) < 1e-3,
    'T4 eps*/n equals lam*/n as advertised: %.4f (amplitude damping) and %.4f '
    '(coherent).  This is the DEPTH-1 reading; a depth-d recovery tolerates '
    'roughly this divided by d'
    % (_w_ad['eps_per_qubit_star'], _w_co['eps_per_qubit_star']))
# The magnitude budget is the honest limit on the claim: the advantage is halved
# long before it changes sign, so quoting lam* alone would oversell it.
chk(all(r['lam_half'] * 5.0 < r['lam_star'] for r in _live),
    'T4 lam_half < lam*/5 on all %d non-degenerate rows (min ratio %.1f): the '
    'MAGNITUDE of the advantage erodes long before its sign flips'
    % (len(_live), min(r['lam_star'] / r['lam_half'] for r in _live)))
# Universality of the erosion: the FRACTION of advantage lost depends only on
# lam*R, not on which channel or which learned table produced it.
_prod = [r['lam_half_times_R'] for r in _live
         if r['lam_half_times_R'] is not None]
chk(bool(_prod) and max(abs(x - math.log(2.0)) for x in _prod)
    < 0.05 * math.log(2.0),
    'T4 lam_half*R = %.4f..%.4f against ln2 = %.6f: the erosion RATE is set by the '
    'round count, not by the channel' % (min(_prod), max(_prod), math.log(2.0)))
chk(all(r['exp_law_max_dev'] < 1.5e-2 for r in _live),
    'T4 advantage(lam)/advantage(0) tracks exp(-lam*R) to %.1e for lam<=%.2f'
    % (max(r['exp_law_max_dev'] for r in _live), _live[0]['exp_law_lam_max']))
_lam_u = [i for i, g in enumerate(_live[0]['lam_grid'])
          if g['lam'] <= _live[0]['exp_law_lam_max']]
_ref = [_live[0]['lam_grid'][i]['advantage'] / _live[0]['advantage_lam0']
        for i in _lam_u]
_uni = max(abs(r['lam_grid'][i]['advantage'] / r['advantage_lam0'] - _ref[j])
           for r in _live for j, i in enumerate(_lam_u))
chk(_uni < 5e-3,
    'T4 that normalised curve is shared by all %d non-degenerate rows to %.1e over '
    'lam<=%.2f: the erosion is channel-blind, while lam* (which spans %.3f..%.3f) '
    'is not' % (len(_live), _uni, _live[0]['exp_law_lam_max'],
                min(r['lam_star'] for r in _live),
                max(r['lam_star'] for r in _live)))
# (7) main.tex and extended_data.tex quote the thresholds they are entitled to.
# Same re-derivation discipline as 13c and 14a/14b: every mantissa below is rendered
# out of the artifact with the format string the prose uses, so a regenerated scan
# that moves a digit fails HERE instead of leaving a stale number inside a published
# claim.  Note the two erosion quantities are locked separately because they are
# different quantities and the prose must not conflate them: `exp_law_max_dev` is
# one row's deviation from exp(-lam*R), `_uni` is the spread of the normalised curve
# ACROSS rows.  Quoting the second where the first belongs would understate the
# approximation by a factor of four.
for _q, _why in (
        ('%.4f' % _w_ad['lam_star'], 'the amplitude-damping sign-change lam*'),
        ('%.4f' % _w_co['lam_star'], 'the coherent sign-change lam*'),
        ('%.4f' % _w_ad['eps_per_qubit_star'],
         'the amplitude-damping depth-1 eps*/n'),
        ('%.4f' % _w_co['eps_per_qubit_star'], 'the coherent depth-1 eps*/n'),
        ('%.5f' % _w_ad['lam_half'],
         'the amplitude-damping half-magnitude lam'),
        ('%.5f' % _w_co['lam_half'], 'the coherent half-magnitude lam'),
        ('%.1f' % (_w_ad['lam_star'] / _w_ad['lam_half']),
         'the amplitude-damping lam*/lam_half ratio'),
        ('%.1f' % (_w_co['lam_star'] / _w_co['lam_half']),
         'the coherent lam*/lam_half ratio'),
        ('%.1f' % min(r['lam_star'] / r['lam_half'] for r in _live),
         'the tightest lam*/lam_half ratio over all non-degenerate rows'),
        (_tex_sci(max(r['exp_law_max_dev'] for r in _live), 1),
         'the worst single-row deviation of adv(lam)/adv(0) from exp(-lam*R)'),
        (_tex_sci(_uni, 1), 'the cross-row spread of the normalised curve'),
        ('%.4f' % min(_prod), 'the low end of lam_half*R'),
        ('%.4f' % max(_prod), 'the high end of lam_half*R'),
        ('%.6f' % math.log(2.0), 'ln 2'),
        ('%.3f' % min(r['lam_star'] for r in _live),
         'the low end of the lam* span'),
        ('%.3f' % max(r['lam_star'] for r in _live),
         'the high end of the lam* span'),
        (_tex_sci(_w_ad['q_escape_max'], 1),
         'the amplitude-damping escaped fraction'),
        (_tex_sci(_w_co['q_escape_max'], 1), 'the coherent escaped fraction'),
        (_tex_sci(max(r['full_space_check']['max_abs_affine_vs_full']
                      for r in _rn), 1),
         'the worst affine-versus-full-space disagreement')):
    chk(_q in _tfn14, 'main.tex quotes %s as %s' % (_why, _q))
# The two headline rows must be quoted to five decimals as the SAME number, because
# that near-coincidence is what the "lambda_half is channel-blind" sentence rests
# on; if a regenerated scan split them, the prose would be claiming a universality
# the artifact no longer shows.
chk(abs(_w_ad['lam_half'] - _w_co['lam_half']) < 5e-5,
    'T4 the two headline lam_half agree to 5 decimals (%.5f vs %.5f), which is what '
    'licences quoting one number for both channels'
    % (_w_ad['lam_half'], _w_co['lam_half']))
chk('ED Table~14f' in t, 'main.tex points the reader at ED Table 14f')
chk('ED Table 14f:' in e, 'ED Table 14f present (recovery-gate noise)')
# Every one of the 16 scan rows must appear in ED Table 14f, degenerate ones
# included: a table that silently dropped the eight rows with no threshold would
# show only the configurations where the method looks good.  The row prefix
# `channel & $p$ & recovery &` is NOT unique to 14f -- Tables 14a, 14b and 14d
# render the same prefix -- so searching the whole ED file passes vacuously even
# with a 14f row deleted (verified by mutation).  The slice below is bounded to
# 14f's own longtable body, which is the only scope in which these checks mean
# anything.
_i14f = e.index('ED Table 14f:')
_ib14f = e.index(r'\begin{longtable}', _i14f)
_ie14f = e.index(r'\end{longtable}', _ib14f)
_tb14f = e[_ib14f:_ie14f]
for _r in _rn:
    _tag = '%s & $%.2f$ & %s &' % (_r['channel'].replace('_', ' '), _r['p'],
                                    _r['recovery'])
    chk(_tag in _tb14f, 'ED Table 14f renders the %s p=%.2f %s row'
        % (_r['channel'], _r['p'], _r['recovery']))
chk(_tb14f.count('degenerate') == len(_rn) - len(_live),
    'ED Table 14f labels exactly the %d threshold-less rows as degenerate rather '
    'than omitting them or bisecting round-off into a number'
    % (len(_rn) - len(_live)))
# The two headline thresholds must be readable off the table, not only off the
# prose: a reader checking the paper's central robustness claim looks at the table.
for _r, _nm in ((_w_ad, 'amplitude damping p=0.10 warm'),
                (_w_co, 'coherent p=0.10 warm')):
    _row = ('%s & $%.2f$ & %s & $%.6f$ & $%.4f$ & $%.6f$ & $%.4f$ &'
            % (_r['channel'].replace('_', ' '), _r['p'], _r['recovery'],
               _r['lam_star'], _r['eps_per_qubit_star'], _r['lam_half'],
               _r['lam_half_times_R']))
    chk(_row in _tb14f,
        'ED Table 14f renders the full %s threshold row' % _nm)



# ---- 15. hardware feasibility: the shot budget behind every quoted number ----
# Every other section of this audit re-derives a number that some script in the
# repository could regenerate.  The hardware numbers are the exception: they come
# from a device run that costs money and queue time to repeat, so
# hw_feasibility_numbers.json is the terminal artifact and nothing downstream can
# recompute it.  That makes it the section most in need of locks, and it is where
# a stale or mis-read mantissa would otherwise survive indefinitely -- as one did:
# main.tex said the injected-branch syndrome distribution "peaks at s=12", while
# the artifact has the s=8 relaxation satellite as the mode by a factor 1.75.
_hw = json.load(open('hw_feasibility_numbers.json'))
_hwc, _hws = _hw['circuits'], _hw['summary']
_REQ = _hw['shots_requested']
_tfn15 = ' '.join(t.split())
_efn15 = ' '.join(e.split())


def _grp(x):
    """Thousands grouping with a LaTeX thin-space -- the renderer make_ed.py uses."""
    return '{:,}'.format(int(x)).replace(',', BS + ',')


def _wilson95(k, n, z=1.96):
    """Wilson 95% interval, identical to hw_verify_analysis.py:data_hit_stats.

    Re-implemented rather than imported: the audit must not share code with the
    thing it checks, or a bug in the estimator would pass silently.  z=1.96 is the
    value the artifact was written with, and the two circuits that define an
    interval reproduce it to the last bit (locked below).
    """
    if not n:
        return float('nan'), float('nan')
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)


def _n_for_half(p, target, z=1.96):
    """Smallest sample whose Wilson interval at rate p is no wider than +/-target.

    Bisected on the true Wilson half-width, not the normal approximation, so the
    shot costs quoted in the paper are the costs of the interval the paper prints.
    """
    lo, hi = 1, 10 ** 9
    while lo < hi:
        mid = (lo + hi) // 2
        a, b = _wilson95(round(p * mid), mid, z)
        if (b - a) / 2.0 <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


# 15a. the artifact is internally consistent before any prose is trusted -------
chk(_REQ == 400 and ''.join(sorted(_hwc)) == 'ABCD',
    'HW artifact describes %d circuits at %d requested shots each'
    % (len(_hwc), _REQ))
_syn = {}
for _n in 'ABCD':
    _c, _b = _hwc[_n], _hwc[_n]['branch_stats']
    _syn[_n] = {int(k): float(v) for k, v in _c['syndrome_dist_hw'].items()}
    chk(_b['n_total'] == _c['shots_used'] <= _REQ,
        'HW circuit %s: %d usable shots of %d executed, and the post-selected '
        'total is the usable count' % (_n, _c['shots_used'], _REQ))
    chk(0 <= _b['n_hit'] <= _b['n_sel'] <= _b['n_total'],
        'HW circuit %s: counts nest, hit(%d) <= sel(%d) <= usable(%d)'
        % (_n, _b['n_hit'], _b['n_sel'], _b['n_total']))
    chk(abs(sum(_syn[_n].values()) - 1.0) < 1e-12 and len(_syn[_n]) == 16,
        'HW circuit %s: the read-syndrome distribution is normalised over all 16 '
        'syndromes' % _n)
    if _b['n_sel']:
        _lo, _hi = _wilson95(_b['n_hit'], _b['n_sel'])
        chk(abs(_b['F_s'] - _b['n_hit'] / _b['n_sel']) < 1e-12,
            'HW circuit %s: F_s = %d/%d = %.4f as reported'
            % (_n, _b['n_hit'], _b['n_sel'], _b['F_s']))
        chk(abs(_lo - _b['F_s_lo95']) < 1e-12 and abs(_hi - _b['F_s_hi95']) < 1e-12,
            'HW circuit %s: the Wilson interval is the z=1.96 one this audit '
            'recomputes, bit for bit' % _n)
        chk(_lo - 1e-15 <= _b['F_s'] <= _hi + 1e-15
            and -1e-15 <= _lo <= _hi <= 1.0 + 1e-15,
            'HW circuit %s: F_s inside its own clamped interval (at n_hit=0 the '
            'lower endpoint is 0 up to %.1e of round-off, which is not a violation)'
            % (_n, abs(_lo - min(_lo, _b['F_s']))))
    else:
        chk(math.isnan(_b['F_s']) and math.isnan(_b['F_s_lo95'])
            and math.isnan(_b['F_s_hi95']),
            'HW circuit %s: with sel=0 the conditional fidelity is recorded as '
            'undefined (nan), not as 0/0 dressed up as a number' % _n)


# 15b. the summary block agrees with the per-circuit records it summarises -----
_A, _B, _Cc, _D = (_hwc[x] for x in 'ABCD')
chk(_hws['mid_measure_hits_total'] == 0
    == _A['branch_stats']['n_hit'] + _B['branch_stats']['n_hit'],
    'HW summary: the two mid-measure circuits produced 0 data hits between them')
chk(_hws['mid_measure_shots_total'] == _A['shots_used'] + _B['shots_used'],
    'HW summary: the %d shots behind that zero are the usable counts of circuits '
    'A and B' % _hws['mid_measure_shots_total'])
chk(_hws['late_D_F_s'] == _D['branch_stats']['F_s']
    and _hws['late_D_F_s_wilson95']
    == [_D['branch_stats']['F_s_lo95'], _D['branch_stats']['F_s_hi95']],
    'HW summary: the headline F_12 and its interval are circuit D\'s')
_sat = _syn['D'][8] / _syn['D'][12]
chk(abs(_hws['late_D_relaxation_satellite_s8_over_s12'] - _sat) < 1e-12,
    'HW summary: the relaxation satellite ratio is P(s=8)/P(s=12) = %.4f' % _sat)
chk(abs(_hws['ideal_frame_fidelity_v1_model'] - _D['P_expected_key_ideal']) < 1e-12,
    'HW summary: the ideal reference is the exact statevector branch weight of '
    'the identical circuit (%.9f), not a simulation of a different one'
    % _hws['ideal_frame_fidelity_v1_model'])

# 15c. the mode claim, which is the defect this section exists for -------------
_rank = {n: 1 + sum(1 for v in _syn[n].values()
                    if v > _syn[n][_hwc[n]['s']] + 1e-12) for n in 'ABCD'}
_mode = {n: max(sorted(_syn[n]), key=lambda k: _syn[n][k]) for n in 'ABCD'}
_sd = {n: math.sqrt((1.0 / 16) * (15.0 / 16) / _hwc[n]['shots_used'])
       for n in 'ABCD'}
_zed = {n: (_syn[n][_hwc[n]['s']] - 1.0 / 16) / _sd[n] for n in 'ABCD'}
chk(_mode['D'] == 8 and _rank['D'] == 2,
    'HW circuit D: the a1-relaxation satellite s=8 is the MODE (%.3f) and the '
    'correctly read s=12 is second (%.3f)' % (_syn['D'][8], _syn['D'][12]))
chk(_syn['D'][8] > _syn['D'][12] and _sat > 1.0,
    'HW circuit D: the satellite out-occurs the correct syndrome by %.2fx, so no '
    'wording may call s=12 the peak' % _sat)
chk('peaks at $s=12$' not in t and 'hardware peak at $s=12$' not in t,
    'main.tex and the Fig.~4 caption no longer claim a hardware peak at the '
    'correct syndrome, which the artifact contradicts')
chk(r'\emph{second}-most-likely outcome' in _tfn15
    and 'satellite $s=8$ is the mode' in _tfn15,
    'main.tex and its figure caption state the ordering the artifact shows')
chk(_syn['A'][0] == 0.0 and _rank['A'] == 16,
    'HW circuit A: the correct all-zero syndrome is never returned (0 of %d '
    'shots), a %.1f-sigma deficit against code-blind'
    % (_A['shots_used'], abs(_zed['A'])))
chk(_zed['B'] > 4.0 and _B['branch_stats']['n_hit'] == 0,
    'HW circuit B: the syndrome excess survives mid-circuit measurement (%.1f '
    'sigma) while the data register does not, so the failure is localised'
    % _zed['B'])
chk(abs(_zed['C']) < 1.0 and _zed['D'] > 6.0,
    'HW: the identity branch shows no detectable syndrome excess (%.1f sigma) '
    'where the injected branch shows %.1f sigma' % (_zed['C'], _zed['D']))


# 15d. main.tex quotes the budget, and pays for every precision claim ----------
# Each mantissa is re-derived here with the renderer the prose uses, so a
# re-analysis of the raw cloud returns that moves a digit fails the audit instead
# of leaving stale prose behind.  The precision costs are re-derived through the
# SAME Wilson interval the artifact prints, at each circuit's own measured
# selection yield -- quoting one branch's yield for both would understate the
# identity-branch cost by a factor 2.2.
_tot_req = _REQ * len(_hwc)
_tot_use = sum(_hwc[n]['shots_used'] for n in 'ABCD')
_tot_sel = sum(_hwc[n]['branch_stats']['n_sel'] for n in 'ABCD')
_tot_hit = sum(_hwc[n]['branch_stats']['n_hit'] for n in 'ABCD')
_yD = _D['branch_stats']['n_sel'] / float(_REQ)
_yC = _Cc['branch_stats']['n_sel'] / float(_REQ)
_loD, _hiD = _wilson95(_D['branch_stats']['n_hit'], _D['branch_stats']['n_sel'])
_hwD = (_hiD - _loD) / 2.0
_exD = math.ceil(_n_for_half(_D['branch_stats']['F_s'], 0.01) / _yD)
_exC = math.ceil(_n_for_half(_Cc['branch_stats']['F_s'], 0.01) / _yC)
_bench = 2 * 2 * 2 * 16
_1kD = int(round(1000 * _yD))
_1kC = int(round(1000 * _yC))
_h1kD = (_wilson95(round(_D['branch_stats']['F_s'] * _1kD), _1kD)[1]
         - _wilson95(round(_D['branch_stats']['F_s'] * _1kD), _1kD)[0]) / 2.0
_h1kC = (_wilson95(round(_Cc['branch_stats']['F_s'] * _1kC), _1kC)[1]
         - _wilson95(round(_Cc['branch_stats']['F_s'] * _1kC), _1kC)[0]) / 2.0
for _q, _why in (
        ('$%s$ executed in total' % _grp(_tot_req), 'the executed shot count'),
        ('$%s$ shots came back parseable' % _grp(_tot_use), 'the usable count'),
        ('$%d$ of those survived post-selection' % _tot_sel,
         'the post-selected count'),
        ('$%d$ reproduced the decoded logical state' % _tot_hit,
         'the data-hit count'),
        ('end-to-end yield of $%.2f\\%%$' % (100.0 * _tot_hit / _tot_req),
         'the end-to-end yield'),
        ('$0$ hits in $%d$ shots' % _hws['mid_measure_shots_total'],
         'the shots behind the mid-measure zero'),
        ('$0$ of $%d$ shots, a $%.1f\\sigma$ deficit'
         % (_A['shots_used'], abs(_zed['A'])),
         'the never-returned all-zero syndrome on circuit A'),
        ('$%.1f\\sigma$ above code-blind' % _zed['B'],
         'the syndrome excess that survives mid-circuit measurement'),
        ('at $%.3f$ against $1/16=0.063$' % _syn['D'][12],
         'the correctly read syndrome probability on circuit D'),
        ('a factor $%.2f$, or $%.1f\\sigma$ at $%d$ shots'
         % (16.0 * _syn['D'][12], _zed['D'], _D['shots_used']),
         'the code-blind excess factor and its significance'),
        ('by a factor $%.2f$ ($%.3f$ against $%.3f$)'
         % (_sat, _syn['D'][8], _syn['D'][12]),
         'the relaxation satellite ratio and the two probabilities behind it'),
        ('correct in $%d/%d$ shots'
         % (_D['branch_stats']['n_hit'], _D['branch_stats']['n_sel']),
         'the headline 44/55 count'),
        ('$F_{12}=%.2f$' % _D['branch_stats']['F_s'], 'the headline fidelity'),
        ('Those $%d$ shots are the $%.1f\\%%$ of the $%d$ executed'
         % (_D['branch_stats']['n_sel'], 100.0 * _yD, _REQ),
         'the selection yield the headline rests on'),
        ('interval is $\\pm%.3f$ wide' % _hwD, 'the Wilson half-width'),
        ('would cost $%s$ executed shots, $%d\\times$ what was run'
         % (_grp(_exD), round(_exD / float(_REQ))),
         'the shot cost of +/-0.01 on the injected branch'),
        ('only $%.1f\\%%$ of shots' % (100.0 * _Cc['P_expected_key_hw']),
         'the identity-branch expected-bitstring rate'),
        ('$p(s{=}0)=%.3f$' % _syn['C'][0],
         'the identity-branch correct-syndrome probability'),
        ('a factor $%.2f$ over uniform and $%.1f\\sigma$'
         % (16.0 * _syn['C'][0], _zed['C']),
         'the identity-branch excess, or absence of one'),
        ('against $%.1f\\sigma$ on the injected branch' % _zed['D'],
         'the injected-branch significance quoted for contrast'),
        ('$%.2f$ for the injected branch, $%.2f$ for the identity branch'
         % (_D['tv_distance_hw_ideal'], _Cc['tv_distance_hw_ideal']),
         'the two total-variation distances to the ideal joint distribution'),
        ('$%d$ circuits and $%s$ executed shots' % (_bench, _grp(_bench * 1000)),
         'the size of the designed benchmark'),
        ('buys $\\pm%.3f$ on an injected branch and $\\pm%.3f$ on an '
         'identity-like one' % (_h1kD, _h1kC),
         'what the designed 10^3 shots per branch would buy'),
        ('needs for $\\pm0.01$ ($%s$ shots)' % _grp(_exC),
         'the identity-branch cost of +/-0.01'),
        ('($%d$--$%d$ usable' % (min(_hwc[n]['shots_used'] for n in 'ABCD'),
                                 max(_hwc[n]['shots_used'] for n in 'ABCD')),
         'the Methods usable-shot range'),
        ('$%s$ of $%s$ executed, $%d$ post-selected'
         % (_grp(_tot_use), _grp(_tot_req), _tot_sel),
         'the Methods budget summary'),
        ('$%d$ data hits, an end-to-end yield of $%.2f\\%%$'
         % (_tot_hit, 100.0 * _tot_hit / _tot_req),
         'the Methods end-to-end yield'),
        ('$F_{12}=%.2f$ ($%d/%d$ shots, Wilson $95\\%%$ interval)'
         % (_D['branch_stats']['F_s'], _D['branch_stats']['n_hit'],
            _D['branch_stats']['n_sel']),
         'the Fig.~4 panel-(b) caption')):
    chk(_q in _tfn15, 'main.tex quotes %s as %s' % (_why, _q))
chk(_tfn15.count('$[%.2f,%.2f]$' % (_loD, _hiD)) >= 2,
    'main.tex prints the Wilson interval [%.2f,%.2f] in both the abstract and the '
    'hardware section' % (_loD, _hiD))
chk('ED Table~3b' in t and 'ED Table~3c' in t,
    'main.tex points the reader at the budget (3b) and syndrome-shape (3c) tables')
chk('ED Table 3b:' in e and 'ED Table 3c:' in e,
    'ED Tables 3b and 3c present')


# 15e. ED Tables 3b/3c render the artifact, in make_ed.py's own format strings --
# Same discipline as 13c: the RENDERED table is compared against the artifact it
# was rendered from, because a regenerated artifact that is not followed by a
# re-run of make_ed.py leaves the ED file quoting numbers nothing else in the
# repository still says.  Row checks are scoped to each table's own body -- the
# circuit labels A-D also occur in Tables 3 and 4, so a whole-file search would
# pass vacuously with a 3b row deleted.
def _body(tag, env='longtable'):
    """The body of the first `env` after `tag`, plus its column spec."""
    i = e.index(tag)
    b = e.index(BS + 'begin{' + env + '}', i)
    j = b + len(BS + 'begin{' + env + '}')
    spec = e[e.index('{', j):e.index('}', j) + 1]
    return e[b:e.index(BS + 'end{' + env + '}', b)], spec[1:-1]


_tb3b, _spec3b = _body('ED Table 3b:')
_tb3c, _spec3c = _body('ED Table 3c:')
_tp3b, _ = _body('ED Table 3b:', 'tabular')
for _n in 'ABCD':
    _c, _b = _hwc[_n], _hwc[_n]['branch_stats']
    _row = ('%s & %s & %d & %d & %d & %d & %d & %.1f' + BS + '%% & %.1f' + BS
            + '%% & %.2f' + BS + '%% ' + BS * 2) % (
        _n, _c['kind'], _c['s'], _REQ, _c['shots_used'], _b['n_sel'],
        _b['n_hit'], 100.0 * _c['shots_used'] / _REQ,
        100.0 * _b['n_sel'] / _REQ, 100.0 * _b['n_hit'] / _REQ)
    chk(_row in _tb3b, 'ED Table 3b renders circuit %s against the artifact (%s)'
        % (_n, _row))
    _row = ('%s & %s & %d & %d & %.3f & %.3f & %d & %.2f$' + BS + 'times$ & '
            '%+.1f & %.3f & %.3f ' + BS * 2) % (
        _n, _c['kind'], _c['s'], _mode[_n], _syn[_n][_mode[_n]],
        _syn[_n][_c['s']], _rank[_n], 16.0 * _syn[_n][_c['s']], _zed[_n],
        0.5 * sum(abs(v - 1.0 / 16) for v in _syn[_n].values()),
        _c['tv_distance_hw_ideal'])
    chk(_row in _tb3c, 'ED Table 3c renders circuit %s against the artifact (%s)'
        % (_n, _row))
chk(('total & --- & --- & %s & %s & %d & %d & %.1f' + BS + '%% & %.1f' + BS
     + '%% & %.2f' + BS + '%% ' + BS * 2)
    % (_grp(_tot_req), _grp(_tot_use), _tot_sel, _tot_hit,
       100.0 * _tot_use / _tot_req, 100.0 * _tot_sel / _tot_req,
       100.0 * _tot_hit / _tot_req) in _tb3b,
    'ED Table 3b renders the total row against the artifact')
for _tgt in (0.05, 0.02, 0.01):
    _nD = _n_for_half(_D['branch_stats']['F_s'], _tgt)
    _nC = _n_for_half(_Cc['branch_stats']['F_s'], _tgt)
    _row = ('$' + BS + 'pm%.2f$ & %d & %s & %d & %s ' % (
        _tgt, _nD, _grp(math.ceil(_nD / _yD)), _nC,
        _grp(math.ceil(_nC / _yC)))) + BS * 2
    chk(_row in _tp3b,
        'ED Table 3b prices +/-%.2f at both measured yields (%s)' % (_tgt, _row))
for _tab, _spec, _nm, _nrow in ((_tb3b, _spec3b, '3b', 5),
                                (_tb3c, _spec3c, '3c', 4)):
    _rows = [r for r in _tab.split(BS * 2) if '&' in r and 'toprule' not in r]
    chk(len(_rows) == _nrow,
        'ED Table %s renders exactly its %d rows (found %d): a table that quietly '
        'dropped a circuit would show only the ones that look good'
        % (_nm, _nrow, len(_rows)))
    chk(all(r.count('&') == len(_spec) - 1 for r in _rows),
        'ED Table %s: every body row has the %d columns its preamble declares '
        '(a mismatch no absent TeX engine would catch)' % (_nm, len(_spec)))
for _q, _why in (
        ('$%s$ shots bought $%d$ of them, $%.2f\\%%$'
         % (_grp(_tot_req), _tot_hit, 100.0 * _tot_hit / _tot_req),
         'the end-to-end efficiency'),
        ('rests on $%d$ of the $%d$ shots executed'
         % (_D['branch_stats']['n_sel'], _REQ), 'what the headline rests on'),
        ('a Wilson half-width of $\\pm%.3f$' % _hwD, 'the headline half-width'),
        ('injected branch $%.4f$, identity branch $%.4f$, a factor $%.1f$ apart'
         % (_yD, _yC, _yD / _yC), 'the two measured selection yields'),
        ('$%d$ circuits and $%s$ executed shots in total'
         % (_bench, _grp(_bench * 1000)), 'the designed benchmark size'),
        ('buys $\\pm%.3f$ per injected branch and $\\pm%.3f$ per identity-like one'
         % (_h1kD, _h1kC), 'what the design buys'),
        ('needs for $\\pm0.01$ ($%s$ shots)' % _grp(_exC),
         'the identity-branch cost of +/-0.01'),
        ('out-occurred by a factor $%.2f$' % _sat, 'the satellite ratio'),
        ('still a $%.1f\\sigma$ excess' % _zed['D'], 'the injected-branch excess'),
        ('the correct syndrome shows $%.1f\\sigma$' % _zed['C'],
         'the identity-branch non-excess'),
        ('keeps a $%.1f\\sigma$ syndrome' % _zed['B'],
         'the mid-measure syndrome excess'),
        ('$0$ of $%d$ shots, a $%.1f\\sigma$ deficit'
         % (_A['shots_used'], abs(_zed['A'])), 'the circuit-A zero')):
    chk(_q in _efn15, 'ED Tables 3b/3c quote %s as %s' % (_why, _q))


# ---- 16. the effective hardware error budget --------------------------------
# hw_error_budget.json is derived, not measured: it re-expresses the counts of
# hw_feasibility_numbers.json as per-ancilla error rates plus an exact gate
# inventory rebuilt offline.  So the audit re-derives every rate a second time,
# from the counts, and requires the two derivations to agree bit for bit -- a
# derived artifact that nothing checks is how the "~160-gate circuit" in main.tex
# survived next to a 204-instruction circuit for as long as it did.
_eb = json.load(open('hw_error_budget.json'))
_ebc, _ebi, _ebh = _eb['circuits'], _eb['gate_inventory'], _eb['headline']
_asy = _eb['readout_asymmetry']
chk(_eb['meta']['qpu_time_spent'] == 0,
    'the error budget spent no QPU time: it is derived from counts already on file')
chk(_eb['meta']['inputs']['counts'] == 'hw_feasibility_numbers.json'
    and _eb['meta']['inputs']['angles'].endswith('vscr_angles_dep.npz'),
    'the error budget names its inputs: the feasibility counts and the v1 angle '
    'snapshot the run actually used (%s)' % _eb['meta']['inputs']['angles'])
_cal = _eb['vendor_calibration']
chk((_cal['available'] and len(_cal.get('block_qubits', {})) == 9)
    or (not _cal['available'] and _cal.get('error')),
    'the vendor-calibration record states either all 9 block qubits or why it is '
    'empty (%s)' % ('available' if _cal['available'] else _cal.get('error')))
chk('Unauthorized' in str(_cal.get('error', '')) or _cal['available'],
    'the paper may only say the calibration query was refused if the artifact '
    'records that refusal')
for _n in 'ABCD':
    _g = _ebi[_n]
    chk(_g['single_qubit_gates'] + _g['two_qubit_gates'] + _g['measurements']
        == _g['instructions'] == sum(_g['by_section'].values()),
        'HW gate inventory %s adds up three ways: %d instructions = %d 1Q + %d 2Q '
        '+ %d meas = sum of its sections'
        % (_n, _g['instructions'], _g['single_qubit_gates'],
           _g['two_qubit_gates'], _g['measurements']))
    chk(_g['by_section']['encoder'] == _g['by_section']['decoder'] == 36
        and _g['by_section']['extraction'] == 32
        and _g['by_section']['recovery'] == 90
        and _g['by_section']['measure_ancilla'] == 4
        and _g['by_section']['measure_data'] == 5,
        'HW circuit %s has the architecture the paper describes: 36-gate encoder '
        'and decoder, 32-gate extraction, 90-gate recovery, 4+5 measurements' % _n)
chk(_ebi['D']['instructions'] - _ebi['C']['instructions'] == 1
    and _ebi['B']['instructions'] - _ebi['A']['instructions'] == 1,
    'HW the injected circuits are one frame gate longer than the identity ones, '
    'so the two pairs are compared at equal cost')
chk('160' not in t.split('Feasibility demonstration')[1].split('Discussion')[0]
    and '$204$-instruction circuit' in _tfn15,
    'main.tex quotes the exact instruction count (204) and no longer the '
    'placeholder "~160-gate circuit"')


# 16b. every rate re-derived from the counts, independently of the script ------
def _bits(s):
    """Syndrome integer -> (a0, a1, a2, a3), the layout the run established."""
    return [(s >> 3) & 1, (s >> 2) & 1, (s >> 1) & 1, s & 1]


_e = {}
for _n in 'ABCD':
    _d = _syn[_n]
    _ref = _hwc[_n]['s']
    _marg = [sum(v for s, v in _d.items() if _bits(s)[i] != _bits(_ref)[i])
             for i in range(4)]
    _e[_n] = _marg
    _sc = _ebc[_n]['syndrome_channel']
    _joint, _prod = _d[_ref], math.prod(1.0 - x for x in _marg)
    _w = {}
    for _s, _v in _d.items():
        _k = sum(1 for i in range(4) if _bits(_s)[i] != _bits(_ref)[i])
        _w[_k] = _w.get(_k, 0.0) + _v
    _p = sum(_marg) / 4.0
    _tv = 0.5 * sum(abs(_w.get(_k, 0.0)
                       - math.comb(4, _k) * _p ** _k * (1 - _p) ** (4 - _k))
                    for _k in range(5))
    for _i, _bn in enumerate(('a0', 'a1', 'a2', 'a3')):
        chk(abs(_sc['per_bit_error'][_bn] - _marg[_i]) < 1e-15,
            'HW circuit %s: the effective %s error rate %.6f is the marginal of '
            'the measured syndrome distribution against a deterministic ideal'
            % (_n, _bn, _marg[_i]))
    chk(abs(_sc['joint_all_correct'] - _joint) < 1e-15
        and abs(_sc['product_of_marginals'] - _prod) < 1e-15
        and abs(_sc['independence_ratio'] - _joint / _prod) < 1e-12,
        'HW circuit %s: joint %.6f = product %.6f x ratio %.4f, all re-derived '
        'from the counts' % (_n, _joint, _prod, _joint / _prod))
    chk(abs(_sc['mean_hamming_weight'] - sum(_marg)) < 1e-15
        and abs(_sc['symmetric_rate_ml'] - _p) < 1e-15
        and abs(_sc['weight_tv_vs_symmetric_binomial'] - _tv) < 1e-15,
        'HW circuit %s: mean weight %.4f, symmetric rate %.4f and the %.4f TV '
        'distance to Bin(4,p) all re-derived' % (_n, sum(_marg), _p, _tv))
    chk(abs(_sc['joint_all_correct']
            - _hwc[_n]['branch_stats']['n_sel'] / _hwc[_n]['shots_used']) < 1e-15,
        'HW circuit %s: the all-four-correct rate is n_sel/n_total from the run'
        % _n)
    _b = _ebc[_n]['budget']
    _lm = 1.0 - _hwc[_n]['P_expected_key_ideal']
    _lt = 1.0 - _hwc[_n]['P_expected_key_hw']
    chk(abs(_b['end_to_end_expected_key'] - _hwc[_n]['P_expected_key_hw']) < 1e-15
        and abs(_b['method_share_of_loss'] - _lm / _lt) < 1e-12
        and abs(_b['method_share_of_loss'] + _b['device_share_of_loss']
                - 1.0) < 1e-12,
        'HW circuit %s: the method and device shares of the loss are %.2e and '
        '%.6f and sum to 1' % (_n, _lm / _lt, _b['device_share_of_loss']))
    if _hwc[_n]['branch_stats']['n_sel']:
        chk(_b['factorisation_residual'] < 1e-12,
            'HW circuit %s: P(expected key) = P(correct syndrome) x F_s to %.1e, '
            'an identity of the post-selection rather than a fit'
            % (_n, _b['factorisation_residual']))
    if _hwc[_n]['P_expected_key_hw'] > 0:
        chk(abs(_b['implied_mean_instruction_error']
                - (1.0 - _hwc[_n]['P_expected_key_hw']
                   ** (1.0 / _ebi[_n]['instructions']))) < 1e-15,
            'HW circuit %s: implied mean per-instruction error %.5f over %d '
            'instructions' % (_n, _b['implied_mean_instruction_error'],
                              _ebi[_n]['instructions']))
for _bn, _i in (('a0', 0), ('a1', 1)):
    _r = _asy['per_bit'][_bn]
    chk(abs(_r['zero_to_one_late'] - _e['C'][_i]) < 1e-15
        and abs(_r['one_to_zero_late'] - _e['D'][_i]) < 1e-15
        and abs(_r['zero_to_one_mid'] - _e['A'][_i]) < 1e-15
        and abs(_r['one_to_zero_mid'] - _e['B'][_i]) < 1e-15,
        'HW asymmetry %s: both directions are the marginals of the identity and '
        'injected circuits, in both measurement schemes' % _bn)
    chk(abs(_r['bias_late'] - (_e['D'][_i] - _e['C'][_i])) < 1e-15
        and abs(_r['bias_mid'] - (_e['B'][_i] - _e['A'][_i])) < 1e-15
        and abs(_r['bias_swing_on_deferral']
                - (_r['bias_late'] - _r['bias_mid'])) < 1e-15,
        'HW asymmetry %s: bias late %+.4f, mid %+.4f, swing %+.4f'
        % (_bn, _r['bias_late'], _r['bias_mid'], _r['bias_swing_on_deferral']))
chk(_asy['bits_scored'] == ['a0', 'a1']
    and _asy['relaxation_biased_bits_late'] == ['a1'],
    'HW a1 is the ONLY ancilla whose effective 1->0 rate exceeds its 0->1 rate '
    'under late measurement, which is what licenses the relaxation attribution')
chk(sum(1 for x in _e['D'] if x > 0.5) == 1 and _e['D'][1] > 0.5,
    'HW circuit D: exactly one ancilla (%s, %.3f) is worse than a coin flip'
    % (_ebc['D']['syndrome_channel']['worst_bit'], _e['D'][1]))
chk(abs(_ebh['worst_bit_error_D'] - max(_e['D'])) < 1e-15
    and _ebc['D']['syndrome_channel']['worst_bit'] == 'a1',
    'HW headline: the worst bit on circuit D is a1 at %.6f' % max(_e['D']))


# 16c. main.tex quotes the budget, including the limitation that makes it honest
_dD = _ebc['D']
for _q, _why in (
        ('over the $%d$-instruction circuit' % _ebi['D']['instructions'],
         'the exact instruction count of the branch circuit'),
        ('the four per-ancilla rates are $%.3f$, $%.3f$, $%.3f$ and $%.3f$'
         % tuple(_e['D']), 'the four effective per-ancilla error rates'),
        ('a mean of $%.2f$ wrong bits out' % _dD['syndrome_channel'][
            'mean_hamming_weight'], 'the mean Hamming weight per shot'),
        ('all-four-correct rate is $%.3f\\times$ the product'
         % _dD['syndrome_channel']['independence_ratio'],
         'the independence ratio of the syndrome bits'),
        ('$%.1f\\%%$ of the shots never read the syndrome correctly'
         % (100.0 * _dD['budget']['syndrome_channel_loss']),
         'the syndrome-channel share of the loss'),
        ('a further $%.1f\\%%$ of the logical information'
         % (100.0 * _dD['budget']['conditional_data_error']),
         'the conditional data-register error'),
        ('loses $%s$ end to end; that is $%s$ of the loss observed on hardware, '
         'one part in $%s$'
         % (_tex_sci(1.0 - _hwc['D']['P_expected_key_ideal'], 1),
            _tex_sci(_ebh['method_share_of_loss_D'], 1),
            _tex_sci(1.0 / _ebh['method_share_of_loss_D'], 2)),
         'the method\'s own end-to-end loss, its share of the observed loss, and '
         'the reciprocal of that share'),
        ("the circuit's $%d$ instructions ($%d$ single-qubit, $%d$ CNOT, $%d$ "
         'measurements)' % (_ebi['D']['instructions'],
                            _ebi['D']['single_qubit_gates'],
                            _ebi['D']['two_qubit_gates'],
                            _ebi['D']['measurements']),
         'the gate inventory of the circuit the headline comes from'),
        ('per-instruction error of $%.1f\\%%$ under an independent-error model'
         % (100.0 * _ebh['implied_mean_instruction_error_D']),
         'the implied mean per-instruction error'),
        ('whose effective $1\\to0$ rate exceeds its $0\\to1$ rate ($%.3f$ '
         'against $%.3f$)' % (_asy['per_bit']['a1']['one_to_zero_late'],
                              _asy['per_bit']['a1']['zero_to_one_late']),
         'the a1 asymmetry that identifies the relaxing ancilla'),
        ('lengthens its idle wait by the $%d$'
         % (_ebi['D']['by_section']['recovery']
            + _ebi['D']['by_section']['decoder']),
         'the idle wait the deferred readout adds'),
        ('flips that asymmetry from $%+.3f$ to $%+.3f$ while $a_0$\'s stays at '
         '$%+.3f$' % (_asy['per_bit']['a1']['bias_mid'],
                      _asy['per_bit']['a1']['bias_late'],
                      _asy['per_bit']['a0']['bias_late']),
         'the asymmetry swing that only a1 shows'),
        ('No vendor calibration accompanies the run',
         'the limitation that makes the budget effective rather than calibrated'),
        ('ED Table~3d', 'the pointer at the error-budget table')):
    chk(_q in _tfn15, 'main.tex quotes %s as %s' % (_why, _q))
chk('ED Tables~3d--3e' in _tfn15 and 'ED Table~3e' not in _tfn15.replace(
    'ED Tables~3d--3e', ''),
    'Methods points at both budget tables and at nothing else')
chk('ED Table 3d:' in e and 'ED Table 3e:' in e,
    'ED Tables 3d and 3e present')
chk(_cal['available'] or 'refused' in _tfn15,
    'main.tex says the calibration query was refused only because it was')


# 16d. ED Tables 3d/3e render the artifact, in make_ed.py's own format strings --
def _ed_sci(x, digits=2):
    """make_ed.py's `sci` renderer: ED cells carry their own $ and a thin space."""
    mnt, ex = ('%.*e' % (digits, abs(float(x)))).split('e')
    return '$' + mnt + BS + 'times 10^{' + str(int(ex)) + '}$'


_tb3d, _spec3d = _body('ED Table 3d:')
_tb3e, _spec3e = _body('ED Table 3e:')
_ta3e, _spece = _body('ED Table 3e:', 'tabular')
for _n in 'ABCD':
    _g, _c = _ebi[_n], _ebc[_n]
    _s = _c['syndrome_channel']
    _fs = ('%.2f' % _c['data_channel']['F_s']) if _c['data_channel']['defined'] \
        else 'undef.'
    _row = ('%s & %s & %d & %d & %d & %d & %.4f & %s & %.5f & %s '
            % (_n, _g['kind'], _g['instructions'], _g['single_qubit_gates'],
               _g['two_qubit_gates'], _g['measurements'],
               _s['joint_all_correct'], _fs,
               _c['budget']['end_to_end_expected_key'],
               _ed_sci(_c['budget']['method_share_of_loss']))) + BS * 2
    chk(_row in _tb3d, 'ED Table 3d renders circuit %s (%s)' % (_n, _row))
    _row = ('%s & %s & %.3f & %.3f & %.3f & %.3f & %.3f & %.4f & %.4f & %.3f & '
            '%.4f ' % (_n, ''.join(str(b) for b in _s['injected_bits']),
                       _e[_n][0], _e[_n][1], _e[_n][2], _e[_n][3],
                       _s['mean_hamming_weight'], _s['joint_all_correct'],
                       _s['product_of_marginals'], _s['independence_ratio'],
                       _s['weight_tv_vs_symmetric_binomial'])) + BS * 2
    chk(_row in _tb3e, 'ED Table 3e renders circuit %s (%s)' % (_n, _row))
for _bn in _asy['bits_scored']:
    _r = _asy['per_bit'][_bn]
    _row = ('$a_%s$ & %.3f & %.3f & %+.3f & %.3f & %.3f & %+.3f & %+.3f '
            % (_bn[1], _r['zero_to_one_mid'], _r['one_to_zero_mid'],
               _r['bias_mid'], _r['zero_to_one_late'], _r['one_to_zero_late'],
               _r['bias_late'], _r['bias_swing_on_deferral'])) + BS * 2
    chk(_row in _ta3e, 'ED Table 3e renders the %s asymmetry row (%s)'
        % (_bn, _row))
for _tab, _spec, _nm, _nrow in ((_tb3d, _spec3d, '3d', 4),
                                (_tb3e, _spec3e, '3e', 4),
                                (_ta3e, _spece, '3e asymmetry', 2)):
    _rows = [r for r in _tab.split(BS * 2) if '&' in r and 'toprule' not in r]
    chk(len(_rows) == _nrow,
        'ED Table %s renders exactly its %d rows (found %d)'
        % (_nm, _nrow, len(_rows)))
    chk(all(r.count('&') == len(_spec) - 1 for r in _rows),
        'ED Table %s: every body row has the %d columns its preamble declares'
        % (_nm, len(_spec)))
_secD = _ebi['D']['by_section']
for _q, _why in (
        ('share of the observed loss is %s'
         % _ed_sci(_ebh['method_share_of_loss_D'], 1),
         'the method share of the loss'),
        ('%d instructions for the injected late-measure'
         % _ebi['D']['instructions'], 'the instruction count'),
        ('split $%d$ single-qubit, $%d$ CNOT and $%d$ measurements'
         % (_ebi['D']['single_qubit_gates'], _ebi['D']['two_qubit_gates'],
            _ebi['D']['measurements']), 'the gate-type split'),
        ('by section ' + ', '.join('%s $%d$' % (k.replace('_', ' '), v)
                                   for k, v in _secD.items()),
         'the per-section gate inventory'),
        ('$%.1f\\%%$ of the shots are lost before the'
         % (100.0 * _dD['budget']['syndrome_channel_loss']),
         'the syndrome-channel loss'),
        ('further $%.1f\\%%$ of the logical information'
         % (100.0 * _dD['budget']['conditional_data_error']),
         'the conditional data-register loss'),
        ("method's own contribution to the same end-to-end quantity is %s"
         % _ed_sci(_ebh['method_loss_D'], 1), 'the method loss'),
        ('implied mean per-instruction error of $%.2f\\%%$'
         % (100.0 * _ebh['implied_mean_instruction_error_D']),
         'the implied per-instruction error'),
        ('($%.2f\\%%$ on the identity circuit)'
         % (100.0 * _ebh['implied_mean_instruction_error_C']),
         'the same on the identity circuit'),
        ('($%.3f$) but the only one worse than a coin flip'
         % _ebh['worst_bit_error_D'], 'the worst ancilla rate'),
        ('$%.3f\\times$ the independent prediction'
         % _dD['syndrome_channel']['independence_ratio'],
         'the independence ratio'),
        ('rate ($%.3f$ against $%.3f$)'
         % (_asy['per_bit']['a1']['one_to_zero_late'],
            _asy['per_bit']['a1']['zero_to_one_late']),
         'the a1 directional asymmetry'),
        ('sections, $%d$ instructions'
         % (_secD['recovery'] + _secD['decoder']), 'the added idle wait'),
        ('flips its asymmetry from $%+.3f$ to $%+.3f$'
         % (_asy['per_bit']['a1']['bias_mid'], _asy['per_bit']['a1']['bias_late']),
         'the a1 asymmetry flip'),
        ("$a_0$'s stays at $%+.3f$" % _asy['per_bit']['a0']['bias_late'],
         'the a0 contrast')):
    chk(_q in _efn15, 'ED Tables 3d/3e quote %s as %s' % (_why, _q))


print('=== OVERALL:', 'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT', '===')
raise SystemExit(0 if ok else 1)

