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
  9. all nine ED tables are present with balanced tabulars;
 10. the key literature is in the bibliography;
 11. every \\includegraphics target exists on disk.

Exit status is 0 iff all checks pass, so it composes with `&&` in CI.
"""
import json
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
          'ancilla_recovery.py',
          'paper/make_ed.py', 'paper/fill_numbers.py',
          'paper/main.tex', 'paper/extended_data.tex', 'paper/refs.bib',
          'paper/README.md',
          'scaling_results.json', 'stationarity_boundary.json',
          'ancilla_recovery.json'):
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
print('=== OVERALL:', 'ALL CHECKS PASSED' if ok else 'FAILURES PRESENT', '===')
raise SystemExit(0 if ok else 1)
