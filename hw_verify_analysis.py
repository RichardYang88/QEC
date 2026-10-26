"""
hw_verify_analysis.py — Paper analysis of the WK_C180 hardware feasibility
run (raw_verify_options.json, 4 circuits x 400 shots, 2026-09-10).

Circuits (qcloud_verify_options.py, v1 angles = vscr_angles_dep.npz):
   A  mid-measure,  frame IIIII, branch s=0
   B  mid-measure,  frame IIXII, branch s=12
   C  late-measure, frame IIIII, branch s=0
   D  late-measure, frame IIXII, branch s=12

Bitstring layout (empirically established, see QCLOUD_SETUP.md sec.6):
   raw_key = [a3 a2 a1 a0][d4 d3 d2 d1 d0]
   equivalently: reverse(raw_key)[5:9] = a0a1a2a3 (syndrome),
                 reverse(raw_key)[0:5] = d0..d4 (decoded data).

Outputs:
   hw_feasibility_numbers.json     — every number quoted in the manuscript
   paper/figures/fig_hw_feasibility.pdf|png
"""
import json, math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import qcloud_vscr_new as q
import ssvr_qec as m

RAW = os.path.join(ROOT, 'raw_verify_options.json')
OUT_JSON = os.path.join(ROOT, 'hw_feasibility_numbers.json')
ANC_SLICE, REV = slice(5, 9), True          # corrected layout

CASES = [('A', 'mid', 'IIIII', 0), ('B', 'mid', 'IIXII', 12),
         ('C', 'late', 'IIIII', 0), ('D', 'late', 'IIXII', 12)]


def raw_key_of_index(idx):
    """Map a 512-dim statevector index (q0=MSB ... q8=LSB; q0..q4 data,
    q5..q8 ancilla) to the raw cloud keystring [a3a2a1a0][d4d3d2d1d0]."""
    bit = lambda qubit: (idx >> (8 - qubit)) & 1
    anc = ''.join(str(bit(5 + i)) for i in (3, 2, 1, 0))   # a3 a2 a1 a0
    dat = ''.join(str(bit(j)) for j in (4, 3, 2, 1, 0))    # d4 d3 d2 d1 d0
    return anc + dat


def ideal_distribution(gates, s_idx):
    """Exact ideal joint distribution over raw keys for a branch circuit."""
    psi, P_s = q.branch_amplitudes(gates, s_idx)
    probs = np.abs(psi) ** 2
    tot = float(probs.sum())
    dist = {}
    for idx in np.nonzero(probs > 1e-14)[0]:
        dist[raw_key_of_index(int(idx))] = float(probs[idx]) / tot
    return dist, tot


def parse_key(key):
    """raw key -> (syndrome_bits a0a1a2a3, data_bits d0..d4)."""
    r = key[::-1]
    return r[5:9], r[0:5]


def syndrome_distribution(counts):
    tot = sum(counts.values()) or 1
    dist = {s: 0.0 for s in range(16)}
    for key, c in counts.items():
        sbits, _ = parse_key(key)
        dist[int(sbits, 2)] += c / tot
    return dist


def data_hit_stats(counts, s_idx):
    """Post-selected branch statistics with the corrected layout."""
    n_sel, n_hit = q.counts_to_branch_stats(counts, s_idx, ANC_SLICE, REV)
    tot = sum(counts.values())
    fs = n_hit / n_sel if n_sel else float('nan')
    if n_sel:                      # Wilson 95% interval
        z = 1.96
        den = 1 + z * z / n_sel
        ctr = (fs + z * z / (2 * n_sel)) / den
        half = z * math.sqrt(fs * (1 - fs) / n_sel +
                             z * z / (4 * n_sel * n_sel)) / den
        lo, hi = max(0.0, ctr - half), min(1.0, ctr + half)
    else:
        lo = hi = float('nan')
    return {'n_total': tot, 'n_sel': n_sel, 'n_hit': n_hit, 'F_s': fs,
            'F_s_lo95': lo, 'F_s_hi95': hi}


def tv_distance(counts, ideal):
    tot = sum(counts.values()) or 1
    keys = set(counts) | set(ideal)
    return 0.5 * sum(abs(counts.get(k, 0) / tot - ideal.get(k, 0.0))
                     for k in keys)



def main():
    q.setup()
    phi = q.load_angles('depolarizing')          # v1 hardware snapshot
    counts_list = json.load(open(RAW))
    assert len(counts_list) == 4

    out = {'layout': 'raw=[a3a2a1a0][d4d3d2d1d0]; rev[5:9]=syndrome',
           'chip': 'WK_C180', 'shots_requested': 400, 'circuits': {}}

    for (name, kind, frame, s), counts in zip(CASES, counts_list):
        gates = q.build_branch_circuit(q.ENC5, '0', frame, s, phi,
                                       late_measure=(kind == 'late'))
        ideal, P_s = ideal_distribution(gates, s)
        tot = sum(counts.values())
        exp_key = format(s, '04b')[::-1] + '00000'        # [a3..a0]+zeros
        stats = data_hit_stats(counts, s)
        synd_hw = syndrome_distribution(counts)
        top3 = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        rec = {
            'kind': kind, 'frame': frame, 's': s, 'shots_used': tot,
            'ideal_branch_weight_P_s': P_s,
            'expected_raw_key': exp_key,
            'P_expected_key_hw': counts.get(exp_key, 0) / tot,
            'P_expected_key_ideal': ideal.get(exp_key, 0.0),
            'top3_hw': top3,
            'ideal_top3': sorted(ideal.items(), key=lambda kv: -kv[1])[:3],
            'tv_distance_hw_ideal': tv_distance(counts, ideal),
            'branch_stats': stats,
            'syndrome_dist_hw': {str(k): v for k, v in synd_hw.items()},
        }
        out['circuits'][name] = rec
        print(f"\n=== circuit {name} ({kind}, frame={frame}, s={s}) ===")
        print(f"  shots={tot}  expected key '{exp_key}': "
              f"HW={rec['P_expected_key_hw']:.3f} "
              f"ideal={rec['P_expected_key_ideal']:.3f}")
        print(f"  TV distance (HW, ideal) = {rec['tv_distance_hw_ideal']:.3f}")
        stats_r = {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in stats.items()}
        print(f"  branch stats: {stats_r}")
        print(f"  HW top3: {top3}")
        print(f"  ideal top3: {rec['ideal_top3']}")
        nz = {k: round(v, 4) for k, v in synd_hw.items() if v > 1e-9}
        print(f"  syndrome dist (HW, nonzero): {nz}")

    d = out['circuits']['D']
    mid_shots = (out['circuits']['A']['shots_used'] +
                 out['circuits']['B']['shots_used'])
    out['summary'] = {
        'mid_measure_hits_total': (out['circuits']['A']['branch_stats']['n_hit'] +
                                   out['circuits']['B']['branch_stats']['n_hit']),
        'mid_measure_shots_total': mid_shots,
        'late_D_F_s': d['branch_stats']['F_s'],
        'late_D_F_s_wilson95': [d['branch_stats']['F_s_lo95'],
                                d['branch_stats']['F_s_hi95']],
        'late_D_relaxation_satellite_s8_over_s12':
            d['syndrome_dist_hw']['8'] / max(d['syndrome_dist_hw']['12'], 1e-12),
        'ideal_frame_fidelity_v1_model': float(
            q.simulator_fidelity_frame('0', 'IIXII', phi)),
    }
    print('\n=== summary ===')
    print(json.dumps(out['summary'], indent=2))

    with open(OUT_JSON, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nsaved {OUT_JSON}')
    make_figure(out)



def make_figure(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # One typographic scale for all four panels: 8 pt ticks, 9 pt axis labels,
    # 9.5 pt titles, 7.5 pt legends and in-plot annotations.  The previous mix
    # (7 pt annotations against 10 pt titles over 8 pt two-line tick labels, in
    # a 2x2 grid with default spacing) is what made the figure read as clutter:
    # the (a)/(b) titles touched, the (c)/(d) titles sat on the row above's tick
    # labels, and the two annotations in (a) lay on the bars they name and on
    # each other.  Every label now has its own space.
    plt.rcParams.update({'font.size': 8, 'axes.labelsize': 9,
                         'axes.titlesize': 9.5, 'legend.fontsize': 7.5,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'savefig.dpi': 600, 'savefig.bbox': 'tight'})
    os.makedirs('paper/figures', exist_ok=True)
    C_HW, C_ID = '#1f4e9c', '#e41a1c'
    xs = np.arange(16)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.0))
    fig.subplots_adjust(hspace=0.62, wspace=0.28)

    # (a) circuit D syndrome distribution HW vs ideal
    ax = axes[0][0]
    d = out['circuits']['D']
    hw = np.array([d['syndrome_dist_hw'][str(s)] for s in range(16)])
    idl = np.zeros(16); idl[12] = 1.0
    ax.bar(xs - 0.2, idl, 0.4, color=C_ID, alpha=0.75, label='ideal')
    ax.bar(xs + 0.2, hw, 0.4, color=C_HW, alpha=0.85, label='WK_C180')
    # Each annotation sits in empty space with a leader line to its own bar:
    # s=8 labelled from the left, s=12 from the right, so neither crosses the
    # unit ideal bar at s=12 nor the other's text.
    ax.annotate('$a_1$ relaxation: $s=8$', xy=(7.8, hw[8] + 0.02),
                xytext=(0.6, 0.60), fontsize=7.5, ha='left', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='0.3',
                                shrinkA=2, shrinkB=2))
    ax.annotate('injected $X_2$: $s=12$', xy=(12.2, hw[12] + 0.02),
                xytext=(15.6, 0.60), fontsize=7.5, ha='right', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.8, color='0.3',
                                shrinkA=2, shrinkB=2))
    ax.set_xlabel('syndrome $s$'); ax.set_ylabel('probability $P(s)$')
    ax.set_title('(a) circuit D, frame $IIXII$: syndrome distribution')
    # the band above 1.05 is empty in every panel, so a frameless legend centred
    # there labels the colours without covering a single bar
    ax.set_ylim(0, 1.28)
    ax.legend(loc='upper center', ncol=2, frameon=False)
    ax.grid(alpha=0.3, axis='y', lw=0.5)

    # (b) conditional data outcomes given s=12
    ax = axes[0][1]
    st = d['branch_stats']
    vals = [st['F_s'], 1 - st['F_s']]
    err = np.array([[st['F_s'] - st['F_s_lo95'],
                     (1 - st['F_s']) - (1 - st['F_s_hi95'])],
                    [st['F_s_hi95'] - st['F_s'],
                     (1 - st['F_s_lo95']) - (1 - st['F_s'])]]).T
    ax.bar([0, 1], [1.0, 0.0], 0.5, color=C_ID, alpha=0.35, label='ideal')
    ax.bar([0, 1], vals, 0.5, yerr=err, capsize=3, color=C_HW, alpha=0.85,
           error_kw=dict(ecolor='black', elinewidth=1.0, capthick=1.0),
           label=f"WK_C180 ({st['n_hit']}/{st['n_sel']} shots)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['data = 00000', 'data $\\neq$ 00000'])
    # The Wilson interval the whiskers draw, printed once above each bar with
    # the outcome it belongs to, instead of two-line tick labels that used to
    # collide with panel (d)'s title.
    # side margins so the centred interval labels never touch a spine
    ax.set_xlim(-0.8, 1.8)
    # the left label starts above the unit ideal bar so it sits on no colour
    ax.text(0, 1.02,
            '$%.3f\\;[%.3f,\\,%.3f]$\ncorrect recovery'
            % (st['F_s'], st['F_s_lo95'], st['F_s_hi95']),
            ha='center', va='bottom', fontsize=7.5, linespacing=1.5)
    ax.text(1, (1 - st['F_s_lo95']) + 0.05,
            '$%.3f\\;[%.3f,\\,%.3f]$\nlogical error'
            % (1 - st['F_s'], 1 - st['F_s_hi95'], 1 - st['F_s_lo95']),
            ha='center', va='bottom', fontsize=7.5, linespacing=1.5)
    ax.set_ylabel('probability given $s=12$')
    ax.set_title('(b) post-selected branch fidelity $F_{12}$')
    ax.set_ylim(0, 1.42); ax.legend(loc='upper center', ncol=2, frameon=False)
    ax.grid(alpha=0.3, axis='y', lw=0.5)

    # (c) circuit C syndrome distribution (identity branch)
    ax = axes[1][0]
    c = out['circuits']['C']
    hw = np.array([c['syndrome_dist_hw'][str(s)] for s in range(16)])
    idl = np.zeros(16); idl[0] = 1.0
    ax.bar(xs - 0.2, idl, 0.4, color=C_ID, alpha=0.75, label='ideal')
    ax.bar(xs + 0.2, hw, 0.4, color=C_HW, alpha=0.85, label='WK_C180')
    ax.set_xlabel('syndrome $s$'); ax.set_ylabel('probability $P(s)$')
    ax.set_title('(c) circuit C, frame $IIIII$: identity branch')
    ax.set_ylim(0, 1.28)
    ax.legend(loc='upper center', ncol=2, frameon=False)
    ax.grid(alpha=0.3, axis='y', lw=0.5)

    # (d) mid vs late summary
    ax = axes[1][1]
    names = ['A mid\n$s$=0', 'B mid\n$s$=12', 'C late\n$s$=0', 'D late\n$s$=12']
    pk = [out['circuits'][n]['P_expected_key_hw'] for n in 'ABCD']
    pi = [out['circuits'][n]['P_expected_key_ideal'] for n in 'ABCD']
    x4 = np.arange(4)
    ax.bar(x4 - 0.2, pi, 0.4, color=C_ID, alpha=0.75, label='ideal')
    ax.bar(x4 + 0.2, pk, 0.4, color=C_HW, alpha=0.85, label='WK_C180')
    for i, v in enumerate(pk):
        ax.text(i + 0.2, v + 0.03, f'{v:.3f}', ha='center', fontsize=7.5)
    ax.set_xticks(x4); ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel('probability of expected bitstring')
    ax.set_title('(d) mid-circuit vs late measurement')
    ax.set_ylim(0, 1.28); ax.legend(loc='upper center', ncol=2, frameon=False)
    ax.grid(alpha=0.3, axis='y', lw=0.5)

    for tag in ('pdf', 'png'):
        fig.savefig(f'paper/figures/fig_hw_feasibility.{tag}')
    plt.close(fig)
    print('saved paper/figures/fig_hw_feasibility.pdf|png')


if __name__ == '__main__':
    main()
