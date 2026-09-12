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
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10,
                         'axes.titlesize': 10, 'legend.fontsize': 8,
                         'savefig.dpi': 600, 'savefig.bbox': 'tight'})
    os.makedirs('paper/figures', exist_ok=True)
    C_HW, C_ID = '#1f4e9c', '#e41a1c'
    xs = np.arange(16)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))

    # (a) circuit D syndrome distribution HW vs ideal
    ax = axes[0][0]
    d = out['circuits']['D']
    hw = np.array([d['syndrome_dist_hw'][str(s)] for s in range(16)])
    idl = np.zeros(16); idl[12] = 1.0
    ax.bar(xs - 0.2, idl, 0.4, color=C_ID, alpha=0.75, label='ideal')
    ax.bar(xs + 0.2, hw, 0.4, color=C_HW, alpha=0.85, label='WK_C180')
    ax.annotate('injected $X_2$: $s$=12', (12, hw[12] + 0.03), ha='center',
                fontsize=7)
    ax.annotate('$a_1$ relaxation: $s$=8', (8, hw[8] + 0.03), ha='center',
                fontsize=7)
    ax.set_xlabel('syndrome $s$'); ax.set_ylabel('probability $P(s)$')
    ax.set_title('(a) late-measure, frame $IIXII$: syndrome distribution')
    ax.legend(loc='upper right'); ax.set_ylim(0, 1.15)
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
    ax.bar([0, 1], vals, 0.5, yerr=err, capsize=4, color=C_HW, alpha=0.85,
           label=f"WK_C180 ({st['n_hit']}/{st['n_sel']} shots)")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['data = 00000\n(correct recovery)',
                        'other\n(logical error)'])
    ax.set_ylabel('probability given $s=12$')
    ax.set_title('(b) post-selected branch fidelity $F_{12}$')
    ax.set_ylim(0, 1.18); ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y', lw=0.5)

    # (c) circuit C syndrome distribution (identity branch)
    ax = axes[1][0]
    c = out['circuits']['C']
    hw = np.array([c['syndrome_dist_hw'][str(s)] for s in range(16)])
    idl = np.zeros(16); idl[0] = 1.0
    ax.bar(xs - 0.2, idl, 0.4, color=C_ID, alpha=0.75, label='ideal')
    ax.bar(xs + 0.2, hw, 0.4, color=C_HW, alpha=0.85, label='WK_C180')
    ax.set_xlabel('syndrome $s$'); ax.set_ylabel('probability $P(s)$')
    ax.set_title('(c) late-measure, frame $IIIII$: identity branch')
    ax.legend(loc='upper right'); ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.3, axis='y', lw=0.5)

    # (d) mid vs late summary
    ax = axes[1][1]
    names = ['A mid $s$=0', 'B mid $s$=12', 'C late $s$=0', 'D late $s$=12']
    pk = [out['circuits'][n]['P_expected_key_hw'] for n in 'ABCD']
    pi = [out['circuits'][n]['P_expected_key_ideal'] for n in 'ABCD']
    x4 = np.arange(4)
    ax.bar(x4 - 0.2, pi, 0.4, color=C_ID, alpha=0.75, label='ideal')
    ax.bar(x4 + 0.2, pk, 0.4, color=C_HW, alpha=0.85, label='WK_C180')
    for i, v in enumerate(pk):
        ax.text(i + 0.2, v + 0.02, f'{v:.3f}', ha='center', fontsize=7)
    ax.set_xticks(x4); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel('probability of expected bitstring')
    ax.set_title('(d) mid-circuit vs late measurement')
    ax.set_ylim(0, 1.18); ax.legend(loc='upper right')
    ax.grid(alpha=0.3, axis='y', lw=0.5)

    for tag in ('pdf', 'png'):
        fig.savefig(f'paper/figures/fig_hw_feasibility.{tag}')
    plt.close(fig)
    print('saved paper/figures/fig_hw_feasibility.pdf|png')


if __name__ == '__main__':
    main()
