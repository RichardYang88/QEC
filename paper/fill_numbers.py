"""fill_numbers.py — replace @TOKEN@ placeholders in paper/main.tex with the
numbers produced by vscr_paper.py (paper_numbers.json) and
hw_verify_analysis.py (hw_feasibility_numbers.json)."""
import json, os, re

ROOT = os.path.dirname(os.path.abspath(__file__))
NUM = json.load(open(os.path.join(ROOT, '..', 'paper_numbers.json')))
HW = json.load(open(os.path.join(ROOT, '..', 'hw_feasibility_numbers.json')))
TEX = os.path.join(ROOT, 'main.tex')


def sci(x, digits=1):
    m, e = f'{x:.{digits}e}'.split('e')
    return f'{m}\\times 10^{{{int(e)}}}'


infos = NUM['infos']['depolarizing']
best = max(infos, key=lambda d: d['min_cf'])

repl = {
    '@SEEDVAL@': f"{best['val_fid']:.4f}",
    '@MINCF15WARM@': f"{NUM['min_cf_dep_p015']['warm']:.3f}",
    '@MINCF15COLD@': f"{NUM['min_cf_dep_p015']['cold_v1']:.3f}",
    '@MINCF15DEC@': f"{NUM['min_cf_dep_p015']['decoder']:.3f}",
    '@DEP010WARM@': f"{NUM['dep_p010']['VSCR warm (ours)']:.4f}",
    '@DEP010DEC@': f"{NUM['dep_p010']['Perfect-code decoder']:.4f}",
    '@DEP010RAW@': f"{NUM['dep_p010']['Raw']:.4f}",
    '@AD010WARM@': f"{NUM['ad_p010']['VSCR warm (ours)']:.4f}",
    '@AD010DEC@': f"{NUM['ad_p010']['Perfect-code decoder']:.4f}",
    '@MX010WARM@': f"{NUM['mx_p010']['VSCR warm (ours)']:.4f}",
    '@MX010DEC@': f"{NUM['mx_p010']['Perfect-code decoder']:.4f}",
    '@COH010WARM@': f"{NUM['coh_p010']['VSCR warm (ours)']:.4f}",
    '@COH010DEC@': f"{NUM['coh_p010']['Perfect-code decoder']:.4f}",
    '@WORSTVER@': sci(NUM['hardware_frame_numbers']['worst_circuit_vs_simulator']),
}

src = open(TEX).read()
missing = [t for t in repl if t not in src]
for t, v in repl.items():
    src = src.replace(t, v)
open(TEX, 'w').write(src)
print('replaced:', len(repl) - len(missing), 'missing:', missing)
left = re.findall(r'@[A-Z0-9_]+@', src)
print('remaining tokens:', set(left))
