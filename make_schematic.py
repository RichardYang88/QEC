"""
make_schematic.py — Manuscript Fig.1: VSCR method + [[5,1,3]] code +
hardware benchmark protocol schematic (vector PDF).

Panels:
  (a) VSCR instrument: encode -> noise -> projective syndrome extraction ->
      hypernetwork-conditioned variational recovery R_s -> decode/readout.
  (b) Recovery ansatz (3 layers: RzRxRz per qubit + Rzz ring; 60 params)
      and warm-start parameterisation phi_s = phi_dec(s) + phi_base + hnet(e_s).
  (c) Branch-fixed hardware protocol: one fixed circuit per syndrome branch,
      Pauli-frame averaging emulates the depolarizing channel exactly,
      late measurement + post-selection estimates F_s.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

os.makedirs('paper/figures', exist_ok=True)
INK = '#1a1a1a'
C_DATA = '#1f4e9c'
C_ANC = '#b3541e'
C_LEARN = '#2e7d32'
C_BOX = '#f2f4f8'


def box(ax, x, y, w, h, text, fc=C_BOX, ec=INK, fs=7.5, tc=INK, lw=1.0,
        style='round,pad=0.02,rounding_size=0.06'):
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
            fontsize=fs, color=tc, linespacing=1.25)


def arrow(ax, p0, p1, color=INK, style='-|>', lw=1.0, ls='-', rad=0.0,
          ms=7):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=ms,
                        color=color, lw=lw, linestyle=ls,
                        connectionstyle=f'arc3,rad={rad}',
                        shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def wires(ax, x0, x1, ys, color=INK, lw=0.9):
    for y in ys:
        ax.plot([x0, x1], [y, y], color=color, lw=lw, zorder=0)


fig = plt.figure(figsize=(7.2, 6.6))

# =====================================================================
# Panel (a): VSCR instrument
# =====================================================================
ax = fig.add_axes([0.03, 0.63, 0.94, 0.34])
ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis('off')
ax.text(0.02, 3.42, '(a) Variational syndrome-conditioned recovery (VSCR)',
        fontsize=8.5, fontweight='bold', va='top')

yd = [2.6, 2.3, 2.0, 1.7, 1.4]        # 5 data wires
ya = [1.0, 0.75, 0.5, 0.25]           # 4 ancilla wires
wires(ax, 0.35, 9.7, yd, color=C_DATA)
wires(ax, 3.15, 6.05, ya, color=C_ANC)

ax.text(0.12, 2.0, '$|\\psi\\rangle$', fontsize=10, color=C_DATA, va='center')
ax.text(0.12, 0.62, '$|0\\rangle^{\\otimes 4}$', fontsize=8, color=C_ANC,
        va='center')

box(ax, 0.55, 1.25, 1.15, 1.5, 'encoder\n$U_{\\rm enc}$\n[[5,1,3]]', fs=7)
box(ax, 2.0, 1.25, 1.0, 1.5, 'noise\n$\\mathcal{N}_p$\n(depol./AD/\nmixed)',
    fs=7, fc='#fdecea')
box(ax, 3.3, 0.05, 2.6, 2.9, '', fc='#ffffff00', ec='none')
ax.add_patch(Rectangle((3.3, 0.05), 2.6, 2.9, fc='#f7f2fa', ec=INK, lw=1.0,
                       zorder=0))
ax.text(4.6, 2.78, 'syndrome extraction', fontsize=7.5, ha='center',
        fontweight='bold')
ax.text(4.6, 2.52, '$g_1{=}XZZXI,\\ g_2{=}IXZZX,$\n$g_3{=}XIXZZ,\\ g_4{=}ZXIXZ$',
        fontsize=6.3, ha='center', va='top')
# CNOT ladder decoration inside extraction block (clean)
for i, yy in enumerate(ya):
    cx = 4.05 + 0.35 * i
    ctrl = yd[(i + 1) % 5]
    ax.plot([cx, cx], [yy, ctrl], color=C_ANC, lw=0.8)
    ax.add_patch(plt.Circle((cx, ctrl), 0.045, fc=C_ANC, ec='none'))
    ax.add_patch(plt.Circle((cx, yy), 0.075, fc='none', ec=C_ANC, lw=0.9))
    ax.plot([cx - 0.075, cx + 0.075], [yy, yy], color=C_ANC, lw=0.8)
    ax.plot([cx, cx], [yy - 0.075, yy + 0.075], color=C_ANC, lw=0.8)
ax.text(4.6, -0.12, 'projective measurement $\\rightarrow P_s$ (16 branches)',
        fontsize=6.8, ha='center', color=INK)

box(ax, 6.3, 0.05, 1.55, 1.15,
    'hypernetwork\n$e_s \\rightarrow \\Delta\\phi_s$\n(label-free)', fs=7,
    fc='#e8f5e9', ec=C_LEARN, tc=C_LEARN)
arrow(ax, (5.98, 0.62), (6.28, 0.62), color=C_ANC, lw=1.1)
ax.text(6.13, 0.78, '$s$', fontsize=8, color=C_ANC, ha='center')

box(ax, 6.3, 1.5, 1.55, 1.25,
    'recovery $R_s(\\phi_s)$\n$\\phi_s=\\phi^{\\rm dec}_s+\\phi_0+\\Delta\\phi_s$\n(60 rot. params)',
    fs=6.8, fc='#e8f0fd', ec=C_DATA)
arrow(ax, (7.07, 1.22), (7.07, 1.48), color=C_LEARN, lw=1.1)

box(ax, 8.25, 1.55, 1.05, 1.15, 'decode\n$U_{\\rm enc}^{\\dagger}$ +\nreadout', fs=7)
arrow(ax, (7.87, 2.1), (8.23, 2.1), color=C_DATA, lw=1.1)
ax.text(9.55, 2.1, '$F$', fontsize=10, va='center')


# =====================================================================
# Panel (b): recovery ansatz + warm start
# =====================================================================
ax = fig.add_axes([0.03, 0.315, 0.94, 0.30])
ax.set_xlim(0, 10); ax.set_ylim(0, 3.4); ax.axis('off')
ax.text(0.02, 3.25, '(b) Learned recovery ansatz $R_s$ and warm-start training',
        fontsize=8.5, fontweight='bold', va='top')

yb = [2.5, 2.1, 1.7, 1.3, 0.9]
wires(ax, 0.4, 9.6, yb, color=C_DATA)
layer_x = [(1.3, 2.1), (3.3, 4.1), (5.3, 6.1)]
for li, (xa, xb) in enumerate(layer_x):
    for y in yb:
        for name, xx in (('Rz', xa), ('Rx', (xa + xb) / 2), ('Rz', xb)):
            ax.add_patch(Rectangle((xx - 0.14, y - 0.11), 0.28, 0.22,
                                   fc='#e8f0fd', ec=C_DATA, lw=0.7))
            ax.text(xx, y, name, fontsize=5.4, ha='center', va='center',
                    color=C_DATA)
    # Rzz ring entanglers: adjacent pairs + closing pair (4,0)
    for qi in range(5):
        q2 = (qi + 1) % 5
        y1, y2 = yb[qi], yb[q2]
        cx = 6.75 + 0.22 * qi
        ax.plot([cx, cx], [min(y1, y2), max(y1, y2)], color=C_ANC, lw=0.8)
        for y in (y1, y2):
            ax.add_patch(plt.Circle((cx, y), 0.05, fc='none', ec=C_ANC, lw=0.9))
            ax.add_patch(plt.Circle((cx, y), 0.02, fc=C_ANC, ec='none'))
ax.text(1.7, 2.86, 'layer $\\ell=1$', fontsize=6.5, ha='center')
ax.text(3.7, 2.86, 'layer $\\ell=2$', fontsize=6.5, ha='center')
ax.text(5.7, 2.86, 'layer $\\ell=3$', fontsize=6.5, ha='center')
ax.text(7.2, 2.86, '$R_{zz}$ ring', fontsize=6.5, ha='center', color=C_ANC)
ax.text(8.35, 2.9, '3 layers $\\times$ 20 params\n$=60$ parameters $\\phi_s$',
        fontsize=6.8, ha='left', va='top')
ax.text(8.35, 1.6, 'warm start:\n$\\phi^{\\rm dec}_s$ = Pauli decoder\n$C_s$ (exact at epoch 0)',
        fontsize=6.8, ha='left', va='top', color=C_LEARN)

# training loop annotation
box(ax, 0.3, 0.02, 9.4, 0.62,
    'unsupervised loss:  $\\mathcal{L} = 1 - \\sum_s \\langle\\psi| R_s P_s\\, '
    '\\mathcal{N}_p[\\psi] P_s R_s^{\\dagger} |\\psi\\rangle$   '
    '(no error labels; Adam + curriculum; multi-seed; label-free selection)',
    fs=6.4, fc='#fffde7', ec='#b39700')

# =====================================================================
# Panel (c): hardware protocol
# =====================================================================
ax = fig.add_axes([0.03, 0.02, 0.94, 0.28])
ax.set_xlim(0, 10); ax.set_ylim(0, 3.1); ax.axis('off')
ax.text(0.02, 2.95, '(c) Branch-fixed hardware benchmark on a superconducting QPU',
        fontsize=8.5, fontweight='bold', va='top')

steps = [
    ('1. compile', 'one FIXED circuit per\nbranch $s$: $U_{\\rm enc}$, frame,\nextraction, $R_s(\\phi_s)$,\n$U_{\\rm enc}^\\dagger$ (no feedback)'),
    ('2. frame average', 'sample Pauli frames\n$I{:}1-p,\\ X/Y/Z{:}p/3$\n$\\Rightarrow$ exact depolarizing\nchannel on data'),
    ('3. late measure', 'all 9 qubits measured\nat circuit end\n(mid-circuit readout\ndysfunctional on chip)'),
    ('4. post-select', 'keep shots with\nancilla $=s$;\n$F_s=P(00000\\,|\\,s)$,\nWilson 95% intervals'),
]
for i, (t, d) in enumerate(steps):
    x0 = 0.35 + i * 2.42
    box(ax, x0, 0.85, 2.12, 1.75, '', fc=C_BOX, ec=INK)
    ax.text(x0 + 1.06, 2.38, t, fontsize=7.3, ha='center', fontweight='bold')
    ax.text(x0 + 1.06, 1.62, d, fontsize=6.3, ha='center', va='center')
    if i < 3:
        arrow(ax, (x0 + 2.16, 1.72), (x0 + 2.38, 1.72), lw=1.1)
ax.text(5.0, 0.42,
        '$F(p) = \\mathbb{E}_{\\rm frames}\\,\\sum_s P(s)\\,F_s$  —  hardware estimate vs. exact circuit-level ideal reference '
        '(verified $<10^{-9}$ vs. density-matrix simulator)',
        fontsize=6.8, ha='center')

for tag in ('pdf', 'png'):
    fig.savefig(f'paper/figures/fig_schematic.{tag}')
print('saved paper/figures/fig_schematic.pdf|png')


ax.text(1.12, 3.05, '5 data qubits', fontsize=6.5, color=C_DATA)
ax.text(3.3, 1.12, '4 ancillae', fontsize=6.5, color=C_ANC)
