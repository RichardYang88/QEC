import sys, faulthandler
faulthandler.enable()
sys.path.insert(0,'/home/yqc/github/QEC')
import numpy as np, torch, math
import ssvr_qec as m
import vscr_paper as vp
import vscr_paper_abl as ab
print('torch threads =', torch.get_num_threads(), flush=True)
print('(a) identity ceiling...', flush=True)
r = ab.sdp_ceiling('depolarizing', 0.0)
print('   F_cptp =', r['F_cptp'], flush=True)
print('(b) depolarizing p=0.05 ceiling...', flush=True)
r = ab.sdp_ceiling('depolarizing', 0.05)
print('   F_cptp =', r['F_cptp'], flush=True)
print('(c) branch qform/exact_F...', flush=True)
for noise, p in (('depolarizing', 0.05), ('amplitude_damping', 0.10)):
    for s in (0, 1, 5, 9, 15):
        _, p_M = ab._branch_M(noise, p, s)
        Q, const, p_Q = ab._branch_qform(noise, p, s)
        assert abs(p_Q - p_M) < 1e-12
        fr = sum(float(np.real(np.trace(np.asarray(A).conj().T @ A))) for A in ab._branch_A(noise, p, s))
        assert abs(const - fr) < 1e-12
    Fe, _ = ab.exact_F(vp.PHI_DEC, noise, p)
    print('   %-20s p=%.2f exact_F=%.8f' % (noise, p, Fe), flush=True)
print('   eval_phi_table(400)...', flush=True)
Fm, Fsem = ab.eval_phi_table(vp.PHI_DEC, 'depolarizing', [0.05], n_test=400)
print('   MC =', Fm, Fsem, flush=True)
print('(d) single-Kraus nuclear norm...', flush=True)
for noise, p in (('amplitude_damping', 0.10), ('coherent', 0.10)):
    for s in range(16):
        As = [np.asarray(A) for A in ab._branch_A(noise, p, s)]
        if len(As) != 1 or p <= 0: continue
        Ju, _ = ab.opt_unitary_branch(noise, p, s)
        ref = float(np.linalg.svd(As[0])[1].sum()) ** 2
        assert abs(Ju - ref) < 1e-9, (noise, p, s, Ju, ref)
print('   OK', flush=True)
print('(e) ceilings...', flush=True)
for noise, p in (('amplitude_damping', 0.30), ('amplitude_damping', 0.06), ('depolarizing', 0.30), ('coherent', 0.10), ('mixed', 0.15)):
    u = ab.opt_unitary_ceiling(noise, p)
    print('   %-20s p=%.2f unit done: F_dec=%.8f F_unit=%.8f' % (noise, p, u['F_dec'], u['F_unit']), flush=True)
    c = ab.sdp_ceiling(noise, p)
    print('      sdp F_cptp=%.8f' % c['F_cptp'], flush=True)
