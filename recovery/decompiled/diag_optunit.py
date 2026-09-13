# Source Generated with Decompyle++
# File: diag_optunit.cpython-311.pyc (Python 3.11)

'''diag_optunit.py -- exact optimal UNITARY recovery per syndrome branch.

For branch s reduce the problem to 2x2 operators A_k = W_s^dag K_k V.  The
Haar-averaged (unnormalised) branch fidelity of a recovery whose action on the
branch is the 2x2 unitary G is, exactly as in vscr_paper_abl._branch_M,

    F_s(G) = 2 v(G)^dag M_s v(G),      v(G)[a*2+b] = G[b,a]/sqrt(2)
           = [ sum_k |Tr(G A_k)|^2 + Tr(sum_k A_k^dag A_k) ] / 6 .

The second term is G-independent, so the optimum over UNITARY recoveries is

    max_{G unitary} J(G) = sum_k |Tr(G A_k)|^2 ,

which for a single Kraus operator (coherent channel) has the closed form
J_max = ||A||_*^2 (nuclear norm squared) attained at G = V_A U_A^dag from the
SVD A = U_A Sigma V_A^dag.  This is the information-theoretic optimum *inside
the VSCR ansatz family* and the natural target the trained model should reach.
'''
import math
import numpy as np
import torch
import ssvr_qec as m
import vscr_paper as vp
import vscr_paper_abl as ab

def _qform(As):
    '''Q (4x4) with J(G) = g^T Q conj(g), g = vec(G) row-major.

    Tr(G A) = sum_ij G_ij A_ji = g . vec(A^T) =: g^T v, so
    J = sum_k |g^T v_k|^2 = g^T (sum_k v_k conj(v_k)^T) conj(g).
    Collapsing the Kraus sum into one 4x4 matrix makes the objective O(1)
    and removes any dependence on the Kraus count (1024 for depolarizing).
    '''
    Q = np.zeros((4, 4), dtype = complex)
    for A in As:
        v = np.asarray(A, dtype = complex).T.reshape(4)
        Q += np.outer(v, v.conj())
        return 0.5 * (Q + Q.conj().T)

_PAULI = [
    np.eye(2, dtype = complex),
    np.array([
        [
            0,
            1],
        [
            1,
            0]], dtype = complex),
    np.array([
        [
            0,
            (-0+-1j)],
        [
            (0+1j),
            0]], dtype = complex),
    np.array([
        [
            1,
            0],
        [
            0,
            -1]], dtype = complex)]

def _G_of(h):
    '''2x2 unitary exp(i H) with H = h0 I + h . sigma (4 real parameters).'''
    expm = expm
    import scipy.linalg
    H = (lambda .0: pass# WARNING: Decompyle incomplete
)(zip(h, _PAULI)())
    return expm((0+1j) * H)


def _Jval(h, Q):
    g = _G_of(h).reshape(4)
    return float(np.real(g @ Q @ g.conj()))


def opt_unitary_branch(As, n_start = (12,)):
    '''max over 2x2 unitaries G of J(G) = sum_k |Tr(G A_k)|^2.'''
    expm = expm
    import scipy.linalg
    minimize = minimize
    import scipy.optimize
    Q = _qform(As)
    best_G = None
    best_val = -(np.inf)
    
    def try_G(G = None):
        pass
    # WARNING: Decompyle incomplete

    if len(As) == 1:
        (U, S, Vh) = np.linalg.svd(np.asarray(As[0], dtype = complex))
        G0 = Vh.conj().T @ U.conj().T
        best_G = G0
        best_val = try_G(G0)
        if not abs(best_val - float(np.sum(S) ** 2)) < 1e-09:
            raise AssertionError
    for P in _PAULI + [
        np.array([
            [
                0,
                (0+1j)],
            [
                (0+1j),
                0]], dtype = complex)]:
        v = try_G(P)
        if v > best_val:
            best_G = P
            best_val = v
        rng = np.random.default_rng(0)
        starts = range(n_start)()
        starts.append(np.zeros(4))
        for h0 in starts:
            r = None((lambda h = None: pass# WARNING: Decompyle incomplete
), h0, method = 'L-BFGS-B', options = {
                'maxiter': 800,
                'ftol': 1e-15,
                'gtol': 1e-14 })
            G = _G_of(r.x)
            v = try_G(G)
            if v > best_val:
                best_G = G
                best_val = v
            for _ in range(3):
                r = None((lambda h = None: pass# WARNING: Decompyle incomplete
), np.concatenate([
                    [
                        rng.normal() * 0.3],
                    rng.normal(size = 3)]), method = 'Nelder-Mead', options = {
                    'maxiter': 4000,
                    'fatol': 1e-16,
                    'xatol': 1e-14 })
                G = _G_of(r.x)
                v = try_G(G)
                if v > best_val:
                    best_G = G
                    best_val = v
                return (best_val, best_G)


def main():
    print(f'''{'channel':22s} {'p':>5s} {'F_decoder':>11s} {'F_opt-unit':>11s} {'gain':>11s} {'F_CPTP':>11s} {'frac of gap':>12s}''')
    rows = []
    for noise, p in (('depolarizing', 0.1), ('depolarizing', 0.3), ('amplitude_damping', 0.1), ('amplitude_damping', 0.3), ('mixed', 0.1), ('coherent', 0.1), ('coherent', 0.15), ('coherent', 0.3)):
        cf_d = np.zeros(16)
        cf_u = np.zeros(16)
        ps = []
        for s in range(16):
            As = ab._branch_A(noise, p, s)()
            Q = _qform(As)
            const = float(np.real(np.trace(Q)))
            p_s = const / 2
            ps.append(p_s)
            if p_s < 1e-13:
                cf_u[s] = 1
                cf_d[s] = 1
                continue
            (Ju, _) = opt_unitary_branch(As)
            cf_u[s] = (Ju + const) / 6 / p_s
            Gd = ab.V_ISO.conj().T @ m.C_SYNDS[s].numpy() @ ab.W_BASIS[s]
            gd = Gd.reshape(4)
            cf_d[s] = (float(np.real(gd @ Q @ gd.conj())) + const) / 6 / p_s
            ps = np.array(ps)
            F_u = float(np.dot(ps, cf_u))
            F_d = float(np.dot(ps, cf_d))
            ceil = ab.sdp_ceiling(noise, p)['F_cptp']
        frac = (F_u - F_d) / (ceil - F_d) if ceil - F_d > 1e-12 else float('nan')
        print(f'''{noise:22s} {p:5.2f} {F_d:11.7f} {F_u:11.7f} {F_u - F_d:+11.2e} {ceil:11.7f} {frac:12.4f}''', flush = True)
        rows.append({
            'noise': noise,
            'p': p,
            'F_dec': F_d,
            'F_opt_unit': F_u,
            'F_cptp': ceil,
            'cf_dec': cf_d.tolist(),
            'cf_opt_unit': cf_u.tolist(),
            'p_s': ps.tolist() })
        F_mc = ab._decoder_ref_F(noise, p, n = 200)
        if not abs(F_mc - F_d) < 0.0003:
            raise (noise, p, F_mc, F_d)()
        print('\n[ok] analytic decoder branch-sum matches Monte-Carlo to <3e-4')
        import json
        json.dump(rows, open('diag_optunit.json', 'w'), indent = 1)
        return None

if __name__ == '__main__':
    main()
    return None
