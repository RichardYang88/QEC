"""
qcloud_vscr.py — Migrate the VSCR [[5,1,3]] experiment to the OriginQ Quantum
Cloud (本源量子云, http://qcloud.originqc.com.cn/) so that the trained
Variational Syndrome-Conditioned Recovery runs on a REAL superconducting QPU.

【本版本：已迁移到 pyqpanda3】
  - 支持字符串芯片 ID（如 'WK_C180_2'），不再使用 int
  - 云提交使用 QCloudService / backend.run()，本地采样使用 pyqpanda3.core.CPUQVM
  - 其余量子线路逻辑与原脚本完全一致

Hardware protocol (per experiment point):
  9 qubits:  data q0..q4  +  syndrome ancillas a0..a3  (qubits 5..8)
  1. logical prep: rotate q4 to the test state psi, apply Clifford encoder U_enc
  2. noise injection: sampled Pauli frame E (I/X/Y/Z per data qubit, drawn from
     the exact per-qubit depolarizing distribution: I w.p. 1-p, X/Y/Z w.p. p/3)
  3. syndrome extraction: per stabilizer g_i — H(a_i), parity CNOT couplings,
     H(a_i), MID-CIRCUIT Measure(a_i -> cbit i)
  4. recovery: branch-fixed R_s built from the TRAINED VSCR angles
     (Rz/Rx per qubit + ring Rzz decomposed as CNOT-Rz-CNOT)
  5. decode: U_enc^dagger, then a logical readout rotation V_val(psi) on q4
  6. measure data q0..q4 -> cbits 4..8
  Post-selection: shots with ancilla bits == s; branch fidelity
     F_s = P(data bits == 00000 | ancilla == s);  total F = sum_s P(s) F_s.
  The frame-averaged, branch-summed value reproduces EXACTLY the simulator
  quantity  F = sum_s <psi| R_s P_s rho_noisy P_s R_s^dag |psi>.

Modes:
  --mode verify   exact numpy(512-dim) verification of every circuit branch
                  against the ssvr_qec density-matrix simulator (no account)
  --mode sample   local pyqpanda3 CPUQVM sampling smoke test (no account)
  --mode cloud    submit to the real chip via QCloud (needs API token)

Token: pass --token or set environment variable ORIGINQ_TOKEN.
Register at http://qcloud.originqc.com.cn/, get the API Key from
`用户中心 -> API密钥`, and make sure real-chip quota is available.
"""
import sys, os, json, time, math, itertools, argparse
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssvr_qec as m

RNG = np.random.default_rng(12345)

# ---------------------------------------------------------------------
# pyqpanda3 imports (cloud submission + local CPU sampling)
# ---------------------------------------------------------------------
try:
    from pyqpanda3.core import QProg, CPUQVM, H, X, Y, Z, RX, RZ, CNOT
    try:
        from pyqpanda3.core import measure as QMEASURE
    except ImportError:
        from pyqpanda3.core import Measure as QMEASURE
    from pyqpanda3.qcloud import QCloudService
    try:
        from pyqpanda3.qcloud import JobStatus
    except ImportError:
        JobStatus = None
    HAS_PYQPANDA3 = True
    _PYQPANDA_IMPORT_ERR = None
except ImportError as exc:
    HAS_PYQPANDA3 = False
    _PYQPANDA_IMPORT_ERR = exc


# ---------------------------------------------------------------------
# qubit / cbit layout
# ---------------------------------------------------------------------
NQ_DATA = 5
NQ_ANC = 4
NQ = NQ_DATA + NQ_ANC            # 9 total; ancilla i is qubit 5+i
CB_ANC = list(range(4))          # cbits 0..3 : mid-circuit ancilla measures
CB_DATA = list(range(4, 9))      # cbits 4..8 : final data measures
DIM9 = 2 ** NQ

TEST_STATES = {                  # name -> (prep rotation on q4, validation rotation)
    '0':  ([],                 []),
    '1':  ([('X', 4)],         [('X', 4)]),
    '+':  ([('H', 4)],         [('H', 4)]),
    '-':  ([('X', 4), ('H', 4)], [('H', 4), ('X', 4)]),
    '+i': ([('RX', 4, -math.pi / 2)], [('RX', 4, math.pi / 2)]),
    '-i': ([('RX', 4, math.pi / 2)],  [('RX', 4, -math.pi / 2)]),
}
LOG_VEC = {'0': np.array([1, 0], dtype=complex), '1': np.array([0, 1], dtype=complex),
           '+': np.array([1, 1], dtype=complex) / math.sqrt(2),
           '-': np.array([1, -1], dtype=complex) / math.sqrt(2),
           '+i': np.array([1, 1j], dtype=complex) / math.sqrt(2),
           '-i': np.array([1, -1j], dtype=complex) / math.sqrt(2)}


# ---------------------------------------------------------------------
# 1) Clifford encoder for the [[5,1,3]] code
# ---------------------------------------------------------------------
def _swap_to_cnots(a, b):
    return [('CNOT', a, b), ('CNOT', b, a), ('CNOT', a, b)]


def synthesize_encoder():
    from qiskit.quantum_info import StabilizerState
    gens = m.STAB_STR + ['ZZZZZ']
    qc = StabilizerState.from_stabilizer_list(gens).clifford.to_circuit()
    gates = []
    for inst in qc.data:
        name = inst.operation.name.lower()
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == 'cx':
            gates.append(('CNOT', qs[0], qs[1]))
        elif name == 'swap':
            gates += _swap_to_cnots(qs[0], qs[1])
        elif name in ('h', 'x', 'y', 'z'):
            gates.append((name.upper(), qs[0]))
        elif name == 's':
            gates.append(('RZ', qs[0], math.pi / 2))
        elif name == 'sdg':
            gates.append(('RZ', qs[0], -math.pi / 2))
        else:
            raise ValueError(f'unexpected encoder gate: {name}')
    U = gates_to_unitary(gates, n=5)
    a = np.vdot(m.ONE_L.numpy(), U[:, 1])
    b = np.vdot(m.ZERO_L.numpy(), U[:, 0])
    theta = float(np.angle(b) - np.angle(a))
    gates = [('RZ', 4, theta)] + gates
    return gates


def _gate_matrix(g):
    name = g[0]
    if name == 'H':
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    if name == 'X':
        return np.array([[0, 1], [1, 0]], dtype=complex)
    if name == 'Y':
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    if name == 'Z':
        return np.array([[1, 0], [0, -1]], dtype=complex)
    if name == 'RZ':
        t = g[2]
        return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])
    if name == 'RX':
        t = g[2]
        return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)],
                         [-1j * np.sin(t / 2), np.cos(t / 2)]])
    raise ValueError(name)


def embed1(gate2, q, n=NQ):
    out = None
    for i in range(n):
        o = gate2 if i == q else np.eye(2, dtype=complex)
        out = o if out is None else np.kron(out, o)
    return out


CNOT4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def embed2(gate4, qa, qb, n=NQ):
    g = gate4.reshape(2, 2, 2, 2)
    e = [np.array([1, 0], dtype=complex), np.array([0, 1], dtype=complex)]
    I2 = np.eye(2, dtype=complex)
    out = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    c = g[i, j, k, l]
                    if c == 0:
                        continue
                    blk = None
                    for q in range(n):
                        if q == qa:
                            o = np.outer(e[i], e[k])
                        elif q == qb:
                            o = np.outer(e[j], e[l])
                        else:
                            o = I2
                        blk = o if blk is None else np.kron(blk, o)
                    out += c * blk
    return out


def gates_to_unitary(gates, n=NQ):
    U = np.eye(2 ** n, dtype=complex)
    for g in gates:
        if g[0] == 'CNOT':
            U = embed2(CNOT4, g[1], g[2], n) @ U
        else:
            U = embed1(_gate_matrix(g), g[1], n) @ U
    return U


def invert_gates(gates):
    out = []
    for g in reversed(gates):
        if g[0] == 'CNOT' or g[0] in ('H', 'X', 'Y', 'Z'):
            out.append(g)
        elif g[0] in ('RZ', 'RX'):
            out.append((g[0], g[1], -g[2]))
        else:
            raise ValueError(g[0])
    return out


def verify_encoder(U_enc):
    v0 = U_enc[:, 0]
    v1 = U_enc[:, 1]
    ph0 = complex(np.vdot(m.ZERO_L.numpy(), v0))
    ph1 = complex(np.vdot(m.ONE_L.numpy(), v1))
    rel = ph1 / ph0 if abs(ph0) > 1e-6 else complex(0)
    ok = (abs(ph0) > 1 - 1e-9) and (abs(ph1) > 1 - 1e-9) and (abs(rel - 1) < 1e-9)
    return ok, dict(overlap0=abs(ph0), overlap1=abs(ph1), rel_phase=rel)


# ---------------------------------------------------------------------
# 2) trained VSCR angles -> recovery gate lists R_s
# ---------------------------------------------------------------------
def load_angles(noise='depolarizing', retrain=False):
    env = os.environ.get('VSCR_ANGLES_FILE')
    path = env if env else f'vscr_angles_{"dep" if noise == "depolarizing" else noise}.npz'
    if os.path.exists(path) and not retrain:
        return np.load(path)['phi']
    torch.manual_seed(0)
    model, hist, infos = m.train_vscr_best(noise)
    with torch.no_grad():
        phi = model().cpu().numpy()
    np.savez(path, phi=phi, val_fidelity=np.array([hist['fidelity'][-1]]))
    print(f'[angles] trained {noise} -> {path} (val F {hist["fidelity"][-1]:.4f})')
    return phi


def _recovery_slots_index():
    slots = []
    for _ in range(m.N_LAYERS):
        for q in range(5):
            slots += [('RZ', q), ('RX', q), ('RZ', q)]
        for q in range(5):
            b = (q + 1) % 5
            a, b2 = (q, b) if q < b else (b, q)
            slots.append(('RZZ', a, b2))
    assert len(slots) == m.PHI_DIM
    return slots


SLOTS = _recovery_slots_index()


def recovery_gates(phi_s):
    gates = []
    for k, slot in enumerate(SLOTS):
        t = float(phi_s[k])
        if slot[0] == 'RZZ':
            _, a, b = slot
            gates += [('CNOT', a, b), ('RZ', b, t), ('CNOT', a, b)]
        else:
            gates.append((slot[0], slot[1], t))
    return gates


# ---------------------------------------------------------------------
# 3) full 9-qubit branch circuit
# ---------------------------------------------------------------------
def sample_pauli_frame(p):
    choices = RNG.choice(4, size=5, p=[1 - p, p / 3, p / 3, p / 3])
    return ''.join('IXYZ'[c] for c in choices)


def frame_gates(frame):
    return [(ch, q) for q, ch in enumerate(frame) if ch != 'I']


def frame_unitary5(frame):
    U = np.array([[1.0 + 0j]])
    mats = {'I': np.eye(2, dtype=complex),
            'X': np.array([[0, 1], [1, 0]], dtype=complex),
            'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
            'Z': np.array([[1, 0], [0, -1]], dtype=complex)}
    for ch in frame:
        U = np.kron(U, mats[ch])
    return U


def _basis_rot_gates(ch, j):
    if ch == 'X':
        return [('H', j)]
    if ch == 'Y':
        return [('RZ', j, -math.pi / 2), ('H', j)]
    return []


def extraction_gates():
    gates = []
    for i, s in enumerate(m.STAB_STR):
        a = NQ_DATA + i
        pre = []
        for j, ch in enumerate(s):
            pre += _basis_rot_gates(ch, j)
        gates += pre
        for j, ch in enumerate(s):
            if ch != 'I':
                gates.append(('CNOT', j, a))
        gates += invert_gates(pre)
    return gates


def extraction_measures():
    return [('MEAS', NQ_DATA + i, CB_ANC[i]) for i in range(NQ_ANC)]


def data_measures():
    return [('MEAS', q, CB_DATA[q]) for q in range(NQ_DATA)]


def build_branch_circuit(enc5, psi_name, frame, s_idx, phi, late_measure=False):
    prep, val = TEST_STATES[psi_name]
    gates = []
    gates += list(prep)
    gates += list(enc5)
    gates += frame_gates(frame)
    gates += extraction_gates()
    if not late_measure:
        gates += extraction_measures()
    gates += recovery_gates(phi[s_idx])
    gates += invert_gates(list(enc5))
    gates += list(val)
    if late_measure:
        gates += extraction_measures()
    gates += data_measures()
    return gates


def split_unitary_sections(gates):
    blocks = []
    for g in gates:
        kind = 'M' if g[0] == 'MEAS' else 'U'
        if blocks and blocks[-1][0] == kind:
            blocks[-1][1].append(g)
        else:
            blocks.append((kind, [g]))
    return blocks


# ---------------------------------------------------------------------
# 4) EXACT verification (numpy state-vector, 9 qubits)
# ---------------------------------------------------------------------
def apply_gates(psi, gates, n=NQ):
    psi = psi.reshape([2] * n)
    for g in gates:
        if g[0] == 'CNOT':
            ctrl, targ = g[1], g[2]
            idx = [slice(None)] * n
            idx[ctrl] = 1
            sub = psi[tuple(idx)]
            ax = targ if targ < ctrl else targ - 1
            psi[tuple(idx)] = np.flip(sub, axis=ax)
        else:
            mat = _gate_matrix(g)
            q = g[1]
            psi = np.tensordot(mat, psi, axes=([1], [q]))
            psi = np.moveaxis(psi, 0, q)
    return psi.reshape(-1)


def branch_amplitudes(gates, s_idx):
    blocks = split_unitary_sections(gates)
    psi = np.zeros(DIM9, dtype=complex)
    psi[0] = 1.0
    n_meas_seen = 0
    for kind, gs in blocks:
        if kind == 'U':
            psi = apply_gates(psi, gs)
        else:
            n_meas_seen += len(gs)
            if n_meas_seen == NQ_ANC:
                keep = np.array([(idx & 15) == s_idx for idx in range(DIM9)])
                psi = psi * keep
    P_s = float(np.vdot(psi, psi).real)
    return psi, P_s


def circuit_fidelity_exact(psi_name, frame, phi, s_list=None):
    gates0 = None
    total = 0.0
    per_s = {}
    for s in (s_list if s_list is not None else range(16)):
        gates = build_branch_circuit(ENC5, psi_name, frame, s, phi)
        psi_f, _ = branch_amplitudes(gates, s)
        F_s = float(abs(psi_f[s]) ** 2)
        per_s[s] = F_s
        total += F_s
    return total, per_s


def simulator_fidelity_frame(psi_name, frame, phi):
    psi = torch.tensor(LOG_VEC[psi_name], dtype=m.DTYPE)
    psi_enc = m.encode(psi)
    rho0 = torch.outer(psi_enc, psi_enc.conj())
    E = torch.tensor(frame_unitary5(frame), dtype=m.DTYPE)
    rho = E @ rho0 @ E.conj().T
    R_all = m.recovery_unitary_batch(torch.tensor(phi, dtype=m.DTYPE))
    _, F = m.vscr_fidelity(psi_enc, rho, R_all)
    return float(F)


ENC5 = None
ENC5_U = None


def setup():
    global ENC5, ENC5_U
    ENC5 = synthesize_encoder()
    ENC5_U = gates_to_unitary(ENC5, n=5)
    ok, info = verify_encoder(ENC5_U)
    print(f'[encoder] gates={len(ENC5)}  ok={ok}  {info}')
    assert ok, f'encoder verification failed: {info}'
    return ok, info


def run_verify(phi, p_values=(0.0, 0.05), n_frames=3, psis=('0', '+')):
    print('\n=== EXACT VERIFICATION: circuit vs simulator ===')
    worst = 0.0
    for p in p_values:
        frames = ['IIIII'] if p == 0.0 else [sample_pauli_frame(p) for _ in range(n_frames)]
        for psi in psis:
            for fr in frames:
                F_c, _ = circuit_fidelity_exact(psi, fr, phi)
                F_s = simulator_fidelity_frame(psi, fr, phi)
                d = abs(F_c - F_s)
                worst = max(worst, d)
                print(f'  p={p:<5} psi={psi:<3} frame={fr}  circuit={F_c:.6f}  sim={F_s:.6f}  |diff|={d:.2e}')
    print(f'worst |circuit - simulator| = {worst:.3e}')
    assert worst < 1e-9, 'circuit/simulator mismatch!'
    print('VERIFICATION PASSED')


# ---------------------------------------------------------------------
# 5) pyqpanda3 QProg builder + local CPUQVM sampling
# ---------------------------------------------------------------------
def _append_gate(prog, g):
    name = g[0]
    if name == 'H':
        prog << H(g[1])
    elif name == 'X':
        prog << X(g[1])
    elif name == 'Y':
        prog << Y(g[1])
    elif name == 'Z':
        prog << Z(g[1])
    elif name == 'RX':
        prog << RX(g[1], g[2])
    elif name == 'RZ':
        prog << RZ(g[1], g[2])
    elif name == 'CNOT':
        prog << CNOT(g[1], g[2])
    elif name == 'MEAS':
        prog << QMEASURE(g[1], g[2])
    else:
        raise ValueError(f'unknown gate: {name}')


def build_qprog(gates):
    """Convert our gate list into a pyqpanda3 QProg on 9 qubits / 9 cbits."""
    prog = QProg()
    for g in gates:
        _append_gate(prog, g)
    return prog


def cpuqvm_sample(gates, shots=500):
    """Sample a branch circuit on the local pyqpanda3 CPU simulator.

    Returns dict bitstring(cb0..cb8 order) -> count."""
    if not HAS_PYQPANDA3:
        raise RuntimeError(f'pyqpanda3 not installed: {_PYQPANDA_IMPORT_ERR}')
    prog = build_qprog(gates)
    qvm = CPUQVM()
    try:
        qvm.run(prog, shots=shots)
    except TypeError:
        qvm.run(prog, shot=shots)
    result = qvm.result()
    raw = result.get_counts() if hasattr(result, 'get_counts') else result
    counts = {}
    for key, cnt in raw.items():
        key = str(key).replace(' ', '')
        counts[key] = counts.get(key, 0) + int(cnt)
    return counts


def counts_to_branch_stats(counts, s_idx, anc_slice=slice(0, 4), rev=False):
    sbits = format(s_idx, '04b')
    data_slice = slice(4, 9) if anc_slice.start == 0 else slice(0, 5)
    n_sel = n_hit = 0
    for key, cnt in counts.items():
        if rev:
            key = key[::-1]
        anc = key[anc_slice]
        data = key[data_slice]
        if anc == sbits:
            n_sel += cnt
            if data == '00000':
                n_hit += cnt
    return n_sel, n_hit


def detect_layout(meta, results):
    hypotheses = [(slice(0, 4), False), (slice(5, 9), False),
                  (slice(0, 4), True), (slice(5, 9), True)]
    best, best_score = hypotheses[0], (-1, -1)
    for hyp in hypotheses:
        anc_slice, rev = hyp
        hits = sel = 0
        for (psi, p, fr, s), counts in zip(meta, results):
            n_sel, n_hit = counts_to_branch_stats(counts, s, anc_slice, rev)
            hits += n_hit
            sel += n_sel
        print(f'[layout] hypothesis anc@key[{anc_slice.start}:{anc_slice.stop}] rev={rev}: '
              f'data-hits {hits}, selected {sel}')
        # hits are the discriminating signal (sel alone favors all-zero keys)
        if (hits, sel) > best_score:
            best, best_score = hyp, (hits, sel)
    anc_slice, rev = best
    print(f'[layout] CHOSEN: ancilla bits at key[{anc_slice.start}:{anc_slice.stop}] rev={rev} '
          f'(data-hits {best_score[0]}, selected {best_score[1]})')
    return anc_slice, rev


def run_sample(phi, shots=400):
    print('\n=== CPUQVM SAMPLING SMOKE TEST ===')
    for frame, note in [('IIIII', 'no error'), ('IIXII', 'X on q2 -> syndrome of SYND_TABLE')]:
        s_exp = None
        for s, bits in enumerate(m.SYND_BITS):
            if m.SYND_TABLE[bits] == frame:
                s_exp = s
        gates = build_branch_circuit(ENC5, '0', frame, s_exp if s_exp is not None else 0, phi)
        counts = cpuqvm_sample(gates, shots)
        top = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        print(f'  frame={frame} ({note}): top outcomes {top}')
        s_use = s_exp if s_exp is not None else 0
        n_sel, n_hit = counts_to_branch_stats(counts, s_use)
        print(f'    branch s={s_use}({format(s_use,"04b")}): selected {n_sel}/{shots}, data-hit {n_hit} '
              f'-> F_s={n_hit / max(n_sel, 1):.3f}')


# ---------------------------------------------------------------------
# 6) OriginQ Quantum Cloud submission (pyqpanda3, string chip_id)
# ---------------------------------------------------------------------
def build_plan(psis, p_values, n_frames):
    plan = []
    for psi in psis:
        for p in p_values:
            frames = ['IIIII'] if p == 0.0 else [sample_pauli_frame(p) for _ in range(n_frames)]
            for fr in frames:
                plan.append(dict(psi=psi, p=p, frame=fr, syndromes=list(range(16))))
    return plan


def plan_progs(plan, phi, late_measure=False):
    progs, meta = [], []
    for pt in plan:
        for s in pt['syndromes']:
            gates = build_branch_circuit(ENC5, pt['psi'], pt['frame'], s, phi,
                                         late_measure=late_measure)
            progs.append(gates)
            meta.append((pt['psi'], pt['p'], pt['frame'], s))
    return progs, meta


def _make_qcloud_service(token):
    """Build QCloudService; try keyword then positional then login()."""
    # Try keyword api_key=...
    try:
        return QCloudService(api_key=token)
    except TypeError:
        pass
    # Try positional
    try:
        return QCloudService(token)
    except Exception:
        pass
    # Try login()
    try:
        svc = QCloudService()
        if hasattr(svc, 'login'):
            svc.login(token)
            return svc
    except Exception:
        pass
    raise RuntimeError('failed to construct QCloudService; check pyqpanda3 version and token')


def _build_options(block=None):
    """QCloudOptions for real-chip submission.

    Enables server-side amend/mapping/optimization; if a physical qubit
    block is given, pins the task to it via set_specified_block.
    """
    from pyqpanda3.qcloud import QCloudOptions
    opt = QCloudOptions()
    for setter in ('set_amend', 'set_mapping', 'set_optimization'):
        try:
            getattr(opt, setter)(True)
        except Exception as e:
            print(f'[cloud] warning: {setter}(True) failed: {e}', flush=True)
    if block:
        opt.set_specified_block([int(q) for q in block])
    return opt


def _run_job(backend, prog_or_list, shots, options=None):
    """Submit a QProg (or list of QProgs) and return the job object."""
    if options is not None:
        try:
            return backend.run(prog_or_list, shots, options=options)
        except TypeError:
            try:
                return backend.run(prog_or_list, shots, options)
            except TypeError:
                print('[cloud] warning: backend.run rejected options; '
                      'falling back to plain run', flush=True)
    try:
        return backend.run(prog_or_list, shots=shots)
    except TypeError:
        return backend.run(prog_or_list, shot=shots)


def _wait_job(job, poll=5.0, timeout=7200):
    """Block until job completes."""
    t0 = time.time()
    while True:
        status = None
        if hasattr(job, 'status'):
            try:
                status = job.status()
            except Exception:
                status = None
        if status is None and hasattr(job, 'is_finished'):
            try:
                if job.is_finished():
                    return
            except Exception:
                pass
            time.sleep(poll)
            continue
        # Interpret status
        done = False
        failed = False
        if JobStatus is not None:
            try:
                if status == JobStatus.FINISHED:
                    done = True
                elif hasattr(JobStatus, 'FAILED') and status == JobStatus.FAILED:
                    failed = True
            except Exception:
                pass
        if not done and not failed:
            s = str(status).lower()
            if any(k in s for k in ('finish', 'success', 'complete', 'done')):
                done = True
            elif any(k in s for k in ('fail', 'error', 'cancel')):
                failed = True
        if done:
            return
        if failed:
            raise RuntimeError(f'QCloud job failed with status: {status}')
        if time.time() - t0 > timeout:
            raise TimeoutError(f'QCloud job did not finish within {timeout}s')
        time.sleep(poll)


def _probs_to_counts(probs, shots):
    if probs is None:
        return {}
    out = {}
    for k, v in probs.items():
        k = str(k).replace(' ', '')
        if set(k) <= {'0', '1'}:
            out[k] = out.get(k, 0.0) + float(v)
    total = sum(out.values())
    if shots and total <= 1.0001 and total > 0:
        return {k: int(round(v * shots)) for k, v in out.items()}
    return {k: int(round(v)) for k, v in out.items()}


def _extract_counts_list(result, n_expected, shots):
    """Extract per-circuit counts dict from a QCloudResult."""
    # Try list-type getters first (batch tasks)
    for attr in ('get_counts_list', 'get_probs_list'):
        if hasattr(result, attr):
            try:
                data = getattr(result, attr)()
            except Exception:
                data = None
            if isinstance(data, (list, tuple)):
                out = [_probs_to_counts(d, shots) for d in data]
                if len(out) == n_expected:
                    return out
            elif isinstance(data, dict):
                return [_probs_to_counts(data, shots)]
    # Single-result getters
    for attr in ('get_counts', 'get_probs'):
        if hasattr(result, attr):
            try:
                data = getattr(result, attr)()
            except Exception:
                data = None
            if isinstance(data, dict) and data:
                return [_probs_to_counts(data, shots)]
    # Fall back to origin_data JSON
    if hasattr(result, 'origin_data'):
        try:
            raw = result.origin_data()
            parsed = _parse_origin_data(raw, shots)
            if parsed:
                return parsed
        except Exception:
            pass
    # Last resort: try str(result)
    try:
        parsed = _parse_origin_data(str(result), shots)
        if parsed:
            return parsed
    except Exception:
        pass
    raise ValueError(f'cannot extract counts from result type {type(result)}: {str(result)[:400]}')


def _parse_origin_data(raw, shots):
    if not isinstance(raw, str):
        try:
            raw = json.dumps(raw, default=str)
        except Exception:
            return None
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    results = []
    # Format 1: {'taskResult': ['{"key":[...],"value":[...]}', ...]}
    if isinstance(obj, dict) and 'taskResult' in obj:
        tr = obj['taskResult']
        if isinstance(tr, str):
            tr = [tr]
        for item in tr:
            inner = json.loads(item) if isinstance(item, str) else item
            probs = dict(zip(inner.get('key', []), inner.get('value', [])))
            results.append(_probs_to_counts(probs, shots))
        return results or None
    # Format 2: {'key': [...], 'value': [...]}
    if isinstance(obj, dict) and 'key' in obj and 'value' in obj:
        probs = dict(zip(obj['key'], obj['value']))
        return [_probs_to_counts(probs, shots)]
    return None


def submit_cloud(gate_lists, token, chip_id='WK_C180_2', shots=4000,
                 batch=True, dump_raw=None, chunk=32, job_timeout=21600,
                 use_options=True, block=None):
    """Submit all branch circuits to the real chip. Returns list of counts dicts.

    use_options: build QCloudOptions(amend+mapping+optimization) for real-chip
                 tasks. block: explicit physical qubit block; if None and
                 use_options, the best block is auto-queried via
                 backend.best_qubit_blocks(NQ).
    """
    if not HAS_PYQPANDA3:
        raise RuntimeError(f'pyqpanda3 not installed: {_PYQPANDA_IMPORT_ERR}')

    service = _make_qcloud_service(token)
    try:
        backend = service.backend(chip_id)
    except TypeError:
        backend = service.backend(chip_id=chip_id)
    print(f'[cloud] backend: {chip_id}', flush=True)

    options = _build_options(block) if use_options else None
    if options is not None and block is None:
        try:
            bb = backend.best_qubit_blocks(NQ)
            blocks = list(bb.qubit_blocks) if bb.qubit_blocks else []
            if blocks:
                block = [int(q) for q in blocks[0]]
                options.set_specified_block(block)
        except Exception as e:
            print(f'[cloud] best_qubit_blocks unavailable ({e}); '
                  f'server will choose qubits', flush=True)
    if options is not None:
        print(f'[cloud] options: amend+mapping+optimization=True, '
              f'specified_block={block}', flush=True)

    results = []
    if batch:
        raw_all = []
        total = len(gate_lists)
        for off in range(0, total, chunk):
            part = gate_lists[off:off + chunk]
            progs = [build_qprog(g) for g in part]
            print(f'[cloud]   batch {off // chunk + 1}: submitting {len(part)} circuits ...',
                  flush=True)
            job = _run_job(backend, progs, shots, options=options)
            _wait_job(job, timeout=job_timeout)
            res = job.result()
            raw_all.extend(_extract_counts_list(res, len(part), shots))
            print(f'[cloud]   batch {off // chunk + 1}: received', flush=True)
            if dump_raw:  # incremental save: keep partial data if quota dies mid-run
                with open(dump_raw, 'w') as f:
                    json.dump(raw_all, f, default=str, ensure_ascii=False, indent=1)
        results = raw_all
    else:
        for i, gates in enumerate(gate_lists):
            prog = build_qprog(gates)
            job = _run_job(backend, prog, shots, options=options)
            print(f"[cloud] job submitted, id = {job.job_id()}", flush=True)  # <--- 添加这行
            _wait_job(job, timeout=job_timeout)
            res = job.result()
            counts = _extract_counts_list(res, 1, shots)[0]
            if dump_raw and i == 0:
                try:
                    raw_txt = res.origin_data() if hasattr(res, 'origin_data') else str(res)
                except Exception:
                    raw_txt = str(res)
                with open(dump_raw, 'w') as f:
                    f.write(str(raw_txt))
            results.append(counts)
            print(f'  [{i+1}/{len(gate_lists)}] done', flush=True)
    return results


def _normalize_counts(r, shots=None):
    """Legacy-compatible normalizer (kept for safety)."""
    if isinstance(r, dict):
        return _probs_to_counts(r, shots)
    try:
        return _extract_counts_list(r, 1, shots)[0]
    except Exception:
        pass
    raise ValueError(f'cannot parse QCloud result: {str(r)[:500]}')


def aggregate(plan, meta, results, shots, anc_slice=slice(0, 4), rev=False):
    from collections import defaultdict
    hits = defaultdict(int)
    sel = defaultdict(int)
    tot = defaultdict(int)
    for (psi, p, fr, s), counts in zip(meta, results):
        n_sel, n_hit = counts_to_branch_stats(counts, s, anc_slice, rev)
        key = (psi, p, fr)
        hits[key] += n_hit
        sel[key] += n_sel
        tot[key] += sum(counts.values())
    out = {}
    for (psi, p, fr) in hits:
        N = tot[(psi, p, fr)] / 16.0
        F_hw = hits[(psi, p, fr)] / max(N, 1)
        F_hw_err = math.sqrt(max(F_hw * (1 - F_hw), 1e-12) / max(N, 1))
        F_ideal = simulator_fidelity_frame(psi, fr, PHI_GLOBAL)
        out.setdefault((psi, p), []).append((fr, F_hw, F_hw_err, F_ideal,
                                             sel[(psi, p, fr)] / max(tot[(psi, p, fr)], 1)))
    return out


PHI_GLOBAL = None


# ---------------------------------------------------------------------
# 7) dense simulator reference curves + outputs + CLI
# ---------------------------------------------------------------------
def dense_reference_curves(phi, psis, p_grid, n_log=80):
    R_all = m.recovery_unitary_batch(torch.tensor(phi, dtype=m.DTYPE))
    curves = {k: [] for k in ('vscr', 'decoder', 'raw')}
    torch.manual_seed(7)
    for p in p_grid:
        acc = {k: 0.0 for k in curves}
        for _ in range(n_log):
            psi = m.random_logical_state()
            psi_enc = m.encode(psi)
            rho = m.noisy_state(psi_enc, p, 'depolarizing')
            acc['vscr'] += float(m.vscr_fidelity(psi_enc, rho, R_all)[1])
            acc['decoder'] += float(m.perfect_code_decoder_fidelity(psi_enc, rho))
            acc['raw'] += float(m.baseline_raw(psi_enc, rho))
        for k in curves:
            curves[k].append(acc[k] / n_log)
    return curves


def save_and_plot(agg, curves, p_grid, tag):
    os.makedirs('figures', exist_ok=True)
    out = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'hardware': {f'{k[0]}|{k[1]}': v for k, v in agg.items()},
        'reference_curves': {k: list(map(float, v)) for k, v in curves.items()},
        'p_grid': list(map(float, p_grid)),
    }
    with open(f'qcloud_results_{tag}.json', 'w') as f:
        json.dump(out, f, default=str, ensure_ascii=False, indent=1)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    a = ax[0]
    a.plot(p_grid, curves['raw'], ':', color='gray', label='Raw (sim)')
    a.plot(p_grid, curves['decoder'], '-.', color='navy', label='Perfect-code decoder (sim)')
    a.plot(p_grid, curves['vscr'], '-', color='crimson', label='VSCR (sim)')
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for i, (key, entries) in enumerate(sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0]))):
        psi, p = key
        F_hw = float(np.mean([e[1] for e in entries]))
        err = float(np.std([e[1] for e in entries]) / max(len(entries) - 1, 1) ** 0.5) if len(entries) > 1 else np.mean([e[2] for e in entries])
        F_id = float(np.mean([e[3] for e in entries]))
        a.errorbar([p], [F_hw], yerr=[err], fmt='o', color=colors[i % 10], capsize=3,
                   label=f'HW {psi} (QPU)')
        a.plot([p], [F_id], marker='x', color=colors[i % 10], ms=7)
    a.set_xlabel('Noise parameter p (depolarizing)')
    a.set_ylabel('Recovery fidelity')
    a.set_title('[[5,1,3]] VSCR: OriginQ real QPU vs simulation\n(x = ideal-protocol sim at the same Pauli frames)')
    a.legend(fontsize=7)
    a.grid(alpha=0.3)
    b = ax[1]
    labels, hw_vals, hw_errs, id_vals = [], [], [], []
    for key, entries in sorted(agg.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        labels.append(f'{key[0]}@p={key[1]}')
        hw_vals.append(float(np.mean([e[1] for e in entries])))
        hw_errs.append(float(np.mean([e[2] for e in entries])))
        id_vals.append(float(np.mean([e[3] for e in entries])))
    xs = np.arange(len(labels))
    b.bar(xs - 0.2, id_vals, 0.4, label='ideal protocol (sim)', color='crimson', alpha=0.75)
    b.bar(xs + 0.2, hw_vals, 0.4, yerr=hw_errs, label='OriginQ QPU', color='steelblue', capsize=3)
    b.set_xticks(xs); b.set_xticklabels(labels, rotation=30, fontsize=7)
    b.set_ylabel('Recovery fidelity'); b.legend(fontsize=8); b.grid(alpha=0.3, axis='y')
    b.set_title('Hardware vs ideal at matched Pauli frames')
    plt.tight_layout()
    fig.savefig(f'figures/fig_qcloud_hardware_{tag}.png', dpi=150)
    plt.close(fig)
    print(f'[out] qcloud_results_{tag}.json + figures/fig_qcloud_hardware_{tag}.png')


def main():
    global PHI_GLOBAL, ENC5, ENC5_U
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['verify', 'sample', 'cloud'], default='verify')
    ap.add_argument('--token', default=os.environ.get('ORIGINQ_TOKEN', ''))
    ap.add_argument('--chip-id', type=str, default='WK_C180_2',
                    help="芯片 ID（字符串），如 'WK_C180_2'、'WK_C180'、'PQPUMESH8'")
    ap.add_argument('--shots', type=int, default=4000)
    ap.add_argument('--p-values', default='0.02,0.08')
    ap.add_argument('--frames', type=int, default=2)
    ap.add_argument('--states', default='0,+')
    ap.add_argument('--noise', default='depolarizing')
    ap.add_argument('--no-batch', action='store_true')
    ap.add_argument('--late-measure', action='store_true',
                    help='move ancilla measurement to circuit end (fallback '
                         'if the real-chip compiler rejects mid-circuit M; '
                         'statistics are identical)')
    ap.add_argument('--dump-raw', default=None)
    ap.add_argument('--chunk', type=int, default=32,
                    help='circuits per batch submission (batch mode only)')
    ap.add_argument('--job-timeout', type=int, default=21600,
                    help='max seconds to wait per cloud job (queue + execution)')
    ap.add_argument('--no-options', action='store_true',
                    help='submit WITHOUT QCloudOptions (plain backend.run; '
                         'not recommended for real chips)')
    ap.add_argument('--block', type=str, default=None,
                    help='comma-separated physical qubit block for '
                         'set_specified_block (default: auto-query '
                         'best_qubit_blocks(9) on the chip)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--analyze', default=None,
                    help='offline mode: path to a raw dump JSON '
                         '(--dump-raw output); analyze it instead of '
                         'submitting to the cloud (no token needed)')
    args = ap.parse_args()

    if args.mode in ('cloud', 'sample') and not HAS_PYQPANDA3:
        print(f'ERROR: pyqpanda3 is required for --mode {args.mode}: {_PYQPANDA_IMPORT_ERR}')
        sys.exit(3)

    setup()
    phi = load_angles(args.noise)
    PHI_GLOBAL = phi
    print(f'[angles] loaded phi {phi.shape}')

    if args.mode == 'verify':
        run_verify(phi)
        return
    if args.mode == 'sample':
        run_verify(phi, p_values=(0.05,), n_frames=1, psis=('0',))
        run_sample(phi)
        return

    # ---- cloud mode ----
    psis = args.states.split(',')
    p_values = [float(x) for x in args.p_values.split(',')]
    plan = build_plan(psis, p_values, args.frames)
    gate_lists, meta = plan_progs(plan, phi, late_measure=args.late_measure)
    print(f'[cloud] plan: {len(plan)} points x 16 branches = {len(gate_lists)} circuits, '
          f'{args.shots} shots each')
    if args.analyze:
        results = json.load(open(args.analyze))
        print(f'[analyze] loaded {len(results)} count-dicts from {args.analyze}')
        assert len(results) == len(gate_lists), \
            f'dump has {len(results)} circuits but plan expects {len(gate_lists)}'
        anc_slice, rev = detect_layout(meta, results)
        agg = aggregate(plan, meta, results, args.shots, anc_slice, rev)
        for key, entries in sorted(agg.items()):
            hw = np.mean([e[1] for e in entries])
            ide = np.mean([e[3] for e in entries])
            print(f'  psi={key[0]} p={key[1]}: HW={hw:.4f}  '
                  f'ideal-sim={ide:.4f}  gap={hw - ide:+.4f}')
        p_grid = np.linspace(0.0, max(p_values) * 1.25, 12)
        curves = dense_reference_curves(phi, psis, p_grid)
        tag = args.tag or ('analyze_' +
                           os.path.basename(args.analyze).replace('.json', ''))
        save_and_plot(agg, curves, p_grid, tag)
        return
    if not args.token:
        print('ERROR: no token. Register at http://qcloud.originqc.com.cn/, '
              'get API Key, then pass --token or set ORIGINQ_TOKEN.')
        sys.exit(2)
    t0 = time.time()
    block = None
    if args.block:
        block = [int(x) for x in args.block.replace(' ', '').split(',') if x]
    results = submit_cloud(gate_lists, args.token, chip_id=args.chip_id,
                           shots=args.shots, batch=not args.no_batch,
                           dump_raw=args.dump_raw, chunk=args.chunk,
                           job_timeout=args.job_timeout,
                           use_options=not args.no_options, block=block)
    print(f'[cloud] all results received in {time.time() - t0:.0f}s')
    anc_slice, rev = detect_layout(meta, results)
    agg = aggregate(plan, meta, results, args.shots, anc_slice, rev)
    for key, entries in sorted(agg.items()):
        hw = np.mean([e[1] for e in entries]); ide = np.mean([e[3] for e in entries])
        print(f'  psi={key[0]} p={key[1]}: HW={hw:.4f}  ideal-sim={ide:.4f}  gap={hw - ide:+.4f}')
    p_grid = np.linspace(0.0, max(p_values) * 1.25, 12)
    curves = dense_reference_curves(phi, psis, p_grid)
    tag = args.tag or datetime.now().strftime('%Y%m%d_%H%M%S')
    save_and_plot(agg, curves, p_grid, tag)


if __name__ == '__main__':
    main()
