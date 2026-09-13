# Recovered API contract (read from the surviving .pyc oracle)


==============================================================================
# MODULE ssvr_qec
==============================================================================


def Rx(t):

def Ry(t):

def Rz(t):
    """Rz(t) = diag(e^{-it/2}, e^{it/2}).  t: real scalar tensor.
    """

def Rzz(t):
    """exp(-i t/2 Z⊗Z) = diag(e^{-it/2}, e^{it/2}, e^{it/2}, e^{-it/2}).
    """

class VSCR:   # bases: ['Module', 'object']
    def __init__(self, n_synd=16, phi_dim=60, hidden=64, ctx=24):
        """Initialize internal Module state, shared by both nn.Module and ScriptModule.
        """
    def forward(self):

def _anticommute(P_full, g):
    """True if P_full and g anticommute.
    """

def _build_recovery_slots(n=5, n_layers=3):

def _logical_basis():
    """Return |0_L>, |1_L> as 32-dim complex vectors (columns of E_code).
    
    Convention: |0_L> is the +1 eigenstate of Z_L inside the code space and
    |1_L> = X_L |0_L>  (so X_L flips 0<->1 exactly, removing global-phase
    ambiguity that would otherwise break an equality assertion).
    """

def _pauli_features(rho):
    """Vectorised Pauli-expectation feature vector (no grad).
    """

def _pauli_index_set(n=5, max_weight=5):

def _print_and_save(res_dep, res_ad, res_mx, p_values, hist_real, hist_rand, t0):

def _reconstruct_state(coeffs):
    """rho = (1/2^n) sum_P c_P P  (c_I forced to 1 by the caller).  Vectorised.
    """

def _selftest_action_cols():

def _selftest_gates():
    """Sanity checks for embed_gate / pauli_string conventions.
    """

def _selftest_quadrature(n_u=3, n_phi=7, n_mc=None, seed=7):
    """Prove the quadrature is EXACT for the objective's algebraic degree.
    
    Test 1 (algebraic, machine precision): for random 2x2 operators A, B the
    Haar second moment is  E[<psi|A|psi><psi|B|psi>] = (TrA TrB + TrAB)/6.
    Test 2 (end to end): the quadrature-averaged recovery fidelity equals
    (i) the independent analytic branch sum built from `_branch_M`'s exact Haar
    matrix M_s, and (ii) `vscr_paper_abl.exact_F`'s quadratic-form route, both
    to machine precision, and (iii) a Monte-Carlo average to within 6 sigma.
    Test 3: the noise-strength rule CONVERGES.  rho(p) is not polynomial (see
    the comment block above), so the test measures the Gauss-Legendre error
    against a 40-node reference and pins n_p = 6 to machine precision while
    recording how badly the naive 2-node rule is biased.
    
    n_mc=None picks a per-channel sample count: the depolarizing channel builds
    1024 Kraus operators per call (~0.3 s), so a large MC average there would
    dominate the runtime while adding nothing -- the analytic comparisons are
    already exact to 1e-11.
    """

def _synd_projectors():
    """16 syndrome projectors  P_s = prod_i (I + (-1)^{s_i} g_i)/2  (rank 2 each).
    """

def _syndrome_table():

def amp_kraus(gamma, q):
    """Two Kraus ops for amplitude damping on qubit q (rate gamma).
    """

def apply_amplitude_damping(rho, gamma):

def apply_coherent(rho, eps):
    """Systematic coherent over-rotation: identical Rx(eps) on every qubit
    (calibration-type control error).  NOT a Pauli mixture.
    """

def apply_depolarizing(rho, p):
    """Independent single-qubit depolarizing of rate p on every qubit.
    """

def apply_mixed(rho, p, gamma):

def baseline_raw(psi_enc, rho_noisy):

def clifford_logical_states():
    """The 6 single-qubit Pauli eigenstates (|0>,|1>,|+>,|->,|+i>,|-i>).
    """

def embed_gate(U, targets, n=5):

def encode(psi):
    """psi: length-2 complex tensor (logical state) -> 32-dim encoded vector.
    """

def evaluate_methods(model, A_lindr, p_values, noise, n_test=40, seed=1241):
    """Return dict: method -> {'F': list, 'LER': list}.
    """

def features_of(rho):
    """rho: 32x32 complex -> float64 feature tensor (vectorised, no grad).
    """

def fidelity(psi_enc, rho):
    """Fidelity of mixed state rho w.r.t. pure |psi_enc> in the 32-dim space.
    """

def haar_quadrature(n_u=3, n_phi=7):
    """Deterministic exact Haar quadrature over single-logical-qubit states.
    
    Returns (states, weights): states is (n_u*n_phi, 2) complex128, weights is
    (n_u*n_phi,) float64 summing to 1.  For any observable polynomial of degree
    <= 2 in the Bloch vector, sum_i w_i <psi_i|O|psi_i> equals the Haar average.
    """

def lindr_fidelity(psi_enc, rho_noisy, A):

def main():

def noisy_state(psi_enc, p, noise='depolarizing'):
    """Build the noisy density matrix from a pure encoded state.
    """

def p_quadrature(p_range, n=6):
    """Gauss-Legendre nodes/weights for averaging over a noise-strength interval.
    
    n = 6 is exact to machine precision for all four channels used here (see the
    convergence table in the comment block above `haar_quadrature`); the
    integrand is analytic in p but NOT polynomial -- `apply_depolarizing`
    composes five single-qubit maps (degree 5), amplitude damping enters through
    sqrt(1-gamma), and `apply_coherent` is trigonometric -- so a low-order rule
    such as n = 2 leaves a bias of up to 5.6e-5.
    """

def pauli_string(string, n=5):
    """Build 2^n x 2^n Pauli from a string like 'XZZXI' (q0 = leftmost = MSB).
    """

def perfect_code_decoder_fidelity(psi_enc, rho_noisy):
    """Exact expected fidelity of the optimal single-error lookup decoder:
    
        F = sum_s <psi| C_s P_s rho P_s C_s^dag |psi>     with P_s the
    syndrome-s projector (rank 2) and C_s the rigid Pauli correction.
    Uses precomputed P_SYNDS / C_SYNDS.
    """

def plot_curves(p_values, results, title, fname, ylabel='Avg. fidelity'):

def plot_ler(p_values, results, title, fname):

def plot_summary_bar(p_values, results, p_idx, fname, noise_label):

def plot_training(hist_real, hist_rand, fname):

def quad_cache(p_range, noise, n_u=3, n_phi=7, n_p=6):
    """Precompute everything the exact quadrature objective needs, ONCE.
    
    Returns a dict Q with
      'psi'  (nq, 32) complex128   encoded probe states V|psi_j>   (constant)
      'rho'  (nq, 32, 32)          E_p(V|psi_j><psi_j|V^†)         (constant)
      'w'    (nq,) float64         Haar x noise-strength weights, sum = 1
      'W'    (16, 32, 2)           orthonormal syndrome bases      (constant)
      'BS'   (16, nq, 2, 2)        W_s^† rho_j W_s                 (constant)
      'p_syn' (16,) float64        syndrome probabilities sum_j w_j Tr[B_sj]
      'meta' dict                  node bookkeeping
    nq = n_u * n_phi * n_p = 126 for the defaults.  The channel is therefore
    applied exactly once per training stage instead of once per epoch per
    sample, and `BS` folds the density matrices down to the 2x2 syndrome
    blocks so that a training epoch only propagates 2 columns per syndrome
    through the 60-gate ansatz.
    """

def random_logical_state(n=1, rng=None):
    """Uniform-Haar random n-qubit logical pure state as a length-2^n complex tensor.
    """

def recovery_action_cols(phi_batch, X):
    """(B, PHI_DIM) real angles, (B, DIM, r) complex columns -> R(phi_b) X_b.
    
    Identical to `recovery_unitary_batch(phi_batch) @ X` but O(PHI_DIM * DIM^2 * r)
    instead of O(PHI_DIM * DIM^3), and differentiable w.r.t. phi_batch.
    """

def recovery_action_cols_dag(phi_batch, X):
    """(B, PHI_DIM), (B, DIM, r) -> R(phi_b)^† X_b  (adjoint column propagation).
    
    R(phi) = G_{59} ... G_0 with G_k = exp(-i phi_k P_k / 2), so
    R^† = G_0^† G_1^† ... G_59^† and the gates must be applied in REVERSE order
    with the sign of the sin term flipped.  Verified against
    `recovery_unitary_batch(phi).conj().transpose(1,2) @ X` in
    `_selftest_action_cols`.
    """

def recovery_unitary(phi):
    """phi: real tensor of length PHI_DIM -> 32x32 complex unitary R.
    """

def recovery_unitary_batch(phi_batch):
    """phi_batch: (B, PHI_DIM) real -> (B, 32, 32) complex unitaries (batched).
    """

def syndrome_bases():
    """W[s] = (32, 2) orthonormal basis of range(P_s), the rank-2 syndrome space.
    
    Only needed by the fast quadrature loss.  The final objective is invariant
    under the choice of basis because it enters exclusively through
    W_s W_s^† = P_s (see `vscr_fidelity_quad`).
    """

def syndrome_conditional_fidelity(model, noise, p=0.07, M=150, rng_seed=1239):
    """LABEL-FREE per-syndrome diagnostic: the mean recovery fidelity
    CONDITIONED on each syndrome occurring,
    
        cf_s = E[ <psi| R_s P_s rho P_s R_s^dag |psi> ] / E[ Tr[P_s rho] ].
    
    A healthy instrument has cf_s close to 1 for every syndrome that occurs
    with non-negligible probability.  Used to detect the occasional syndrome
    that falls into a bad local optimum and to select the best seed.  No
    error labels or decoder table are involved.
    """

def syndrome_conditional_fidelity_exact(phi, Q):
    """EXACT (zero-variance) label-free per-syndrome conditional fidelity.
    
        cf_s = sum_j w_j <psi_j| R_s P_s rho_j P_s R_s^dag |psi_j> / p_s
             = F_s / p_s ,
    
    i.e. the same quantity as `syndrome_conditional_fidelity` but with the
    Haar average and the syndrome probabilities both replaced by the exact
    quadrature in `Q`.  This matters: the Monte-Carlo version uses M = 150
    probe states, so its sem is ~1e-3 -- an order of magnitude LARGER than the
    1.9e-4 non-Pauli headroom the warm start is being selected on, which makes
    seed selection essentially a coin flip.  Returns (cf, p_s) as float64
    arrays of shape (16,).
    """

def train_lindr(n_train=1500, p_range=(0.01, 0.12), noise='depolarizing', n_val=200, rng_seed=1333):
    """Ridge least-squares fit  A : noisy-Pauli-features -> ideal-Pauli-features.
    
    The ridge strength is chosen on a held-out validation set by
    reconstruction fidelity.  Returns A with the column convention y = A @ x.
    """

def train_vscr(model, n_epochs=400, batch=48, p_range=(0.02, 0.15), noise='depolarizing', lr=0.01, lam=0.1, rng_seed=1234, use_real_grad=True, verbose=True, synd_w=None, quad=False):
    """Train VSCR.  If use_real_grad=False, parameters get RANDOM Gaussian
    updates (replicating the broken ACE-QEC reference in main.py) to ablate
    the importance of genuine gradients.  synd_w: optional per-syndrome loss
    weights (gradient balancing; see vscr_fidelity).
    
    quad=True replaces the Monte-Carlo probe-state average by the EXACT
    deterministic Haar/noise-strength quadrature of `haar_quadrature` /
    `p_quadrature` (see the comment block above those functions), evaluated by
    `vscr_fidelity_quad`.  Two reasons this matters for Fig.3/Fig.4:
    
      * Accuracy.  The signal those figures are meant to resolve -- the
        non-Pauli headroom of the optimal syndrome-conditioned unitary over the
        rigid Pauli decoder -- is 1.9e-4 (amplitude damping, p=0.06) to 3.4e-3
        (coherent / depolarizing).  The measured per-sample spread of the
        recovery fidelity gives a 48-sample Monte-Carlo SEM of 1.1e-4 at
        amplitude damping p=0.10 and 3.2e-5 at p=0.05, i.e. the estimator noise
        is the same order as the effect for exactly the channel where the effect
        is smallest.  The quadrature gradient is zero-variance and zero-bias.
      * Speed.  The 126 (psi, rho) pairs are cached once per call, and because
        <psi|R_s P_s rho P_s R_s^dag|psi> only ever probes the 2-dimensional
        syndrome subspace, the 60-gate ansatz is applied to 2 columns instead of
        a full 32x32 unitary (see `vscr_fidelity_quad`).
    
    The default remains False so that pre-existing Monte-Carlo runs reproduce
    bit-for-bit.
    
    The p = 0.06 model-selection criterion (`best_val`) is evaluated with the
    exact quadrature in BOTH modes, so MC and quad runs are selected by the
    same deterministic, label-free number.
    """

def train_vscr_best(noise, seeds=(1234, 2024, 777), schedule=None, batch=48, lam=0.0, verbose=True, val_gate=0.85):
    """Train VSCR for several seeds and keep the best model.
    
    Selection is LABEL-FREE: candidates that actually trained (validation
    fidelity >= val_gate) are ranked by the worst per-syndrome conditional
    fidelity (tie-break: validation fidelity); if no candidate passes the
    gate, the highest validation fidelity is used.  Rationale: training
    occasionally leaves ONE syndrome in a bad local optimum (which syndrome
    depends on the seed); with >=3 seeds, a candidate that is healthy on ALL
    syndromes is available.  The gate rejects degenerate runs whose cf values
    are uniformly mediocre (flat), which would otherwise inflate min_cf.
    """

def virtual_distillation_fidelity(psi_enc, rho_noisy, k=2):
    """rho_VD ~ rho^k (unnormalised) projected toward the purest component.
    """

def vscr_fidelity(psi_enc, rho_noisy, R_all, lam=0.0, synd_w=None):
    """Expected recovery fidelity given the 16 recovery unitaries R_all:
    
    F = sum_s <psi| R_s P_s rho P_s R_s^dag |psi>     (exact expectation
    over the projective syndrome outcome).  Vectorised over 16 syndromes.
    R_all can be SHARED across samples (built once per epoch/eval).
    
    synd_w: optional length-16 weight vector rescaling the per-syndrome
    terms (training-time gradient balancing; the optimum is unchanged
    because each R_s only appears in its own syndrome term).
    """

def vscr_fidelity_quad(phi_batch, Q, synd_w=None):
    """EXACT Haar-quadrature recovery fidelity, vectorised over syndromes AND
    probe states, propagating only TWO columns per syndrome.
    
        F = sum_j w_j sum_s <psi_j| R_s P_s rho_j P_s R_s^dag |psi_j>
    
    Writing P_s = W_s W_s^dag and G_s = R_s W_s (32x2),
    
        u_{sj} = P_s R_s^dag psi_j = W_s G_s^dag psi_j = W_s z_{sj},
        term   = u^dag rho_j u = z_{sj}^dag B_{sj} z_{sj},   B_{sj} = W_s^dag rho_j W_s,
    
    so the 60-gate ansatz is applied to the 2 columns of W_s ONLY, and every
    rho_j-dependent quantity (B) is a constant precomputed once per training
    stage by `quad_cache`.  This is the whole point: the exact objective costs
    ~1/20 of the Monte-Carlo path per epoch *and* has zero gradient variance,
    whereas the non-Pauli headroom it is meant to resolve (1.9e-4 .. 3.4e-3
    over the Pauli decoder) sits right at the 48-sample Monte-Carlo SEM
    (1.1e-4 for amplitude damping at p = 0.10).
    
    Returns (loss, F, F_s) with F_s the exact per-syndrome averaged fidelity
    (used for the label-free worst-syndrome model-selection criterion).
    
    NOTE: the code-population regulariser (`lam` in `vscr_fidelity`) needs the
    full 32x32 R_s and is not supported here; every paper configuration trains
    with lam = 0, which `train_vscr` asserts.
    """

def vscr_forward(model, psi_enc, rho_noisy, lam=0.1):
    """Convenience wrapper: build R_all from the model, then score.
    """

def zne_fidelity(psi_enc, p, noise):
    """Richardson ZNE at noise scales 1,2,3 (state-level extrapolation).
    """

==============================================================================
# MODULE vscr_paper
==============================================================================


class VSCRWarm:   # bases: ['Module', 'object']
    def __init__(self, n_synd=16, phi_dim=60, hidden=64, ctx=24, emb_scale=0.5):
        """Initialize internal Module state, shared by both nn.Module and ScriptModule.
        """
    def forward(self):
    def syndrome_spread(self):
        """max_{s,s'} ||hnet(e_s) - hnet(e_s')||_inf : the size of the learned
        SYNDROME-DEPENDENT deviation from the decoder.  Exactly 0 for a dead
        hypernetwork (the old double-zero-init bug), O(1e-3..1e-1) when healthy.
        NOTE: this deliberately excludes `phi_dec`, which already differs across
        branches, and `phi_base`, which is syndrome-independent by construction.
        """
    def branch_deviation(self):
        """max_s ||phi_s - phi_dec[s]||_inf : total learned correction size.
        """
    def set_phi(self, phi_table):
        """Install an explicit per-syndrome angle table via `phi_extra`.
        
        The refined table produced by `refine_per_syndrome` is a free
        (16, PHI_DIM) array, which need not lie in the image of `hnet`; storing
        the difference in the residual buffer makes `forward()` return it
        exactly, so `evaluate_paper`, `cf_table`, `hardware_frame_numbers` and
        the plotting code all see the refined recovery unchanged.
        """

def _psd_project(rho):

def _save_all(curves, hists, hist_rw, cf, infos_all, hw, t0):

def _selftest_hypernet_trainable(noise='amplitude_damping', steps=60, batch=12):
    """Guard against re-introducing the dead-hypernetwork bug.
    
    Asserts (a) the warm start is exactly the decoder, (b) EVERY trainable
    parameter of the syndrome pathway moves under the real label-free loss, and
    (c) the 16 branch angle sets become mutually distinct.  Without (b)/(c) the
    model can only apply one global rotation, and both the "syndrome-conditioned
    recovery" claim and the K-dependence of the low-data ablation are vacuous.
    """

def _verify_warm_start():

def cf_table(R_all, noise, p, M=400, rng_seed=1239):
    """Label-free per-syndrome conditional fidelity cf_s and weights w_s.
    """

def decoder_angles():
    """(16, PHI_DIM) angles realising R_s = C_s (Pauli decoder) up to phase.
    Layer-1 slots per qubit q: [Rz(3q), Rx(3q+1), Rz(3q+2)]; X -> Rx(pi),
    Z -> Rz(pi), Y -> Rz(pi)Rx(pi) (phase-irrelevant).
    """

def evaluate_paper(model_warm, A_lindr, p_values, noise, n_test=200, seed=1241, cold_R=None):
    """method -> {'F': mean, 'F_sem': sem, 'LER': mean, 'LER_sem': sem}.
    
    Baselines are all scored on PHYSICAL states (ZNE-phys, LinDR-phys) so that
    F <= 1 for every method.  On channels listed in VD_DEGENERATE_FOR the VD
    baseline is omitted after asserting F_VD == F_Raw to machine precision
    (rho^2 = rho for a unitary channel), instead of plotting a duplicate curve.
    Returns (out, audit) where audit records the VD/Raw degeneracy check.
    """

def hardware_frame_numbers(phi_warm_dep):
    """Exact ideal reference for the future full QPU benchmark, computed with
    the NEW warm angles: for psi in {0,+} and all 16 single-error Pauli
    frames, the per-frame circuit fidelity F = sum_s F_s and the per-branch
    F_s (the quantity the hardware post-selection estimates).  Also
    re-verifies circuit == density-matrix simulator to <1e-9.
    """

def lindr_phys_fidelity(psi_enc, rho_noisy, A):

def main():

def plot_benchmark(curves_by_noise, fname='fig_sim_benchmark', p_vals=None):

def plot_branch_cf(cf_dec, cf_cold, cf_warm, fname='fig_branch_cf'):

def plot_ler(curves_by_noise, fname='fig_sim_ler', p_vals=None):

def plot_training(hist_warm, fname='fig_training'):
    """Warm curve (this run) + cold real-grad + cold random-walk (v1 npz).
    """

def save_fig(fig, name):

def train_lindr_phys(n_train=1500, p_range=(0.01, 0.12), noise='depolarizing', n_val=200, rng_seed=1333):
    """Same ridge fit as ssvr_qec.train_lindr but the ridge strength is
    selected on PHYSICALLY PROJECTED (PSD, trace-1) validation fidelity.
    """

def train_warm_best(noise, seeds=(1234, 2024), verbose=True, quad=True):
    """Multi-seed warm training with the same LABEL-FREE selection as the
    prototype: rank by worst per-syndrome conditional fidelity.
    
    quad=True (default) makes the whole pipeline exact and fast:
    
      * every epoch's objective is the deterministic Haar x noise-strength
        quadrature of `ssvr_qec.quad_cache`, propagated through only the
        2-dimensional syndrome subspace -- 1.5 ms/epoch against 67 ms for the
        equivalent full-32x32 reference loop (47x), and the gradient has zero
        variance, which is what makes a 1.9e-4 non-Pauli headroom resolvable
        instead of being buried in a 48-sample Monte-Carlo SEM of 1.1e-4;
      * the seed-selection diagnostic uses
        `ssvr_qec.syndrome_conditional_fidelity_exact` rather than the M=150
        Monte-Carlo version, whose ~1e-3 sem is larger than the effect.
    
    quad=False restores the original Monte-Carlo path bit-for-bit.
    """

def zne_phys_fidelity(psi_enc, p, noise):
    """Richardson ZNE (scales 1,2,3) with the extrapolated operator projected
    onto the nearest physical (PSD, trace-1) state before scoring.  Identical
    treatment to LinDR-phys; without it the reported "fidelity" is an unbounded
    estimator artefact that exceeds 1 on the coherent channel.
    """

def zne_unphysical_overshoot(p_values, noise, n_test=60, seed=1245):
    """Audit of the raw (unprojected) ZNE estimator: returns {p: max F} where
    F = <psi|rho_zne|psi> may exceed 1.  Recorded in paper_numbers.json so the
    reader can see exactly why the physical projection was introduced.
    """

==============================================================================
# MODULE vscr_paper_abl
==============================================================================


class GlobalRec:   # bases: ['Module', 'object']
    def __init__(self):
        """Initialize internal Module state, shared by both nn.Module and ScriptModule.
        """
    def forward(self):

class VQRInd:   # bases: ['Module', 'object']
    def __init__(self, warm=True):
        """Initialize internal Module state, shared by both nn.Module and ScriptModule.
        """
    def forward(self):

def _G_of_h(h):

def _W_basis():
    """(16, 32, 2) orthonormal bases of the syndrome subspaces.
    """

def _branch_A(noise, p, s):
    """2x2 reduced branch operators {A_k = W_s^† K_k V}.
    """

def _branch_M(noise, p, s):
    """Exact Haar-averaged objective matrix M_s and branch weight p_s.
    """

def _branch_qform(noise, p, s):
    """Q_s (4x4), the G-independent constant Tr(sum_k A_k^† A_k), and p_s.
    """

def _decoder_ref_F(noise, p, n=200, seed=1241):

def _kraus_ad(gamma):

def _kraus_coherent(eps):

def _kraus_dep(p):
    """1024 Kraus ops of the per-qubit-sequential depolarizing channel.
    """

def _lmin_grad(A, lifts):
    """lambda_min(A) and d lambda_min / d y_k = Re Tr[v v^dag  dA/dy_k].
    
    lambda_min is a concave function of a Hermitian matrix, and its subgradient
    at a simple eigenvalue is the rank-one projector onto the eigenvector.  Both
    constraints of the dual below are of this form, so each feasible set
    {y : lambda_min(A(y)) >= 0} is CONVEX and SLSQP sees exact gradients.
    """

def _sdp_branch(M, G_dec, G_ref=None):
    """Max 2Tr[CM] s.t. C=LL^† ⪰ 0, I/2 − Tr_out C ⪰ 0.  Returns F_unnorm.
    
    All three functions are supplied to scipy together with their EXACT
    closed-form gradients from `_sdp_valgrad` (see its docstring for the ~40x
    speedup and the feasibility argument); `fun` and `jac` share one memoised
    evaluation because scipy calls them back to back at the same point.
    
    Returns (F_unnorm, x_star) where x_star is the length-32 real parameter
    vector attaining F_unnorm, so callers can re-verify feasibility.
    """

def _sdp_dual_bound(M, ntry=8):
    """RIGOROUS upper bound on `_sdp_branch`'s optimum, via LP/SDP duality.
    
    The branch problem is, in terms of the Choi matrix C = L L^dag directly,
    
        (P)  max 2 <M, C>   s.t.  C >= 0,   Tr_out(C) <= I/2 .
    
    Its Lagrangian dual, with multiplier Y >= 0 on the trace constraint and
    using Tr[Y Tr_out(C)] = <kron(Y, I_2), C>, is
    
        (D)  min (1/2) Tr[Y]  s.t.  Y >= 0,   kron(Y, I_2) - 2M >= 0 .
    
    Weak duality gives <P> <= <D> for every DUAL-FEASIBLE Y, no matter how (P)
    was solved.  So this is a certificate, not a second opinion: if the
    multi-start SLSQP primal had stalled in a local trap, (P) < (D) by a visible
    margin.  Slater's condition holds for (P) (C = eps*I is strictly feasible),
    so strong duality applies and the true gap is zero -- measured worst gap over
    25 branches x 5 channels is < 1e-9.
    
    This replaces the trust-constr "polish" that used to run on every branch.
    trust-constr needed Hessians, and `c_det = Re det[I/2 - Tr_out(L L^dag)]` is
    QUARTIC in L (det of a matrix that is itself quadratic in L), not quadratic
    as an earlier version of this file assumed -- so the "constant" Hessian it
    was handed was simply wrong, off by exactly 2.0 at a non-zero point.  A
    wrong Hessian is worse than none: it silently biases the trust region.  It
    also never improved on SLSQP (both returned F = 0.014085885 on
    depolarizing p=0.05, branch s=7) while costing ~7x more per solve, and the
    finite-difference Hessian path scipy falls back on is what segfaulted
    Section B.  Dual certification is both cheaper and mathematically stronger.
    """

def _sdp_valgrad(x, M):
    """EXACT values and gradients of the three functions `_sdp_branch` optimises.
    
        L     = (x[:16] + i x[16:]).reshape(4,4)      Choi factor, C = L L^dag
        obj   = 2 Re Tr[L L^dag M]                                 (maximised)
        T     = I/2 - Tr_out(C)   (2x2 Hermitian)
        c_tr  = Re Tr[T]                                           (>= 0)
        c_det = Re det[T]                                          (>= 0)
    
    Closed forms.  Writing L = X + iY and P = A L, the identity
    `(L^dag A)_{ba} = conj((A^dag L)_{ab})` gives, for any fixed matrix A,
    
        d Re Tr[L^dag A dL] = Re sum_ab conj((A^dag L)_ab) dL_ab ,
    
    and with dL = dX + i dY the real part of conj(P) dL is
    Re(P) dX + Im(P) dY.  Applying this three times:
    
    * obj = 2 Re Tr[L^dag M dL]|_{dL->L}: since M is Hermitian, A^dag = M and
          grad_X obj = 4 Re(ML),   grad_Y obj = 4 Im(ML).
    * c_tr = 1 - Tr[C] = 1 - ||L||_F^2 exactly (Tr[I/2] = 1 and the trace of a
      partial trace is the trace), hence
          grad_X c_tr = -2X,       grad_Y c_tr = -2Y.
    * c_det: d det[T] = sum_ab adj(T)^T_ab dT_ab and dT = -dS with
      S = Tr_out(C), so with Ghat = adj(T)^T and K = Ghat (x) I_2 (the 4x4 lift
      satisfying sum_ab Ghat_ab dS_ab = Tr[K^T dC]) one gets
          d c_det = -2 Re sum_ab conj((K^T L)_ab) dL_ab ,
      i.e. grad_X c_det = -2 Re(K^T L), grad_Y c_det = -2 Im(K^T L).
      (K is Hermitian because T is, so K^* = K^T and the two dL and dL^dag
      terms combine into twice the real part.)
    
    x is the length-32 real parameter vector.  Returns
    (obj, g_obj, c_tr, g_c_tr, c_det, g_c_det), each g_* a length-32 float64
    array in the SAME layout as x.
    
    Why this exists at all: scipy was differentiating these three scalars by
    finite differences, i.e. 3 x 33 = 99 extra evaluations per Jacobian, inside
    ~22 solves per syndrome x 16 syndromes x every (noise, p) point -- about
    10 min per `sdp_ceiling` call and ~70 min for the E3 sweep in `main`.  The
    exact gradient is both ~50x faster and more reliable: SLSQP's default
    finite-difference step is at the level where the trace/det constraints are
    already saturated, so the numerical gradient routinely lost feasibility and
    forced the solver to `maxiter`.
    """

def _sdp_valgrad_autograd(x, M_t):
    """Reference implementation of `_sdp_valgrad` via torch autograd.
    
    Kept ONLY as an independent cross-check: `_selftest_sdp` (g) asserts the
    closed-form gradients below agree with these to 1e-12.  Autograd is the
    definition-correct way to differentiate the three functions but costs
    ~1.9 ms per call, which dominates a solve that does hundreds of them, so
    the production path uses the closed form (~20 us).
    """

def _selftest_refine(noise='amplitude_damping', n_start=3, steps=900, p_range=(0.02, 0.15), p_test=0.06):
    """Prove the separable refinement is EXACT and captures the headroom.
    
    Test 1 (identity): `pavg_F` -- which cancels p_s against cf_s and splits the
    objective into 16 independent blocks -- must reproduce the PRODUCTION
    training objective `ssvr_qec.vscr_fidelity_quad` on `quad_cache(p_range,
    noise)`.  If these disagreed, the refinement would be optimising something
    other than what training and Fig.3/Fig.4 report.
    
    Test 2 (stationarity): the exact gradient of the p=0.06 training objective
    w.r.t. all 960 warm-start angles must vanish at PHI_DEC to round-off.  This
    is the mechanism behind 0% headroom capture -- the decoder is a stationary
    point, not a merely flat region -- so it is asserted, not just observed.
    
    Test 3 (monotone + capture): `refine_per_syndrome` must never lower F̄, must
    stay at or below the certified ceiling F̄_unit, and must capture most of the
    certified headroom F̄_unit - F̄_dec.
    """

def _selftest_sdp():

def channel_kraus(noise, p):

def eval_global(phi, noise, p_values, n_test=None, seed=1241):
    """F̄ ± sem of a single global unitary (no projection).
    """

def eval_phi_table(phi_table, noise, p_values, n_test=None, seed=1241):
    """F̄ ± sem of the syndrome-projected recovery R_s = ansatz(phi_s).
    """

def exact_F(phi_table, noise, p):
    """Noiseless Haar-averaged F̄ of an arbitrary per-syndrome unitary table.
    
    phi_table is (16, PHI_DIM) of ansatz angles; R_s = ansatz(phi_s).  Uses the
    closed second-moment identity
        F_s = [ sum_k |Tr(G_s A_k)|^2 + Tr(sum_k A_k^† A_k) ] / 6 / p_s ,
        G_s = V^† R_s W_s ,
    so the result has ZERO Monte-Carlo error -- gains of order 1e-5 (the exact
    coherent-over-rotation headroom) are resolvable, which the n_test=400
    protocol in `eval_phi_table` cannot do.  Cross-checked against that MC
    evaluator and against `sdp_ceiling`'s branch data in `_selftest_sdp`.
    """

def main():

def opt_unitary_branch(noise, p, s, n_start=16):
    """Exact max over 2x2 unitaries of J(G) = sum_k |Tr(G A_k)|^2 for branch s.
    
    Returns (J_max, G_opt).  Single-Kraus branches use the polar-factor closed
    form, cross-checked against the nuclear norm, so they are provably global;
    multi-Kraus branches use multi-start L-BFGS-B over the Lie algebra plus a
    derivative-free Nelder-Mead polish, seeded with the identity, the decoder
    unitary, the 6 Pauli/phase matrices and random points.
    """

def opt_unitary_ceiling(noise, p):
    """F̄ of the best PER-BRANCH UNITARY recovery (the VSCR family optimum).
    
    Returns {'F_unit', 'F_dec', 'cf_unit', 'cf_dec', 'p_s', 'G_opt'}.  F_dec is
    the analytic decoder fidelity (validated against Monte-Carlo in
    `_selftest_sdp`), so F_unit - F_dec is the exact, noiseless non-Pauli
    headroom available to a trained warm start.
    """

def pauli_lookup(noise, p, M=200, seed=1239):
    """Unsupervised exhaustive search over Pauli recoveries per branch.
    Any Pauli P with synd(P) != s maps image(P_s) orthogonal to the code
    space (fidelity 0); the 64 Paulis with synd(P) = s fall into 4 logical
    classes C_s · {I, X_L, Z_L, X_L Z_L} acting identically (up to phase)
    on the branch.  So the search is exactly 4 candidates per branch, and
    a brute-force check over random Pauli strings verifies the reduction.
    """

def pavg_F(phi_table, B):
    """Exact p-averaged F̄ and its 16 separable branch contributions.
    
    Returns (F̄_avg, obj) with obj[s] = the whole contribution of syndrome s, so
    sum(obj) == F̄_avg exactly.  `B` is a `pavg_bundle`.
    """

def pavg_bundle(noise, p_range=(0.02, 0.15), n_p=6):
    """Precompute the separable p-averaged objective for one channel.
    
    The training objective is the Gauss-Legendre p-average of F̄,
        F̄_avg = sum_p w_p sum_s p_s(p) cf_s(G_s; p) ,
    and because `cf_s = (J_s + const_s)/6/p_s` the branch weight CANCELS:
        p_s(p) cf_s(G_s; p) = (J_s(G_s; p) + const_s(p)) / 6 .
    Two consequences drive everything below.
    
    (1) SEPARABILITY.  The term for syndrome s involves only G_s, hence only
        R_s = ansatz(phi_s).  So F̄_avg = sum_s obj_s(phi_s) with the 16 blocks
        mutually independent -- even after the p-average, since reordering the
        two sums leaves each block a function of phi_s alone.  Optimising the
        blocks separately therefore reaches the GLOBAL optimum of the VSCR
        family, with no cross-syndrome coupling to coordinate.
    
    (2) CONDITIONING.  Dividing by p_s is what makes cf_s explode on rare
        syndromes; the cancelled form has no such division and stays exact even
        where p_s underflows.
    
    `_branch_qform` rebuilds the branch Kraus operators (about 0.3 s per call
    for depolarizing), so the 16 x n_p grid is materialised ONCE here and reused
    by every refinement start.  `_selftest_refine` asserts sum_s obj_s reproduces
    `ssvr_qec.vscr_fidelity_quad` on `quad_cache(p_range, noise)` to ~1e-14.
    """

def plot_ablation(bar_vals, lowdata, fname='fig_ablation'):

def refine_per_syndrome(phi_init, noise, p_range=(0.02, 0.15), n_p=6, n_start=3, steps=1500, lr=0.01, seed=0, bundle=None, verbose=False):
    """Escape the decoder saddle by optimising each syndrome INDEPENDENTLY.
    
    WHY THIS IS NEEDED.  Each branch term is a 4x4 Rayleigh quotient,
        obj_s ∝ g_s^† Q_s g_s ,   g_s = vec(V^† R_s W_s) ,  ||g_s|| fixed,
    and the gradient of a Rayleigh quotient vanishes EXACTLY at the eigenvectors
    of Q_s.  The Pauli decoder's vec(G_dec) is such an eigenvector on every
    branch of every channel (measured eigen-residual ~1e-16 on depolarizing; on
    coherent the decoder sits in Q's null space, i.e. an eigenvector with
    eigenvalue ~0).  So `PHI_DEC` is an exact stationary point of the warm-start
    objective -- not a small-gradient region but a true zero, confirmed by
    central finite differences as well as autograd.  Wherever
    λ(G_dec) < λ_max(Q_s) the decoder is a SADDLE, and the certified headroom
    F̄_unit − F̄_dec is precisely that eigenvalue gap.  No first-order method
    started at PHI_DEC can move off it, which is why warm training captured 0%
    of the headroom while `Adam` still visibly moved the parameters: it was
    normalising round-off-level gradients and drifting, not descending.
    
    Small symmetry-breaking perturbations do NOT fix this -- measured captured
    headroom of −0.52% to −0.03% for eps in [1e-4, 1e-1], because a random
    60-angle offset is almost entirely in the ansatz's null directions and Adam
    then random-walks.  What DOES work is exploiting separability: optimise each
    of the 16 blocks on its own from a few LARGE symmetry-broken starts.  The
    ansatz is provably expressive enough -- on amplitude damping p=0.06 branch 2
    it reaches `opt_unitary_ceiling`'s certified optimum to −1.0e−15 -- and the
    full table then captures 99.9999% of the certified headroom
    (F̄ 0.994310805 → 0.994994328 against F̄_unit = 0.994994328).
    
    Start 0 is always `phi_init[s]` and a candidate is only accepted if it
    strictly improves that block's objective, so the result can NEVER be worse
    than the input table -- the refinement is monotone by construction.
    
    Returns (phi, info); phi is (16, PHI_DIM).
    """

def sdp_ceiling(noise, p, ntry_dual=4):
    """F̄^CPTP ceiling and per-branch data for a channel at strength p.
    
    BUG FIX: this used to take `fu, _ = _sdp_branch(...)` unconditionally and
    divide by p_s.  `_sdp_branch` returned -1.0 whenever SLSQP reported
    `success=False` (which it does at many perfectly converged KKT points, and
    always once it hits `maxiter`), so `F_s_max` became -1/p_s -- e.g. the
    amplitude-damping p=0.30 ceiling was printed as F_CPTP = -0.1113, a
    negative fidelity, and the derived "non-Pauli headroom" was meaningless.
    The solver now never returns below its feasible starts, and this function
    additionally seeds it with the exact unitary-family optimum and asserts
    F̄^CPTP >= max(F̄^unitary, F̄^decoder).
    
    Every branch is now also CERTIFIED globally optimal by `_sdp_dual_bound`:
    `gap_max` in the returned dict is the largest |dual − primal| over the 16
    branches and must be ~0.  A large POSITIVE gap means the multi-start SLSQP
    stalled in a local trap (the primal is not provably optimal); a large
    NEGATIVE one would violate weak duality and means a dual-infeasible Y was
    accepted.  Both are failures, so the reported quantity is the absolute gap;
    the per-branch weak-duality direction is asserted separately below.
    """

def train_global(noise, seed=1234):

def train_ind(noise, seeds=(1234,)):

def train_pool(make_model, pool, noise, seed=1234, epochs=None, lr=0.003, p_range=(0.02, 0.15)):
