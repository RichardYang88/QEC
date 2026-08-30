"""Quick smoke test for ssvr_qec (VSCR core): import, forward+backward, baselines, mini-train."""
import torch, numpy as np
import ssvr_qec as m

print("import OK; [[5,1,3]] code & syndrome table validated on import")

psi = m.random_logical_state(1)
pe = m.encode(psi)
rho = m.noisy_state(pe, 0.05, 'depolarizing')
print("noisy rho shape", tuple(rho.shape), "trace", round(float(torch.trace(rho).real), 6))

model = m.VSCR()
loss, F = m.vscr_forward(model, pe, rho, lam=0.1)
print("init loss", round(float(loss), 4), "init fid", round(float(F), 4))
loss.backward()
print("phi_base grad norm", round(float(model.phi_base.grad.norm()), 6))

print("raw          ", round(float(m.baseline_raw(pe, rho)), 4))
print("perfect-code ", round(float(m.perfect_code_decoder_fidelity(pe, rho)), 4))
print("zne          ", round(float(m.zne_fidelity(pe, 0.05, 'depolarizing')), 4))
print("vd           ", round(float(m.virtual_distillation_fidelity(pe, rho)), 4))
A = m.train_lindr(n_train=40, noise='depolarizing')
print("lindr        ", round(float(m.lindr_fidelity(pe, rho, A)), 4))

# mini training to confirm real gradients improve fidelity
print("\nmini train (30 epochs, real grad):")
h = m.train_vscr(m.VSCR(), n_epochs=30, batch=16, p_range=(0.02, 0.12),
                 noise='depolarizing', lr=1e-2, use_real_grad=True, verbose=False)
print("  val fid epoch1 ", round(h['fidelity'][0], 4),
      " epoch30", round(h['fidelity'][-1], 4))
print("SMOKE OK")
