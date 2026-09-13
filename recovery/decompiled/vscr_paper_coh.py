# Source Generated with Decompyle++
# File: vscr_paper_coh.cpython-311.pyc (Python 3.11)

'''vscr_paper_coh.py — coherent-control-error channel supplement.

Trains the warm-started VSCR on a systematic coherent over-rotation channel
(identical Rx(eps) on all five data qubits; NOT a Pauli mixture) and
benchmarks it against the perfect-code Pauli decoder, the cold-started v1
model and the raw state.  For incoherent (Pauli-mixture) noise the decoder
point is a stationary point of the Haar-averaged branch-fidelity loss, so
VSCR reproduces it exactly; for coherent errors the learned non-Pauli
corrections can and do exceed rigid Pauli decoding.

Outputs: vscr_angles_paper_coh.npz, paper/figures/fig_coherent.pdf|png,
and merges coh_* fields into paper_numbers.json.
'''
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import pyplot as plt
import ssvr_qec as m
import vscr_paper as vp
vp.WARM_SCHEDULES['coherent'] = [
    (400, 0.003, (0.02, 0.08)),
    (500, 0.003, (0.02, 0.15)),
    (800, 0.0003, (0.02, 0.15))]

def main():
    pass
# WARNING: Decompyle incomplete

if __name__ == '__main__':
    main()
    return None
