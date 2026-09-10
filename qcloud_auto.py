#!/usr/bin/env python
"""OriginQ auto-retry daemon.

Real chips answer "Quantum computer under maintenance" at night.  This daemon
periodically probes the available chips with a REAL 9-qubit branch circuit
(s=0, frame IIIII, psi=|0_L>, 100 shots) -- which simultaneously validates
mid-circuit-measurement acceptance by the amend/mapping pipeline -- and as
soon as one chip accepts a task, launches the full experiment:

    qcloud_vscr.py --mode cloud --chip-id <chip> --shots 4000 \
        --p-values 0.02,0.08 --frames 2 --states 0,+   (batch; --no-batch retry)

Usage:  nohup ./qenv/bin/python -u qcloud_auto.py > qcloud_auto.log 2>&1 &
"""
import subprocess
import sys
import time
import datetime

BASE = '/home/yqc/github/QEC'
PY = f'{BASE}/qenv/bin/python'
TOK = open(f'{BASE}/.originq_token').read().strip()
CHIPS = [2, 72, 5]            # 悟源D5, 悟空72, 悟源D4  (chip 7: resource is null)
RETRY_SEC = 15 * 60           # probe every 15 min while under maintenance
PROBE_SHOTS = 100


def log(msg):
    print(f'[{datetime.datetime.now():%m-%d %H:%M:%S}] {msg}', flush=True)


def main():
    sys.path.insert(0, BASE)
    import qcloud_vscr as q
    q.setup()
    phi = q.load_angles()
    log(f'probe circuit: {len(q.build_branch_circuit(q.ENC5, "0", "IIIII", 0, phi))} gates on {q.NQ} qubits')

    chip_ok = None
    late_mode = False
    attempt = 0
    while chip_ok is None:
        attempt += 1
        for chip in CHIPS:
            for late in (False, True):
                log(f'attempt #{attempt}: probing chip {chip} (late_measure={late}) ...')
                gates0 = q.build_branch_circuit(q.ENC5, '0', 'IIIII', 0, phi,
                                                late_measure=late)
                try:
                    res = q.submit_cloud([gates0], TOK, chip_id=chip,
                                         shots=PROBE_SHOTS, batch=False,
                                         dump_raw=f'{BASE}/raw_probe_chip{chip}.json')
                    log(f'CHIP {chip} ACCEPTED TASK (late_measure={late}) -> counts {res[0]}')
                    chip_ok, late_mode = chip, late
                    break
                except Exception as e:                  # noqa: BLE001
                    msg = str(e)
                    log(f'chip {chip} late={late}: {type(e).__name__}: {msg}')
                    busy = ('maintenance' in msg or 'resource is null' in msg
                            or 'busy' in msg.lower())
                    if busy:
                        break          # chip-level issue: late variant won't help
                    # non-maintenance error (e.g. compiler rejects mid-circuit
                    # measurement): fall through and try the late variant
            if chip_ok is not None:
                break
        if chip_ok is None:
            log(f'all chips busy/maintenance; retry in {RETRY_SEC // 60} min')
            time.sleep(RETRY_SEC)

    # ---- full experiment on the responsive chip ----
    common = ['--mode', 'cloud', '--token', TOK, '--chip-id', str(chip_ok),
              '--shots', '4000', '--p-values', '0.02,0.08', '--frames', '2',
              '--states', '0,+', '--dump-raw', f'{BASE}/qcloud_raw_full.json',
              '--tag', f'full_chip{chip_ok}']
    if late_mode:
        common.append('--late-measure')
    for extra in ([], ['--no-batch']):
        cmd = [PY, f'{BASE}/qcloud_vscr.py'] + common + extra
        log('launching: ' + ' '.join(a if a != TOK else '<TOKEN>' for a in cmd))
        rc = subprocess.call(cmd, cwd=BASE)
        log(f'qcloud_vscr.py exited with code {rc}')
        if rc == 0:
            log('FULL EXPERIMENT COMPLETED')
            return 0
        log('retrying without batch ...' if not extra else 'both attempts failed')
    return 1


if __name__ == '__main__':
    sys.exit(main())