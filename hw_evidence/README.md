# hw_evidence/ -- the run logs behind the manuscript's hardware section

`*.log` is gitignored everywhere else in this repository, and this directory is the
deliberate exception (see `.gitignore`).  The reason is that a paid QPU submission
produces exactly one record of what it actually did, and that record is its log:

* the wall time and the shots returned **per chunk**, which is how a cut-off job is
  distinguished from a short-changed one;
* the `[budget]` line printed before the spend, i.e. the dashboard reading the
  submission was priced against and the rate it was priced at;
* the verdict printed at the time (`SCREEN: ... USABLE`, `VERDICT: A/B PASSED`,
  `VERDICT: BOUND EXPERIMENT`), which is what the next gate was allowed to rely on.

None of that is reconstructible from the artifact: `hw_mixedness.json` records the
*analysis*, and the raw dumps record the *counts*, but neither records what the
operator was told before spending, or what the account said afterwards.

## Naming

```
hw_evidence/<UTC>_<chip>_<gate>.log        e.g. 20261014T160604Z_WK_C180_wcprobe.log
```

`launch_mixedness.sh` creates the file, writes the header (chip, stage, gate flags,
`dashboard_before`, block, shots, angles, vendor table) and appends the run's stdout
and stderr to it.

## The dashboard protocol

The remaining allocation cannot be queried: `QCloudService` exposes
`backend`/`backends`/`best_qubit_blocks`/`setup_logging` and `ChipInfo` only topology
and calibration, with no balance, quota or usage call anywhere in the SDK.  So every
gate is bracketed by a human reading, and the pair of readings across a known shot
count is the **only measurement of the billed rate that exists**:

```
ms/shot = 1000 * (dashboard_before - dashboard_after) / shots_bought
```

Record the second reading in the log as soon as the job lands:

```bash
echo 'dashboard_after=<seconds>' >> hw_evidence/<the log>
```

and pass the resulting rate to the next gate as `MSPS=<ms/shot>`.  Until that is
done, every cost in this repository is priced at the pessimistic `MS_PER_SHOT = 14`,
which is not reproducible from anything on disk (the 2026-09 wall clock implies
~1.0 ms/shot) and is therefore an upper bound rather than a measurement.

## What is already here

The pre-2026-10-14 runs logged to `qcloud_mixedness_<stage>.log` in the repository
root, where `*.log` left them untracked.  Their surviving content is quoted
directly in `QCLOUD_SETUP.md` §8 and in the manuscript's hardware section; the
numbers those runs produced are frozen in `qcloud_raw_*.json` and in
`hw_mixedness.json` (whose previous records `archive_previous` preserves under
`hardware_run_archive` rather than overwriting).
