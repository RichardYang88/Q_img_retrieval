# Q_img_retrieval
quantum image retrieval system

## Files
- `AE-QIP.ipynb` — original notebook (as in the manuscript).
- `AE-QIP-fixed.ipynb` — corrected, runnable copy (Qiskit >= 2.0): fixed `Us_gate`
  state-preparation circuit, `aux_qubit = 8`, `shots = 1024`, and a
  maximum-likelihood read-out; reports `Err rate` below 0.1 % on all six pairs.
- `ae_qip_fixed.py` — standalone corrected implementation with a self-test:
  `python3 ae_qip_fixed.py --selftest` (exact state preparation, ~1e-16),
  `python3 ae_qip_fixed.py --h 8 --shots 1024` (sampled), `--mode exact` (noiseless).

## Why the original notebook reports a nonzero `Err rate` in pure simulation
1. **State-preparation bug in `Us_gate`** (dominant): the uniformly-controlled
   RY tree that loads each normalized image vector is mis-wired (wrong
   control/X-flip sequence), so the embedded state is not
   (|0>|e> + |1>|f>)/sqrt(2) and every estimate is biased.
2. **QAE grid quantization**: with `aux_qubit = h` the argmax read-out
   `q_out(y*) = -cos(2*pi*y/2^h)` carries an irreducible error ~|ds/dy|/2
   (mean 2.15 % / max 3.88 % at h = 7 even with a perfect circuit).
3. **Shot noise**: `shots = 1` makes a single sample the whole estimate.

Fix 1 is mandatory; combining it with h = 8 and the MLE read-out (or h >= 9 with
the argmax read-out) brings the error below 1 % — details and validation numbers
are in the `ae_qip_fixed.py` docstring and in the markdown cell of
`AE-QIP-fixed.ipynb`.
