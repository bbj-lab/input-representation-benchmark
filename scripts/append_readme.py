

## Paper Audit Trail

This section documents the explicit mapping between the claims in `paper.tex` and the verifiable experimental outputs in this repository.

### Experiment 1: The Granularity Paradox
The "representational collapse" phenomenon observed with Fused Centiles (Figure 1, Table 2) is tracked by comparing the following scripts:
- **Fused vs. Unfused Deciles/Centiles JSONs:** Check `outputs/eval_maxlen_deciles.json` and `outputs/eval_maxlen_centiles.json`. The `trunc_rate_delta` validates that pushing sequence length constraints increases the failure rate of high-precision representations. 

### Experiment 2: Representation Mechanics (Diagnostics)
The mechanistic interpretability results diagnosing *why* models fail are substantiated in `outputs/diagnostics/`:
* **Soft Discretization Autoregressive Washout (Table 5):** Validated in `attention_washout_results.json`. The discrete encoder maintains $\approx 0.033$ share to numeric tokens vs softmax degradation shown in cross entropy degradation (`ce_degradation: -2.0` vs `-5.5`).
* **Clinical Boundary Probe (Table 4):** Validated in `clinical_boundary_probe_results.json`. Fused deciles achieve Potassium accuracy 90%, Glucose 70%, Creatinine 40%, matching the exact columns in the manuscript probe.
* **Embedding Geometry (Figure 4, Table 3):** Validated in `embedding_geometry_results.json` and `sparsity_distance_results.json`. The U-shape curve for the ordinal manifold is `-0.0026`. The number of frozen tokens is $10,866/83,927$ ($12.9\%$) for Fused Centiles, validating the statement in Section 5.1.
* **xVal Variance Collapse:** Validated in `xval_zero_out_results.json` and `xval_scale_dist_xval_rope.png` where the scale factor strictly clusters around $1.0$ instead of passing through zero.
