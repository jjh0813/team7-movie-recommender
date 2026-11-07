# Additional improvement experiment

This extends the submitted Team7_code.zip. Teammate source files remain intact.

## What changed

The submitted enhanced version is retained as the intermediate comparator. Instead of choosing weights from the external test, the new experiment uses two recent periods of MovieLens 32M only. For each cutoff (2022-01-01 and 2023-01-01), it fits only earlier ratings and tags and evaluates the next six months. It samples 600 eligible cases in each period (not necessarily 1,200 distinct people).

48 configurations were searched: sum or reciprocal-rank fusion, six CF weights (0.50, 0.75, 0.85, 0.90, 0.95, 0.99), and four popularity weights (0, 0.05, 0.10, 0.20). Equal-period mean NDCG@10 selects the setting, with Recall as the tie-breaker. CF settings and content construction remain fixed. The selected RRF setting has CF weight 0.85 and popularity weight 0.10, replacing 0.75 and 0.20. Validation mean NDCG increases from 0.096741 to 0.100308.

The frozen setting is then evaluated on all 1,274 eligible Beliefs users. Intermediate user-level results are reproduced to numerical tolerance 1e-12 before comparison. A paired user bootstrap with 2,000 replicates and seed 2026 estimates uncertainty.

| Same external cohort | Recall@10 | NDCG@10 |
|---|---:|---:|
| Original team combination under the corrected external protocol | 0.072587 | 0.076017 |
| Submitted enhanced version, now intermediate | 0.082231 | 0.081034 |
| Additional recent-validation hybrid, new default | 0.083220 | 0.084547 |
| CF alone | 0.085114 | 0.086918 |

Versus intermediate: NDCG +4.34%, Recall +1.20%. NDCG difference 95% bootstrap interval [0.000719, 0.006314]; Recall interval [-0.003412, 0.005368]. Recall improvement is uncertain. These are observed-cohort intervals, not proof of generalization to independent users.

No-tags ablation NDCG 0.078123; no-popularity-prior ablation 0.084154. Tags help this hybrid on these cases. The prior slightly helps NDCG while slightly lowering Recall. CF alone remains stronger, so do not claim the hybrid beats every baseline.

## Run and reproduce

`python main.py` loads team_models_v2.pkl when available and produces the enhanced Top-10. `python main.py evaluate` writes current results under results/current, preserving the submitted snapshots.

To rebuild from raw data, first run download_data.py and build_additions.py, then `python -m experiments.recent_validation --raw data/raw --users 600`, then `python -m experiments.external_comparison`. The validation script creates private caches; these are intentionally omitted from packages. Raw downloads and calibration can take substantial time and memory. The ready-to-run package contains both model versions and saved results.

Frozen settings, full search, user-level results, bootstrap intervals and plots are in results/additional. Older README/IMPROVEMENTS descriptions refer to the intermediate version; this file describes the additional experiment.

## Next experiments

Before looking at another untouched test: validate request-specific gating on 32M only (short versus long histories), compare full-history versus recent-history profiles, and test content features that add independent information such as plot text. Report CF alone alongside any hybrid. Broader catalog support and coverage/diversity are separate objectives and require fresh evaluation. These are future ideas, not completed improvements.
