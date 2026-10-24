# Query-aware KTV

A small exploratory comparison of KTV and query-aware frame selection on **400 NExT-QA validation questions / 291 videos**, using LLaVA-v1.6-Vicuna-7B. All methods retain **six distinct frames and 936 visual tokens**.

| Method | Accuracy | Mean complete-request time* |
|---|---:|---:|
| KTV: six DINOv2 clustering representatives | 63.75% | 31.47 s |
| Four representatives + CLIP query match + temporal neighbor | 63.50% | 50.48 s |
| Four representatives + two query matches | 62.50% | 50.79 s |
| Six query matches | 62.25% | 24.56 s |

The neighbor hybrid corrected 16 baseline mistakes and introduced 17. Its temporal-question gain was +3.10 points, with uncertainty including zero. Overall improvement was not established. *Timing covers eight videos, one request per method/video, with resident models and features recomputed; model loading is excluded. This is an adapted reproduction, not verified parity with KTV's published score.

## Reproduce (Linux, CUDA)

Tested on a 32 GB V100 with Python 3.12, PyTorch 2.5.1/CUDA 12.1 and Transformers 4.37.2. Check your user quota, storage permissions and GPU availability first. Model/video downloads require substantial space; runtime files stay outside Git.

```bash
export STUDY_BASE=/absolute/path/to/your/private/runtime
bash study/setup.sh
# Set this to an allowance you verified from quota -s -v (example: 80 GiB).
export STUDY_STORAGE_BUDGET_GIB=80
# Alternatively, set STUDY_QUOTA_FILESYSTEM to the filesystem in quota -w -v.
bash study/prepare_assets.sh
tmux new-session -d -s query-ktv 'bash study/run_research.sh auto subset400'
python3 study/progress.py
```

The runner waits for an idle GPU, resumes saved work, runs the pilot before the final split, and retains pilot predictions. `study/run_e2e.sh` runs the additional direct-cost benchmark after evaluation. The neighbor uses a nominal ±1-second offset with question wording/CLIP determining direction; collisions are deduplicated and filled by CLIP rank. Selection details and known deviations are in `config/experiment.json`.

`results/` contains predictions, paired statistics, frame comparisons, a six-case visual review, and all direct timing measurements. Check them with `python study/verify_results.py`; regenerate diagnostics and plots with `study/selection_diagnostics.py`, `study/summarize_e2e.py`, and `study/plot_e2e.py`. To re-run accuracy analysis, copy the published prediction files into your prepared runtime's `data/results/`, then run `study/analyze.py --root "$STUDY_BASE/data" --split subset400`.

Upstream code is fetched at the commits in `config/sources.json`; model/dataset revisions are in `config/model_revisions.json`. Upstream code, models and NExT-QA data retain their respective licenses and terms. No weights or source videos are included. Related work: [KTV](https://github.com/hktk07/KTV), [QCA](https://github.com/hktk07/QCA), [Q-Frame](https://github.com/xiaomi-research/q-frame), [NExT-QA](https://github.com/doc-doc/NExT-QA).
