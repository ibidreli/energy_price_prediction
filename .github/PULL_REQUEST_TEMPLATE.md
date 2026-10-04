## Summary

<!-- Briefly explain what changes were made and why. Quote the issue or context. -->

Closes #<!-- Issue number -->

## Type of Change

- [ ] 📊 **EDA / Data Quality**: New analysis, distribution profiling, anomaly checks
- [ ] ⚙️ **Feature Engineering**: New feature, lag transformation, aggregation
- [ ] 🧪 **Model Experiment**: Model architecture, training run, uncertainty estimation
- [ ] 🐛 **Bug Fix**: Fix for pipeline failure, data parsing, or test error
- [ ] 🔨 **Refactoring / Maintenance**: Code cleanup, test expansion, Makefile/CI tweak
- [ ] 📝 **Documentation**: Updates to `docs/` or notebooks

## Data Science & Leakage Checklist

> [!IMPORTANT]
> The challenge mandates forecast emission at **D-1 11:00** for all 96 quarter-hours of day D. Zero tolerance for look-ahead leakage.

- [ ] **Temporal Leakage Check**:
  - Swissgrid prices & balance data strictly $\le$ D-2 24:00.
  - ECMWF weather runs strictly $\le$ D-2 18 UTC.
  - Transformers/scalers fit *only* on training splits (no pre-split transformations).
- [ ] **Reproducibility**:
  - Random seeds explicitly set (`numpy`, `torch`, `lightgbm`, etc.).
  - Environment runs on pinned `requirements.lock` via `make test`.
- [ ] **Uncertainty Quantification** (if modeling):
  - Model outputs uncertainty bounds or prediction intervals alongside point estimates.
  - Evaluation includes calibration / coverage error or pinball loss.
- [ ] **Tests & Quality**:
  - `make test` executed locally and all tests pass.
  - No new linter/formatting errors introduced.
- [ ] **Documentation**:
  - Relevant findings or documentation updated in `docs/`.
  - No `STATUS.md` created.

## Metrics / Evaluation Results (if applicable)

| Metric | Baseline | This PR | Delta |
|---|---|---|---|
| **MAE** (ct/kWh) | | | |
| **RMSE** (ct/kWh) | | | |
| **Coverage** (e.g. 80%) | | | |
| **Pinball Loss / Winkler** | | | |

## How Has This Been Tested?

```bash
# Paste command and output proof, e.g. pytest tests/
make test
```
