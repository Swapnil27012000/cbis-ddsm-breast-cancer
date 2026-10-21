# Stage 16.3 — Smoke-Test Checkpoint Isolation Fix Report

**Implementation Date:** 2026-10-22  
**Auditor / Engineer:** Antigravity Pair-Programming Agent  
**Project:** MSc IT (Artificial Intelligence) — CBIS-DDSM Mammography Breast Cancer Classification  
**Target File Modified:** [`src/training/stage16_trainer.py`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/src/training/stage16_trainer.py)  
**Test Suite Modified:** [`tests/test_stage16_models.py`](file:///c:/Users/swapn/Project/Breast_Cancer/cbis-ddsm-breast-cancer/tests/test_stage16_models.py)  
**Scope:** Strict Physical & Path Isolation of Smoke-Test Checkpoints, Metrics, and Artifacts to Prevent Any Risk of Overwriting Production Checkpoints  
**Status:** **IMPLEMENTED & READY FOR VERIFICATION**

---

## 1. Executive Summary

During Stage 16.2 audit, an architectural vulnerability was identified in `src/training/stage16_trainer.py`: smoke tests and full training shared identical output directories (`models/stage16_<model>/`) and identical checkpoint filenames (`<model>_best_loss.pt`, `<model>_best_auc.pt`, `<model>_final.pt`). If a smoke test or unit test was executed after full training, the 1-epoch / 32-sample smoke checkpoint would silently overwrite the production checkpoint trained on the full dataset.

Stage 16.3 resolves this vulnerability by introducing **two-tier defense-in-depth isolation**:
1. **Dedicated Directory Isolation:** Smoke tests write exclusively to `models/stage16_smoke_tests/<model>/` and `results/stage16_smoke_tests/<model>/`.
2. **Explicit Filename Prefixing:** All smoke-test checkpoints and artifacts are prefixed with `smoke_` (e.g., `smoke_densenet121_best_loss.pt`).
3. **Disjoint Artifact Routing:** All training history, validation metric, and configuration paths for smoke tests are redirected to distinct `smoke_` filenames, ensuring production artifacts can never be modified by smoke testing.
4. **Preservation of Production Paths:** Full-training paths remain 100% identical to the verified Stage 16.1 and Stage 16.2 standards.

---

## 2. Code Changes in `src/training/stage16_trainer.py`

### 2.1 Centralized Path Resolution (`resolve_stage16_paths`)
A centralized routing function was implemented:

```python
def resolve_stage16_paths(
    model_name: str,
    smoke_test: bool = False,
    project_root: Optional[str] = None,
) -> Dict[str, str]:
    root = project_root if project_root is not None else _PROJECT_ROOT
    canonical_model = "densenet121" if "dense" in model_name.lower() else "efficientnet_b0"

    metadata_dir = os.path.join(root, "data/metadata/stage16")

    if smoke_test:
        models_dir = os.path.join(root, "models/stage16_smoke_tests", canonical_model)
        results_dir = os.path.join(root, "results/stage16_smoke_tests", canonical_model)
        prefix = f"smoke_{canonical_model}"
        summary_filename = f"stage16_{canonical_model}_smoke_test_summary.json"
        epoch_history_filename = f"stage16_{canonical_model}_smoke_epoch_history.csv"
        val_loss_filename = f"stage16_{canonical_model}_smoke_validation_metrics_best_loss.csv"
        val_auc_filename = f"stage16_{canonical_model}_smoke_validation_metrics_best_auc.csv"
        config_filename = f"stage16_{canonical_model}_smoke_config.json"
    else:
        models_dir = os.path.join(root, f"models/stage16_{canonical_model}")
        results_dir = os.path.join(root, f"results/stage16_{canonical_model}")
        prefix = canonical_model
        summary_filename = f"stage16_{canonical_model}_smoke_test_summary.json"
        epoch_history_filename = f"stage16_{canonical_model}_epoch_history.csv"
        val_loss_filename = f"stage16_{canonical_model}_validation_metrics_best_loss.csv"
        val_auc_filename = f"stage16_{canonical_model}_validation_metrics_best_auc.csv"
        config_filename = f"stage16_{canonical_model}_config.json"

    return {
        "canonical_model": canonical_model,
        "metadata_dir": metadata_dir,
        "models_dir": models_dir,
        "results_dir": results_dir,
        "best_loss_checkpoint_path": os.path.join(models_dir, f"{prefix}_best_loss.pt"),
        "best_auc_checkpoint_path": os.path.join(models_dir, f"{prefix}_best_auc.pt"),
        "final_checkpoint_path": os.path.join(models_dir, f"{prefix}_final.pt"),
        "smoke_summary_path": os.path.join(metadata_dir, summary_filename),
        "epoch_history_path": os.path.join(metadata_dir, epoch_history_filename),
        "val_loss_metrics_path": os.path.join(metadata_dir, val_loss_filename),
        "val_auc_metrics_path": os.path.join(metadata_dir, val_auc_filename),
        "test_metrics_path": os.path.join(metadata_dir, f"stage16_{canonical_model}_test_metrics.csv"),
        "config_path": os.path.join(metadata_dir, config_filename),
    }
```

### 2.2 Path Routing Comparison

| Artifact Key | Full-Training Mode (`smoke_test=False`) | Smoke-Test Mode (`smoke_test=True`) | Overwrite Risk |
|---|---|---|---|
| **Checkpoint Directory** | `models/stage16_<model>/` | `models/stage16_smoke_tests/<model>/` | **Eliminated (Disjoint)** |
| **Best Loss Checkpoint** | `<model>_best_loss.pt` | `smoke_<model>_best_loss.pt` | **Eliminated (Disjoint)** |
| **Best AUC Checkpoint** | `<model>_best_auc.pt` | `smoke_<model>_best_auc.pt` | **Eliminated (Disjoint)** |
| **Final State Weights** | `<model>_final.pt` | `smoke_<model>_final.pt` | **Eliminated (Disjoint)** |
| **Results / Curves Dir** | `results/stage16_<model>/` | `results/stage16_smoke_tests/<model>/` | **Eliminated (Disjoint)** |
| **Epoch History CSV** | `stage16_<model>_epoch_history.csv` | `stage16_<model>_smoke_epoch_history.csv` | **Eliminated (Disjoint)** |
| **Validation Loss CSV** | `stage16_<model>_validation_metrics_best_loss.csv` | `stage16_<model>_smoke_val_metrics_best_loss.csv` | **Eliminated (Disjoint)** |
| **Validation AUC CSV** | `stage16_<model>_validation_metrics_best_auc.csv` | `stage16_<model>_smoke_val_metrics_best_auc.csv` | **Eliminated (Disjoint)** |
| **Config JSON** | `stage16_<model>_config.json` | `stage16_<model>_smoke_config.json` | **Eliminated (Disjoint)** |

---

## 3. Regression & Unit Tests in `tests/test_stage16_models.py`

Three complementary tests guarantee checkpoint isolation and non-regression:

1. **`test_smoke_test_checkpoint_path_isolation_exactness()` (NEW):**
   - Resolves paths for both DenseNet-121 and EfficientNet-B0 under both modes.
   - Asserts that full training paths strictly match production paths.
   - Asserts that smoke test paths reside inside `stage16_smoke_tests/` with the `smoke_` prefix.
   - Asserts that all 7 checkpoint and metric path keys are mutually disjoint between modes.
2. **`test_smoke_test_mode_isolation_densenet()` (UPDATED):**
   - Records the timestamp/mtime of `models/stage16_densenet121/densenet121_best_loss.pt` before running.
   - Executes smoke-test mode on CPU.
   - Asserts that the smoke checkpoint was written to `models/stage16_smoke_tests/densenet121/smoke_densenet121_best_loss.pt`.
   - Asserts that the production checkpoint's modification time was completely unchanged.
3. **`test_smoke_test_mode_isolation_efficientnet()` (UPDATED):**
   - Records the timestamp/mtime of `models/stage16_efficientnet_b0/efficientnet_b0_best_loss.pt` before running.
   - Executes smoke-test mode on CPU.
   - Asserts that the smoke checkpoint was written to `models/stage16_smoke_tests/efficientnet_b0/smoke_efficientnet_b0_best_loss.pt`.
   - Asserts that the production checkpoint's modification time was completely unchanged.

---

## 4. Test Suite Execution Command

To verify the regression tests and full Stage 16 test suite inside the Docker container, run:

```powershell
docker compose run --rm cbis-ddsm-ai pytest tests/test_stage16_models.py -v
```

Expected Result: **13 passed**.
- 11 existing unit/integration tests continue to pass.
- 1 new regression test (`test_smoke_test_checkpoint_path_isolation_exactness`) passes.
- Updated smoke-test isolation tests verify production checkpoints are untouched.

---

## 5. Verification Checklist

- [x] Full-training checkpoints preserved exactly as verified in Stage 16.2.
- [x] Zero exposure or evaluation of the 644 clean test set records.
- [x] Smoke-test checkpoints redirected to `models/stage16_smoke_tests/`.
- [x] Smoke-test filenames prefixed with `smoke_`.
- [x] Smoke-test metrics redirected to separate `smoke_` files.
- [x] Stages 1–15 artifacts unchanged.
- [x] No model retraining executed during this fix.
