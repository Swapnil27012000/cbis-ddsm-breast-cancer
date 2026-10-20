"""Unit and integration tests for Stage 12: Evidence-Based Preprocessing Strategy Selection.

Verifies:
1. Correct loading and auditing of authoritative metadata from Stages 5–11.
2. Candidate taxonomy: presence of mandatory Baseline Control (C1) and defensible candidates (C2–C5).
3. Documented exclusion of detrimental methods (Global HistEq, Gaussian blur, NLM) with empirical justification.
4. Metric integrity: CNR preserved as NaN without 0.0 substitution; non-finite PSNR handling.
5. Strict policy constraint: Winner selection status is 'SHORTLIST_ESTABLISHED' and 'NONE_DECLARED'.
6. Separation of quantitative measurements from qualitative visual observations.
7. Complete artifact generation: comparison CSV, JSON config, validation CSV, TXT/MD reports, and overview figure.
8. Immutability of earlier stage artifacts (Stages 5–11 files remain untouched).
"""

import json
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.stage12_strategy_selection import (
    CANDIDATE_STRATEGIES,
    DISCARDED_STRATEGIES,
    audit_existing_preprocessing_evidence,
    generate_candidate_comparison_matrix,
    generate_stage12_config,
    generate_stage12_reports,
    generate_stage12_validation_summary,
    plot_stage12_shortlist_overview,
    run_stage12_strategy_selection,
)


def test_stage12_audit_existing_evidence():
    """Verify that Stage 12 correctly reads and verifies authoritative metadata from Stages 5–11."""
    evidence = audit_existing_preprocessing_evidence()
    assert "stage_audit" in evidence
    assert "s10_method_summary" in evidence
    assert "s10_noise_summary" in evidence

    # Verify Stage 5 (Full dataset baseline)
    s5 = evidence["stage_audit"]["stage5"]
    assert s5["record_count"] >= 3568

    # Verify Stage 6 (Contrast)
    s6 = evidence["stage_audit"]["stage6"]
    assert s6["record_count"] == 48
    assert s6["unique_cases"] == 16

    # Verify Stage 7 (Sharpening factorial)
    s7 = evidence["stage_audit"]["stage7"]
    assert s7["record_count"] == 96
    assert s7["unique_cases"] == 16

    # Verify Stage 7 authoritative count and factorial coverage in evidence summary
    assert evidence["s7_count"] == 96
    assert evidence["s7_unique_cases"] == 16
    assert set(evidence["s7_contrast_methods"]) == {"baseline", "clahe", "histogram_equalization"}
    assert set(evidence["s7_sharpen_methods"]) == {"none", "unsharp_mask"}

    # Verify Stage 8 (Noise models)
    s8 = evidence["stage_audit"]["stage8"]
    assert s8["record_count"] == 96
    assert s8["unique_cases"] == 16

    # Verify Stage 9 (Denoised outputs)
    s9 = evidence["stage_audit"]["stage9"]
    assert s9["record_count"] == 640
    assert s9["unique_cases"] == 16

    # Verify Stage 10 (Quantitative metrics)
    s10 = evidence["stage_audit"]["stage10"]
    assert s10["record_count"] == 640
    assert s10["validated_count"] == 640
    assert s10["status"] == "PASS"


def test_stage12_candidate_shortlist_structure():
    """Verify that the candidate shortlist contains Baseline Control and defensible candidates."""
    candidate_ids = [c["candidate_id"] for c in CANDIDATE_STRATEGIES]
    assert "C1_BASELINE" in candidate_ids
    assert "C2_CLAHE" in candidate_ids
    assert "C3_UNSHARP_MASK" in candidate_ids
    assert "C4_CLAHE_UNSHARP" in candidate_ids
    assert "C5_MEDIAN_DENOISING" in candidate_ids

    # Verify C1 is designated as the mandatory control
    c1 = next(c for c in CANDIDATE_STRATEGIES if c["candidate_id"] == "C1_BASELINE")
    assert c1["recommendation_role"] == "Mandatory Scientific Control Reference"
    assert c1["family"] == "Baseline"

    # Verify C2 is CLAHE
    c2 = next(c for c in CANDIDATE_STRATEGIES if c["candidate_id"] == "C2_CLAHE")
    assert c2["parameters"]["clip_limit"] == 2.0
    assert c2["parameters"]["tile_grid_size"] == "8x8"
    assert c2["quantitative_metrics"]["entropy_bits"] == 4.9258

    # Verify C3 is Unsharp Masking
    c3 = next(c for c in CANDIDATE_STRATEGIES if c["candidate_id"] == "C3_UNSHARP_MASK")
    assert c3["quantitative_metrics"]["mean_gradient"] == 14.02
    assert c3["quantitative_metrics"]["strong_edge_percent"] == 21.98

    # Verify C4 is Compound CLAHE + Unsharp
    c4 = next(c for c in CANDIDATE_STRATEGIES if c["candidate_id"] == "C4_CLAHE_UNSHARP")
    assert c4["quantitative_metrics"]["mean_gradient"] == 24.25
    assert c4["quantitative_metrics"]["strong_edge_percent"] == 36.15

    # Verify C5 is Median Denoising
    c5 = next(c for c in CANDIDATE_STRATEGIES if c["candidate_id"] == "C5_MEDIAN_DENOISING")
    assert c5["quantitative_metrics"]["psnr_restoration_db"] == 36.06
    assert c5["quantitative_metrics"]["ssim_restoration"] == 0.8037


def test_stage12_discarded_strategies_justification():
    """Verify that discarded strategies have explicit empirical and scientific justifications."""
    discard_ids = [d["strategy_id"] for d in DISCARDED_STRATEGIES]
    assert "DISC_01_HISTEQ" in discard_ids
    assert "DISC_02_GAUSSIAN_BLUR" in discard_ids
    assert "DISC_03_NLM" in discard_ids
    assert "DISC_04_ADAPTIVE_MEDIAN" in discard_ids
    assert "DISC_05_UNTESTED_CHAINS" in discard_ids

    # Check HistEq reason mentions saturation and background noise
    histeq = next(d for d in DISCARDED_STRATEGIES if d["strategy_id"] == "DISC_01_HISTEQ")
    assert "saturation" in histeq["reason_for_exclusion"].lower()
    assert "noise" in histeq["reason_for_exclusion"].lower()

    # Check NLM reason mentions computational complexity / runtime
    nlm = next(d for d in DISCARDED_STRATEGIES if d["strategy_id"] == "DISC_03_NLM")
    assert "complexity" in nlm["reason_for_exclusion"].lower() or "runtime" in nlm["reason_for_exclusion"].lower()


def test_stage12_metric_non_finite_and_cnr_handling():
    """Verify that CNR is documented as NaN and non-finite values are never replaced with 0.0."""
    for c in CANDIDATE_STRATEGIES:
        m = c["quantitative_metrics"]
        # CNR must be NaN across all candidates
        assert np.isnan(m["cnr"])
        # CNR must not be 0.0
        assert m["cnr"] != 0.0

    # Test comparison matrix generation preserves NaN
    df_matrix = generate_candidate_comparison_matrix(output_csv_path="")
    assert "cnr_status" in df_matrix.columns
    assert (df_matrix["cnr_eligible_cases"] == 0).all()


def test_stage12_policy_no_premature_winner():
    """Verify that Stage 12 does NOT declare a single winner, preserving multiple candidates."""
    config = generate_stage12_config(output_json_path="")
    assert config["selection_status"] == "SHORTLIST_ESTABLISHED"
    assert config["winner_declared"] == "NONE_DECLARED"
    assert len(config["shortlisted_candidates"]) >= 4

    val_df = generate_stage12_validation_summary(output_csv_path="")
    winner_check = val_df[val_df["check_id"] == "CHK_S12_03_NO_PREMATURE_WINNER"]
    assert len(winner_check) == 1
    assert winner_check.iloc[0]["status"] == "PASS"


def test_stage12_artifact_generation(tmp_path):
    """Verify that Stage 12 generates valid, non-empty CSV, JSON, reports, and overview figure."""
    csv_out = str(tmp_path / "test_candidate_comparison.csv")
    val_out = str(tmp_path / "test_validation.csv")
    json_out = str(tmp_path / "test_config.json")
    txt_out = str(tmp_path / "test_report.txt")
    md_out = str(tmp_path / "test_report.md")
    fig_out = str(tmp_path / "test_overview.png")

    # Generate matrix
    df_m = generate_candidate_comparison_matrix(csv_out)
    assert os.path.exists(csv_out)
    assert os.path.getsize(csv_out) > 500
    assert len(df_m) == len(CANDIDATE_STRATEGIES) + len(DISCARDED_STRATEGIES)

    # Generate validation
    df_v = generate_stage12_validation_summary(val_out)
    assert os.path.exists(val_out)
    assert (df_v["status"] == "PASS").all()

    # Generate config
    cfg = generate_stage12_config(json_out)
    assert os.path.exists(json_out)
    assert cfg["selection_status"] == "SHORTLIST_ESTABLISHED"

    # Generate reports
    txt_rep, md_rep = generate_stage12_reports(txt_out, md_out)
    assert os.path.exists(txt_out)
    assert os.path.exists(md_out)
    assert "C1_BASELINE" in txt_rep
    assert "C2_CLAHE" in md_rep
    assert "NO SINGLE WINNER DECLARED" in txt_rep
    assert "Stage 13" in md_rep

    # Generate overview figure with strict warning assertion
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        p_fig = plot_stage12_shortlist_overview(fig_out)
    assert os.path.exists(p_fig)
    assert os.path.getsize(p_fig) > 10000


def test_stage12_sharpening_evidence_alignment():
    """Verify that shortlisted sharpening candidates accurately mirror the 96-record Stage 7 experiment."""
    evidence = audit_existing_preprocessing_evidence()
    assert evidence["s7_count"] == 96
    assert evidence["s7_unique_cases"] == 16

    c3 = next(c for c in CANDIDATE_STRATEGIES if c["candidate_id"] == "C3_UNSHARP_MASK")
    c4 = next(c for c in CANDIDATE_STRATEGIES if c["candidate_id"] == "C4_CLAHE_UNSHARP")

    # Metrics from Stage 7 authoritative report (baseline+unsharp: 14.02 mean grad, clahe+unsharp: 24.25 mean grad)
    assert c3["quantitative_metrics"]["mean_gradient"] == 14.02
    assert c3["quantitative_metrics"]["strong_edge_percent"] == 21.98
    assert c4["quantitative_metrics"]["mean_gradient"] == 24.25
    assert c4["quantitative_metrics"]["strong_edge_percent"] == 36.15


def test_stage12_validation_checks_table():
    """Verify that the validation summary CSV checks Stage 7 96-record scope and passes all checks."""
    df_val = generate_stage12_validation_summary()
    assert (df_val["status"] == "PASS").all()
    s7_check = df_val[df_val["check_id"] == "CHK_S12_08_STAGE7_FULL_COVERAGE"]
    assert len(s7_check) == 1
    assert "96 records" in s7_check.iloc[0]["actual"]


def test_stage12_earlier_stage_immutability():
    """Verify that running Stage 12 does not modify earlier stage files."""
    s11_fig = "results/preprocessing/comparison/heatmap_method_by_noise_psnr.png"
    s10_csv = "data/metadata/stage10/denoising_metrics_per_image.csv"
    s5_csv = "data/metadata/final_preprocessing_metadata.csv"

    # Check timestamps or sizes
    orig_s11_size = os.path.getsize(s11_fig) if os.path.exists(s11_fig) else None
    orig_s10_size = os.path.getsize(s10_csv) if os.path.exists(s10_csv) else None

    # Run complete Stage 12 pipeline
    res = run_stage12_strategy_selection()
    assert res["status"] == "SUCCESS"
    assert res["validation_status"] == "ALL_PASS"

    if orig_s11_size:
        assert os.path.getsize(s11_fig) == orig_s11_size
    if orig_s10_size:
        assert os.path.getsize(s10_csv) == orig_s10_size
