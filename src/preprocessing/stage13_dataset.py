"""Stage 13 module export under preprocessing namespace for invocation convenience."""
from src.data.stage13_dataset import *
from src.data.stage13_dataset import (
    STAGE12_CANDIDATES,
    LABEL_MAP,
    LABEL_NAMES,
    load_raw_case_descriptions,
    load_stage5_baseline_metadata,
    map_pathology_labels,
    audit_cross_partition_patient_conflicts,
    create_patient_level_splits,
    audit_images_and_traceability,
    generate_class_distribution_summary,
    generate_missing_image_and_invalid_label_report,
    generate_stage13_validation_csv,
    generate_stage13_config_json,
    generate_stage13_reports,
    run_stage13_pipeline,
)

if __name__ == "__main__":
    result = run_stage13_pipeline()
    print(f"[Stage 13 Preprocessing] Pipeline status: {result['status']}")
