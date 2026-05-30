"""
Cross-Modal Knowledge Distillation (KD) package.

Distils a federated CNN-LSTM teacher (wrist signals) into a lightweight
MLP student that runs on Fitbit-grade HRV features only.
"""

from .fitbit_evaluator import evaluate_fitbit_pilot, load_wesad_scaler
from .fitbit_preprocessor import (
    generate_quality_report,
    parse_fitbit_json,
    preprocess_fitbit_session,
)
from .hrv_feature_extractor import (
    extract_extended_features,
    extract_fitbit_features,
    extract_primary_features,
    features_to_array,
)
from .kd_config import (
    ALL_SUBJECT_IDS,
    DEVICE,
    KD_CHECKPOINT_DIR,
    KD_DEFAULTS,
    KD_FIGURES_DIR,
    KD_PAIRED_DIR,
    KD_RESULTS_DIR,
    KDHyperparameters,
    ensure_dirs,
)
from .kd_evaluate import (
    build_comparison_table,
    extract_confusion_matrices,
    generate_evaluation_report,
    load_results,
    print_comparison_table,
    run_statistical_tests,
    save_results,
)
from .kd_loss import kd_loss, kd_loss_with_features
from .kd_teacher_loading import (
    get_teacher_loader,
    make_centralised_teacher_loader,
    make_fedrep_teacher_loader,
)
from .kd_trainer import run_full_loso, run_loso_fold
from .kd_visualizations import (
    plot_ablation_bar,
    plot_ablation_line,
    plot_confidence_scatter,
    plot_confusion_matrices,
    plot_embedding_panels,
    plot_embeddings,
    plot_per_subject_f1,
    plot_personalisation_gain,
    plot_roc_curves,
    save_figure,
)
from .paired_data_generator import generate_all_paired_data, verify_paired_data
from .student_model import StudentBase, StudentHead, StudentModel, build_student
from .teacher_wrapper import KDTeacherWrapper

__all__ = [
    # Config
    "KDHyperparameters",
    "KD_DEFAULTS",
    "ALL_SUBJECT_IDS",
    "DEVICE",
    "KD_PAIRED_DIR",
    "KD_CHECKPOINT_DIR",
    "KD_RESULTS_DIR",
    "KD_FIGURES_DIR",
    "ensure_dirs",
    # Models
    "StudentModel",
    "StudentBase",
    "StudentHead",
    "build_student",
    "KDTeacherWrapper",
    # Loss
    "kd_loss",
    "kd_loss_with_features",
    # Training
    "run_loso_fold",
    "run_full_loso",
    "get_teacher_loader",
    "make_fedrep_teacher_loader",
    "make_centralised_teacher_loader",
    # Evaluation
    "build_comparison_table",
    "print_comparison_table",
    "run_statistical_tests",
    "extract_confusion_matrices",
    "save_results",
    "load_results",
    "generate_evaluation_report",
    # Data pipeline
    "generate_all_paired_data",
    "verify_paired_data",
    "extract_primary_features",
    "extract_extended_features",
    "extract_fitbit_features",
    "features_to_array",
    # Fitbit
    "parse_fitbit_json",
    "preprocess_fitbit_session",
    "generate_quality_report",
    "evaluate_fitbit_pilot",
    "load_wesad_scaler",
    # Visualisations
    "plot_per_subject_f1",
    "plot_confusion_matrices",
    "plot_roc_curves",
    "plot_ablation_line",
    "plot_ablation_bar",
    "plot_personalisation_gain",
    "plot_confidence_scatter",
    "plot_embeddings",
    "plot_embedding_panels",
    "save_figure",
]
