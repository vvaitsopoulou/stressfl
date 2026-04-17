# KD Visualisation

# Purpose:
# Generate all dissertation figures for the cross-modal KD chapter:

#   1. t-SNE/UMAP of student penultimate representations (3 panels)
#   2. Ablation plots: macro-F1 vs T, α, feature set, window size, K
#   3. Per-subject comparison: 15 subjects' F1 for all four conditions
#   4. Confusion matrices: one per condition
#   5. ROC curves: overlay all four conditions
#   6. Teacher-student agreement: confidence scatter
#   7. Personalisation gain: per-subject F1 improvement (c) → (d)

# All functions return matplotlib Figure objects for flexible saving.
# Default style follows academic publication conventions (no grid,
# minimal decoration, colourblind-safe palette).

# References:
# [vanderMaaten08]  van der Maaten & Hinton 2008 — t-SNE
# [McInnes18]       McInnes et al. 2018 — UMAP

from __future__ import annotations

import os

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend for saving
    from matplotlib.figure import Figure
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from kd.kd_config import (
    KD_FIGURES_DIR,
    ensure_dirs,
)

CONDITION_COLOURS = {
    "teacher": "#4477AA",  # blue
    "no_kd": "#EE6677",  # red
    "kd": "#228833",  # green
    "personal": "#CCBB44",  # yellow
}
CONDITION_LABELS = {
    "teacher": "Teacher (CNN-LSTM)",
    "no_kd": "Student (no KD)",
    "kd": "Student (KD)",
    "personal": "Student (KD + personal)",
}
CONDITIONS = ["teacher", "no_kd", "kd", "personal"]


def _check_mpl():
    if not HAS_MPL:
        raise ImportError("matplotlib is required for visualisations")


#  1. t-SNE / UMAP EMBEDDING PLOTS


def plot_embeddings(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "Student Representations",
    method: str = "tsne",
    perplexity: float = 30.0,
    seed: int = 42,
) -> Figure:
    """t-SNE or UMAP visualisation of student penultimate representations.

    Parameters
    ----------
    features : np.ndarray (N, 32)
        Student base output (intermediate representation).
    labels : np.ndarray (N,)
        Binary labels (0=non-stress, 1=stress).
    title : str
    method : str
        "tsne" or "umap".
    perplexity : float
        t-SNE perplexity (ignored for UMAP).
    seed : int

    Returns
    -------
    Figure
    """
    _check_mpl()

    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2, perplexity=perplexity, random_state=seed, init="pca"
        )
    elif method == "umap":
        from umap import UMAP

        reducer = UMAP(n_components=2, random_state=seed)
    else:
        raise ValueError(f"Unknown method: {method}")

    embedding = reducer.fit_transform(features)

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    for label_val, label_name, colour in [
        (0, "Non-stress", "#4477AA"),
        (1, "Stress", "#EE6677"),
    ]:
        mask = labels == label_val
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=colour,
            label=label_name,
            s=10,
            alpha=0.6,
        )

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    return fig


def plot_embedding_panels(
    features_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    method: str = "tsne",
    seed: int = 42,
) -> Figure:
    """Three-panel embedding plot comparing conditions.

    Parameters
    ----------
    features_dict : dict
        condition_name → (features_array, labels_array).
        Typically: {"no_kd": (...), "kd": (...), "personal": (...)}.
    method : str
    seed : int

    Returns
    -------
    Figure
    """
    _check_mpl()

    n_panels = len(features_dict)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (cond, (feats, labs)) in zip(axes, features_dict.items()):
        if method == "tsne":
            from sklearn.manifold import TSNE

            emb = TSNE(
                n_components=2, perplexity=30, random_state=seed, init="pca"
            ).fit_transform(feats)
        else:
            from umap import UMAP

            emb = UMAP(n_components=2, random_state=seed).fit_transform(feats)

        for lv, ln, c in [(0, "Non-stress", "#4477AA"), (1, "Stress", "#EE6677")]:
            m = labs == lv
            ax.scatter(emb[m, 0], emb[m, 1], c=c, label=ln, s=8, alpha=0.5)

        ax.set_title(CONDITION_LABELS.get(cond, cond))
        ax.set_xlabel(f"{method.upper()} 1")
        if ax == axes[0]:
            ax.set_ylabel(f"{method.upper()} 2")
        ax.legend(loc="best", frameon=False, fontsize=8)

    fig.tight_layout()
    return fig


#  2. ABLATION PLOTS


def plot_ablation_line(
    x_values: list | np.ndarray,
    y_values: list | np.ndarray,
    x_label: str,
    y_label: str = "Macro F1",
    title: str = "",
    y_err: list | np.ndarray | None = None,
) -> Figure:
    """Generic ablation line plot with optional error bars.

    Parameters
    ----------
    x_values : array-like
    y_values : array-like
        Mean metric values.
    x_label, y_label, title : str
    y_err : array-like, optional
        Standard deviation for error bars.

    Returns
    -------
    Figure
    """
    _check_mpl()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.array(x_values)
    y = np.array(y_values)

    if y_err is not None:
        ax.errorbar(
            x,
            y,
            yerr=y_err,
            marker="o",
            capsize=4,
            color="#228833",
            linewidth=2,
            markersize=6,
        )
    else:
        ax.plot(x, y, marker="o", color="#228833", linewidth=2, markersize=6)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ablation_bar(
    categories: list[str],
    values: list[float],
    errors: list[float] | None = None,
    y_label: str = "Macro F1",
    title: str = "",
) -> Figure:
    """Bar chart for categorical ablations (e.g. feature sets).

    Returns
    -------
    Figure
    """
    _check_mpl()

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(categories))
    colours = ["#4477AA", "#EE6677", "#228833", "#CCBB44"][: len(categories)]

    if errors:
        ax.bar(x, values, yerr=errors, color=colours, capsize=5, edgecolor="black")
    else:
        ax.bar(x, values, color=colours, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=0)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


#  3. PER-SUBJECT COMPARISON


def plot_per_subject_f1(
    loso_results: list[dict],
    metric: str = "macro_f1",
) -> Figure:
    """Per-subject F1 for all four conditions, grouped bar chart.

    Parameters
    ----------
    loso_results : list[dict]
    metric : str

    Returns
    -------
    Figure
    """
    _check_mpl()

    subjects = [r["meta"]["test_subject"] for r in loso_results]
    n_subjects = len(subjects)
    n_conds = len(CONDITIONS)
    bar_width = 0.8 / n_conds
    x = np.arange(n_subjects)

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, cond in enumerate(CONDITIONS):
        vals = []
        for r in loso_results:
            if cond in r:
                vals.append(r[cond].get(metric, 0))
            else:
                vals.append(0)

        ax.bar(
            x + i * bar_width,
            vals,
            bar_width,
            label=CONDITION_LABELS.get(cond, cond),
            color=CONDITION_COLOURS.get(cond, "#999999"),
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x + bar_width * (n_conds - 1) / 2)
    ax.set_xticklabels([f"S{s}" for s in subjects], fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_xlabel("Test Subject")
    ax.set_title("Per-Subject Performance — All Conditions")
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig


#  4. CONFUSION MATRICES


def plot_confusion_matrices(
    matrices: dict[str, np.ndarray],
    class_names: list[str] | None = None,
) -> Figure:
    """Plot confusion matrices side-by-side, one per condition.

    Parameters
    ----------
    matrices : dict[str, np.ndarray]
        condition_name → (n_classes, n_classes) array.
    class_names : list[str], optional
        Default ["Non-stress", "Stress"].

    Returns
    -------
    Figure
    """
    _check_mpl()

    if class_names is None:
        class_names = ["Non-stress", "Stress"]

    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (cond, cm) in zip(axes, matrices.items()):
        # im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(CONDITION_LABELS.get(cond, cond), fontsize=10)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # Annotate cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]}",
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12,
                )

    fig.tight_layout()
    return fig


#  5. ROC CURVES


def plot_roc_curves(
    roc_data: dict[str, tuple[np.ndarray, np.ndarray, float]],
) -> Figure:
    """Overlay ROC curves for all conditions.

    Parameters
    ----------
    roc_data : dict
        condition_name → (fpr_array, tpr_array, auroc_value).

    Returns
    -------
    Figure
    """
    _check_mpl()

    fig, ax = plt.subplots(figsize=(6, 5.5))

    for cond in CONDITIONS:
        if cond not in roc_data:
            continue
        fpr, tpr, auroc = roc_data[cond]
        ax.plot(
            fpr,
            tpr,
            color=CONDITION_COLOURS.get(cond, "#999999"),
            linewidth=2,
            label=f"{CONDITION_LABELS.get(cond, cond)} (AUC={auroc:.3f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Conditions")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    return fig


#  6. TEACHER-STUDENT AGREEMENT


def plot_confidence_scatter(
    teacher_probs: np.ndarray,
    student_probs: np.ndarray,
    correct: np.ndarray,
    title: str = "Teacher-Student Agreement",
) -> Figure:
    """Scatter plot of teacher vs student confidence, coloured by correctness.

    Parameters
    ----------
    teacher_probs : np.ndarray (N,)
        Teacher P(stress) for each window.
    student_probs : np.ndarray (N,)
        Student P(stress) for each window.
    correct : np.ndarray of bool (N,)
        Whether the student predicted correctly.

    Returns
    -------
    Figure
    """
    _check_mpl()

    fig, ax = plt.subplots(figsize=(6, 5.5))

    ax.scatter(
        teacher_probs[correct],
        student_probs[correct],
        c="#228833",
        s=8,
        alpha=0.4,
        label="Correct",
    )
    ax.scatter(
        teacher_probs[~correct],
        student_probs[~correct],
        c="#EE6677",
        s=15,
        alpha=0.7,
        label="Incorrect",
        marker="x",
    )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Teacher P(stress)")
    ax.set_ylabel("Student P(stress)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    return fig


#  7. PERSONALISATION GAIN


def plot_personalisation_gain(
    loso_results: list[dict],
    metric: str = "macro_f1",
) -> Figure:
    """Per-subject F1 improvement from KD (c) to KD+personal (d).

    Parameters
    ----------
    loso_results : list[dict]
    metric : str

    Returns
    -------
    Figure
    """
    _check_mpl()

    subjects = []
    gains = []
    for r in loso_results:
        if "kd" in r and "personal" in r:
            sid = r["meta"]["test_subject"]
            f1_kd = r["kd"].get(metric, 0)
            f1_personal = r["personal"].get(metric, 0)
            subjects.append(f"S{sid}")
            gains.append(f1_personal - f1_kd)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(subjects))
    colours = ["#228833" if g >= 0 else "#EE6677" for g in gains]

    ax.bar(x, gains, color=colours, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhline(
        np.mean(gains),
        color="#4477AA",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean gain: {np.mean(gains):+.3f}",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(subjects, fontsize=9)
    ax.set_ylabel(f"Δ {metric.replace('_', ' ').title()}")
    ax.set_xlabel("Test Subject")
    ax.set_title("Personalisation Gain: KD → KD + Personal Head")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    return fig


#  8. SAVE UTILITY


def save_figure(
    fig: Figure,
    name: str,
    output_dir: str = KD_FIGURES_DIR,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
) -> list[str]:
    """Save a figure in multiple formats.

    Parameters
    ----------
    fig : Figure
    name : str
        Filename without extension.
    output_dir : str
    formats : tuple of str
    dpi : int

    Returns
    -------
    list[str]
        Paths to saved files.
    """
    ensure_dirs()
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for fmt in formats:
        path = os.path.join(output_dir, f"{name}.{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)

    plt.close(fig)
    print(f"  Saved: {', '.join(paths)}")
    return paths
