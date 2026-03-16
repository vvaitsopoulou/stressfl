# WESAD ML Classifier: Random Forest with LOSO-CV
#
# Implements the full training and evaluation pipeline for stress detection
# using Leave-One-Subject-Out Cross-Validation (LOSOCV) with nested
# hyperparameter tuning via RandomizedSearchCV.
#
# Pipeline order per fold:
#   1. LOSO split: train = other subjects, test = held-out subject.
#   2. Drop structurally-NaN features (>80% missing across full dataset, done once).
#   3. Imputation: fit median on training fold only, apply to both folds.
#   4. Normalization: StandardScaler fit on training fold only, apply to both folds.
#   5. Feature selection: fit on training fold only, transform both folds.
#   6. Nested RandomizedSearchCV: tune RF on the training subjects.
#   7. Evaluate: best model predicts the held-out subject.
#
# Normalization strategy:
#   StandardScaler (Z-score) fit inside the LOSO fold on training data only.
#   This avoids the data leakage caused by the previous per-subject baseline min-max normalization which
#   computed scaling parameters from the test subject's own data before the
#   LOSO split.
#
# Imputation strategy:
#   Median imputation fit on the training fold only.  NaN values arise from
#   NeuroKit2 failing to compute certain features (e.g. HRV frequency-domain
#   when R-peak detection fails on a noisy window).  Training-fold median is
#   used to fill both train and test NaNs so no test data influences the fill
#   values.
#
#   Features that are structurally NaN (>80% missing across all subjects) are
#   dropped entirely before the LOSO loop.  These are features that almost
#   never compute successfully (e.g. HRV_SDANN1, HRV_SDNNI1 which require
#   ~5-minute recordings but we have 60s windows).  Imputing them adds noise.
#
# Overlapped windows:
#   Controlled by the train_independent_only flag:
#     - False (default): all windows in training, independent only in test.
#     - True: independent windows only in BOTH training and test.
#   The 5s stride on 60s windows produces 91.7% overlap, meaning 11 out of
#   every 12 training windows are near-duplicates.  This inflates inner CV
#   scores and biases hyperparameter selection toward overfitting configs.
#   Setting train_independent_only=True produces the cleanest evaluation
#   but reduces training data from ~5,600 to ~535 windows.
#
# Metrics:
#   Because stress data is inherently imbalanced (subjects spend more time at
#   baseline), F1-Score (Weighted or Macro) is the primary metric.  Accuracy
#   alone is misleading if the model just predicts "No Stress" for every window.
#
#   Per-fold: Accuracy, Precision, Recall, F1 (weighted), confusion matrix.
#   Aggregate: Mean ± STD across all 15 folds.
#
# Label mapping:
#   Labels are mapped dynamically at training time (not permanently in the dataset):
#     Binary:  1.0 (baseline) → 0, 2.0 (stress) → 1, 3.0 (amusement) → 0
#     3-class: 1.0 → 0, 2.0 → 1, 3.0 → 2
#
# References:
#   Schmidt, P. et al. (2018). Introducing WESAD. ICMI '18.
#   Probst, P., Wright, M. N. & Boulesteix, A.-L. (2019). Hyperparameters
#       and tuning strategies for random forest. WIREs DMKD, 9(3), e1301.
#   Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter
#       optimization. JMLR, 13, 281–305.
#   Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
#   Scikit-learn: StratifiedGroupKFold, RandomizedSearchCV, RandomForestClassifier.

import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from src.ml_pipeline.feature_selection import FeatureSelectionPipeline

#  Label mapping

# Binary: baseline + amusement → 0, stress → 1
LABEL_MAP_BINARY = {1.0: 0, 2.0: 1, 3.0: 0}

# 3-class: baseline → 0, stress → 1, amusement → 2
LABEL_MAP_3CLASS = {1.0: 0, 2.0: 1, 3.0: 2}

_META_COLS = ["sid", "label", "window_idx", "is_overlapped"]


def _map_labels(labels: pd.Series, label_map: dict) -> pd.Series:
    # Map float labels to integer classes
    return labels.map(label_map).astype(int)


#  LOSO-CV Pipeline


class WESADClassifier:
    """
    Random Forest classifier with LOSO-CV for WESAD stress detection.

    Parameters
    ----------
    df_features : pd.DataFrame
        Feature matrix from run_feature_pipeline(). Must contain columns:
        sid, label, window_idx, is_overlapped, plus feature columns.
        IMPORTANT NOTE TO REMEMBER: Should be the UNNORMALIZED feature file!!  Normalization
        is now handled inside the LOSO loop via StandardScaler.
    label_map : dict
        Label mapping to apply (LABEL_MAP_BINARY or LABEL_MAP_3CLASS).
    feature_selection_method : str
        Feature selection method: "tree", "kbest", "pca", or "none".
    feature_selection_kwargs : dict
        Additional kwargs for the feature selector.
    param_grid : dict, optional
        Hyperparameter distributions for RandomizedSearchCV. If None, uses
        a default grid following Probst, Wright & Boulesteix (2019).
    scoring : str
        Scoring metric for RandomizedSearchCV. Default "f1_weighted".
    n_iter : int
        Number of random hyperparameter combinations to sample per fold
        (default: 60). Bergstra & Bengio (2012) show random search finds
        equally good solutions as grid search in fewer evaluations.
    n_inner_cv : int
        Number of inner CV folds for RandomizedSearchCV (default: 3).
    random_state : int
        Random seed for reproducibility.
    output_dir : str
        Directory to save results.
    train_independent_only : bool
        If True, use only non-overlapped (independent) windows for training.
        If False (default), include overlapped windows in training.
        Test always uses independent windows only regardless of this flag.
    nan_threshold : float
        Drop features where the fraction of NaN values exceeds this threshold
        across the entire dataset.  Default 0.8 (drop features that are >80%
        NaN).  These are structurally missing features that NeuroKit2 cannot
        compute reliably on 60s windows (e.g. HRV_SDANN1 requires ~5 min).
    normalize : bool
        If True (default), apply StandardScaler normalization inside each
        LOSO fold (fit on train, transform both).  Set to False for ablation
        to test the effect of normalization.
    """

    def __init__(
        self,
        df_features: pd.DataFrame,
        label_map: dict | None = None,
        feature_selection_method: str = "tree",
        feature_selection_kwargs: dict | None = None,
        param_grid: dict | None = None,
        scoring: str = "f1_weighted",
        n_iter: int = 60,
        n_inner_cv: int = 3,
        random_state: int = 42,
        output_dir: str = "results/",
        train_independent_only: bool = False,
        nan_threshold: float = 0.8,
        normalize: bool = True,
    ):
        if label_map is None:
            label_map = LABEL_MAP_BINARY

        self.label_map = label_map
        self.fs_method = feature_selection_method
        self.fs_kwargs = feature_selection_kwargs or {}
        self.scoring = scoring
        self.n_iter = n_iter
        self.n_inner_cv = n_inner_cv
        self.random_state = random_state
        self.output_dir = output_dir
        self.train_independent_only = train_independent_only
        self.nan_threshold = nan_threshold
        self.normalize = normalize

        os.makedirs(output_dir, exist_ok=True)

        # Default hyperparameter distributions for RandomizedSearchCV.
        #
        # Justification:
        #   n_estimators: sklearn default (100) through stability plateau (~500).
        #       Probst & Boulesteix (2017) recommend searching until OOB error
        #       plateaus.
        #   max_depth: 5–15 range prevents overfitting in LOSO (critical when
        #       every test subject is entirely unseen).  None (fully grown trees)
        #       is excluded: combined with correlated overlapped windows and
        #       min_samples_leaf=1 it produces near-perfect memorization.
        #   min_samples_split / min_samples_leaf: Probst, Wright & Boulesteix
        #       (2019) identify min node size as one of the two highest-impact
        #       RF hyperparameters. Pushed higher than sklearn defaults (2/1)
        #       to force each split to represent a statistically meaningful
        #       pattern rather than individual window noise.
        #   max_features: Identified as the single most impactful RF hyperparameter
        #       by Probst et al. (2019). "sqrt" is Breiman's (2001) original
        #       recommendation; "log2" adds more per-split randomness, beneficial
        #       when many features have correlated predictive power (HRV features).
        #
        # References:
        #   Probst, P., Wright, M. N. & Boulesteix, A.-L. (2019). Hyperparameters
        #       and tuning strategies for random forest. WIREs DMKD, 9(3), e1301.
        #   Bergstra, J. & Bengio, Y. (2012). Random search for hyper-parameter
        #       optimization. JMLR, 13, 281–305.
        #   Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5–32.
        if param_grid is None:
            self.param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15],
                "min_samples_split": [10, 20, 50],
                "min_samples_leaf": [5, 10, 20],
                "max_features": ["sqrt", "log2"],
            }
        else:
            self.param_grid = param_grid

        # Store the full feature matrix
        self.df = df_features.copy()

        # Drop structurally-NaN features
        # Features where >nan_threshold fraction of values are NaN are almost
        # certainly features that NeuroKit2 cannot compute reliably on 60s
        # windows
        # Imputing them with a constant adds uninformative noise.
        # This is computed on the FULL dataset before any split AND it is NOT
        # leakage because we are only looking at the MISSINGNESS PATTERN
        # (which columns have NaN), not at the actual feature VALUES.
        # The decision to drop a column is structural (based on window length
        # vs. feature requirements) not statistical.
        feat_cols = [c for c in self.df.columns if c not in _META_COLS]
        nan_frac = self.df[feat_cols].isna().mean()
        self._dropped_nan_cols = nan_frac[nan_frac > nan_threshold].index.tolist()
        if self._dropped_nan_cols:
            self.df = self.df.drop(columns=self._dropped_nan_cols)
            print(
                f"  Dropped {len(self._dropped_nan_cols)} structurally-NaN features "
                f"(>{nan_threshold*100:.0f}% missing):"
            )
            # Print first 10 for visibility
            for col in self._dropped_nan_cols[:10]:
                print(f"    - {col} ({nan_frac[col]*100:.1f}% NaN)")
            if len(self._dropped_nan_cols) > 10:
                print(f"    ... and {len(self._dropped_nan_cols) - 10} more")

        # Map labels dynamically
        self.df["mapped_label"] = _map_labels(self.df["label"], label_map)

        # Get subject IDs
        self.subjects = sorted(self.df["sid"].unique())

        # Report remaining NaN count for transparency
        remaining_feat_cols = [
            c for c in self.df.columns if c not in _META_COLS + ["mapped_label"]
        ]
        remaining_nan = self.df[remaining_feat_cols].isna().sum().sum()
        total_values = len(self.df) * len(remaining_feat_cols)

        print("\nWESADClassifier initialized:")
        print(f"  Subjects: {len(self.subjects)}")
        print(f"  Total windows: {len(self.df)}")
        print(f"  Features: {len(remaining_feat_cols)}")
        print(
            f"  Remaining NaN: {remaining_nan} ({remaining_nan/total_values*100:.2f}% of values)"
        )
        print(f"  Label map: {label_map}")
        print(f"  Feature selection: {feature_selection_method}")
        print(
            f"  Normalization: {'StandardScaler (inside LOSO fold)' if normalize else 'None'}"
        )
        print(f"  Train independent only: {train_independent_only}")
        print(f"  Scoring: {scoring}")
        print("  Label distribution:")
        print(f"    {self.df['mapped_label'].value_counts().to_dict()}")

    def run_losocv(self) -> dict:
        """
        Run the full LOSO-CV pipeline.

        For each subject:
        1. Split: train = all other subjects, test = held-out subject
           (independent windows only).
        2. Imputation: fit median on training fold, apply to both.
        3. Normalization: fit StandardScaler on training fold, apply to both.
        4. Feature selection: fit on training fold, transform both.
        5. Nested RandomizedSearchCV: tune RF hyperparameters on the training
           subjects using inner K-fold CV.
        6. Evaluate: best model predicts the held-out subject.

        Returns
        -------
        dict
            Results with per-fold metrics, aggregate metrics, and
            best hyperparameters per fold.
        """
        all_fold_results = []
        all_y_true = []
        all_y_pred = []
        start_time = time.time()

        n_folds = len(self.subjects)

        for fold_idx, held_out_sid in enumerate(self.subjects):
            fold_start = time.time()
            print("\n")
            print(
                f"FOLD {fold_idx + 1}/{n_folds}: Hold out Subject {int(held_out_sid)}\n"
            )

            # ── Split ──
            # Training: all subjects except held-out
            train_mask = self.df["sid"] != held_out_sid

            # Optionally restrict training to independent windows only
            if self.train_independent_only:
                train_mask = train_mask & (~self.df["is_overlapped"])

            # Validation/test: held-out subject, independent windows only
            val_mask = (self.df["sid"] == held_out_sid) & (~self.df["is_overlapped"])

            df_train = self.df.loc[train_mask].reset_index(drop=True)
            df_val = self.df.loc[val_mask].reset_index(drop=True)

            if len(df_val) == 0:
                print(
                    f"  WARNING: No independent windows for Subject "
                    f"{int(held_out_sid)}. Skipping."
                )
                continue

            y_train = df_train["mapped_label"]
            y_val = df_val["mapped_label"]

            print(
                f"  Train: {len(df_train)} windows "
                f"(labels: {y_train.value_counts().to_dict()})"
            )
            print(
                f"  Val:   {len(df_val)} windows "
                f"(labels: {y_val.value_counts().to_dict()})"
            )

            # Imputation (fit on train only to prevent leakage)
            _feature_cols = [
                c for c in df_train.columns if c not in _META_COLS + ["mapped_label"]
            ]
            n_nan_train = df_train[_feature_cols].isna().sum().sum()
            n_nan_val = df_val[_feature_cols].isna().sum().sum()
            if n_nan_train > 0 or n_nan_val > 0:
                fill_values = df_train[_feature_cols].median()
                fill_values = fill_values.fillna(0.0)
                df_train[_feature_cols] = df_train[_feature_cols].fillna(fill_values)
                df_val[_feature_cols] = df_val[_feature_cols].fillna(fill_values)
                print(
                    f"  Imputation: {n_nan_train} train NaN, {n_nan_val} val NaN "
                    f"→ filled with train-fold median"
                )

            # Normalization (fit on train only to prevent leakage)
            # StandardScaler: Z-score normalization (mean=0, std=1) fit on
            # training fold only.  This is the tabular-feature equivalent of
            # the per-window instance Z-score used in the DL pipeline.
            #
            # Why StandardScaler and not the old per-subject baseline min-max:
            #   The old normalize_per_subject() computed min/max from each
            #   subject's own baseline windows BEFORE the LOSO split.  This
            #   meant the test subject's features were scaled using their own
            #   physiological data — data leakage.
            #
            #   StandardScaler fit inside the LOSO fold uses ONLY the 14
            #   training subjects' statistics.  The held-out subject's features
            #   are transformed using parameters derived entirely from other
            #   subjects — zero leakage.
            #
            # Note: RF is scale-invariant (threshold-based splits), so this
            # does not affect RF performance directly.  It is included for:
            #   1. PCA-based feature selection (PCA is scale-sensitive)
            #   2. Future classifier substitution (SVM, LDA, etc.)
            #   3. Methodological consistency with the DL pipeline
            if self.normalize:
                scaler = StandardScaler()
                df_train[_feature_cols] = scaler.fit_transform(df_train[_feature_cols])
                df_val[_feature_cols] = scaler.transform(df_val[_feature_cols])

                # Report any zero-variance features (scaler divides by 0 → NaN)
                zero_var = (scaler.scale_ == 0).sum()
                if zero_var > 0:
                    print(
                        f"  Normalization: StandardScaler applied. "
                        f"WARNING: {zero_var} features have zero variance in "
                        f"training fold (set to 0 after scaling)."
                    )
                    # Replace any NaN/inf from zero-variance division
                    df_train[_feature_cols] = df_train[_feature_cols].fillna(0.0)
                    df_val[_feature_cols] = df_val[_feature_cols].fillna(0.0)
                else:
                    print("  Normalization: StandardScaler applied.")
            else:
                print("  Normalization: SKIPPED (normalize=False)")

            # Feature Selection (fit on train only)
            print(f"\n  Feature selection ({self.fs_method}):")
            fs_pipeline = FeatureSelectionPipeline(
                method=self.fs_method, **self.fs_kwargs
            )
            df_train_fs = fs_pipeline.fit_transform(df_train, y_train)
            df_val_fs = fs_pipeline.transform(df_val)

            selected_features = fs_pipeline.selected_features
            print(f"  Selected {len(selected_features)} features")

            # Prepare numpy arrays for sklearn
            X_train = df_train_fs[selected_features].values
            X_val = df_val_fs[selected_features].values

            # Nested RandomizedSearchCV (inner CV on training subjects)
            # Bergstra & Bengio (2012) show random search finds equally good
            # solutions as grid search in fewer evaluations because not all
            # hyperparameters are equally important and random search doesn't
            # waste evaluations exhaustively covering low-impact ones.
            print(
                f"\n  RandomizedSearchCV (n_iter={self.n_iter}, "
                f"inner {self.n_inner_cv}-fold, scoring={self.scoring})..."
            )

            train_groups = df_train["sid"].values

            # Inner CV uses StratifiedGroupKFold to respect both subject
            # boundaries and label distribution, prevents inner folds where
            # stress/baseline proportions vary wildly due to small subject
            # counts per fold.
            inner_cv = StratifiedGroupKFold(
                n_splits=min(self.n_inner_cv, len(df_train["sid"].unique()))
            )

            # n_jobs=1 on the RF: parallelism is owned by RandomizedSearchCV.
            rf = RandomForestClassifier(
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=1,
            )
            random_search = RandomizedSearchCV(
                estimator=rf,
                param_distributions=self.param_grid,
                n_iter=self.n_iter,
                cv=inner_cv,
                scoring=self.scoring,
                n_jobs=-1,
                refit=True,
                random_state=self.random_state,
            )
            random_search.fit(X_train, y_train, groups=train_groups)

            best_params = random_search.best_params_
            best_score = random_search.best_score_
            print(f"  Best inner CV score: {best_score:.4f}")
            print(f"  Best params: {best_params}")

            # ── Evaluate on held-out subject ──
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(X_val)
            y_pred_proba = best_model.predict_proba(X_val)

            # Explicit labels= prevents crashes when a held-out subject is
            # missing a class (e.g. no amusement windows) and ensures
            # confusion matrices are the same shape across all folds and
            # log_loss columns align with predict_proba output.
            all_classes = sorted(self.df["mapped_label"].unique())

            # Compute metrics
            # labels=all_classes ensures consistent shape across folds and
            # penalises folds where a class is absent from the held-out
            # subject (correct for LOSO).
            acc = accuracy_score(y_val, y_pred)
            f1_w = f1_score(
                y_val,
                y_pred,
                average="weighted",
                zero_division=0,
                labels=all_classes,
            )
            f1_m = f1_score(
                y_val,
                y_pred,
                average="macro",
                zero_division=0,
                labels=all_classes,
            )
            prec = precision_score(
                y_val,
                y_pred,
                average="weighted",
                zero_division=0,
                labels=all_classes,
            )
            rec = recall_score(
                y_val,
                y_pred,
                average="weighted",
                zero_division=0,
                labels=all_classes,
            )
            cm = confusion_matrix(y_val, y_pred, labels=all_classes)
            logloss = log_loss(y_val, y_pred_proba, labels=all_classes)

            fold_time = time.time() - fold_start
            print(f"\n  Results for Subject {int(held_out_sid)}:")
            print(f"    Accuracy:        {acc:.4f}")
            print(f"    F1 (weighted):   {f1_w:.4f}")
            print(f"    F1 (macro):      {f1_m:.4f}")
            print(f"    Precision:       {prec:.4f}")
            print(f"    Recall:          {rec:.4f}")
            print(f"    Log Loss:        {logloss:.4f}")
            print(f"    Confusion Matrix:\n{cm}")
            print(f"    Fold time:       {fold_time:.1f}s")

            # Save the trained model to a separate file to avoid bloating
            # the results pickle
            model_path = os.path.join(
                self.output_dir, f"model_S{int(held_out_sid)}.pkl"
            )
            with open(model_path, "wb") as fh:
                pickle.dump(best_model, fh)

            fold_result = {
                "subject": int(held_out_sid),
                "accuracy": acc,
                "f1_weighted": f1_w,
                "f1_macro": f1_m,
                "precision": prec,
                "recall": rec,
                "log_loss": logloss,
                "confusion_matrix": cm,
                "best_params": best_params,
                "best_inner_score": best_score,
                "best_model_path": model_path,
                "n_train": len(df_train),
                "n_val": len(df_val),
                "n_features": len(selected_features),
                "selected_features": selected_features,
                "fold_time": fold_time,
            }
            all_fold_results.append(fold_result)
            all_y_true.extend(y_val.tolist())
            all_y_pred.extend(y_pred.tolist())

        # Aggregate results
        total_time = time.time() - start_time
        results = self._aggregate_results(
            all_fold_results, all_y_true, all_y_pred, total_time
        )

        # Save
        self._save_results(results)

        return results

    def _aggregate_results(
        self,
        fold_results: list[dict],
        all_y_true: list[int],
        all_y_pred: list[int],
        total_time: float,
    ) -> dict:
        """Compute aggregate metrics across all LOSO folds."""
        metrics = [
            "accuracy",
            "f1_weighted",
            "f1_macro",
            "precision",
            "recall",
            "log_loss",
        ]

        print(f"\n{'='*60}")
        print(f"AGGREGATE RESULTS ({len(fold_results)} folds)")
        print(f"{'='*60}")

        aggregate = {}
        for metric in metrics:
            values = [f[metric] for f in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            aggregate[f"{metric}_mean"] = mean_val
            aggregate[f"{metric}_std"] = std_val
            print(f"  {metric:<20s}: {mean_val:.4f} ± {std_val:.4f}")

        # Global confusion matrix (concatenated predictions)
        global_cm = confusion_matrix(all_y_true, all_y_pred)
        global_report = classification_report(all_y_true, all_y_pred, zero_division=0)
        print(f"\nGlobal confusion matrix:\n{global_cm}")
        print(f"\nGlobal classification report:\n{global_report}")

        print(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")

        return {
            "fold_results": fold_results,
            "aggregate": aggregate,
            "global_confusion_matrix": global_cm,
            "global_classification_report": global_report,
            "all_y_true": all_y_true,
            "all_y_pred": all_y_pred,
            "total_time": total_time,
            "config": {
                "label_map": self.label_map,
                "feature_selection": self.fs_method,
                "scoring": self.scoring,
                "param_grid": self.param_grid,
                "n_iter": self.n_iter,
                "n_inner_cv": self.n_inner_cv,
                "search_method": "RandomizedSearchCV",
                "normalize": self.normalize,
                "train_independent_only": self.train_independent_only,
                "nan_threshold": self.nan_threshold,
            },
        }

    def _save_results(self, results: dict):
        """Save results to disk."""
        # Save full results as pickle
        results_path = os.path.join(self.output_dir, "losocv_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nSaved full results → {results_path}")

        # Save summary as CSV
        summary_rows = []
        for fold in results["fold_results"]:
            summary_rows.append(
                {
                    "subject": fold["subject"],
                    "accuracy": fold["accuracy"],
                    "f1_weighted": fold["f1_weighted"],
                    "f1_macro": fold["f1_macro"],
                    "precision": fold["precision"],
                    "recall": fold["recall"],
                    "log_loss": fold["log_loss"],
                    "n_features": fold["n_features"],
                    "n_train": fold["n_train"],
                    "n_val": fold["n_val"],
                    "fold_time": fold["fold_time"],
                }
            )

        # Add aggregate rows (mean and std)
        agg = results["aggregate"]
        summary_rows.append(
            {
                "subject": "MEAN",
                "accuracy": agg["accuracy_mean"],
                "f1_weighted": agg["f1_weighted_mean"],
                "f1_macro": agg["f1_macro_mean"],
                "precision": agg["precision_mean"],
                "recall": agg["recall_mean"],
                "log_loss": agg["log_loss_mean"],
                "n_features": "",
                "n_train": "",
                "n_val": "",
                "fold_time": results["total_time"],
            }
        )
        summary_rows.append(
            {
                "subject": "STD",
                "accuracy": agg["accuracy_std"],
                "f1_weighted": agg["f1_weighted_std"],
                "f1_macro": agg["f1_macro_std"],
                "precision": agg["precision_std"],
                "recall": agg["recall_std"],
                "log_loss": agg["log_loss_std"],
                "n_features": "",
                "n_train": "",
                "n_val": "",
                "fold_time": "",
            }
        )

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(self.output_dir, "losocv_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved summary → {summary_path}")


#  Ablation runner: compare feature selection methods


def run_fs_ablation(
    df_features: pd.DataFrame,
    label_map: dict | None = None,
    methods: list[str] | None = None,
    output_base: str = "results/ablation/",
    **kwargs,
) -> dict:
    """
    Run LOSO-CV with multiple feature selection methods for comparison.

    Parameters
    ----------
    df_features : pd.DataFrame
        Feature matrix.
    label_map : dict
        Label mapping.
    methods : list[str]
        Feature selection methods to compare.
    output_base : str
        Base directory for results.
    **kwargs
        Additional kwargs passed to WESADClassifier.

    Returns
    -------
    dict
        Mapping method name → results dict.
    """
    if label_map is None:
        label_map = LABEL_MAP_BINARY
    if methods is None:
        methods = ["none", "tree", "kbest", "pca"]

    all_results = {}

    for method in methods:
        print("\n")
        print(f"# ABLATION: feature_selection = {method}")
        print("\n")

        output_dir = os.path.join(output_base, f"fs_{method}")

        classifier = WESADClassifier(
            df_features=df_features,
            label_map=label_map,
            feature_selection_method=method,
            output_dir=output_dir,
            **kwargs,
        )
        results = classifier.run_losocv()
        all_results[method] = results

    # Print comparison
    print("\n")
    print("FEATURE SELECTION ABLATION SUMMARY")
    print("\n")
    print(f"{'Method':<12} {'Accuracy':<18} {'F1 (weighted)':<18} {'F1 (macro)':<18}")
    print("\n")

    for method, res in all_results.items():
        agg = res["aggregate"]
        acc = f"{agg['accuracy_mean']:.4f} ± {agg['accuracy_std']:.4f}"
        f1w = f"{agg['f1_weighted_mean']:.4f} ± {agg['f1_weighted_std']:.4f}"
        f1m = f"{agg['f1_macro_mean']:.4f} ± {agg['f1_macro_std']:.4f}"
        print(f"{method:<12} {acc:<18} {f1w:<18} {f1m:<18}")

    # Save comparison
    comparison_path = os.path.join(output_base, "fs_ablation_comparison.pkl")
    with open(comparison_path, "wb") as f:
        pickle.dump(all_results, f)
    print(f"\nSaved ablation comparison → {comparison_path}")

    return all_results
