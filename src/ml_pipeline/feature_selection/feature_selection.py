# WESAD Feature Selection
#
# Implements multiple feature selection strategies for the Random Forest pipeline.
# All selectors are designed to be fit inside the CV loop on training data only
# to prevent data leakage.
#
# Strategies implemented:
#
#   1. Variance Threshold: removes features with near-zero variance (uninformative).
#      Applied first before more expensive methods.
#
#   2. Tree-based Feature Selection (Tree FS): fits a preliminary Random Forest and
#      selects features whose importance exceeds the mean importance.
#      Literature (see sources in project notes) reports Tree FS outperforms PCA
#      for stress detection because HRV/EDA features are too heterogeneous to compress
#      into shared principal components, while Tree FS correctly identifies and removes
#      noisy features.
#
#   3. SelectKBest (univariate): selects top-K features via ANOVA F-test.
#      Useful as a comparison baseline and for dimensionality reduction before RF.
#
#   4. PCA: included as a secondary option for comparison, though literature suggests
#      it underperforms Tree FS for this domain.
#
# Usage:
#   All selectors follow the sklearn fit/transform pattern. They are called inside
#   the LOSO-CV loop: fit on training fold, transform both train and test fold.
#
# References:
#   Schmidt, P. et al. (2018). Introducing WESAD. ICMI '18.
#   Sriramprakash et al. (2017). Stress detection in working people.
#   Pourmohammadi & Maleki (2020). Stress detection using ECG and EMG signals.

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
)

#  Metadata column handling

_META_COLS = ["sid", "label", "window_idx", "is_overlapped"]


def _split_meta_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Separate metadata columns from feature columns."""
    meta_cols = [c for c in _META_COLS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in _META_COLS]
    return df[meta_cols], feat_cols


#  1. Variance Threshold


class VarianceSelector:
    # Remove features with variance below a threshold.

    # Features that are near-constant across all training windows carry no discriminative information.

    # Parameters
    # threshold : float
    #     Minimum variance to keep a feature. Default 0.0 removes only
    #     zero-variance (constant) features. A small positive value like
    #     0.01 is more aggressive.

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self._selector = VarianceThreshold(threshold=threshold)
        self._feature_names: list[str] = []

    # extract feature columns, keep the ones based on the boolean array returned in the mask
    def fit(self, X: pd.DataFrame, _: object = None) -> "VarianceSelector":
        _, feat_cols = _split_meta_features(X)
        self._selector.fit(X[feat_cols])
        mask = self._selector.get_support()
        self._feature_names = [feat_cols[i] for i, keep in enumerate(mask) if keep]
        n_dropped = len(feat_cols) - len(self._feature_names)
        print(
            f"  VarianceSelector: {len(feat_cols)} → {len(self._feature_names)} "
            f"features (dropped {n_dropped}, threshold={self.threshold})"
        )
        return self

    # reassemble df with the feature columns being kept by concatenationg with the metadata ones
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        meta, _ = _split_meta_features(X)
        X_selected = X[self._feature_names].copy()
        return pd.concat(
            [meta.reset_index(drop=True), X_selected.reset_index(drop=True)], axis=1
        )

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    @property
    def selected_features(self) -> list[str]:
        return self._feature_names


#  2. Tree-based Feature Selection


class TreeFeatureSelector:
    # Select features using a preliminary Random Forest's feature importances.

    # Fits a Random Forest on the training data, computes mean decrease
    # impurity (MDI) for each feature, and keeps features whose importance
    # exceeds the mean importance (sklearn's default threshold="mean").

    # Parameters
    # n_estimators : int
    #     Number of trees in the preliminary RF (default: 200).
    # threshold : str or float
    #     Feature importance threshold. "mean" keeps features above the
    #     mean importance. Can also be a float or "median".
    # random_state : int
    #     Random seed for reproducibility.

    def __init__(
        self,
        n_estimators: int = 200,
        threshold: str = "mean",
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.random_state = random_state
        self._rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._selector: SelectFromModel | None = None
        self._feature_names: list[str] = []
        self._importances: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TreeFeatureSelector":
        _, feat_cols = _split_meta_features(X)
        X_feat = X[feat_cols]

        self._rf.fit(X_feat, y)
        self._selector = SelectFromModel(
            self._rf, threshold=self.threshold, prefit=True
        )

        mask = self._selector.get_support()
        self._feature_names = [feat_cols[i] for i, keep in enumerate(mask) if keep]

        # Store importances for inspection / plotting
        self._importances = pd.Series(
            self._rf.feature_importances_, index=feat_cols
        ).sort_values(ascending=False)

        n_dropped = len(feat_cols) - len(self._feature_names)
        print(
            f"  TreeFeatureSelector: {len(feat_cols)} → {len(self._feature_names)} "
            f"features (dropped {n_dropped}, threshold={self.threshold})"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        meta, _ = _split_meta_features(X)
        X_selected = X[self._feature_names].copy()
        return pd.concat(
            [meta.reset_index(drop=True), X_selected.reset_index(drop=True)], axis=1
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    @property
    def selected_features(self) -> list[str]:
        return self._feature_names

    @property
    def importances(self) -> pd.Series:
        """Feature importances from the preliminary RF, sorted descending."""
        return self._importances


#  3. SelectKBest (univariate ANOVA F-test)


class KBestSelector:
    # Select top-K features using univariate ANOVA F-test.

    # Each feature is scored independently by its F-statistic between
    # class labels. (ratio of between-class variance to within-class variance). high Fstatistic is desired for classes that separate well

    # Parameters

    # k : int or "all"
    #     Number of features to select. Default 50.

    def __init__(self, k: int = 50):
        self.k = k
        self._selector = SelectKBest(score_func=f_classif, k=k)
        self._feature_names: list[str] = []
        self._scores: pd.Series | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "KBestSelector":
        _, feat_cols = _split_meta_features(X)
        X_feat = X[feat_cols]

        # in case there are less than 50 features after variance threshoold
        actual_k = min(self.k, len(feat_cols))
        if actual_k != self.k:
            print(
                f"  KBestSelector: k={self.k} clamped to {actual_k} (only {len(feat_cols)} features)"
            )
            self._selector.k = actual_k

        self._selector.fit(X_feat, y)
        mask = self._selector.get_support()
        self._feature_names = [feat_cols[i] for i, keep in enumerate(mask) if keep]

        # Store scores for inspection
        self._scores = pd.Series(self._selector.scores_, index=feat_cols).sort_values(
            ascending=False
        )

        print(
            f"  KBestSelector: {len(feat_cols)} → {len(self._feature_names)} "
            f"features (k={actual_k})"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        meta, _ = _split_meta_features(X)
        X_selected = X[self._feature_names].copy()
        return pd.concat(
            [meta.reset_index(drop=True), X_selected.reset_index(drop=True)], axis=1
        )

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    @property
    def selected_features(self) -> list[str]:
        return self._feature_names

    @property
    def scores(self) -> pd.Series:
        """ANOVA F-scores per feature, sorted descending."""
        return self._scores


#  4. PCA


class PCASelector:
    # Dimensionality reduction via PCA.

    # Included as a comparison baseline.

    # Parameters

    # n_components : float or int
    #     If float in (0, 1), selects enough components to explain that
    #     fraction of variance (default: 0.95 = 95% variance explained).
    #     If int, selects exactly that many components.

    def __init__(self, n_components: float = 0.95):
        self.n_components = n_components
        self._pca = PCA(n_components=n_components)
        self._component_names: list[str] = []

    def fit(self, X: pd.DataFrame, _: object = None) -> "PCASelector":
        _, feat_cols = _split_meta_features(X)
        X_feat = X[feat_cols].values

        self._pca.fit(X_feat)
        n_out = self._pca.n_components_
        self._component_names = [f"PC_{i+1}" for i in range(n_out)]
        variance_explained = self._pca.explained_variance_ratio_.sum()

        print(
            f"  PCASelector: {len(feat_cols)} → {n_out} components "
            f"({variance_explained:.1%} variance explained)"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        meta, feat_cols = _split_meta_features(X)
        X_pca = self._pca.transform(X[feat_cols].values)
        df_pca = pd.DataFrame(X_pca, columns=self._component_names)
        return pd.concat(
            [meta.reset_index(drop=True), df_pca.reset_index(drop=True)], axis=1
        )

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    @property
    def selected_features(self) -> list[str]:
        return self._component_names


class FeatureSelectionPipeline:
    #     Chains VarianceThreshold (always first) with a configurable second
    #     stage selector.

    #     This is the object that gets called inside each LOSO-CV fold:
    #         pipeline.fit(X_train, y_train)
    #         X_train_selected = pipeline.transform(X_train)
    #         X_val_selected = pipeline.transform(X_val)

    #     Parameters
    #     method : str
    #         Second-stage method: "tree", "kbest", "pca", or "none".
    #     variance_threshold : float
    #         Minimum variance for the first stage (default: 0.0).
    #     **kwargs
    #         Passed to the second-stage selector (e.g. k=50, n_components=0.95).

    def __init__(
        self,
        method: str = "tree",
        variance_threshold: float = 0.0,
        **kwargs,
    ):
        self.method = method
        self._variance = VarianceSelector(threshold=variance_threshold)
        self._selector: TreeFeatureSelector | KBestSelector | PCASelector | None

        match method:
            case "tree":
                self._selector = TreeFeatureSelector(**kwargs)
            case "kbest":
                self._selector = KBestSelector(**kwargs)
            case "pca":
                self._selector = PCASelector(**kwargs)
            case "none":
                self._selector = None
            case _:
                raise ValueError(f"Unknown feature selection method: {method}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelectionPipeline":
        # Stage 1: variance threshold
        X_var = self._variance.fit_transform(X, y)

        # Stage 2: method-specific
        if self._selector is not None:
            self._selector.fit(X_var, y)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_var = self._variance.transform(X)
        if self._selector is not None:
            return self._selector.transform(X_var)
        return X_var

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.fit(X, y)
        return self.transform(X)

    @property
    def selected_features(self) -> list[str]:
        if self._selector is not None:
            return self._selector.selected_features
        return self._variance.selected_features
