"""
data/preprocess.py

Metadata preprocessing pipeline for HAM10000:
  - Handles missing values (age → median imputation)
  - One-hot encodes categorical features (sex, localization)
  - Scales numerical features (age)
  - Returns a fitted MetadataPreprocessor that can transform new samples
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import pickle
from pathlib import Path

from config import META_CATEGORICAL, META_NUMERICAL, METADATA_CSV


class MetadataPreprocessor:
    """
    Fits a scikit-learn ColumnTransformer on HAM10000 metadata and
    exposes `transform()` for use at inference time.

    Categorical features: sex, localization  → one-hot encoded
    Numerical features:   age                → median-imputed + z-scaled

    Usage::

        prep = MetadataPreprocessor()
        X_train = prep.fit_transform(df_train)
        X_val   = prep.transform(df_val)
        prep.save("data/preprocessor.pkl")
    """

    def __init__(self):
        numerical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        self.transformer = ColumnTransformer([
            ("num", numerical_pipeline,  META_NUMERICAL),
            ("cat", categorical_pipeline, META_CATEGORICAL),
        ])
        self._fitted = False

    # ── Fit / transform ───────────────────────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> "MetadataPreprocessor":
        self.transformer.fit(df[META_NUMERICAL + META_CATEGORICAL])
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() or load() before transform().")
        return self.transformer.transform(
            df[META_NUMERICAL + META_CATEGORICAL]
        ).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def output_dim(self) -> int:
        """Number of features after encoding."""
        return self.transformer.transform(
            pd.DataFrame([[0, "unknown", "unknown"]],
                         columns=META_NUMERICAL + META_CATEGORICAL)
        ).shape[1]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Preprocessor] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "MetadataPreprocessor":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[Preprocessor] Loaded from {path}")
        return obj


# ── Standalone helper ─────────────────────────────────────────────────────────

def load_and_split_metadata(csv_path: str = str(METADATA_CSV),
                             train_frac: float = 0.80,
                             val_frac: float = 0.10,
                             seed: int = 42):
    """
    Loads HAM10000 metadata CSV and returns stratified train / val / test splits.

    Returns:
        df_train, df_val, df_test — DataFrames with all original columns.
    """
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset="image_id")    # remove duplicate lesion images

    # Stratified split on dx (diagnosis label)
    from sklearn.model_selection import train_test_split

    df_train, df_temp = train_test_split(
        df, train_size=train_frac, stratify=df["dx"], random_state=seed
    )
    relative_val = val_frac / (1 - train_frac)
    df_val, df_test = train_test_split(
        df_temp, train_size=relative_val, stratify=df_temp["dx"], random_state=seed
    )

    print(f"[Preprocess] Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


if __name__ == "__main__":
    df_train, df_val, df_test = load_and_split_metadata()
    prep = MetadataPreprocessor()
    X_train = prep.fit_transform(df_train)
    print(f"Metadata feature dim: {X_train.shape[1]}")
    prep.save("data/preprocessor.pkl")