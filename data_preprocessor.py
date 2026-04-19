from __future__ import annotations
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


class DataPreprocessor:
    """Cleans, validates, transforms, and normalizes a dataset."""

    DEFAULT_INVALID_VALUES = [-99, -9998, -9999]
    DEFAULT_ID_COLUMNS = ["HS_CPF"]

    def __init__(
        self,
        df: pd.DataFrame,
        invalid_values: list[int] | None = None,
        missing_threshold: float = 0.80,
        target_column: str = "TARGET",
        id_columns: list[str] | None = None,
        scaler_method: str = "robust",
    ) -> None:
        """Initializes the preprocessor.

        Args:
            df: Raw dataset to preprocess.
            invalid_values: Values that must be interpreted as missing.
            missing_threshold: Maximum accepted missing ratio per column.
            target_column: Name of the target variable.
            id_columns: Identifier columns that should be removed from model-ready data.
            scaler_method: Numeric scaling method. Supported values are "robust",
                "standard", and "none".

        Returns:
            None.
        """
        self.raw_df = df.copy()
        self.df = df.copy()
        self.invalid_values = invalid_values or self.DEFAULT_INVALID_VALUES
        self.missing_threshold = missing_threshold
        self.target_column = target_column
        self.id_columns = id_columns or self.DEFAULT_ID_COLUMNS
        self.scaler_method = scaler_method.lower()

        self.invalid_values_report = pd.DataFrame()
        self.outliers_report = pd.DataFrame()
        self.normalization_report = pd.DataFrame()
        self.text_cleaning_report = pd.DataFrame()
        self.columns_to_drop: list[str] = []
        self.removed_identifier_columns: list[str] = []
        self.imputation_values: dict[str, object] = {}
        self.normalized_columns: list[str] = []

    def analyze_invalid_values(self) -> pd.DataFrame:
        """Calculates invalid value counts and percentages per column.

        Args:
            None.

        Returns:
            DataFrame with invalid counts, invalid percentages, and drop flags.
        """
        total_rows = len(self.df)
        rows = []

        for column in self.df.columns:
            invalid_count = int(self.df[column].isin(self.invalid_values).sum())
            invalid_percentage = invalid_count / total_rows if total_rows else 0
            rows.append(
                {
                    "column": column,
                    "total_records": total_rows,
                    "invalid_count": invalid_count,
                    "invalid_percentage": invalid_percentage,
                    "invalid_percentage_label": f"{invalid_percentage:.2%}",
                    "drop_column": invalid_percentage > self.missing_threshold,
                }
            )

        self.invalid_values_report = pd.DataFrame(rows).sort_values(
            by=["invalid_percentage", "invalid_count"],
            ascending=[False, False],
        )
        self.columns_to_drop = self.get_columns_above_missing_threshold()
        return self.invalid_values_report

    def replace_invalid_values(self) -> pd.DataFrame:
        """Replaces configured invalid values with NaN.

        Args:
            None.

        Returns:
            DataFrame with invalid values replaced by NaN.
        """
        self.df = self.df.replace(self.invalid_values, np.nan)
        return self.df

    def get_columns_above_missing_threshold(self) -> list[str]:
        """Identifies columns whose invalid ratio is above the configured threshold.

        Args:
            None.

        Returns:
            List of column names to be removed.
        """
        if self.invalid_values_report.empty:
            return []

        columns = self.invalid_values_report.loc[
            self.invalid_values_report["drop_column"],
            "column",
        ].tolist()
        return [column for column in columns if column != self.target_column]

    def drop_high_missing_columns(self) -> pd.DataFrame:
        """Drops columns with missing ratio above the configured threshold.

        Args:
            None.

        Returns:
            DataFrame without high-missing columns.
        """
        existing_columns = [column for column in self.columns_to_drop if column in self.df.columns]
        self.df = self.df.drop(columns=existing_columns)
        return self.df

    def drop_identifier_columns(self) -> pd.DataFrame:
        """Drops identifier columns that should not be used as model features.

        Args:
            None.

        Returns:
            DataFrame without configured identifier columns.
        """
        existing_columns = [column for column in self.id_columns if column in self.df.columns]
        self.removed_identifier_columns = existing_columns
        self.df = self.df.drop(columns=existing_columns)
        return self.df

    def clean_text_columns(self) -> pd.DataFrame:
        """Cleans and standardizes text columns.

        Args:
            None.

        Returns:
            DataFrame with standardized text columns.
        """
        rows = []

        for column in self.df.select_dtypes(include=["object", "string"]).columns:
            before_unique = int(self.df[column].nunique(dropna=False))
            before_missing = int(self.df[column].isna().sum())

            self.df[column] = self.df[column].apply(self._clean_text_value)
            self.df[column] = self.df[column].replace("", np.nan)
            self.df[column] = self.df[column].fillna("DESCONHECIDO")

            after_unique = int(self.df[column].nunique(dropna=False))
            after_missing = int(self.df[column].isna().sum())
            changed_count = int(
                (
                    self.raw_df[column].astype("string").fillna("")
                    != self.df[column].astype("string").fillna("")
                ).sum()
            ) if column in self.raw_df.columns else 0

            rows.append(
                {
                    "column": column,
                    "unique_before": before_unique,
                    "unique_after": after_unique,
                    "missing_before": before_missing,
                    "missing_after": after_missing,
                    "changed_records": changed_count,
                }
            )

        self.text_cleaning_report = pd.DataFrame(rows)
        return self.df

    def detect_outliers_iqr(self) -> pd.DataFrame:
        """Detects outliers using the interquartile range method.

        Args:
            None.

        Returns:
            DataFrame with outlier counts, percentages, and IQR limits.
        """
        rows = []

        for column in self._outlier_candidate_columns():
            series = self.df[column].dropna()
            if series.empty:
                continue

            q1 = float(series.quantile(0.25))
            q3 = float(series.quantile(0.75))
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if iqr == 0:
                outlier_mask = pd.Series(False, index=series.index)
            else:
                outlier_mask = (series < lower_bound) | (series > upper_bound)

            rows.append(
                self._build_outlier_report_row(
                    column=column,
                    method="iqr",
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    outlier_count=int(outlier_mask.sum()),
                    non_null_count=int(series.shape[0]),
                )
            )

        return pd.DataFrame(rows)

    def detect_outliers_zscore(self) -> pd.DataFrame:
        """Detects outliers using the Z-score method.

        Args:
            None.

        Returns:
            DataFrame with outlier counts, percentages, mean, and standard deviation.
        """
        rows = []

        for column in self._outlier_candidate_columns():
            series = self.df[column].dropna()
            if series.empty:
                continue

            mean = float(series.mean())
            std = float(series.std(ddof=0))
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std

            if std == 0:
                outlier_mask = pd.Series(False, index=series.index)
            else:
                zscores = (series - mean) / std
                outlier_mask = zscores.abs() > 3

            rows.append(
                self._build_outlier_report_row(
                    column=column,
                    method="zscore",
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    outlier_count=int(outlier_mask.sum()),
                    non_null_count=int(series.shape[0]),
                    mean=mean,
                    std=std,
                )
            )

        return pd.DataFrame(rows)

    def treat_outliers(self, method: str = "iqr") -> pd.DataFrame:
        """Treats outliers using column-appropriate clipping rules.

        Args:
            method: Outlier detection method to apply. Supported values are "iqr"
                and "zscore".

        Returns:
            DataFrame with treated outliers.

        Raises:
            ValueError: If an unsupported outlier treatment method is provided.
        """
        method = method.lower()
        if method not in {"iqr", "zscore"}:
            raise ValueError("Metodo de outlier invalido. Use 'iqr' ou 'zscore'.")

        iqr_report = self.detect_outliers_iqr()
        zscore_report = self.detect_outliers_zscore()
        reports = [report for report in [iqr_report, zscore_report] if not report.empty]
        self.outliers_report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()

        selected_report = iqr_report if method == "iqr" else zscore_report
        for _, row in selected_report.iterrows():
            column = row["column"]
            lower_bound = row["lower_bound"]
            upper_bound = row["upper_bound"]
            if column not in self.df.columns:
                continue
            self.df[column] = self.df[column].clip(lower=lower_bound, upper=upper_bound)

        if not self.outliers_report.empty:
            self.outliers_report["treatment_applied"] = self.outliers_report["method"] == method

        return self.df

    def impute_missing_values(self) -> pd.DataFrame:
        """Imputes missing values after invalid value replacement.

        Args:
            None.

        Returns:
            DataFrame with missing values imputed.
        """
        for column in self.df.columns:
            if column == self.target_column:
                continue

            if pd.api.types.is_numeric_dtype(self.df[column]):
                median = self.df[column].median()
                fill_value = 0 if pd.isna(median) else median
            else:
                mode = self.df[column].mode(dropna=True)
                fill_value = mode.iloc[0] if not mode.empty else "DESCONHECIDO"

            self.imputation_values[column] = self._json_safe_value(fill_value)
            self.df[column] = self.df[column].fillna(fill_value)

        return self.df

    def normalize_numeric_columns(self) -> pd.DataFrame:
        """Normalizes selected numeric columns.

        Args:
            None.

        Returns:
            DataFrame with normalized numeric columns.

        Raises:
            ValueError: If an unsupported scaler method is provided.
        """
        if self.scaler_method not in {"robust", "standard", "none"}:
            raise ValueError("Scaler invalido. Use 'robust', 'standard' ou 'none'.")

        if self.scaler_method == "none":
            self.normalization_report = pd.DataFrame()
            return self.df

        rows = []

        for column in self._normalization_candidate_columns():
            series = self.df[column].astype(float)
            original_stats = self._series_stats(series, prefix="original")

            if self.scaler_method == "robust":
                median = series.median()
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                scale = q3 - q1
                if scale == 0 or pd.isna(scale):
                    continue
                self.df[column] = (series - median) / scale
                center = median
                center_label = "median"
                scale_label = "iqr"
            else:
                mean = series.mean()
                std = series.std(ddof=0)
                if std == 0 or pd.isna(std):
                    continue
                self.df[column] = (series - mean) / std
                center = mean
                center_label = "mean"
                scale = std
                scale_label = "std"

            self.normalized_columns.append(column)
            normalized_stats = self._series_stats(self.df[column].astype(float), prefix="normalized")
            rows.append(
                {
                    "column": column,
                    "scaler_method": self.scaler_method,
                    "center_type": center_label,
                    "center_value": center,
                    "scale_type": scale_label,
                    "scale_value": scale,
                    **original_stats,
                    **normalized_stats,
                }
            )

        self.normalization_report = pd.DataFrame(rows)

        return self.df

    def preprocess(self, outlier_method: str = "iqr") -> pd.DataFrame:
        """Runs the full preprocessing pipeline.

        Args:
            outlier_method: Outlier treatment method. Supported values are "iqr"
                and "zscore".

        Returns:
            Fully cleaned and normalized DataFrame.
        """
        print("\n=== Pre-processamento iniciado ===")
        self.analyze_invalid_values()
        self.replace_invalid_values()
        self.drop_high_missing_columns()
        self.drop_identifier_columns()
        self.clean_text_columns()
        self.treat_outliers(method=outlier_method)
        self.impute_missing_values()
        self.normalize_numeric_columns()
        print("=== Pre-processamento concluido ===")
        return self.df

    def save_outputs(self, output_dir: str = "outputs") -> None:
        """Saves cleaned data and preprocessing reports.

        Args:
            output_dir: Directory where output files will be saved.

        Returns:
            None.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.df.to_csv(output_path / "cleaned_train.csv", index=False)

        if not self.invalid_values_report.empty:
            self.invalid_values_report.to_csv(output_path / "invalid_values_report.csv", index=False)

        if not self.outliers_report.empty:
            self.outliers_report.to_csv(output_path / "outliers_report.csv", index=False)

        if not self.normalization_report.empty:
            self.normalization_report.to_csv(output_path / "normalization_report.csv", index=False)

        if not self.text_cleaning_report.empty:
            self.text_cleaning_report.to_csv(output_path / "text_cleaning_report.csv", index=False)

        summary = self._build_summary()
        with (output_path / "preprocessing_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2, ensure_ascii=False)

        print(f"Arquivos salvos em: {output_path.resolve()}")

    def print_report_summary(self) -> None:
        """Displays a compact summary of preprocessing reports.

        Args:
            None.

        Returns:
            None.
        """
        print("\n=== Resumo do pre-processamento ===")
        print(f"Shape original: {self.raw_df.shape}")
        print(f"Shape final: {self.df.shape}")
        print(f"Valores invalidos tratados como NaN: {self.invalid_values}")
        print(f"Threshold de remocao: > {self.missing_threshold:.0%}")
        print(f"Colunas removidas por invalidos: {self.columns_to_drop}")
        print(f"Colunas identificadoras removidas: {self.removed_identifier_columns}")
        self.print_outliers_summary()
        self.print_normalization_summary()

    def print_outliers_summary(self) -> None:
        """Displays outlier detection and treatment details.

        Args:
            None.

        Returns:
            None.
        """
        print("\n=== Outliers ===")
        if self.outliers_report.empty:
            print("Nenhuma coluna elegivel para analise de outliers.")
            return

        method_summary = (
            self.outliers_report.groupby("method")
            .agg(
                analyzed_columns=("column", "count"),
                columns_with_outliers=("outlier_count", lambda values: int((values > 0).sum())),
                total_outliers=("outlier_count", "sum"),
            )
            .reset_index()
        )
        print("\nResumo por metodo:")
        print(method_summary.to_string(index=False))

        applied = self.outliers_report[
            (self.outliers_report["treatment_applied"]) & (self.outliers_report["outlier_count"] > 0)
        ].copy()

        if applied.empty:
            print("\nNenhum outlier tratado pelo metodo aplicado.")
            return

        applied = applied.sort_values(
            by=["outlier_percentage", "outlier_count"],
            ascending=[False, False],
        )
        display_columns = [
            "column",
            "method",
            "lower_bound",
            "upper_bound",
            "non_null_count",
            "outlier_count",
            "outlier_percentage_label",
        ]
        print("\nColunas com outliers tratados:")
        print(
            applied[display_columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    def print_normalization_summary(self) -> None:
        """Displays detailed normalization information for each normalized column.

        Args:
            None.

        Returns:
            None.
        """
        print("\n=== Normalizacao ===")
        if self.scaler_method == "none":
            print("Normalizacao desativada por configuracao.")
            return

        if self.normalization_report.empty:
            print("Nenhuma coluna foi normalizada.")
            return

        display_columns = [
            "column",
            "scaler_method",
            "center_type",
            "center_value",
            "scale_type",
            "scale_value",
            "original_min",
            "original_median",
            "original_max",
            "normalized_min",
            "normalized_median",
            "normalized_max",
        ]
        print(
            self.normalization_report[display_columns].to_string(
                index=False,
                float_format=lambda value: f"{value:.4f}",
            )
        )

    def _outlier_candidate_columns(self) -> list[str]:
        """Identifies numeric columns eligible for outlier detection.

        Args:
            None.

        Returns:
            List of numeric columns eligible for outlier detection.
        """
        candidates = []

        for column in self.df.select_dtypes(include=[np.number]).columns:
            if column == self.target_column or column in self.id_columns:
                continue

            series = self.df[column].dropna()
            unique_count = series.nunique()
            if unique_count > 20:
                candidates.append(column)

        return candidates

    def _normalization_candidate_columns(self) -> list[str]:
        """Identifies numeric columns eligible for normalization.

        Args:
            None.

        Returns:
            List of numeric columns eligible for normalization.
        """
        candidates = []

        for column in self.df.select_dtypes(include=[np.number]).columns:
            if column == self.target_column or column in self.id_columns:
                continue

            series = self.df[column].dropna()
            if series.nunique() > 20:
                candidates.append(column)

        return candidates

    def _build_outlier_report_row(
        self,
        column: str,
        method: str,
        lower_bound: float,
        upper_bound: float,
        outlier_count: int,
        non_null_count: int,
        mean: float | None = None,
        std: float | None = None,
    ) -> dict[str, object]:
        """Builds a standardized outlier report row.

        Args:
            column: Column analyzed for outliers.
            method: Outlier detection method used.
            lower_bound: Lower outlier threshold.
            upper_bound: Upper outlier threshold.
            outlier_count: Number of detected outliers.
            non_null_count: Number of non-null records evaluated.
            mean: Column mean, when available.
            std: Column standard deviation, when available.

        Returns:
            Dictionary representing one outlier report row.
        """
        outlier_percentage = outlier_count / non_null_count if non_null_count else 0
        return {
            "column": column,
            "method": method,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "mean": np.nan if mean is None else mean,
            "std": np.nan if std is None else std,
            "non_null_count": non_null_count,
            "outlier_count": outlier_count,
            "outlier_percentage": outlier_percentage,
            "outlier_percentage_label": f"{outlier_percentage:.2%}",
        }

    def _build_summary(self) -> dict[str, object]:
        """Builds a JSON-serializable preprocessing summary.

        Args:
            None.

        Returns:
            Dictionary with preprocessing metadata and output statistics.
        """
        return {
            "original_shape": list(self.raw_df.shape),
            "final_shape": list(self.df.shape),
            "invalid_values": self.invalid_values,
            "missing_threshold": self.missing_threshold,
            "columns_removed_by_missing_threshold": self.columns_to_drop,
            "identifier_columns_removed": self.removed_identifier_columns,
            "target_column": self.target_column,
            "scaler_method": self.scaler_method,
            "normalized_columns": self.normalized_columns,
            "imputation_values": self.imputation_values,
        }

    @staticmethod
    def _series_stats(series: pd.Series, prefix: str) -> dict[str, float]:
        """Calculates descriptive statistics for a numeric series.

        Args:
            series: Numeric series to summarize.
            prefix: Prefix to add to each statistic name.

        Returns:
            Dictionary with count, mean, standard deviation, quartiles, min, and max.
        """
        return {
            f"{prefix}_count": float(series.count()),
            f"{prefix}_mean": float(series.mean()),
            f"{prefix}_std": float(series.std(ddof=0)),
            f"{prefix}_min": float(series.min()),
            f"{prefix}_q1": float(series.quantile(0.25)),
            f"{prefix}_median": float(series.median()),
            f"{prefix}_q3": float(series.quantile(0.75)),
            f"{prefix}_max": float(series.max()),
        }

    @staticmethod
    def _clean_text_value(value: object) -> object:
        """Standardizes a single text value.

        Args:
            value: Raw text value to standardize.

        Returns:
            Standardized text value or NaN when the original value is missing.
        """
        if pd.isna(value):
            return np.nan

        text = str(value).strip().upper()
        text = unicodedata.normalize("NFKD", text)
        text = "".join(char for char in text if not unicodedata.combining(char))
        text = re.sub(r"\s+", " ", text)
        return text

    @staticmethod
    def _json_safe_value(value: object) -> object:
        """Converts numpy and pandas scalar values into JSON-safe values.

        Args:
            value: Value to convert.

        Returns:
            JSON-serializable value.
        """
        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        return value
