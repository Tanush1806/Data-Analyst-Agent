import numpy as np
import pandas as pd
import duckdb
from typing import Dict, Any, List, Optional
from scipy.stats import chi2_contingency
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression


class EDAProfiler:
    """
    Improved EDA Profiler
    - Same logic as original
    - DuckDB used for heavy aggregations
    - Pandas retained where appropriate
    """

    def __init__(
        self,
        schema: Optional[Dict[str, Any]] = None,
        top_k_categories: int = 20,
        corr_threshold: float = 0.3
    ):
        self.schema = schema
        self.top_k = top_k_categories
        self.corr_threshold = corr_threshold

    # ============================================================================
    # 1. MAIN PROFILE
    # ============================================================================

    def profile(self, df: pd.DataFrame) -> Dict[str, Any]:
        column_types = self.detect_column_types(df)
        summary = self.generate_summary(df, column_types)
        return {"column_types": column_types, "summary": summary}

    # ============================================================================
    # TYPE DETECTION (UNCHANGED LOGIC)
    # ============================================================================

    def detect_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        columns = df.columns
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime64[ns]"]).columns.tolist()

        boolean = [c for c in columns if df[c].dropna().isin([0, 1, True, False]).all()]
        categorical = [c for c in columns if df[c].dtype == "object" and c not in boolean]

        text_cols = [
            c for c in categorical
            if df[c].dropna().astype(str).str.len().mean() > 30
        ]

        id_cols = [c for c in columns if df[c].nunique() >= len(df) * 0.90]
        categorical = [c for c in categorical if c not in text_cols + id_cols]

        return {
            "numeric": numeric,
            "categorical": categorical,
            "datetime": datetime_cols,
            "boolean": boolean,
            "text": text_cols,
            "id": id_cols
        }

    # ============================================================================
    # HELPER: SQL ESCAPING
    # ============================================================================

    def _escape(self, identifier: str) -> str:
        """Escape SQL identifiers (double quotes)"""
        return '"' + identifier.replace('"', '""') + '"'

    # ============================================================================
    # SUMMARY STATISTICS (DUCKDB OPTIMIZED)
    # ============================================================================

    def generate_summary(self, df: pd.DataFrame, column_types: Dict[str, List[str]]) -> Dict[str, Any]:
        con = duckdb.connect()
        con.register("data", df)

        summary = {
            "shape": df.shape,
            "missing": {
                c: con.execute(
                    f"SELECT COUNT(*) FROM data WHERE {self._escape(c)} IS NULL"
                ).fetchone()[0]
                for c in df.columns
            },
            "dtypes": df.dtypes.astype(str).to_dict()
        }

        if column_types["numeric"]:
            numeric_summary = {}
            for c in column_types["numeric"]:
                row = con.execute(f"""
                    SELECT
                        AVG({self._escape(c)}) AS mean,
                        STDDEV({self._escape(c)}) AS std,
                        MIN({self._escape(c)}) AS min,
                        MAX({self._escape(c)}) AS max
                    FROM data
                """).df().iloc[0].to_dict()
                numeric_summary[c] = row
            summary["numeric_summary"] = numeric_summary

        for c in column_types["categorical"]:
            counts = con.execute(f"""
                SELECT {self._escape(c)}, COUNT(*) AS cnt
                FROM data
                GROUP BY {self._escape(c)}
                ORDER BY cnt DESC
                LIMIT {self.top_k}
            """).df()
            summary[f"{c}_value_counts"] = dict(zip(counts[c], counts["cnt"]))

        con.close()
        return summary

    # ============================================================================
    # 2. UNIVARIATE ANALYSIS (DUCKDB + PANDAS)
    # ============================================================================

    def univariate(self, df: pd.DataFrame) -> Dict[str, Any]:
        con = duckdb.connect()
        con.register("data", df)

        result = {}
        for col in df.columns:
            info = {
                "missing_pct": float(
                    con.execute(
                        f"SELECT COUNT(*) FROM data WHERE {self._escape(col)} IS NULL"
                    ).fetchone()[0] / len(df)
                ),
                "unique": int(df[col].nunique())
            }

            if np.issubdtype(df[col].dtype, np.number):
                stats = con.execute(f"""
                    SELECT
                        AVG({self._escape(col)}) AS mean,
                        STDDEV({self._escape(col)}) AS std,
                        MIN({self._escape(col)}) AS min,
                        MAX({self._escape(col)}) AS max
                    FROM data
                """).df().iloc[0].to_dict()
                stats["skew"] = float(df[col].skew())
                info.update(stats)
            else:
                vc = df[col].value_counts().head(self.top_k)
                info["top_categories"] = vc.to_dict()

            result[col] = info

        con.close()
        return result

    # ============================================================================
    # 3. BIVARIATE ANALYSIS (PRUNED + DUCKDB)
    # ============================================================================

    def bivariate(self, df: pd.DataFrame) -> Dict[str, Any]:
        con = duckdb.connect()
        con.register("data", df)

        numeric = df.select_dtypes(include=[np.number]).columns
        categorical = df.select_dtypes(include=["object", "category"]).columns

        result = {"numeric_numeric": {}, "categorical_numeric": {}, "categorical_categorical": {}}

        # numeric vs numeric (thresholded)
        for i, c1 in enumerate(numeric):
            for c2 in numeric[i + 1:]:
                corr = con.execute(
                    f"SELECT corr({self._escape(c1)}, {self._escape(c2)}) FROM data"
                ).fetchone()[0]
                if corr is not None and abs(corr) >= self.corr_threshold:
                    result["numeric_numeric"][f"{c1}__{c2}"] = float(corr)

        # categorical vs numeric (mean + count)
        for c in categorical:
            for n in numeric:
                grp = df.groupby(c)[n].agg(["mean", "count"]).head(self.top_k)
                result["categorical_numeric"][f"{c}__{n}"] = grp.to_dict()

        # categorical vs categorical (contingency)
        for i, c1 in enumerate(categorical):
            for c2 in categorical[i + 1:]:
                table = pd.crosstab(df[c1], df[c2])
                result["categorical_categorical"][f"{c1}__{c2}"] = table.to_dict()

        con.close()
        return result

    # ============================================================================
    # 4. CORRELATION ANALYSIS (UNCHANGED LOGIC, OPTIMIZED)
    # ============================================================================

    def correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        result = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        result["pearson"] = (
            df[numeric_cols].corr().round(3).to_dict()
            if len(numeric_cols) >= 2 else {}
        )

        result["cramers_v"] = self.cramers_v_matrix(df)
        result["mutual_information"] = self.mutual_information(df)

        return result

    # ============================================================================
    # Cramér’s V (UNCHANGED)
    # ============================================================================

    def cramers_v(self, x, y):
        table = pd.crosstab(x, y)
        chi2 = chi2_contingency(table)[0]
        n = table.sum().sum()
        phi2 = chi2 / n
        r, k = table.shape
        return np.sqrt(phi2 / max(1, min(k - 1, r - 1)))

    def cramers_v_matrix(self, df):
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        result = {}
        for i, c1 in enumerate(cat_cols):
            for c2 in cat_cols[i + 1:]:
                try:
                    result[f"{c1}__{c2}"] = float(self.cramers_v(df[c1], df[c2]))
                except Exception:
                    result[f"{c1}__{c2}"] = None
        return result

    # ============================================================================
    # Mutual Information (UNCHANGED LOGIC)
    # ============================================================================

    def mutual_information(self, df):
        result = {}
        numeric = df.select_dtypes(include=[np.number]).columns

        if len(numeric) > 1:
            target = numeric[0]
            X = df[numeric[1:]].fillna(0)
            y = df[target].fillna(0)
            mi = mutual_info_regression(X, y)
            for col, score in zip(X.columns, mi):
                result[f"{col}__{target}"] = float(score)

        return result

    # ============================================================================
    # SUPERVISED PROFILING (UNCHANGED)
    # ============================================================================

    def supervised_profile(self, df: pd.DataFrame, target: str) -> Dict[str, Any]:
        y = df[target]
        info = {
            "target_dtype": str(y.dtype),
            "missing_pct": float(y.isnull().mean()),
            "unique": int(y.nunique())
        }

        if np.issubdtype(y.dtype, np.number):
            info.update({
                "mean": float(y.mean()),
                "std": float(y.std()),
                "min": float(y.min()),
                "max": float(y.max()),
                "skew": float(y.skew())
            })
            numeric = df.select_dtypes(include=[np.number]).columns
            info["feature_correlations"] = df[numeric].corr()[target].round(3).to_dict()
        else:
            counts = y.value_counts().to_dict()
            info["class_distribution"] = counts
            info["imbalance_ratio"] = float(max(counts.values()) / min(counts.values()))

        return info

    # ============================================================================
    # BASIC INSIGHTS (STRUCTURED OUTPUT)
    # ============================================================================

    def basic_insights_from_summary(self, summary: Dict[str, Any]) -> List[str]:
        insights = []

        missing_total = sum(summary.get("missing", {}).values())
        if missing_total > 0:
            insights.append(f"Dataset contains {missing_total} missing values.")

        for col, stats in summary.get("numeric_summary", {}).items():
            if abs(stats.get("std", 0)) > 0 and abs(stats.get("skew", 0)) > 1:
                insights.append(f"{col} is highly skewed.")

        if not insights:
            insights.append("No major anomalies detected using rule-based profiling.")

        return insights
