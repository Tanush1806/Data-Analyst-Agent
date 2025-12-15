import os
import json
from typing import Dict, Any, List
import pandas as pd
import plotly.express as px

# Optional dependencies
try:
    from fpdf import FPDF
    HAS_FPDF = True
except ImportError:
    HAS_FPDF = False

try:
    import kaleido
    HAS_KALEIDO = True
except ImportError:
    HAS_KALEIDO = False


# ------------------ CHART RECOMMENDER ------------------

class ChartRecommender:
    def recommend(
        self,
        column_types: Dict[str, List[str]],
        profiling: Dict[str, Any],
        max_plots: int = 30
    ) -> Dict[str, Any]:

        plots = []

        numeric = column_types.get("numeric", [])
        categorical = column_types.get("categorical", [])
        datetime_cols = column_types.get("datetime", [])

        for col in numeric:
            plots.append({"type": "histogram", "column": col, "title": f"Distribution of {col}"})
            plots.append({"type": "box", "column": col, "title": f"Boxplot of {col}"})

        for col in categorical:
            plots.append({"type": "bar", "column": col, "title": f"Count of {col}"})

        for i, x in enumerate(numeric):
            for y in numeric[i+1:]:
                plots.append({"type": "scatter", "x": x, "y": y, "title": f"{x} vs {y}"})

        for dt in datetime_cols:
            for num in numeric:
                plots.append({"type": "line", "x": dt, "y": num, "title": f"{num} over time"})

        return {"plots": plots[:max_plots]}


# ------------------ PLOT GENERATOR ------------------

class PlotGenerator:
    def __init__(self, output_dir: str, data_path: str):
        self.output_dir = output_dir
        self.data_path = data_path
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_from_specs(self, specs: Dict[str, Any]) -> List[Dict[str, Any]]:
        df = pd.read_csv(self.data_path)
        results = []

        for i, spec in enumerate(specs.get("plots", [])):
            fig = self._create_figure(df, spec)
            if not fig:
                continue

            fig_json = fig.to_json()
            img_path = None

            if HAS_KALEIDO:
                try:
                    img_path = os.path.join(self.output_dir, f"plot_{i}.png")
                    fig.write_image(img_path)
                except Exception as e:
                    print(f"Warning: Image export failed for plot {i}: {e}")
                    img_path = None

            results.append({
                "spec": spec,
                "figure_json": fig_json,
                "image_path": img_path
            })

        return results

    def _create_figure(self, df: pd.DataFrame, spec: Dict[str, Any]):
        t = spec["type"]
        title = spec.get("title", "")

        if t == "histogram":
            return px.histogram(df, x=spec["column"], title=title)
        if t == "box":
            return px.box(df, y=spec["column"], title=title)
        if t == "bar":
            counts = df[spec["column"]].value_counts().reset_index()
            counts.columns = ["value", "count"]
            return px.bar(counts, x="value", y="count", title=title)
        if t == "scatter":
            return px.scatter(df, x=spec["x"], y=spec["y"], title=title)
        if t == "line":
            return px.line(df.sort_values(spec["x"]), x=spec["x"], y=spec["y"], title=title)

        return None


# ------------------ REPORT BUILDER ------------------

class ReportBuilder:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def build_report(self, generated_plots: List[Dict[str, Any]], insights: str) -> str:
        out_path = os.path.join(self.output_dir, "eda_report.pdf")

        if not HAS_FPDF:
            txt = os.path.join(self.output_dir, "report_error.txt")
            with open(txt, "w") as f:
                f.write(insights)
            return txt

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "AI Generated EDA Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", "", 10)
        pdf.multi_cell(0, 6, insights)
        pdf.ln(10)

        for p in generated_plots:
            if p["image_path"] and os.path.exists(p["image_path"]):
                pdf.add_page()
                pdf.image(p["image_path"], w=170)

        pdf.output(out_path)
        return out_path
