"""
Data Analyst Orchestrator Agent
Coordinates:
1. EDA Agent
2. Visualization Agent

Single entrypoint for the entire data analyst pipeline.
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime

from agents.eda_agent import run_eda
from agents.visualization_agent import run_visualization


class DataAnalystOrchestrator:
    """
    Orchestrates end-to-end data analysis workflow.
    Starts with a PRE-CLEANED dataset.
    """

    def __init__(
        self,
        cleaned_file_path: str,
        file_type: str = "csv",
        output_root: str = "data"
    ):
        self.cleaned_path = cleaned_file_path
        self.file_type = file_type
        self.output_root = output_root

        self.eda_dir = None
        self.viz_dir = None

        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # ------------------ STEP 1: EDA ------------------

    def run_eda(self, target_column: Optional[str] = None) -> Dict[str, Any]:
        print("▶ Running EDA Agent...")

        eda_dir = os.path.join(self.output_root, "eda")

        result = run_eda(
            cleaned_path=self.cleaned_path,
            target_column=target_column,
            eda_dir=eda_dir
        )

        if not result.get("success"):
            raise RuntimeError(f"EDA failed: {result.get('error')}")

        self.eda_dir = result["eda_dir"]
        print(f"✔ EDA complete → {self.eda_dir}")
        return result

    # ------------------ STEP 2: VISUALIZATION ------------------

    def run_visualization(self, max_plots: int = 30) -> Dict[str, Any]:
        print("▶ Running Visualization Agent...")

        viz_dir = os.path.join(self.output_root, "visualizations")

        result = run_visualization(
            eda_dir=self.eda_dir,
            viz_dir=viz_dir,
            max_plots=max_plots
        )

        if not result.get("success"):
            raise RuntimeError(f"Visualization failed: {result.get('error')}")

        self.viz_dir = result["viz_dir"]
        print(f"✔ Visualization complete → {self.viz_dir}")
        return result

    # ------------------ FULL PIPELINE ------------------

    def run_full_pipeline(
        self,
        target_column: Optional[str] = None,
        max_plots: int = 30
    ) -> Dict[str, Any]:
        """
        Single-call execution of the full Data Analyst workflow.
        """

        print(f"\\n🚀 Starting Data Analyst Pipeline [{self.run_id}]")
        print(f"📂 Using Dataset: {self.cleaned_path}")

        eda_result = self.run_eda(target_column)
        viz_result = self.run_visualization(max_plots)

        print("\\n✅ Data Analyst Pipeline Completed Successfully")

        return {
            "run_id": self.run_id,
            "cleaned_data": self.cleaned_path,
            "eda_dir": self.eda_dir,
            "visualization_dir": self.viz_dir,
            "eda_artifacts": eda_result.get("artifacts"),
            "visualization_report": viz_result.get("report"),
        }
