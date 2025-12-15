"""
Visualization Agent - LangGraph powered
Consumes EDA artifacts and produces:
- visualization_specs.json (AI dashboard)
- generated_plots.json
- plots/*.png (optional)
- eda_report.pdf
"""

import os
import json
from typing import Annotated, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

from utils.visualization_helper import (
    ChartRecommender,
    PlotGenerator,
    ReportBuilder
)


# ------------------ STATE ------------------

class VizState(TypedDict):
    messages: Annotated[list, add_messages]

    eda_dir: str
    viz_dir: str
    plots_dir: str

    visualization_spec_path: str
    generated_plots_path: str
    report_path: str

    max_plots: int
    error: str


# ------------------ NODES ------------------

def load_eda_node(state: VizState) -> VizState:
    try:
        eda_dir = state.get("eda_dir", "data/eda")
        required = ["profiling_summary.json", "column_types.json"]

        for f in required:
            if not os.path.exists(os.path.join(eda_dir, f)):
                raise FileNotFoundError(f"Missing EDA artifact: {f}")

        viz_dir = state.get("viz_dir", "data/visualizations")
        plots_dir = os.path.join(viz_dir, "plots")

        os.makedirs(plots_dir, exist_ok=True)

        return {
            **state,
            "eda_dir": eda_dir,
            "viz_dir": viz_dir,
            "plots_dir": plots_dir,
            "messages": state["messages"] + [
                HumanMessage(content=f"✅ Loaded EDA artifacts from {eda_dir}")
            ]
        }

    except Exception as e:
        return {**state, "error": str(e)}


def recommend_node(state: VizState) -> VizState:
    try:
        with open(os.path.join(state["eda_dir"], "column_types.json")) as f:
            column_types = json.load(f)

        with open(os.path.join(state["eda_dir"], "profiling_summary.json")) as f:
            profiling = json.load(f)

        recommender = ChartRecommender()
        specs = recommender.recommend(
            column_types=column_types,
            profiling=profiling,
            max_plots=state.get("max_plots", 30)
        )

        spec_path = os.path.join(state["viz_dir"], "visualization_specs.json")
        with open(spec_path, "w") as f:
            json.dump(specs, f, indent=2)

        return {
            **state,
            "visualization_spec_path": spec_path,
            "messages": state["messages"] + [
                HumanMessage(content="📊 AI visualization specs generated")
            ]
        }

    except Exception as e:
        return {**state, "error": str(e)}


def generate_plots_node(state: VizState) -> VizState:
    try:
        with open(state["visualization_spec_path"]) as f:
            specs = json.load(f)

        plotter = PlotGenerator(
            output_dir=state["plots_dir"],
            data_path=os.path.join(state["eda_dir"], "temp_loaded.csv")
        )

        generated = plotter.generate_from_specs(specs)

        gen_path = os.path.join(state["viz_dir"], "generated_plots.json")
        with open(gen_path, "w") as f:
            json.dump(generated, f, indent=2)

        return {
            **state,
            "generated_plots_path": gen_path,
            "messages": state["messages"] + [
                HumanMessage(content=f"🖼 Generated {len(generated)} plots")
            ]
        }

    except Exception as e:
        return {**state, "error": str(e)}


def build_report_node(state: VizState) -> VizState:
    try:
        with open(os.path.join(state["eda_dir"], "insights.json")) as f:
            insights_obj = json.load(f)

        insights = insights_obj.get("insights", "")
        if isinstance(insights, list):
            insights = "\n".join(insights)

        with open(state["generated_plots_path"]) as f:
            plots = json.load(f)

        builder = ReportBuilder(output_dir=state["viz_dir"])
        report_path = builder.build_report(
            generated_plots=plots,
            insights=insights
        )

        return {
            **state,
            "report_path": report_path,
            "messages": state["messages"] + [
                HumanMessage(content=f"📄 Report generated: {report_path}")
            ]
        }

    except Exception as e:
        return {**state, "error": str(e)}


# ------------------ ROUTER ------------------

def should_continue(state: VizState) -> str:
    if state.get("error"):
        return "end"
    if not state.get("visualization_spec_path"):
        return "recommend"
    if not state.get("generated_plots_path"):
        return "generate"
    if not state.get("report_path"):
        return "report"
    return "end"


# ------------------ GRAPH ------------------

def create_visualization_agent():
    wf = StateGraph(VizState)

    wf.add_node("load", load_eda_node)
    wf.add_node("recommend", recommend_node)
    wf.add_node("generate", generate_plots_node)
    wf.add_node("report", build_report_node)

    wf.add_edge(START, "load")
    wf.add_conditional_edges("load", should_continue, {
        "recommend": "recommend",
        "end": END
    })
    wf.add_conditional_edges("recommend", should_continue, {
        "generate": "generate",
        "end": END
    })
    wf.add_conditional_edges("generate", should_continue, {
        "report": "report",
        "end": END
    })
    wf.add_edge("report", END)

    return wf.compile()


# ------------------ ENTRYPOINT ------------------

def run_visualization(eda_dir: str, viz_dir: Optional[str] = None, max_plots: int = 30):
    agent = create_visualization_agent()

    state: VizState = {
        "messages": [],
        "eda_dir": eda_dir,
        "viz_dir": viz_dir or "data/visualizations",
        "plots_dir": "",
        "visualization_spec_path": "",
        "generated_plots_path": "",
        "report_path": "",
        "max_plots": max_plots,
        "error": ""
    }

    res = agent.invoke(state)

    return {
        "success": not bool(res.get("error")),
        "viz_dir": res.get("viz_dir"),
        "report": res.get("report_path"),
        "messages": [m.content for m in res.get("messages", [])],
        "error": res.get("error")
    }
