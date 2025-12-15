"""
EDA Agent - LangGraph powered
Produces profiling artifacts for downstream agents (visualization, autoML, training)

Outputs (saved under data/eda/):
- profiling_summary.json
- column_types.json
- univariate_stats.json
- bivariate_stats.json
- correlation_matrix.json
- supervised_target_profile.json (if target provided)
- insights.json (LLM or rule-based)
"""

import os
import json
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env file

from typing import Annotated, Optional, Dict, Any
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import pandas as pd
from agents.data_loader_agent import DataLoader
from utils.eda_helper import EDAProfiler  # implement this helper (methods used below)

# LLM init (same pattern as cleaner agent)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0.1,
        google_api_key=GOOGLE_API_KEY
    )
else:
    llm = None


class EDAState(TypedDict):
    messages: Annotated[list, add_messages]
    cleaned_path: str
    data_loaded: bool
    profiling_done: bool
    eda_dir: str
    schema: Optional[Dict[str, Any]]
    target_column: Optional[str]
    error: str
    # thresholds/options
    top_k_categories: int
    corr_threshold: float
    # outputs
    profiling_summary_path: str
    column_types_path: str
    univariate_path: str
    bivariate_path: str
    correlation_path: str
    supervised_profile_path: str
    insights_path: str


def load_cleaned_node(state: EDAState) -> EDAState:
    """Load cleaned dataset produced by DataCleaner Agent"""
    try:
        file_path = state['cleaned_path']
        loader = DataLoader()
        df = loader.load_csv(file_path)  # cleaned CSV path expected

        # basic validation
        if df.shape[0] == 0:
            return {**state, 'data_loaded': False, 'error': 'Cleaned dataset is empty',
                    'messages': state['messages'] + [HumanMessage(content="❌ Cleaned dataset is empty")]}

        # ensure eda dir exists
        eda_dir = state.get('eda_dir') or "data/eda"
        os.makedirs(eda_dir, exist_ok=True)

        # temporarily save canonical copy for nodes
        tmp_path = os.path.join(eda_dir, "temp_loaded.csv")
        df.to_csv(tmp_path, index=False)

        msg = f"✅ Loaded cleaned dataset: {file_path} ({df.shape[0]} rows × {df.shape[1]} cols)"
        return {
            **state,
            'data_loaded': True,
            'cleaned_path': tmp_path,
            'eda_dir': eda_dir,
            'messages': state['messages'] + [HumanMessage(content=msg)]
        }

    except Exception as e:
        return {
            **state,
            'data_loaded': False,
            'error': str(e),
            'messages': state['messages'] + [HumanMessage(content=f"❌ Failed to load cleaned dataset: {e}")]
        }


def profile_node(state: EDAState) -> EDAState:
    """Run profiler to produce basic summary and column types"""
    try:
        df = pd.read_csv(state['cleaned_path'])
        profiler = EDAProfiler(schema=state.get('schema'),
                               top_k_categories=state.get('top_k_categories', 20),
                               corr_threshold=state.get('corr_threshold', 0.3))
        profile = profiler.profile(df)

        eda_dir = state['eda_dir']
        # write artifacts
        profiling_path = os.path.join(eda_dir, "profiling_summary.json")
        column_types_path = os.path.join(eda_dir, "column_types.json")
        with open(profiling_path, "w", encoding="utf-8") as f:
            json.dump(profile['summary'], f, indent=2, default=str)
        with open(column_types_path, "w", encoding="utf-8") as f:
            json.dump(profile['column_types'], f, indent=2)

        msg = f"📊 Profiling complete. Summary saved to {profiling_path}"
        return {
            **state,
            'profiling_done': True,
            'profiling_summary_path': profiling_path,
            'column_types_path': column_types_path,
            'messages': state['messages'] + [HumanMessage(content=msg)]
        }

    except Exception as e:
        return {
            **state,
            'profiling_done': False,
            'error': str(e),
            'messages': state['messages'] + [HumanMessage(content=f"⚠️ Profiling failed: {e}")]
        }


def univariate_node(state: EDAState) -> EDAState:
    """Generate univariate statistics (per-column)"""
    try:
        df = pd.read_csv(state['cleaned_path'])
        profiler = EDAProfiler()
        univariate = profiler.univariate(df)

        path = os.path.join(state['eda_dir'], "univariate_stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(univariate, f, indent=2, default=str)

        msg = f"📈 Univariate stats generated: {path}"
        return {**state, 'univariate_path': path, 'messages': state['messages'] + [HumanMessage(content=msg)]}

    except Exception as e:
        return {**state, 'error': str(e), 'messages': state['messages'] + [HumanMessage(content=f"⚠️ Univariate failed: {e}")]} 


def bivariate_node(state: EDAState) -> EDAState:
    """Compute bivariate relationships and contingency tables"""
    try:
        df = pd.read_csv(state['cleaned_path'])
        profiler = EDAProfiler()
        bivariate = profiler.bivariate(df)

        path = os.path.join(state['eda_dir'], "bivariate_stats.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(bivariate, f, indent=2, default=str)

        msg = f"🔗 Bivariate relationships saved: {path}"
        return {**state, 'bivariate_path': path, 'messages': state['messages'] + [HumanMessage(content=msg)]}

    except Exception as e:
        return {**state, 'error': str(e), 'messages': state['messages'] + [HumanMessage(content=f"⚠️ Bivariate failed: {e}")]} 


def correlation_node(state: EDAState) -> EDAState:
    """Compute correlation matrix and dependency measures"""
    try:
        df = pd.read_csv(state['cleaned_path'])
        profiler = EDAProfiler()
        corr = profiler.correlation(df)

        path = os.path.join(state['eda_dir'], "correlation_matrix.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(corr, f, indent=2, default=str)

        msg = f"🧭 Correlation matrix saved: {path}"
        return {**state, 'correlation_path': path, 'messages': state['messages'] + [HumanMessage(content=msg)]}

    except Exception as e:
        return {**state, 'error': str(e), 'messages': state['messages'] + [HumanMessage(content=f"⚠️ Correlation failed: {e}")]} 


def supervised_node(state: EDAState) -> EDAState:
    """If target provided, profile target vs features (class balance, regression stats)"""
    try:
        target = state.get('target_column')
        if not target:
            return {**state, 'messages': state['messages'] + [HumanMessage(content="• No target provided — skipping supervised profiling")]}

        df = pd.read_csv(state['cleaned_path'])
        profiler = EDAProfiler()
        supervised = profiler.supervised_profile(df, target)

        path = os.path.join(state['eda_dir'], "supervised_profile.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(supervised, f, indent=2, default=str)

        msg = f"🎯 Supervised profiling saved: {path}"
        return {**state, 'supervised_profile_path': path, 'messages': state['messages'] + [HumanMessage(content=msg)]}

    except Exception as e:
        return {**state, 'error': str(e), 'messages': state['messages'] + [HumanMessage(content=f"⚠️ Supervised profiling failed: {e}")]} 


def insight_node(state: EDAState) -> EDAState:
    """Generate insight summary using LLM if available, otherwise rule-based extraction"""
    try:
        # load a compact snippet for LLM
        with open(state['profiling_summary_path'], "r", encoding="utf-8") as f:
            summary = json.load(f)

        prompt_sys = """You are an expert Senior Data Analyst. Your goal is to provide a comprehensive "Human-in-the-loop" EDA Report.
        
You must structure your analysis strictly using the following 15 sections. Use Markdown headers (##) for each section.
1. Dataset Overview
2. Data Composition Summary
3. Data Quality Observations (EDA perspective, not cleaning)
4. Numeric Feature Distribution Insights
5. Categorical Feature Distribution Insights
6. Outlier & Skewness Analysis
7. Feature Relationships & Correlation Analysis
8. Redundant & Low-Information Features
9. Target Variable Analysis (if applicable, otherwise state N/A)
10. Temporal Patterns & Trends (if applicable, otherwise state N/A)
11. Risk & Bias Indicators
12. Visualization Recommendations
13. Modeling Readiness Assessment
14. Key Takeaways
15. Recommended Next Steps

Instructions:
- Use the provided logical Profiling Summary to populate these sections.
- Be verbose and explanatory where necessary, but keep it professional.
- If data for a section is missing from the summary, provide a reasonable inference or state that more deep-dive analysis is needed.
"""
        # Increase context size to capture more of the summary
        user_prompt = f"Profiling summary: {json.dumps(summary)[:8000]}"

        if llm:
            resp = llm.invoke([SystemMessage(content=prompt_sys), HumanMessage(content=user_prompt)])
            insights = resp.content
        else:
            # fallback simple rule-based insights
            # EDAProfiler also provides simple insights method
            profiler = EDAProfiler()
            insights = profiler.basic_insights_from_summary(summary)

        path = os.path.join(state['eda_dir'], "insights.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"insights": insights}, f, indent=2)

        msg = f"🧠 Insights generated: {path}"
        return {**state, 'insights_path': path, 'messages': state['messages'] + [HumanMessage(content=msg)]}

    except Exception as e:
        return {**state, 'error': str(e), 'messages': state['messages'] + [HumanMessage(content=f"⚠️ Insight generation failed: {e}")]} 


def should_continue_eda(state: EDAState) -> str:
    if state.get('error'):
        return "end"
    if not state.get('data_loaded'):
        return "end"
    # After profiling do the rest sequentially
    if not state.get('profiling_done'):
        return "profile"
    # if profiling done and univariate not done yet, analyze
    messages = state.get('messages', [])
    if not state.get('univariate_path'):
        return "univariate"
    if not state.get('bivariate_path'):
        return "bivariate"
    if not state.get('correlation_path'):
        return "correlation"
    # supervised optional
    if state.get('target_column') and not state.get('supervised_profile_path'):
        return "supervised"
    if not state.get('insights_path'):
        return "insights"
    return "end"


def create_eda_agent():
    wf = StateGraph(EDAState)
    wf.add_node("load", load_cleaned_node)
    wf.add_node("profile", profile_node)
    wf.add_node("univariate", univariate_node)
    wf.add_node("bivariate", bivariate_node)
    wf.add_node("correlation", correlation_node)
    wf.add_node("supervised", supervised_node)
    wf.add_node("insights", insight_node)

    wf.add_edge(START, "load")
    wf.add_conditional_edges("load", should_continue_eda, {"profile": "profile", "end": END})
    wf.add_conditional_edges("profile", should_continue_eda, {"univariate": "univariate", "end": END})
    wf.add_conditional_edges("univariate", should_continue_eda, {"bivariate": "bivariate", "end": END})
    wf.add_conditional_edges("bivariate", should_continue_eda, {"correlation": "correlation", "end": END})
    wf.add_conditional_edges("correlation", should_continue_eda, {"supervised": "supervised", "insights": "insights", "end": END})
    wf.add_conditional_edges("supervised", should_continue_eda, {"insights": "insights", "end": END})
    wf.add_edge("insights", END)

    return wf.compile()


def run_eda(cleaned_path: str, target_column: Optional[str] = None, eda_dir: Optional[str] = None, flags: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    agent = create_eda_agent()
    flags = flags or {}
    eda_dir = eda_dir or "data/eda"
    initial_state: EDAState = {
        'messages': [],
        'cleaned_path': cleaned_path,
        'data_loaded': False,
        'profiling_done': False,
        'eda_dir': eda_dir,
        'schema': flags.get('schema'),
        'target_column': target_column,
        'error': '',
        'top_k_categories': flags.get('top_k_categories', 20),
        'corr_threshold': flags.get('corr_threshold', 0.3),
        'profiling_summary_path': '',
        'column_types_path': '',
        'univariate_path': '',
        'bivariate_path': '',
        'correlation_path': '',
        'supervised_profile_path': '',
        'insights_path': ''
    }

    res = agent.invoke(initial_state)
    return {
        'success': not bool(res.get('error')),
        'eda_dir': res.get('eda_dir'),
        'artifacts': {
            'profiling': res.get('profiling_summary_path'),
            'column_types': res.get('column_types_path'),
            'univariate': res.get('univariate_path'),
            'bivariate': res.get('bivariate_path'),
            'correlation': res.get('correlation_path'),
            'supervised': res.get('supervised_profile_path'),
            'insights': res.get('insights_path')
        },
        'messages': [getattr(m, 'content', '') for m in res.get('messages', [])],
        'error': res.get('error', '')
    }  