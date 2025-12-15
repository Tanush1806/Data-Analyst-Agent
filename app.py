"""
AI Data Analyst Dashboard
"""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from typing import Dict, Any, List

# ------------------ PAGE CONFIG ------------------

st.set_page_config(
    page_title="AI Data Analyst Dashboard",
    page_icon="📊",
    layout="wide"
)

# ------------------ CONSTANTS & PATHS ------------------

CLEANED_DATA_PATH = "data/processed/cleaned_data.csv"
VIZ_SPEC_PATH = "data/visualizations/visualization_specs.json"
INSIGHTS_PATH = "data/eda/insights.json"

st.markdown("""
    <h1 style='text-align: center;'>📊 AI Data Analyst Dashboard</h1>
    <p style='text-align: center; color: gray;'>
        Automated data analysis using AI agents | Human-in-the-loop dashboard editing<br>
        <span style="font-size: 0.8em;">Backend = Agents | Frontend = Visualization & Control</span>
    </p>
    <hr>
""", unsafe_allow_html=True)

# ------------------ UTILS ------------------

# Ensure Backend is in sys.path
import sys
backend_path = os.path.join(os.getcwd(), "Backend")
if backend_path not in sys.path:
    sys.path.append(backend_path)

try:
    from Backend.agents.data_loader_agent import DataLoader
    from Backend.agents.data_analyst_orchestrator import DataAnalystOrchestrator
except ImportError:
    try:
        from agents.data_loader_agent import DataLoader
        from agents.data_analyst_orchestrator import DataAnalystOrchestrator
    except ImportError as e:
        st.error(f"❌ Could not import Agents. Ensure Backend/agents/ exists. Error: {e}")
        st.stop()

def rerun_app():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, data: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# ------------------ SESSION RESET LOGIC ------------------

if "app_initialized" not in st.session_state:
    # Logic to run only once when the app starts or is refreshed
    if os.path.exists(CLEANED_DATA_PATH):
        os.remove(CLEANED_DATA_PATH)
    if os.path.exists(VIZ_SPEC_PATH):
        os.remove(VIZ_SPEC_PATH)
    if os.path.exists(INSIGHTS_PATH):
        os.remove(INSIGHTS_PATH)
    
    # Clear the global function cache
    load_data.clear()
    
    # Mark as initialized so we don't delete immediately after upload (which triggers rerun)
    st.session_state["app_initialized"] = True

# ------------------ DATA LOADING ORCHESTRATION ------------------

# Initialize DataLoader
loader = DataLoader()

# Sidebar for Data Upload and AI Analysis
with st.sidebar:
    st.header("📂 Data Upload")
    st.caption("Upload your cleaned dataset here.")
    
    uploaded_file = st.file_uploader("Upload Cleaned CSV/Excel", type=["csv", "xlsx"])
    
    if uploaded_file:
        try:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if "last_uploaded_file" not in st.session_state or st.session_state["last_uploaded_file"] != file_key:
                # New file uploaded: Clean up and Process
                
                # Clear existing charts/artifacts from previous runs
                if os.path.exists(VIZ_SPEC_PATH):
                    os.remove(VIZ_SPEC_PATH)
                if "charts" in st.session_state:
                    del st.session_state["charts"]
                if os.path.exists(INSIGHTS_PATH):
                    os.remove(INSIGHTS_PATH)
                    
                file_type = "csv" if uploaded_file.name.endswith(".csv") else "xlsx"
                
                # Use DataLoader to load the file object
                df_loaded = loader.load_file_object(uploaded_file, file_type=file_type)
                
                if df_loaded is not None:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
                    # Save to the expected path
                    df_loaded.to_csv(CLEANED_DATA_PATH, index=False)
                    # Clear cache to ensure new data is loaded
                    load_data.clear()
                    
                    st.session_state["last_uploaded_file"] = file_key
                    st.toast(f"✅ Data loaded! ({len(df_loaded)} rows)", icon="✅")
                    
        except Exception as e:
            st.error(f"Error loading file: {e}")

    st.markdown("---")
    st.header("🤖 AI Analysis")
    
    max_plots = st.slider("Max Charts to Generate", min_value=1, max_value=50, value=5)
    
    if st.button("🚀 Auto-Generate Charts & Insights"):
        if os.path.exists(CLEANED_DATA_PATH):
            with st.spinner("🤖 AI Agents are working... (EDA -> Visualization)"):
                try:
                    orchestrator = DataAnalystOrchestrator(
                        cleaned_file_path=CLEANED_DATA_PATH,
                        output_root="data"
                    )
                    orchestrator.run_full_pipeline(max_plots=max_plots)
                    
                    # Force reload of charts by clearing session state
                    if "charts" in st.session_state:
                        del st.session_state["charts"]
                        
                    st.success("✅ Analysis Complete!")
                    rerun_app()
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
        else:
            st.warning("⚠️ Please upload a dataset first.")

# Check if data exists (either just uploaded or previously saved)
if not os.path.exists(CLEANED_DATA_PATH):
    st.info("👈 Please upload a cleaned dataset in the sidebar to start.")
    st.stop()


df = load_data(CLEANED_DATA_PATH)

# ------------------ TABS ------------------

tab_dashboard, tab_insights, tab_preview = st.tabs(["📈 Dashboard", "🧠 Insights", "📄 Data Preview"])

# ------------------ TAB: DATA PREVIEW ------------------

with tab_preview:
    st.subheader("Data Preview")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isna().sum().sum())

    st.dataframe(df.head(50), use_container_width=True)
    st.caption("Showing first 50 rows of cleaned dataset (Read-Only)")

# ------------------ TAB: INSIGHTS ------------------

with tab_insights:
    st.subheader("🤖 AI-Generated Insights")
    
    insights_data = load_json(INSIGHTS_PATH)
    
    if insights_data and "insights" in insights_data:
        insights_text = insights_data["insights"]
        # Assuming insights might be a list or a formatted string. 
        # If it's a string from LLM, render as markdown.
        if isinstance(insights_text, list):
            for i, item in enumerate(insights_text):
                st.info(item, icon="💡")
        else:
            st.markdown(insights_text)
    else:
        st.warning("⚠ No insights found. Ensure EDA agent has run successfully.")

# ------------------ TAB: DASHBOARD ------------------

with tab_dashboard:
    if "charts" not in st.session_state:
        specs = load_json(VIZ_SPEC_PATH)
        if specs and "plots" in specs:
            st.session_state.charts = specs["plots"]
        else:
            st.session_state.charts = []
            st.warning("⚠ No AI-generated visualization specs found. You can add charts manually.")

    # Helper to get columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    all_cols = df.columns.tolist()

    # --- RENDER CHARTS ---
    original_charts = list(st.session_state.charts) # Copy to detect changes? No, we edit in place.

    st.markdown("### 🤖 AI-Generated & Editable Charts")
    
    charts_to_remove = []

    for i, chart in enumerate(st.session_state.charts):
        with st.expander(f"Chart {i+1}: {chart.get('title', 'Untitled')}", expanded=True):
            cols = st.columns([3, 1])
            
            # --- VISUALIZATION ---
            with cols[0]:
                try:
                    c_type = chart.get("type", "bar")
                    title = chart.get("title", "")
                    
                    fig = None
                    if c_type == "histogram":
                        if "column" in chart and chart["column"] in df.columns:
                            fig = px.histogram(df, x=chart["column"], title=title)
                    elif c_type == "box":
                         if "column" in chart and chart["column"] in df.columns:
                            fig = px.box(df, y=chart["column"], title=title)
                    elif c_type == "bar":
                        if "column" in chart and chart["column"] in df.columns:
                            # Simple count aggregation if not specified
                            counts = df[chart["column"]].value_counts().reset_index()
                            counts.columns = [chart["column"], "count"]
                            fig = px.bar(counts, x=chart["column"], y="count", title=title)
                    elif c_type == "scatter":
                         if "x" in chart and "y" in chart and chart["x"] in df.columns and chart["y"] in df.columns:
                            fig = px.scatter(df, x=chart["x"], y=chart["y"], title=title)
                    elif c_type == "line":
                         if "x" in chart and "y" in chart and chart["x"] in df.columns and chart["y"] in df.columns:
                            fig = px.line(df.sort_values(chart["x"]), x=chart["x"], y=chart["y"], title=title)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"⚠️ Unable to render chart. Check configuration.")

                except Exception as e:
                    st.error(f"Error rendering chart: {e}")

            # --- CONTROLS ---
            with cols[1]:
                st.markdown("#### ⚙️ Settings")
                
                # Update Type
                new_type = st.selectbox(
                    "Type",
                    ["histogram", "box", "bar", "scatter", "line"],
                    index=["histogram", "box", "bar", "scatter", "line"].index(chart.get("type", "histogram")),
                    key=f"type_{i}"
                )
                chart["type"] = new_type

                # Update Title
                chart["title"] = st.text_input("Title", chart.get("title", ""), key=f"title_{i}")

                # Update Axes based on Type
                if new_type in ["histogram", "box", "bar"]:
                    chart["column"] = st.selectbox(
                        "Column", 
                        all_cols, 
                        index=all_cols.index(chart.get("column", all_cols[0])) if chart.get("column") in all_cols else 0,
                        key=f"col_{i}"
                    )
                    # Clean up irrelevant keys
                    chart.pop("x", None)
                    chart.pop("y", None)

                elif new_type in ["scatter", "line"]:
                    chart["x"] = st.selectbox(
                        "X Axis", 
                        all_cols, 
                        index=all_cols.index(chart.get("x", all_cols[0])) if chart.get("x") in all_cols else 0,
                        key=f"x_{i}"
                    )
                    chart["y"] = st.selectbox(
                        "Y Axis", 
                        numeric_cols, # Usually Y should be numeric for these
                        index=numeric_cols.index(chart.get("y", numeric_cols[0])) if chart.get("y") in numeric_cols else 0,
                        key=f"y_{i}"
                    )
                    chart.pop("column", None)

                st.markdown("---")
                if st.button("🗑 Delete Chart", key=f"del_{i}", type="secondary"):
                    charts_to_remove.append(i)

    # Remove deleted charts
    if charts_to_remove:
        for index in sorted(charts_to_remove, reverse=True):
            del st.session_state.charts[index]
        rerun_app()

    # --- ADD NEW CHART ---
    st.markdown("### ➕ Add New Chart")
    with st.form("new_chart_form"):
        nc_col1, nc_col2, nc_col3, nc_col4 = st.columns(4)
        
        nc_type = nc_col1.selectbox("New Chart Type", ["histogram", "box", "bar", "scatter", "line"])
        nc_title = nc_col2.text_input("New Title", "New Analysis")
        
        # Dynamic Placeholders logic is hard in a form without rerun. 
        # We will ask for all potentially relevant fields and use them based on type logic post-submission.
        # Or just show all.
        
        nc_annot = st.caption("Select columns below ensuring they match the chart type (e.g. Y must be numeric for Scatter).")
        
        nc_param1 = nc_col3.selectbox("Column / X Axis", all_cols)
        nc_param2 = nc_col4.selectbox("Y Axis (for Scatter/Line)", numeric_cols)

        if st.form_submit_button("Add Chart"):
            new_chart_obj = {"type": nc_type, "title": nc_title}
            
            if nc_type in ["histogram", "box", "bar"]:
                new_chart_obj["column"] = nc_param1
            else:
                new_chart_obj["x"] = nc_param1
                new_chart_obj["y"] = nc_param2
            
            st.session_state.charts.append(new_chart_obj)
            rerun_app()

    # --- SAVE ---
    st.markdown("---")
    col_save, _ = st.columns([2, 5])
    with col_save:
        if st.button("💾 Save Dashboard Changes", type="primary", use_container_width=True):
            save_payload = {"plots": st.session_state.charts}
            save_json(VIZ_SPEC_PATH, save_payload)
            st.success(f"✅ Dashboard saved to {VIZ_SPEC_PATH}")
