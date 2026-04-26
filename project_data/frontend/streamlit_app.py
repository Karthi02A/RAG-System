import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import io
import os
from dotenv import load_dotenv

load_dotenv()
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# --- Page Configuration ---
st.set_page_config(
    page_title="DataForge AI | Dashboard",
    page_icon="⚒️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 🚀 PHYSICAL SPACER
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# --- Global CSS Injection (Selective Stability Mobile) ---
st.markdown("""
<style>
# 🚀 Selective Horizontal Swiping (Fixed Nav & Action Boxes)
    @media (max-width: 768px) {
        /* 🚀 (GLOBAL) Force Horizontal + Enable Swiping */
        div[data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important; 
            gap: 15px !important;
            width: 100% !important;
        }

        /* 🛡️ PHYSICAL ISOLATION: FIXED BOXES (No columns = No swiping) */
        /* We remove columns for these buttons in the code below */
        
        /* Dashboard Fitness (Sizing with Peek UX) */
        div[data-testid="column"] {
            flex: 0 0 auto !important;
            min-width: 280px !important; /* THE GOLDILOCKS PEEK SIZE */
        }

        /* LOGO FIT (Avoid cutting) */
        .hero-title {
            font-size: 2.4rem !important; /* Smaller logo for perfect fit */
        }

        /* PERFECT SIGHT (Medium 16px) */
        button, p, span, label, input, h1, h2, h3, div {
            font-size: 16px !important;
        }

        /* Hide the Scrollbar for a premium look, while keeping it swipeable */
        div[data-testid="stHorizontalBlock"]::-webkit-scrollbar {
            display: none;
        }
        
        .main .block-container {
            padding: 1.5rem !important;
            padding-top: 2rem !important;
        }
    }

    /* 🛡️ PRIVACY & CLEAN LOOK */
    [data-testid="stHeader"], header { display: none !important; }
    footer { display: none !important; }
    #MainMenu { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# --- CSS Injection ---
def local_css(file_name):
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

# --- Session State ---
if "dataset_id" not in st.session_state: st.session_state.dataset_id = None
if "original_df" not in st.session_state: st.session_state.original_df = None
if "cleaned_df" not in st.session_state: st.session_state.cleaned_df = None
if "analysis_results" not in st.session_state: st.session_state.analysis_results = None
if "cleaned_analysis" not in st.session_state: st.session_state.cleaned_analysis = None
if "uploaded_filename" not in st.session_state: st.session_state.uploaded_filename = None
if "view_mode" not in st.session_state: st.session_state.view_mode = "original"
if "analysis_done" not in st.session_state: st.session_state.analysis_done = False # New Guard
if "ml_readiness" not in st.session_state: st.session_state.ml_readiness = None
if "active_tab" not in st.session_state: st.session_state.active_tab = "Data Analysis"
if "ml_results" not in st.session_state: st.session_state.ml_results = None
if "is_processing" not in st.session_state: st.session_state.is_processing = False
if "pending_action" not in st.session_state: st.session_state.pending_action = None
if "target_memory" not in st.session_state: st.session_state.target_memory = None

def reset_session_state():
    st.session_state.dataset_id = None
    st.session_state.original_df = None
    st.session_state.cleaned_df = None
    st.session_state.analysis_results = None
    st.session_state.cleaned_analysis = None
    st.session_state.uploaded_filename = None
    st.session_state.view_mode = "original"
    st.session_state.analysis_done = False
    st.session_state.ml_results = None
    st.session_state.active_tab = "Data Analysis"
    st.session_state.is_processing = False
    st.session_state.pending_action = None
    st.session_state.ml_readiness = None
    st.session_state.target_memory = None

def set_action(action_name):
    st.session_state.pending_action = action_name
    st.session_state.is_processing = True

# --- Helper UI Components ---
def centered_header(title):
    st.markdown(f"""
        <div class="heading-box">
            <h2 style='margin:0;'>{title}</h2>
        </div>
    """, unsafe_allow_html=True)

def topic_header(title):
    st.markdown(f"""
        <div class="topic-box">
            <h3 style='margin:0; font-size: 1.2rem; color: #38BDF8;'>{title}</h3>
        </div>
    """, unsafe_allow_html=True)

def score_display(label, score, status=None):
    st.markdown(f"""
        <div class="score-box">
            <p style='color: #9CA3AF; font-size: 0.9rem; margin-bottom: 0.5rem;'>{label}</p>
            <h1 style='color: #38BDF8; margin:0;'>{score}%</h1>
            {f"<p style='color: #7DD3FC; margin-top: 5px;'>{status}</p>" if status else ""}
        </div>
    """, unsafe_allow_html=True)

def kpi_row(metrics):
    cols_html = "".join([f"""
        <div class="kpi-card">
            <div class="kpi-label">{m['label']}</div>
            <div class="kpi-value">{m['value']}</div>
        </div>
    """ for m in metrics])
    st.markdown(f'<div class="kpi-container">{cols_html}</div>', unsafe_allow_html=True)

# --- HERO SECTION & UPLOAD ---
st.markdown(f"""
    <div class="hero-container">
        <div class="hero-content">
            <div class="hero-title-box">
                <div class="hero-title">DataForge AI</div>
            </div>
            <div class="hero-subtitle">Turning Data Into Intelligent Decisions</div>
            <div class="hero-description">
                A powerful AI-driven platform for analyzing datasets, detecting patterns, and 
                transforming raw data into actionable insights through intelligent analytics.
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- BACKEND WAKE-UP GUARD ---
if st.session_state.original_df is None:
    with st.spinner("⚔️ Contacting the Forge..."):
        try:
            # Short timeout to check if it's already awake
            requests.get(f"{BACKEND_URL}/api/health", timeout=3)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            st.warning("🔮 **The Forge is Currently Hibernating.**")
            st.info("Since this is a free hosting tier, the backend sleeps during inactivity. **Please wait 30-60 seconds** while we wake it up for you...")
            try:
                # Longer timeout to wait for wake-up
                requests.get(f"{BACKEND_URL}/api/health", timeout=60)
                st.success("✅ **The Forge is Awake!**")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to wake up the Forge: {e}")
                st.stop()

# Central Upload Area
if st.session_state.original_df is None:
    col_up1, col_up2, col_up3 = st.columns([1, 2, 1])
    with col_up2:
        st.markdown("<div style='text-align: center; margin-bottom: 10px; color: #7DD3FC; font-weight: 600;'>🚀 READY TO START?</div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv", label_visibility="collapsed")
        
        if uploaded_file:
            with st.spinner("Forging dataset..."):
                try:
                    # Robust Encoding Detection
                    content = uploaded_file.getvalue()
                    df = None
                    for encoding in ["utf-8", "latin-1", "cp1252"]:
                        try:
                            df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                            break
                        except UnicodeDecodeError:
                            continue
                        except Exception:
                            continue
                    
                    if df is None:
                        # Final fallback with replacement
                        df = pd.read_csv(io.BytesIO(content), encoding="utf-8", encoding_errors="replace")

                    # Reset state before setting new data
                    reset_session_state()
                    
                    st.session_state.original_df = df
                    st.session_state.cleaned_df = df.copy()
                    st.session_state.uploaded_filename = uploaded_file.name
                    
                    # Backend Upload
                    uploaded_file.seek(0)
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
                    res = requests.post(f"{BACKEND_URL}/api/upload_dataset", files=files).json()
                    st.session_state.dataset_id = res["dataset_id"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Upload failed: {e}")
    st.info("Please upload a dataset to begin the analysis.")
    st.stop() # Keep stop only for total empty state to prevent UI clutter

# --- DATASET PREVIEW (Only shows after upload) ---
if st.session_state.original_df is not None:
    st.markdown(f"<p style='text-align: center; color: #9CA3AF; margin-top: -10px;'>Working with Intelligence: <b style='color: #7DD3FC;'>{st.session_state.uploaded_filename}</b></p>", unsafe_allow_html=True)
    
    # Change Dataset Button (PHYSICALLY ISOLATED)
    if st.button("🔄 Change Dataset", use_container_width=True, type="secondary", key="btn_change_ds"):
        reset_session_state()
        st.rerun()
            
    centered_header("Dataset Preview")
    st.dataframe(st.session_state.original_df.head(10), use_container_width=True)

    # --- HORIZONTAL NAVIGATION (Swipe Zone) ---
    st.markdown("<div class='swipe-zone'>", unsafe_allow_html=True)
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    tabs = ["Data Analysis", "Machine Learning Intelligence", "Insights Dashboard", "Download"]
    nav_cols = st.columns(len(tabs))
    
    for i, tab in enumerate(tabs):
        with nav_cols[i]:
            is_disabled = (tab != "Data Analysis" and st.session_state.view_mode != "cleaned") or st.session_state.is_processing
            if st.button(tab, use_container_width=True, key=f"btn_{tab}", disabled=is_disabled):
                st.session_state.active_tab = tab
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<hr style='border: 0.5px solid #1F2937; margin: 1.5rem 0;'>", unsafe_allow_html=True)

    # --- DYNAMIC CONTENT (VERTICAL FLOW) ---
    if st.session_state.active_tab == "Data Analysis":
        centered_header("Data Analysis & Diagnosis")
        
        # side-by-side action buttons (SWIPE-ZONE)
        st.markdown("<div class='swipe-zone'>", unsafe_allow_html=True)
        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            diag_label = "🔍 Run Full Diagnosis" if st.session_state.view_mode != "cleaned" else "🔍 Re-run Analysis"
            if st.session_state.pending_action == "analyze":
                try:
                    with st.spinner("Analyzing data..."):
                        res = requests.post(f"{BACKEND_URL}/api/analyze_dataset", params={"dataset_id": st.session_state.dataset_id}).json()
                        if "health_score" in res:
                            st.session_state.analysis_results = res
                            st.session_state.view_mode = "original" if st.session_state.view_mode != "cleaned" else "cleaned"
                            st.session_state.analysis_done = True
                        else:
                            st.error(f"Analysis Failed: {res.get('detail', 'Unknown error')}")
                finally:
                    st.session_state.pending_action = None
                    st.session_state.is_processing = False
                    st.rerun()
            else:
                st.button(diag_label, use_container_width=True, on_click=set_action, args=("analyze",), disabled=st.session_state.is_processing)
            
        with btn_c2:
            if st.session_state.view_mode == "cleaned":
                st.button("✨ ✅ Dataset is Cleaned", use_container_width=True, disabled=True)
            else:
                if st.session_state.pending_action == "clean":
                    try:
                        with st.spinner("Cleansing data..."):
                            res = requests.post(f"{BACKEND_URL}/api/clean_dataset", params={"dataset_id": st.session_state.dataset_id}).json()
                            if "dataset_id" in res:
                                st.session_state.dataset_id = res["dataset_id"]
                                st.session_state.cleaned_df = pd.read_csv(f"{BACKEND_URL}/api/download_clean?dataset_id={res['dataset_id']}")
                                st.session_state.cleaned_analysis = res["analysis"]
                                st.session_state.view_mode = "cleaned"
                                st.success("Successfully cleaned! Your dataset health is now 100%.")
                            else:
                                st.error(f"Cleaning failed: {res.get('detail', 'Unknown error')}")
                    finally:
                        st.session_state.pending_action = None
                        st.session_state.is_processing = False
                        st.rerun()
                else:
                    st.button("✨ Auto Clean Dataset", type="primary", use_container_width=True, on_click=set_action, args=("clean",), disabled=st.session_state.is_processing)
        st.markdown("</div>", unsafe_allow_html=True)

        # Determine which results to show
        show_results = None
        current_df = None
        if st.session_state.view_mode == "original" and st.session_state.analysis_results:
            show_results = st.session_state.analysis_results
            current_df = st.session_state.original_df
        elif st.session_state.view_mode == "cleaned" and st.session_state.cleaned_analysis:
            show_results = st.session_state.cleaned_analysis
            current_df = st.session_state.cleaned_df

        if show_results:
            res = show_results
            df = current_df

            # 1. Data Quality / Health Score
            topic_header("Data Quality / Health Score")
            score_display("Overall Dataset Health", res['health_score'])

            # 2. Missing Values Summary & Duplicates
            topic_header("Data Integrity Summary")
            st.markdown("#### Missing Values & Percentages")
            mv_counts = res.get('missing_values', {})
            mv_percs = res.get('missing_percentages', {})
            
            mv_data = []
            for col in mv_counts.keys():
                mv_data.append({
                    "Column": col,
                    "Missing Count": mv_counts[col],
                    "Missing %": f"{mv_percs.get(col, 0)}%"
                })
            
            mv_df = pd.DataFrame(mv_data)
            st.dataframe(mv_df[mv_df['Missing Count'] > 0] if res.get('total_missing', 0) > 0 else mv_df, use_container_width=True)
            
            d_col1, d_col2 = st.columns(2)
            with d_col1:
                st.markdown("#### Duplicate Rows")
                st.markdown(f"**Duplicates Found:** {res.get('duplicates', 0)} rows")
            with d_col2:
                st.markdown("#### Duplicate Columns")
                dupe_cols = res.get('duplicate_columns', [])
                if dupe_cols:
                    st.warning(f"**Duplicates Found:** {len(dupe_cols)} columns\n({', '.join(dupe_cols)})")
                else:
                    st.success("No duplicate columns detected.")

            # 3. Dataset Overview
            topic_header("Dataset Overview")
            st.markdown("<div class='swipe-zone'>", unsafe_allow_html=True)
            o1, o2, o3 = st.columns(3)
            with o1: st.metric("Rows", res.get('rows_count', 0))
            with o2: st.metric("Columns", res.get('columns_count', 0))
            with o3: st.metric("Dataset Size", f"{res.get('dataset_size_kb', 0)} KB")
            st.markdown("</div>", unsafe_allow_html=True)

            # 4. Column Structure
            topic_header("Column Structure")
            struct = res.get('structure', {})
            s1, s2, s3, s4, s5, s6 = st.columns(6)
            with s1: st.metric("Numeric", len(struct.get('numeric', [])))
            with s2: st.metric("Categorical", len(struct.get('categorical', [])))
            with s3: st.metric("Date", len(struct.get('date', [])))
            with s4: st.metric("ID", len(struct.get('id', [])))
            with s5: st.metric("High Card", len(struct.get('high_cardinality', [])))
            with s6: st.metric("Mixed", len(struct.get('mixed', [])))

            # 5. Summary Statistics (Numeric Columns)
            if res.get('summary_statistics'):
                topic_header("Summary Statistics (Numeric Columns)")
                stats_df = pd.DataFrame(res['summary_statistics']).T
                st.dataframe(stats_df, use_container_width=True)

            # 6. Correlation Analysis (NEW)
            topic_header("Correlation Analysis")
            corr_data = res.get('correlation_matrix')
            if corr_data and len(corr_data) >= 2:
                import plotly.express as px
                corr_df = pd.DataFrame(corr_data)
                
                # Limit to 30 columns for heatmap readability
                if len(corr_df.columns) > 30:
                    st.warning("Displaying first 30 numeric columns for correlation heatmap.")
                    corr_df = corr_df.iloc[:30, :30]
                
                fig = px.imshow(
                    corr_df,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdYlGn',
                    labels=dict(color="Correlation"),
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Correlation Insights
                total_strong = res.get('total_strong_correlations', 0)
                st.markdown(f"**Total Strong Correlations Detected: {total_strong}**")
                
                if res.get('correlation_insights'):
                    st.markdown("#### Showing Top 5 strongest relationships")
                    for insight in res['correlation_insights']:
                        st.markdown(insight)
            else:
                st.info("📊 Correlation analysis requires at least two numeric columns.")

            # 7. Outlier Detection
            topic_header("Outlier Detection")
            out_df = pd.DataFrame(res.get('outliers', {}).items(), columns=['Column', 'Outlier Count'])
            st.dataframe(out_df[out_df['Outlier Count'] > 0] if res.get('total_outliers', 0) > 0 else out_df, use_container_width=True)

            # 8. Statistical Insights
            if res.get('insights'):
                topic_header("Statistical Insights")
                for insight in res['insights']:
                    st.markdown(f"📊 {insight}")

            # 9. Detail Statistics
            topic_header("Detail Statistics")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Unique Values")
                unq_df = pd.DataFrame(res.get('unique_values', {}).items(), columns=['Column', 'Unique Count'])
                st.dataframe(unq_df, use_container_width=True)
            with c2:
                st.markdown("#### Constant Columns")
                const_cols = res.get('constant_columns', [])
                if const_cols:
                    st.warning(f"Found {len(const_cols)} constant columns: {', '.join(const_cols)}")
                else:
                    st.success("No constant columns detected.")

            # --- DATA DISTRIBUTION (Keep but move if needed) ---
            if res.get('distributions'):
                topic_header("Data Distribution Insights")
                for dist in res['distributions']:
                    st.markdown(dist)

            # 10. Top Categories Analysis
            if res.get('top_categories'):
                topic_header("Top performing Categories")
                for top in res['top_categories']:
                    st.markdown(top)

            # 11. Dataset Cleaning Suggestions
            topic_header("Dataset Cleaning Suggestions")
            if res.get('suggestions'):
                for sugg in res['suggestions']:
                    st.info(f"💡 {sugg}")
            else:
                st.success("✨ Your dataset is in excellent shape!")
        else:
            st.info("Click 'Run Full Diagnosis' or 'Auto Clean' to view diagnostics.")

    # --- CONTENT GUARDS ---
    if st.session_state.active_tab != "Data Analysis" and not st.session_state.analysis_done:
        st.warning("⚠ Please run Data Analysis first.")
        st.session_state.active_tab = "Data Analysis"
        st.rerun()

    if st.session_state.active_tab == "Machine Learning Intelligence":
        centered_header("Intelligent Machine Learning Section")
        st.markdown("<p style='text-align: center; color: #9CA3AF;'>The AI Data Scientist has analyzed your clean data. Select a target column below to automatically train and evaluate the best predictive model.</p>", unsafe_allow_html=True)
        df = st.session_state.cleaned_df # ML works on cleaned data
        
        # Phase 1 & 2: Detect & Select ML Eligible Columns
        with st.spinner("Analyzing column eligibility..."):
            try:
                eligible_res = requests.get(f"{BACKEND_URL}/api/ml_eligible_columns", params={"dataset_id": st.session_state.dataset_id}).json()
                eligible_cols = eligible_res.get("eligible_columns", df.columns.tolist())
            except Exception:
                eligible_cols = df.columns.tolist()

        # Determine current index from memory to PERSIST across tab switches
        target_idx = 0
        if st.session_state.target_memory in eligible_cols:
            target_idx = eligible_cols.index(st.session_state.target_memory)
            
        target = st.selectbox("Select Target Column", eligible_cols, index=target_idx)
        st.session_state.target_memory = target

        # Robust Target Validation
        with st.spinner("Validating selection..."):
            try:
                v_res = requests.get(f"{BACKEND_URL}/api/validate_target", params={"dataset_id": st.session_state.dataset_id, "target_column": target}).json()
                if not v_res["valid"]:
                    st.error(f"⚠ {v_res['error']}")
                elif v_res["warnings"]:
                    for w in v_res["warnings"]:
                        st.warning(w)
            except Exception:
                pass

        if st.session_state.pending_action == "forge":
            st.session_state.ml_results = None # Clear old results
            try:
                with st.spinner("Running Intelligent Selection & Training..."):
                    res = requests.post(f"{BACKEND_URL}/api/train_model", params={"dataset_id": st.session_state.dataset_id, "target_column": target}).json()
                    if "detail" in res:
                        st.error(f"❌ Training Failed: {res['detail']}")
                        st.session_state.ml_results = "ERROR"
                    else:
                        st.session_state.ml_results = res
            finally:
                st.session_state.pending_action = None
                st.session_state.is_processing = False
                st.rerun()
        else:
            st.button("🔥 Forge AI Model", use_container_width=True, on_click=set_action, args=("forge",), disabled=st.session_state.is_processing)
        
        if st.session_state.ml_results:
            if st.session_state.ml_results == "ERROR":
                st.info("💡 Model training failed. Check the dataset and try a different target column.")
            else:
                res = st.session_state.ml_results

                # --- COMPACT 2-LINE HEADER (Centered Confidence) ---
                prob_type = "Classification" if res.get("problem_type") == "classification" else "Regression"
                conf = res.get("advanced_insights", {}).get("confidence", "Low")
                
                # Confidence Color & Badge Map
                conf_emoji = {"High": "🟢 High", "Medium": "🟡 Medium", "Low": "🔴 Low"}
                badge = conf_emoji.get(conf, "🔴 Low")
                conf_color = "#10B981" if conf == "High" else "#F59E0B" if conf == "Medium" else "#EF4444"
                
                # Inject Neon CSS
                st.markdown(f"""
                <style>
                @keyframes neonPulse_{conf} {{
                    0% {{ box-shadow: 0 0 5px {conf_color}, 0 0 10px {conf_color}; opacity: 0.8; }}
                    50% {{ box-shadow: 0 0 15px {conf_color}, 0 0 30px {conf_color}; opacity: 1.0; }}
                    100% {{ box-shadow: 0 0 5px {conf_color}, 0 0 10px {conf_color}; opacity: 0.8; }}
                }}
                .neon-badge {{
                    animation: neonPulse_{conf} 2s infinite ease-in-out;
                    background-color: {conf_color};
                    color: white;
                    padding: 6px 16px;
                    border-radius: 8px;
                    font-weight: bold;
                    font-size: 0.95em;
                    border: 1px solid rgba(255,255,255,0.3);
                    display: inline-block;
                }}
                </style>
                """, unsafe_allow_html=True)

                # Center ML Header for Mobile
                st.markdown(f"""
                    <div class="ml-header-mobile">
                        <h4 style='display:inline-block; margin: 0 10px 10px 0;'>🎯 Type: {prob_type}</h4>
                        <div style='margin-bottom: 15px;'><span class='neon-badge'>{badge} Confidence</span></div>
                        <h3 style='margin-top: 5px;'>🧠 {res.get('recommended_algorithm')}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)

                # --- ELIGIBILITY (Target & Features) ---
                topic_header("🔍 Dataset Eligibility")
                for entry in res.get("audit_log", []):
                    if entry: 
                        if "Target:" in entry or "Features Used:" in entry or "Excluded:" in entry:
                            st.markdown(f"**{entry}**")
                        else:
                            st.markdown(entry)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # --- FINAL RESULT ---
                st.info(f"**💡 Final Result:**\n\n{res.get('summary', '')}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # --- MODEL PERFORMANCE & HEALTH ---
                topic_header("📊 Performance & Health")
                
                metric_val = list(res['metrics'].values())[0]
                metric_name = "Accuracy" if res.get("problem_type") == "classification" else "R² Score"
                
                # Stack metrics vertically in a slightly more visible way
                st.markdown("<div class='swipe-zone'>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric(metric_name, f"{metric_val:.1%}" if "Accuracy" in metric_name else f"{metric_val:.2f}")
                    if res.get("problem_type") == "classification":
                        st.metric("Model Lift", f"{res.get('model_lift', 0):.1f}%")
                with c2:
                    if res.get("problem_type") == "classification":
                        st.metric("Baseline", f"{res.get('baseline_score', 0):.1%}")
                    diag = res.get("overfitting_diagnosis", {})
                    st.metric("Model Health", diag.get('status', 'Good'))
                st.markdown("</div>", unsafe_allow_html=True)
                if diag.get("description"): 
                    st.caption(f"Health Detail: {diag['description']}")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # --- ADVANCED AI INSIGHTS ---
                adv = res.get("advanced_insights", {})
                if adv:
                    topic_header("🤖 Advanced AI Insights")
                    st.markdown(f"**📦 Data Size:** {adv.get('data_size')}")
                    st.markdown(f"**🎯 Prediction Difficulty:** {adv.get('prediction_difficulty')}")
                    
                    st.markdown(f"**⚠️ Limitation:** {adv.get('limitation')}")
                    st.markdown(f"**💡 Suggestion:** {adv.get('suggestion')}")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                
                # --- KEY FACTORS ---
                if res.get("feature_importance"):
                    topic_header("📈 Key Predictive Factors")
                    importance_df = pd.DataFrame(res["feature_importance"])
                    
                    first_note = importance_df.iloc[0].get("note", "")
                    if first_note:
                        st.warning(first_note)
                        
                    for _, row in importance_df.head(5).iterrows():
                        feature_clean = str(row['feature']).replace('\n', '').replace('\u200b', '').strip()
                        st.markdown(f"**{feature_clean}** → {row['influence']}")
                        
                    st.markdown("<br>", unsafe_allow_html=True)
                    # Optional visualization stacked underneath
                    fig_fi = px.bar(importance_df.head(5), x="importance", y="feature", orientation='h', template="plotly_dark", color_discrete_sequence=['#38BDF8'], title="Top 5 Features")
                    fig_fi.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0), yaxis={'categoryorder':'total ascending'})
                    st.plotly_chart(fig_fi, use_container_width=True)

                # v11 Model Download
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                model_url = f"{BACKEND_URL}{res.get('model_url')}"
                st.link_button("💾 Download Trained Model (.pkl)", model_url, use_container_width=True, type="primary")

    elif st.session_state.active_tab == "Insights Dashboard":
        centered_header("Visual Insights Dashboard")
        df = st.session_state.cleaned_df  # Visualize cleaned results
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        # ── CHART PALETTE (each chart gets a completely different color) ──
        CLR_HISTOGRAM  = "#34D399"   # emerald green
        CLR_BOXPLOT    = "#FBBF24"   # amber / orange
        CLR_BARCHART   = "#A78BFA"   # violet / purple
        CLR_SCATTER    = "#FB7185"   # rose / pink
        CLR_IMPORTANCE = "#22D3EE"   # cyan / teal
        CLR_DONUT      = ["#34D399", "#A78BFA", "#FBBF24", "#FB7185", "#38BDF8", "#F472B6"]
        CLR_TARGET     = "#F59E0B"   # warm amber
        PLOTLY_DEFAULTS = dict(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#E5E7EB"),
            margin=dict(l=20, r=20, t=40, b=20),
        )

        # ═══════════════════════════════════════════
        # 1. QUICK OVERVIEW (native Streamlit metrics)
        # ═══════════════════════════════════════════
        st.markdown("<div class='swipe-zone'>", unsafe_allow_html=True)
        ov1, ov2, ov3, ov4 = st.columns(4)
        with ov1: st.metric("Total Rows", f"{len(df):,}")
        with ov2: st.metric("Total Columns", f"{len(df.columns)}")
        with ov3: st.metric("Numerical Features", f"{len(num_cols)}")
        with ov4: st.metric("Categorical Features", f"{len(cat_cols)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ═══════════════════════════════════════════
        # 2. DATA TYPE BREAKDOWN (Donut Chart)
        # ═══════════════════════════════════════════
        topic_header("🧩 Data Type Breakdown")

        # Build type counts dynamically from the cleaned dataframe
        type_counts = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                type_counts["Numeric"] = type_counts.get("Numeric", 0) + 1
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                type_counts["Date"] = type_counts.get("Date", 0) + 1
            elif df[col].dtype == "object" or df[col].dtype.name == "category":
                if df[col].nunique() / len(df) > 0.9:
                    type_counts["ID / Unique"] = type_counts.get("ID / Unique", 0) + 1
                else:
                    type_counts["Categorical"] = type_counts.get("Categorical", 0) + 1
            else:
                type_counts["Other"] = type_counts.get("Other", 0) + 1

        donut_df = pd.DataFrame({"Type": list(type_counts.keys()), "Count": list(type_counts.values())})
        fig_donut = px.pie(
            donut_df, names="Type", values="Count",
            hole=0.45,
            color_discrete_sequence=CLR_DONUT,
        )
        fig_donut.update_layout(**PLOTLY_DEFAULTS, height=350, showlegend=True)
        fig_donut.update_traces(textinfo="label+percent", textfont_size=13)
        st.plotly_chart(fig_donut, use_container_width=True)

        # ═══════════════════════════════════════════
        # 2. FEATURE DISTRIBUTION — Histogram & Box Plot
        # ═══════════════════════════════════════════
        topic_header("📊 Feature Distribution")

        if num_cols:
            dist_col = st.selectbox("Select Numerical Feature", num_cols, key="dash_num_col")

            col_hist, col_box = st.columns(2)

            with col_hist:
                st.markdown("##### Histogram")
                fig_hist = px.histogram(
                    df, x=dist_col,
                    color_discrete_sequence=[CLR_HISTOGRAM],
                    nbins=30,
                )
                fig_hist.update_layout(**PLOTLY_DEFAULTS, height=350)
                fig_hist.update_traces(marker_line_width=0.5, marker_line_color="#0c4a6e")
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_box:
                st.markdown("##### Box Plot")
                fig_box = px.box(
                    df, y=dist_col,
                    color_discrete_sequence=[CLR_BOXPLOT],
                )
                fig_box.update_layout(**PLOTLY_DEFAULTS, height=350)
                fig_box.update_traces(marker_color=CLR_BOXPLOT, line_color=CLR_BOXPLOT)
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("No numerical features available for distribution charts.")

        # ═══════════════════════════════════════════
        # 3. CATEGORICAL BAR CHART
        # ═══════════════════════════════════════════
        if cat_cols:
            topic_header("📊 Category Distribution")
            cat_sel = st.selectbox("Select Categorical Feature", cat_cols, key="dash_cat_col")
            value_counts = df[cat_sel].value_counts().head(15).reset_index()
            value_counts.columns = [cat_sel, "Count"]

            fig_bar = px.bar(
                value_counts, x=cat_sel, y="Count",
                color_discrete_sequence=[CLR_BARCHART],
            )
            fig_bar.update_layout(**PLOTLY_DEFAULTS, height=380)
            fig_bar.update_traces(marker_line_width=0, textposition="outside", texttemplate="%{y}")
            st.plotly_chart(fig_bar, use_container_width=True)

        # ═══════════════════════════════════════════
        # 4. CORRELATION HEATMAP
        # ═══════════════════════════════════════════
        topic_header("🔗 Correlation Heatmap")

        if len(num_cols) >= 2:
            corr_matrix = df[num_cols].corr().round(2)
            # Limit to 25 for readability
            if len(corr_matrix.columns) > 25:
                st.caption("Showing first 25 numeric columns for readability.")
                corr_matrix = corr_matrix.iloc[:25, :25]

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=".2f",
                color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1,
                labels=dict(color="Correlation"),
            )
            fig_corr.update_layout(**PLOTLY_DEFAULTS, height=max(400, len(corr_matrix.columns) * 22))
            st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Correlation heatmap requires at least 2 numeric columns.")

        # ═══════════════════════════════════════════
        # TOP 5 STRONGEST CORRELATIONS (Bar Chart)
        # ═══════════════════════════════════════════
        if len(num_cols) >= 2:
            topic_header("🏆 Top Correlations")
            corr_full = df[num_cols].corr()
            pairs = []
            cols_list = corr_full.columns.tolist()
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    val = corr_full.iloc[i, j]
                    pairs.append({"Pair": f"{cols_list[i]}  ↔  {cols_list[j]}", "Correlation": round(abs(val), 3), "Direction": "Positive" if val > 0 else "Negative"})
            pairs_sorted = sorted(pairs, key=lambda x: x["Correlation"], reverse=True)[:5]
            if pairs_sorted:
                top_corr_df = pd.DataFrame(pairs_sorted)
                fig_topc = px.bar(
                    top_corr_df, x="Correlation", y="Pair",
                    orientation="h",
                    color="Direction",
                    color_discrete_map={"Positive": "#34D399", "Negative": "#F87171"},
                )
                fig_topc.update_layout(**PLOTLY_DEFAULTS, height=280, yaxis={"categoryorder": "total ascending"})
                fig_topc.update_traces(textposition="outside", texttemplate="%{x:.3f}")
                st.plotly_chart(fig_topc, use_container_width=True)

        # ═══════════════════════════════════════════
        # 5. SCATTER / TARGET ANALYSIS
        # ═══════════════════════════════════════════
        topic_header("🎯 Relationship / Target Analysis")

        if len(num_cols) >= 2:
            sc1, sc2 = st.columns(2)
            with sc1:
                default_x = 0
                scatter_x = st.selectbox("X-axis Feature", num_cols, index=default_x, key="dash_scatter_x")
            with sc2:
                default_y = min(1, len(num_cols) - 1)
                scatter_y = st.selectbox("Y-axis Feature", num_cols, index=default_y, key="dash_scatter_y")

            fig_scatter = px.scatter(
                df, x=scatter_x, y=scatter_y,
                color_discrete_sequence=[CLR_SCATTER],
                opacity=0.65,
            )
            fig_scatter.update_layout(**PLOTLY_DEFAULTS, height=420)
            fig_scatter.update_traces(marker=dict(size=6, line=dict(width=0.5, color="#0c4a6e")))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Scatter plot requires at least 2 numeric columns.")

        # ═══════════════════════════════════════════
        # TARGET DISTRIBUTION (always visible)
        # ═══════════════════════════════════════════
        topic_header("🎯 Target Distribution")
        all_cols_list = df.columns.tolist()
        target_pick = st.selectbox("Select Target Column", all_cols_list, key="dash_target_col")

        if df[target_pick].nunique() <= 15:
            # Classification-style: count plot
            target_counts = df[target_pick].value_counts().reset_index()
            target_counts.columns = [target_pick, "Count"]
            fig_target = px.bar(
                target_counts, x=target_pick, y="Count",
                color_discrete_sequence=[CLR_TARGET],
            )
            fig_target.update_layout(**PLOTLY_DEFAULTS, height=350)
            fig_target.update_traces(textposition="outside", texttemplate="%{y}")
            st.plotly_chart(fig_target, use_container_width=True)
        else:
            # Regression-style: histogram
            fig_target = px.histogram(
                df, x=target_pick,
                color_discrete_sequence=[CLR_TARGET],
                nbins=30,
            )
            fig_target.update_layout(**PLOTLY_DEFAULTS, height=350)
            st.plotly_chart(fig_target, use_container_width=True)

        # ═══════════════════════════════════════════
        # 6. FEATURE IMPORTANCE (from ML)
        # ═══════════════════════════════════════════
        topic_header("📈 Feature Importance")
        
        ml_res = st.session_state.ml_results if st.session_state.ml_results and st.session_state.ml_results != "ERROR" else None
        if ml_res and ml_res.get("feature_importance"):
            importance_df = pd.DataFrame(ml_res["feature_importance"])
            importance_df["feature"] = importance_df["feature"].apply(
                lambda x: str(x).replace('\n', '').replace('\u200b', '').strip()
            )
            top_features = importance_df.head(10)

            fig_fi = px.bar(
                top_features,
                x="importance", y="feature",
                orientation="h",
                color_discrete_sequence=[CLR_IMPORTANCE],
                labels={"importance": "Importance Score", "feature": "Feature"},
            )
            fig_fi.update_layout(
                **PLOTLY_DEFAULTS,
                height=max(280, len(top_features) * 35),
                yaxis={"categoryorder": "total ascending"},
            )
            fig_fi.update_traces(marker_line_width=0, textposition="outside", texttemplate="%{x:.3f}")
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("💡 Feature Importance identifies the strongest 'drivers' in your data. It will appear here after you train a model for your chosen 'Target' column.")
            # Physically isolated (no columns = no movement)
            if st.button("🚀 Go to ML Intelligence", use_container_width=True, type="primary"):
                st.session_state.active_tab = "Machine Learning Intelligence"
                st.rerun()

    elif st.session_state.active_tab == "Download":
        centered_header("Data Export Center")
        topic_header("Final Export Options")
        st.write("Ready to export your forged dataset and insights?")
        st.link_button("💾 Download Clean Dataset (CSV)", f"{BACKEND_URL}/api/download_clean?dataset_id={st.session_state.dataset_id}", use_container_width=True)
        
        if st.button("📄 Export Analysis Report", use_container_width=True):
            if st.session_state.analysis_results:
                report = pd.DataFrame([st.session_state.analysis_results]).to_csv().encode('utf-8')
                st.download_button("Download Report", report, file_name="dataforge_report.csv")
            else:
                st.warning("Run analysis first to generate a report.")
    

# --- FOOTER ---
st.markdown("""
<div class='footer'>
    <p><b>DataForge AI</b> — Powered by Python, FastAPI & Advanced ML Intelligence</p>
</div>
""", unsafe_allow_html=True)
