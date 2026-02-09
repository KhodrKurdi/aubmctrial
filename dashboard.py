import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AUBMC Physician Performance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# STYLING
# =========================
st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1f77b4; margin-bottom: 0.25rem; }
    .subtle { color: #666; font-size: 0.95rem; margin-bottom: 1rem; }
    .section { margin-top: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HELPERS (keep ALL helpers here - never inside page blocks)
# =========================
def safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pct(n: float, d: float) -> float:
    return (n / d * 100.0) if d else 0.0

def fmt_num(x, decimals=2) -> str:
    if x is None:
        return "N/A"
    try:
        if np.isnan(x):
            return "N/A"
    except Exception:
        pass
    return f"{x:,.{decimals}f}"

def ensure_year(eval_df: pd.DataFrame) -> pd.DataFrame:
    if "Year" in eval_df.columns:
        return eval_df
    date_col = safe_col(eval_df, ["Fillout Date (mm/dd/yy)", "Fillout Date", "Date"])
    if date_col:
        eval_df[date_col] = pd.to_datetime(eval_df[date_col], errors="coerce")
        eval_df["Year"] = eval_df[date_col].dt.year
    return eval_df

def physician_score_summary(eval_df: pd.DataFrame,
                            selected_physician,
                            selected_years: list[int],
                            peer_scope: str = "All") -> dict | None:
    """
    Percentile compares physician mean score vs other physicians' mean scores (fair).
    peer_scope:
      - "All" -> compare to all physicians in selected_years
      - "Department" -> compare only within same department (requires eval_df['Department'])
    """
    if selected_years:
        df = eval_df[eval_df["Year"].isin(selected_years)].copy()
    else:
        df = eval_df.copy()

    df_p = df[df["Subject ID"] == selected_physician].copy()
    if df_p.empty:
        return None

    p_scores = df_p["Response_Numeric"].dropna()
    if len(p_scores) == 0:
        return None

    p_avg = float(p_scores.mean())
    p_std = float(p_scores.std(ddof=1)) if len(p_scores) > 1 else 0.0
    p_min = float(p_scores.min())
    p_max = float(p_scores.max())

    # peer set
    if peer_scope == "Department" and "Department" in df.columns:
        p_dept = None
        if df_p["Department"].notna().any():
            try:
                p_dept = df_p["Department"].mode().iloc[0]
            except Exception:
                p_dept = None
        peers = df[df["Department"] == p_dept] if p_dept is not None else df
    else:
        peers = df

    peer_means = (
        peers.groupby("Subject ID")["Response_Numeric"]
        .mean()
        .dropna()
        .values
    )

    if len(peer_means) == 0:
        return None

    pct_rank = float((peer_means <= p_avg).mean() * 100.0)
    overall_avg = float(np.mean(peer_means))
    delta = p_avg - overall_avg

    return {
        "phys_avg": p_avg,
        "phys_std": p_std,
        "phys_min": p_min,
        "phys_max": p_max,
        "pct_rank": pct_rank,
        "overall_avg": overall_avg,
        "delta": delta,
    }

def make_percentile_gauge(pct_rank: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct_rank,
            number={"suffix": "%"},
            title={"text": "Percentile Rank"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 20]},
                    {"range": [20, 40]},
                    {"range": [40, 60]},
                    {"range": [60, 80]},
                    {"range": [80, 100]},
                ],
                "threshold": {"line": {"width": 4}, "value": pct_rank},
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
    return fig

# =========================
# DATA LOADING (CACHED)
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    # evaluation columns we need (keeps CSV load faster)
    eval_cols = [
        "Subject ID",
        "Raters Group",
        "Fillout Date (mm/dd/yy)",
        "Question",
        "Response",
        "Response_Numeric",
        "Q2_Comments",
        "Q2_Comments\n",
        "Year",
    ]

    def read_eval(path: str) -> pd.DataFrame:
        return pd.read_csv(path, low_memory=False, usecols=lambda c: c in eval_cols)

    # REQUIRED
    dept_2023 = read_eval("All_Departments_2023.csv")
    dept_2024 = read_eval("All_Departments_2024.csv")
    dept_2025 = read_eval("All_Departments_2025.csv")

    eval_df = pd.concat([dept_2023, dept_2024, dept_2025], ignore_index=True)
    eval_df = ensure_year(eval_df)

    # Parse date
    date_col = safe_col(eval_df, ["Fillout Date (mm/dd/yy)"])
    if date_col:
        eval_df[date_col] = pd.to_datetime(eval_df[date_col], errors="coerce")

    # Physicians Indicators (REQUIRED)
    phys_df = pd.read_csv("Physicians_Indicators_Anonymized.csv")

    # Normalize key
    if "Aubnetid" in phys_df.columns and "Subject ID" not in phys_df.columns:
        phys_df = phys_df.rename(columns={"Aubnetid": "Subject ID"})

    # Derive Year from FiscalCycle when available
    if "FiscalCycle" in phys_df.columns and "Year" not in phys_df.columns:
        try:
            phys_df["Year"] = phys_df["FiscalCycle"].str.extract(r"(\d{4})-\d{4}").astype(int) + 1
        except Exception:
            pass

    # Optional doctor stats
    docstats_df = None
    try:
        docstats_df = pd.read_csv("Doctor_Statistics_2025.csv")
    except Exception:
        docstats_df = None

    return eval_df, phys_df, docstats_df

eval_df, phys_df, docstats_df = load_data()

# Normalize comment field name
comment_col = safe_col(eval_df, ["Q2_Comments", "Q2_Comments\n"])
if comment_col is None:
    comment_col = "Q2_Comments"
    eval_df[comment_col] = np.nan

# Merge department into evaluations
if "Department" in phys_df.columns and "Subject ID" in phys_df.columns:
    dept_map = phys_df[["Subject ID", "Department"]].drop_duplicates()
    eval_with_dept = eval_df.merge(dept_map, on="Subject ID", how="left")
else:
    eval_with_dept = eval_df.copy()
    eval_with_dept["Department"] = "Unknown"

# =========================
# SIDEBAR NAV + GLOBAL FILTERS
# =========================
st.sidebar.title("🏥 Navigation")

if "page" not in st.session_state:
    st.session_state.page = "📊 Overview"

pages = ["📊 Overview", "👨‍⚕️ Physician Performance", "🏢 Department Analytics"]
page = st.sidebar.radio(
    "Select View",
    pages,
    index=pages.index(st.session_state.page),
)
st.session_state.page = page

all_years = sorted([int(y) for y in eval_with_dept["Year"].dropna().unique()])
default_years = all_years[-3:] if len(all_years) >= 3 else all_years
selected_years = st.sidebar.multiselect("Year(s)", all_years, default=default_years)

# filtered evaluation data
eval_f = eval_with_dept.copy()
if selected_years:
    eval_f = eval_f[eval_f["Year"].isin(selected_years)]

# filtered physician indicators
phys_f = phys_df.copy()
if "Year" in phys_f.columns and selected_years:
    phys_f = phys_f[phys_f["Year"].isin(selected_years)]

# =========================
# PAGE ROUTING (IMPORTANT: keep ONLY if/elif/elif here)
# =========================
if page == "📊 Overview":
    st.markdown('<div class="main-header">AUBMC Physician Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Executive overview of evaluations, performance, and operational indicators.</div>', unsafe_allow_html=True)
    st.markdown("---")

    total_phys = eval_f["Subject ID"].nunique()
    total_evals = len(eval_f)
    avg_score = eval_f["Response_Numeric"].mean() if total_evals else np.nan
    neg_rate = (eval_f["Response_Numeric"] <= 2).mean() * 100 if total_evals else 0

    visits_col = safe_col(phys_f, ["ClinicVisits", "Visits", "TotalVisits"])
    wait_col = safe_col(phys_f, ["ClinicWaitingTime", "WaitingTime", "AvgWaitingTime"])
    comp_col = safe_col(phys_f, ["PatientComplaints", "Complaints", "ComplaintCount"])

    total_visits = phys_f[visits_col].sum() if visits_col else None
    avg_wait = phys_f[wait_col].mean() if wait_col else None
    total_complaints = phys_f[comp_col].sum() if comp_col else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Physicians", f"{total_phys:,}")
    c2.metric("Evaluations", f"{total_evals:,}")
    c3.metric("Avg Score", f"{avg_score:.2f}/5" if total_evals else "N/A")
    c4.metric("Negative Rate", f"{neg_rate:.1f}%")
    c5.metric("Total Visits", f"{int(total_visits):,}" if total_visits is not None else "N/A")
    c6.metric("Avg Waiting", fmt_num(avg_wait, 1) if avg_wait is not None else "N/A")

    st.markdown("---")

    left, right = st.columns(2)
    with left:
        st.subheader("📈 Evaluations by Year")
        year_counts = eval_f["Year"].value_counts().sort_index()
        fig_year = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            labels={"x": "Year", "y": "Evaluations"},
        )
        fig_year.update_layout(height=330, showlegend=False)
        st.plotly_chart(fig_year, use_container_width=True)

        st.subheader("📊 Score Distribution")
        fig_dist = px.histogram(eval_f, x="Response_Numeric", nbins=6, labels={"Response_Numeric": "Score"})
        fig_dist.update_layout(height=330)
        st.plotly_chart(fig_dist, use_container_width=True)

    with right:
        st.subheader("👥 Evaluations by Rater Group")
        r = eval_f["Raters Group"].value_counts()
        fig_raters = px.pie(values=r.values, names=r.index, hole=0.45)
        fig_raters.update_layout(height=330)
        st.plotly_chart(fig_raters, use_container_width=True)

        st.subheader("🧭 Department Snapshot (Avg Score)")
        dept_avg = (
            eval_f.groupby("Department")["Response_Numeric"]
            .mean()
            .reset_index()
            .sort_values("Response_Numeric", ascending=False)
        )
        fig_dept = px.bar(dept_avg.head(12), x="Department", y="Response_Numeric", labels={"Response_Numeric": "Avg Score"})
        fig_dept.update_layout(height=330, xaxis_tickangle=35, showlegend=False)
        st.plotly_chart(fig_dept, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Outlier Detection — Funnel Plot (Physician Performance vs Sample Size)")

    # Build physician-level aggregates
    phys_agg = (
        eval_f.groupby("Subject ID")
        .agg(
            Avg_Score=("Response_Numeric", "mean"),
            N_Evals=("Response_Numeric", "size")
        )
        .reset_index()
    )
    
    # Remove very small samples (noise)
    min_n = st.slider("Minimum evaluations per physician", 5, 50, 10, 5)
    phys_agg = phys_agg[phys_agg["N_Evals"] >= min_n]
    
    if len(phys_agg) < 5:
        st.info("Not enough physicians after applying minimum evaluation threshold.")
    else:
        # Overall mean
        mu = phys_agg["Avg_Score"].mean()
    
        # Approximate standard error for bounded scores
        # Using empirical SD for stability
        sd = phys_agg["Avg_Score"].std(ddof=1)
    
        # 95% control limits
        phys_agg["SE"] = sd / np.sqrt(phys_agg["N_Evals"])
        phys_agg["Upper"] = mu + 1.96 * phys_agg["SE"]
        phys_agg["Lower"] = mu - 1.96 * phys_agg["SE"]
    
        # Identify outliers
        phys_agg["Outlier"] = np.where(
            phys_agg["Avg_Score"] > phys_agg["Upper"], "Above 95%",
            np.where(phys_agg["Avg_Score"] < phys_agg["Lower"], "Below 95%", "Within")
        )
    
        # Scatter plot
        fig = px.scatter(
            phys_agg,
            x="N_Evals",
            y="Avg_Score",
            color="Outlier",
            color_discrete_map={
                "Above 95%": "#2ecc71",
                "Below 95%": "#e74c3c",
                "Within": "#3498db"
            },
            labels={
                "N_Evals": "Number of Evaluations",
                "Avg_Score": "Average Score"
            },
            hover_data=["Subject ID", "Avg_Score", "N_Evals"],
        )
    
        # Mean line
        fig.add_hline(y=mu, line_dash="solid", line_color="blue", annotation_text="Overall Average")
    
        # Control limits
        fig.add_scatter(
            x=phys_agg["N_Evals"],
            y=phys_agg["Upper"],
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="95% Upper Limit"
        )
        fig.add_scatter(
            x=phys_agg["N_Evals"],
            y=phys_agg["Lower"],
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="95% Lower Limit"
        )
    
        fig.update_layout(
            height=520,
            legend_title_text="Performance Band"
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
        # Summary metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Physicians Analyzed", len(phys_agg))
        c2.metric("Above 95% Limit", (phys_agg["Outlier"] == "Above 95%").sum())
        c3.metric("Below 95% Limit", (phys_agg["Outlier"] == "Below 95%").sum())
    
        with st.expander("📋 View outlier physicians"):
            outliers = phys_agg[phys_agg["Outlier"] != "Within"]
            if outliers.empty:
                st.info("No physicians outside 95% control limits.")
            else:
                st.dataframe(
                    outliers.sort_values("Avg_Score"),
                    use_container_width=True,
                    hide_index=True
                )
    
elif page == "👨‍⚕️ Physician Performance":
    st.markdown('<div class="main-header">👨‍⚕️ Physician Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Filter by physician + year(s) to review trends, percentile ranking, and comments.</div>', unsafe_allow_html=True)
    st.markdown("---")

    physicians_list = sorted(eval_f["Subject ID"].dropna().unique())
    if not physicians_list:
        st.error("No physicians found in the selected year(s).")
        st.stop()

    selected_phys = st.sidebar.selectbox("Physician", physicians_list)

    dfp = eval_f[eval_f["Subject ID"] == selected_phys].copy()

    # Indicators for this physician (visits/waiting/complaints)
    phys_row = phys_f[phys_f["Subject ID"] == selected_phys].copy()
    visits_col = safe_col(phys_row, ["ClinicVisits", "Visits", "TotalVisits"])
    wait_col = safe_col(phys_row, ["ClinicWaitingTime", "WaitingTime", "AvgWaitingTime"])
    comp_col = safe_col(phys_row, ["PatientComplaints", "Complaints", "ComplaintCount"])

    visits_val = phys_row[visits_col].sum() if (visits_col and len(phys_row)) else None
    wait_val = phys_row[wait_col].mean() if (wait_col and len(phys_row)) else None
    comp_val = phys_row[comp_col].sum() if (comp_col and len(phys_row)) else None

    # Comments
    comments = dfp[dfp[comment_col].notna() & (dfp[comment_col].astype(str).str.strip() != "")]
    neg_rows = dfp[dfp["Response_Numeric"] <= 2]
    neg_rate = pct(len(neg_rows), len(dfp))

    # Top KPI row
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Score", f"{dfp['Response_Numeric'].mean():.2f}/5" if len(dfp) else "N/A")
    k2.metric("Evaluations", f"{len(dfp):,}")
    k3.metric("Total Comments", f"{len(comments):,}")
    k4.metric("Neg Score Rate", f"{neg_rate:.1f}%")
    k5.metric("Visits", f"{int(visits_val):,}" if visits_val is not None else "N/A")
    k6.metric("Avg Waiting", fmt_num(wait_val, 1) if wait_val is not None else "N/A")

    st.markdown("---")

    # Behavior Score Analysis + Percentile Gauge
    st.subheader("Behavior Score Analysis")

    peer_scope = st.radio("Compare percentile against:", ["All", "Department"], horizontal=True)

    summary = physician_score_summary(eval_f, selected_phys, selected_years, peer_scope=peer_scope)

    if summary is None:
        st.info("No sufficient data to compute percentile for this physician.")
    else:
        left, right = st.columns([1.15, 1])

        with left:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Average Score", f"{summary['phys_avg']:.3f}")
                st.metric("Min Score", f"{summary['phys_min']:.3f}")
            with c2:
                st.metric("Standard Deviation", f"{summary['phys_std']:.3f}")
                st.metric("Max Score", f"{summary['phys_max']:.3f}")

            if not np.isnan(summary["delta"]):
                if summary["delta"] < 0:
                    st.warning(f"{abs(summary['delta']):.3f} points below peer average")
                else:
                    st.success(f"{abs(summary['delta']):.3f} points above peer average")

            if comp_val is not None:
                st.caption(f"Complaints (from indicators): {int(comp_val):,}")

        with right:
            fig_gauge = make_percentile_gauge(summary["pct_rank"])
            st.plotly_chart(fig_gauge, use_container_width=True)
            st.info(f"Better than **{summary['pct_rank']:.1f}%** of physicians ({peer_scope.lower()} comparison).")

    st.markdown("---")

    # Avg score trend
    st.subheader("📈 Average Score Trend (Year over Year)")
    yoy = dfp.groupby("Year")["Response_Numeric"].mean().reset_index()
    fig_trend = px.line(yoy, x="Year", y="Response_Numeric", markers=True, labels={"Response_Numeric": "Avg Score"})
    fig_trend.update_layout(height=380, yaxis_range=[0, 5])
    st.plotly_chart(fig_trend, use_container_width=True)

    # Distribution + Lowest questions
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Response Distribution")
        order = ["Always", "Most of the time", "Sometimes", "Hardly ever", "Never"]
        rc = dfp["Response"].value_counts()
        rc = rc.reindex([x for x in order if x in rc.index])
        fig_resp = px.bar(x=rc.index, y=rc.values, labels={"x": "Response", "y": "Count"})
        fig_resp.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig_resp, use_container_width=True)

    with col2:
        st.subheader("📌 Selected Physician Summary")

        base = eval_f.copy()
        
        # -------------------------
        # Physician-level statistics
        # -------------------------
        phys_stats = (
            base.groupby("Subject ID")
            .agg(
                Evaluations=("Response_Numeric", "size"),
                Avg_Score=("Response_Numeric", "mean"),
                Std_Score=("Response_Numeric", "std"),
            )
            .reset_index()
        )
        
        # Format
        phys_stats["Avg_Score"] = phys_stats["Avg_Score"].round(2)
        phys_stats["Std_Score"] = phys_stats["Std_Score"].round(2)
        
        # -------------------------
        # Join operational indicators
        # -------------------------
        visits_col = safe_col(phys_f, ["ClinicVisits", "Visits", "TotalVisits"])
        wait_col   = safe_col(phys_f, ["ClinicWaitingTime", "WaitingTime", "AvgWaitingTime"])
        comp_col   = safe_col(phys_f, ["PatientComplaints", "Complaints", "ComplaintCount"])
        
        ind_cols = ["Subject ID"]
        rename_map = {}
        
        if visits_col:
            ind_cols.append(visits_col)
            rename_map[visits_col] = "Visits"
        if wait_col:
            ind_cols.append(wait_col)
            rename_map[wait_col] = "Avg_Waiting"
        if comp_col:
            ind_cols.append(comp_col)
            rename_map[comp_col] = "Complaints"
        
        if len(ind_cols) > 1 and "Subject ID" in phys_f.columns:
            indicators = phys_f[ind_cols].copy().rename(columns=rename_map)
        
            agg_dict = {}
            if "Visits" in indicators.columns:
                agg_dict["Visits"] = "sum"
            if "Avg_Waiting" in indicators.columns:
                agg_dict["Avg_Waiting"] = "mean"
            if "Complaints" in indicators.columns:
                agg_dict["Complaints"] = "sum"
        
            indicators = indicators.groupby("Subject ID", as_index=False).agg(agg_dict)
            phys_stats = phys_stats.merge(indicators, on="Subject ID", how="left")
        
        # Ensure indicator columns always exist
        for col in ["Visits", "Avg_Waiting", "Complaints"]:
            if col not in phys_stats.columns:
                phys_stats[col] = np.nan
        
        if "Avg_Waiting" in phys_stats.columns:
            phys_stats["Avg_Waiting"] = phys_stats["Avg_Waiting"].round(1)
        
        # -------------------------
        # Linked single-row display
        # -------------------------
        selected_row = phys_stats[phys_stats["Subject ID"] == selected_phys]
        
        ordered_cols = [
            "Evaluations",
            "Avg_Score",
            "Std_Score",
            "Visits",
            "Avg_Waiting",
            "Complaints",
        ]
        
        if selected_row.empty:
            st.info("No statistics available for the selected physician (in the selected years).")
        else:
            st.dataframe(
                selected_row[ordered_cols],
                use_container_width=True,
                hide_index=True
            )
        

    st.markdown("---")
    st.subheader("💬 Comment Review (sample)")
    if len(comments):
        show = comments[["Year", "Raters Group", "Response", "Response_Numeric", comment_col]].head(20)
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No comments available for the selected physician/year(s).")

elif page == "🏢 Department Analytics":
    st.markdown('<div class="main-header">🏢 Department Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Department-level comparisons: scores, volume, and (optional) operational indicators.</div>', unsafe_allow_html=True)
    st.markdown("---")

    depts = sorted([d for d in eval_f["Department"].dropna().unique()])
    selected_dept = st.sidebar.selectbox("Department", ["All Departments"] + depts)

    dfd = eval_f.copy()
    if selected_dept != "All Departments":
        dfd = dfd[dfd["Department"] == selected_dept]

    # Operational indicators aggregated for department (if available in phys file)
    phys_dept = phys_f.copy()
    if selected_dept != "All Departments" and "Department" in phys_dept.columns:
        phys_dept = phys_dept[phys_dept["Department"] == selected_dept]

    visits_col = safe_col(phys_dept, ["ClinicVisits", "Visits", "TotalVisits"])
    wait_col = safe_col(phys_dept, ["ClinicWaitingTime", "WaitingTime", "AvgWaitingTime"])
    comp_col = safe_col(phys_dept, ["PatientComplaints", "Complaints", "ComplaintCount"])

    total_visits = phys_dept[visits_col].sum() if visits_col else None
    avg_wait = phys_dept[wait_col].mean() if wait_col else None
    total_complaints = phys_dept[comp_col].sum() if comp_col else None

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Score", f"{dfd['Response_Numeric'].mean():.2f}/5" if len(dfd) else "N/A")
    k2.metric("Evaluations", f"{len(dfd):,}")
    k3.metric("Physicians", f"{dfd['Subject ID'].nunique():,}")
    k4.metric("Total Visits", f"{int(total_visits):,}" if total_visits is not None else "N/A")
    k5.metric("Avg Waiting", fmt_num(avg_wait, 1) if avg_wait is not None else "N/A")
    k6.metric("Complaints", f"{int(total_complaints):,}" if total_complaints is not None else "N/A")

    st.markdown("---")

    if selected_dept == "All Departments":
        st.subheader("🏥 Department Snapshot (Avg Score)")

        dept_avg = (
            eval_f.groupby("Department")["Response_Numeric"]
            .mean()
            .reset_index(name="Avg_Score")
            .sort_values("Avg_Score", ascending=False)
        )
        
        mode = st.radio("View:", ["Top 10", "Bottom 10", "Top & Bottom 10"], horizontal=True)
        
        if mode == "Top 10":
            show = dept_avg.head(10)
        elif mode == "Bottom 10":
            show = dept_avg.tail(10).sort_values("Avg_Score", ascending=True)
        else:
            show = pd.concat([dept_avg.head(10), dept_avg.tail(10)]).sort_values("Avg_Score", ascending=True)
        
        fig = px.bar(
            show,
            y="Department",
            x="Avg_Score",
            orientation="h",
            labels={"Avg_Score": "Avg Score"},
        )
        fig.update_layout(height=520, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


    st.subheader("📈 Score Trend by Year")
    yoy = dfd.groupby("Year")["Response_Numeric"].mean().reset_index()
    fig_tr = px.line(yoy, x="Year", y="Response_Numeric", markers=True, labels={"Response_Numeric": "Avg Score"})
    fig_tr.update_layout(height=360, yaxis_range=[0, 5])
    st.plotly_chart(fig_tr, use_container_width=True)

    st.subheader("👥 Rater Group Breakdown")
    rg = dfd["Raters Group"].value_counts().reset_index()
    rg.columns = ["Rater Group", "Count"]
    fig_rg = px.bar(rg, x="Rater Group", y="Count")
    fig_rg.update_layout(height=330, xaxis_tickangle=35, showlegend=False)
    st.plotly_chart(fig_rg, use_container_width=True)

    st.markdown("---")
    show_heatmap = st.checkbox("Show heatmap (slower on large data)")
    if show_heatmap:
        st.subheader("🔥 Heatmap: Rater Group vs Year (Avg Score)")
        hm = dfd.groupby(["Raters Group", "Year"])["Response_Numeric"].mean().reset_index()
        pivot = hm.pivot(index="Raters Group", columns="Year", values="Response_Numeric")
        fig_hm = px.imshow(pivot, aspect="auto", labels=dict(color="Avg Score"))
        fig_hm.update_layout(height=420)
        st.plotly_chart(fig_hm, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%B %d, %Y')}")
