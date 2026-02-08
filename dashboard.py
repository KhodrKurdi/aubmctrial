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
    initial_sidebar_state="expanded"
)

# =========================
# STYLING
# =========================
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 800; color: #1f77b4; }
    .subtle { color: #666; font-size: 0.95rem; }
    .kpi-box { background: #f6f7fb; border-radius: 14px; padding: 14px 16px; border: 1px solid #eee; }
</style>
""", unsafe_allow_html=True)

# =========================
# HELPERS
# =========================
def safe_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first matching column name from candidates, else None."""
    for c in candidates:
        if c in df.columns:
            return c
    return None

def pct(n, d):
    return (n / d * 100) if d else 0

def fmt_num(x, decimals=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:,.{decimals}f}"

def ensure_year(df_eval: pd.DataFrame) -> pd.DataFrame:
    # if Year exists already, keep it
    if "Year" in df_eval.columns:
        return df_eval
    # otherwise attempt to derive Year from date
    date_col = safe_col(df_eval, ["Fillout Date (mm/dd/yy)", "Fillout Date", "Date"])
    if date_col:
        df_eval[date_col] = pd.to_datetime(df_eval[date_col], errors="coerce")
        df_eval["Year"] = df_eval[date_col].dt.year
    return df_eval

# =========================
# DATA LOADING (CACHED)
# =========================
@st.cache_data(show_spinner=False)
def load_data():
    # --- Evaluation data (dept files) ---
    # Usecols keeps memory + load time low on big CSVs
    eval_cols = [
        "Subject ID", "Raters Group", "Fillout Date (mm/dd/yy)",
        "Question", "Response", "Response_Numeric",
        "Q2_Comments", "Q2_Comments\n", "Year"
    ]

    def read_eval(path):
        # If the dataset has extra columns, usecols will ignore others safely
        return pd.read_csv(path, low_memory=False, usecols=lambda c: (c in eval_cols))

    dept_2023 = read_eval("data/All_Departments_2023.csv")
    dept_2024 = read_eval("data/All_Departments_2024.csv")
    dept_2025 = read_eval("data/All_Departments_2025.csv")

    eval_df = pd.concat([dept_2023, dept_2024, dept_2025], ignore_index=True)
    eval_df = ensure_year(eval_df)

    # Standardize datetime
    date_col = safe_col(eval_df, ["Fillout Date (mm/dd/yy)"])
    if date_col:
        eval_df[date_col] = pd.to_datetime(eval_df[date_col], errors="coerce")

    # --- Physician indicators (visits, waiting time, complaints) ---
    phys_df = pd.read_csv("data/Physicians_Indicators_Anonymized.csv")

    # Map Aubnetid -> Subject ID (your earlier logic)
    if "Aubnetid" in phys_df.columns:
        phys_df = phys_df.rename(columns={"Aubnetid": "Subject ID"})

    # Create Year from FiscalCycle if exists (ex: 2023-2024 => 2024)
    if "FiscalCycle" in phys_df.columns:
        try:
            phys_df["Year"] = phys_df["FiscalCycle"].str.extract(r"(\d{4})-\d{4}").astype(int) + 1
        except Exception:
            pass

    # Optional: doctor-level aggregated stats (sentiment, negative comments, etc.)
    docstats = None
    try:
        docstats = pd.read_csv("data/Doctor_Statistics_2025.csv")
    except Exception:
        docstats = None

    return eval_df, phys_df, docstats

eval_df, phys_df, docstats_df = load_data()

# =========================
# NORMALIZE COMMENT FIELD
# =========================
comment_col = safe_col(eval_df, ["Q2_Comments", "Q2_Comments\n"])
if comment_col is None:
    comment_col = "Q2_Comments"
    eval_df[comment_col] = np.nan

# =========================
# MERGE DEPARTMENT INTO EVALS
# =========================
dept_map = None
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

page = st.sidebar.radio(
    "Select View",
    ["📊 Overview", "👨‍⚕️ Physician Performance", "🏢 Department Analytics"],
    index=["📊 Overview", "👨‍⚕️ Physician Performance", "🏢 Department Analytics"].index(st.session_state.page)
)

st.session_state.page = page

# Global year filter (applies to all pages)
all_years = sorted([y for y in eval_df["Year"].dropna().unique()])
default_years = all_years[-3:] if len(all_years) >= 3 else all_years

selected_years = st.sidebar.multiselect("Year(s)", all_years, default=default_years)

# Filter evaluation data by selected years
eval_f = eval_with_dept[eval_with_dept["Year"].isin(selected_years)].copy()

# Filter physician indicators by selected years if Year exists there too
phys_f = phys_df.copy()
if "Year" in phys_f.columns and selected_years:
    phys_f = phys_f[phys_f["Year"].isin(selected_years)]

# =========================
# OVERVIEW PAGE
# =========================
if page == "📊 Overview":
    st.markdown('<div class="main-header">AUBMC Physician Performance Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Executive overview of evaluations, performance, and operational indicators.</div>', unsafe_allow_html=True)
    st.markdown("---")

    # KPIs
    total_phys = eval_f["Subject ID"].nunique()
    total_evals = len(eval_f)
    avg_score = eval_f["Response_Numeric"].mean()
    neg_rate = (eval_f["Response_Numeric"] <= 2).mean() * 100 if total_evals else 0

    # Operational indicators (from Physicians_Indicators_Anonymized)
    visits_col = safe_col(phys_f, ["ClinicVisits", "Visits", "TotalVisits"])
    wait_col   = safe_col(phys_f, ["ClinicWaitingTime", "WaitingTime", "AvgWaitingTime"])
    comp_col   = safe_col(phys_f, ["PatientComplaints", "Complaints", "ComplaintCount"])

    total_visits = phys_f[visits_col].sum() if visits_col else None
    avg_wait = phys_f[wait_col].mean() if wait_col else None
    total_complaints = phys_f[comp_col].sum() if comp_col else None

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Physicians", f"{total_phys:,}")
    c2.metric("Evaluations", f"{total_evals:,}")
    c3.metric("Avg Score", f"{avg_score:.2f}/5" if total_evals else "N/A")
    c4.metric("Negative Rate", f"{neg_rate:.1f}%")
    c5.metric("Total Visits", f"{int(total_visits):,}" if total_visits is not None else "N/A")
    c6.metric("Avg Waiting", f"{fmt_num(avg_wait, 1)}" if avg_wait is not None else "N/A")

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.subheader("📈 Evaluations by Year")
        year_counts = eval_f["Year"].value_counts().sort_index()
        fig = px.bar(x=year_counts.index, y=year_counts.values, labels={"x": "Year", "y": "Evaluations"})
        fig.update_layout(height=330, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📊 Score Distribution")
        fig2 = px.histogram(eval_f, x="Response_Numeric", nbins=6, labels={"Response_Numeric": "Score"})
        fig2.update_layout(height=330)
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader("👥 Evaluations by Rater Group")
        r = eval_f["Raters Group"].value_counts()
        fig3 = px.pie(values=r.values, names=r.index, hole=0.45)
        fig3.update_layout(height=330)
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("🧭 Department Snapshot (Avg Score)")
        dept_avg = eval_f.groupby("Department")["Response_Numeric"].mean().reset_index().sort_values("Response_Numeric", ascending=False)
        fig4 = px.bar(dept_avg.head(12), x="Department", y="Response_Numeric", labels={"Response_Numeric": "Avg Score"})
        fig4.update_layout(height=330, xaxis_tickangle=35)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        if st.button("👨‍⚕️ Go to Physician Performance", use_container_width=True):
            st.session_state.page = "👨‍⚕️ Physician Performance"
            st.rerun()
    with colB:
        if st.button("🏢 Go to Department Analytics", use_container_width=True):
            st.session_state.page = "🏢 Department Analytics"
            st.rerun()

# =========================
# PHYSICIAN PERFORMANCE
# =========================
elif page == "👨‍⚕️ Physician Performance":
    st.markdown('<div class="main-header">👨‍⚕️ Physician Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Filter by physician + year(s) to review score trends, comments, and operational indicators.</div>', unsafe_allow_html=True)
    st.markdown("---")

    physicians_list = sorted(eval_f["Subject ID"].dropna().unique())
    selected_phys = st.sidebar.selectbox("Physician", physicians_list)

    dfp = eval_f[eval_f["Subject ID"] == selected_phys].copy()

    # Physician indicators row (visits, waiting, complaints)
    phys_row = phys_f[phys_f["Subject ID"] == selected_phys].copy()
    visits_col = safe_col(phys_row, ["ClinicVisits", "Visits", "TotalVisits"])
    wait_col   = safe_col(phys_row, ["ClinicWaitingTime", "WaitingTime", "AvgWaitingTime"])
    comp_col   = safe_col(phys_row, ["PatientComplaints", "Complaints", "ComplaintCount"])

    visits_val = phys_row[visits_col].sum() if (visits_col and len(phys_row)) else None
    wait_val   = phys_row[wait_col].mean() if (wait_col and len(phys_row)) else None
    comp_val   = phys_row[comp_col].sum() if (comp_col and len(phys_row)) else None

    # Comments / negative comments
    comments = dfp[dfp[comment_col].notna() & (dfp[comment_col].astype(str).str.strip() != "")]
    neg_comments = dfp[dfp["Response_Numeric"] <= 2]
    neg_comment_pct = pct(len(neg_comments), len(dfp))

    # docstats (optional)
    neg_comment_count_ds = None
    neg_comment_pct_ds = None
    if docstats_df is not None and "Subject ID" in docstats_df.columns:
        row = docstats_df[docstats_df["Subject ID"] == selected_phys]
        if len(row):
            if "Negative_Comment_Count" in row.columns:
                neg_comment_count_ds = int(row["Negative_Comment_Count"].iloc[0])
            if "Negative_Comment_Pct" in row.columns:
                neg_comment_pct_ds = float(row["Negative_Comment_Pct"].iloc[0])

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Score", f"{dfp['Response_Numeric'].mean():.2f}/5" if len(dfp) else "N/A")
    k2.metric("Evaluations", f"{len(dfp):,}")
    k3.metric("Total Comments", f"{len(comments):,}")
    k4.metric("Neg Score Rate", f"{neg_comment_pct:.1f}%")
    k5.metric("Visits", f"{int(visits_val):,}" if visits_val is not None else "N/A")
    k6.metric("Avg Waiting", f"{fmt_num(wait_val,1)}" if wait_val is not None else "N/A")

    st.markdown("---")

    # Score trend across years
    st.subheader("📈 Average Score Trend (Year over Year)")
    yoy = dfp.groupby("Year")["Response_Numeric"].mean().reset_index()
    fig = px.line(yoy, x="Year", y="Response_Numeric", markers=True, labels={"Response_Numeric": "Avg Score"})
    fig.update_layout(height=380, yaxis_range=[0, 5])
    st.plotly_chart(fig, use_container_width=True)

    # Extra: Response distribution + "worst questions"
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Response Distribution")
        order = ["Always", "Most of the time", "Sometimes", "Hardly ever", "Never"]
        rc = dfp["Response"].value_counts().reindex([x for x in order if x in dfp["Response"].unique()])
        fig2 = px.bar(x=rc.index, y=rc.values, labels={"x": "Response", "y": "Count"})
        fig2.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        st.subheader("⚠️ Lowest-Scoring Questions (Top 10)")
        # shorten question for display
        qshort = dfp.copy()
        qshort["Question_Short"] = qshort["Question"].astype(str).str.replace("_", " ").str[-90:]
        qavg = qshort.groupby("Question_Short")["Response_Numeric"].agg(["mean", "count"]).reset_index()
        qavg = qavg[qavg["count"] >= 5].sort_values("mean", ascending=True).head(10)
        fig3 = px.bar(qavg, y="Question_Short", x="mean", orientation="h",
                      labels={"mean": "Avg Score", "Question_Short": "Question"})
        fig3.update_layout(height=420)
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # Comments table
    st.subheader("💬 Comment Review (sample)")
    st.caption("You can later replace this with NLP topic modeling / sentiment explanation.")
    if len(comments):
        show = comments[["Year", "Raters Group", "Response", "Response_Numeric", comment_col]].head(15)
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("No comments available for the selected physician/year(s).")

    # Optional docstats view
    if neg_comment_count_ds is not None or neg_comment_pct_ds is not None:
        st.markdown("---")
        st.subheader("📌 Notes from Doctor Statistics (Optional file)")
        st.write({
            "Negative_Comment_Count (Doctor_Statistics_2025)": neg_comment_count_ds,
            "Negative_Comment_Pct (Doctor_Statistics_2025)": neg_comment_pct_ds
        })

# =========================
# DEPARTMENT ANALYTICS
# =========================
elif page == "🏢 Department Analytics":
    st.markdown('<div class="main-header">🏢 Department Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Department-level comparisons: scores, volume, and operational indicators.</div>', unsafe_allow_html=True)
    st.markdown("---")

    depts = sorted([d for d in eval_f["Department"].dropna().unique()])
    selected_dept = st.sidebar.selectbox("Department", ["All Departments"] + depts)

    dfd = eval_f.copy()
    if selected_dept != "All Departments":
        dfd = dfd[dfd["Department"] == selected_dept]

    # Operational indicators aggregated for department
    # We aggregate from phys_f (since waiting/complaints/visits live there)
    phys_dept = phys_f.copy()
    if selected_dept != "All Departments" and "Department" in phys_dept.columns:
        phys_dept = phys_dept[phys_dept["Department"] == selected_dept]

    visits_col = safe_col(phys_dept, ["ClinicVisits", "Visits", "TotalVisits"])
    wait_col   = safe_col(phys_dept, ["ClinicWaitingTime", "WaitingTime", "AvgWaitingTime"])
    comp_col   = safe_col(phys_dept, ["PatientComplaints", "Complaints", "ComplaintCount"])

    total_visits = phys_dept[visits_col].sum() if visits_col else None
    avg_wait = phys_dept[wait_col].mean() if wait_col else None
    total_complaints = phys_dept[comp_col].sum() if comp_col else None

    # KPIs
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Score", f"{dfd['Response_Numeric'].mean():.2f}/5" if len(dfd) else "N/A")
    k2.metric("Evaluations", f"{len(dfd):,}")
    k3.metric("Physicians", f"{dfd['Subject ID'].nunique():,}")
    k4.metric("Total Visits", f"{int(total_visits):,}" if total_visits is not None else "N/A")
    k5.metric("Avg Waiting", f"{fmt_num(avg_wait,1)}" if avg_wait is not None else "N/A")
    k6.metric("Complaints", f"{int(total_complaints):,}" if total_complaints is not None else "N/A")

    st.markdown("---")

    # Department comparisons (only if All Departments)
    if selected_dept == "All Departments":
        st.subheader("🏆 Department Performance Comparison")
        dept_perf = eval_f.groupby("Department").agg(
            Avg_Score=("Response_Numeric", "mean"),
            Evaluations=("Response_Numeric", "size"),
            Physicians=("Subject ID", "nunique")
        ).reset_index().sort_values("Avg_Score", ascending=False)

        fig = px.bar(dept_perf, x="Department", y="Avg_Score",
                     hover_data=["Evaluations", "Physicians"],
                     labels={"Avg_Score": "Avg Score"})
        fig.update_layout(height=420, xaxis_tickangle=35)
        st.plotly_chart(fig, use_container_width=True)

    # Trend over time (Year)
    st.subheader("📈 Score Trend by Year")
    yoy = dfd.groupby("Year")["Response_Numeric"].mean().reset_index()
    fig2 = px.line(yoy, x="Year", y="Response_Numeric", markers=True, labels={"Response_Numeric": "Avg Score"})
    fig2.update_layout(height=360, yaxis_range=[0, 5])
    st.plotly_chart(fig2, use_container_width=True)

    # Rater group breakdown
    st.subheader("👥 Rater Group Breakdown")
    rg = dfd["Raters Group"].value_counts().reset_index()
    rg.columns = ["Rater Group", "Count"]
    fig3 = px.bar(rg, x="Rater Group", y="Count")
    fig3.update_layout(height=330, xaxis_tickangle=35)
    st.plotly_chart(fig3, use_container_width=True)

    # Optional heatmap toggle (can be slow on big data)
    st.markdown("---")
    show_heatmap = st.checkbox("Show heatmap (slower on large data)")
    if show_heatmap:
        st.subheader("🔥 Heatmap: Rater Group vs Year (Avg Score)")
        hm = dfd.groupby(["Raters Group", "Year"])["Response_Numeric"].mean().reset_index()
        pivot = hm.pivot(index="Raters Group", columns="Year", values="Response_Numeric")
        fig4 = px.imshow(pivot, aspect="auto", labels=dict(color="Avg Score"))
        fig4.update_layout(height=420)
        st.plotly_chart(fig4, use_container_width=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%B %d, %Y')}")
