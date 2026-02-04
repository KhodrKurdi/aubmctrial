import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AUBMC Physician Performance Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        height: 60px;
        font-size: 18px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        physicians = pd.read_csv('/mnt/user-data/uploads/Physicians_Indicators_Anonymized.csv')
        dept_2023 = pd.read_csv('All_Departments_2023.csv', low_memory=False)
        dept_2024 = pd.read_csv('All_Departments_2024.csv', low_memory=False)
        dept_2025 = pd.read_csv('All_Departments_2025.csv', low_memory=False)
        
        # Combine department data
        dept_all = pd.concat([dept_2023, dept_2024, dept_2025], ignore_index=True)
        
        # Clean and process data
        dept_all['Fillout Date (mm/dd/yy)'] = pd.to_datetime(dept_all['Fillout Date (mm/dd/yy)'], errors='coerce')
        
        # Extract year from FiscalCycle for physicians
        physicians['Year'] = physicians['FiscalCycle'].str.extract(r'(\d{4})-\d{4}').astype(int) + 1
        
        return physicians, dept_all
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Load data
physicians_df, dept_df = load_data()

if physicians_df is None or dept_df is None:
    st.error("Failed to load data. Please check the file paths.")
    st.stop()

# Sidebar navigation
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Select View",
    ["📊 Overview", "👨‍⚕️ Physician Analytics", "🏢 Department Analytics"]
)

# =============================================================================
# OVERVIEW PAGE
# =============================================================================
if page == "📊 Overview":
    st.markdown('<h1 class="main-header">AUBMC Physician Performance Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Physicians",
            dept_df['Subject ID'].nunique(),
            help="Unique physicians in evaluation system"
        )
    
    with col2:
        st.metric(
            "Total Evaluations",
            len(dept_df),
            help="Total number of evaluation responses"
        )
    
    with col3:
        avg_score = dept_df['Response_Numeric'].mean()
        st.metric(
            "Average Score",
            f"{avg_score:.2f}/5",
            help="Overall average performance score"
        )
    
    with col4:
        st.metric(
            "Departments",
            physicians_df['Department'].nunique() if 'Department' in physicians_df.columns else "N/A",
            help="Number of unique departments"
        )
    
    st.markdown("---")
    
    # Overview charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Evaluations by Year")
        year_counts = dept_df['Year'].value_counts().sort_index()
        fig_year = px.bar(
            x=year_counts.index,
            y=year_counts.values,
            labels={'x': 'Year', 'y': 'Number of Evaluations'},
            color=year_counts.values,
            color_continuous_scale='Blues'
        )
        fig_year.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_year, use_container_width=True)
    
    with col2:
        st.subheader("👥 Evaluations by Rater Group")
        rater_counts = dept_df['Raters Group'].value_counts()
        fig_raters = px.pie(
            values=rater_counts.values,
            names=rater_counts.index,
            hole=0.4
        )
        fig_raters.update_layout(height=350)
        st.plotly_chart(fig_raters, use_container_width=True)
    
    st.markdown("---")
    
    # Navigation buttons
    st.subheader("🔍 Explore Detailed Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("👨‍⚕️ View Physician Analytics", use_container_width=True):
            st.session_state.page = "Physician Analytics"
            st.rerun()
    
    with col2:
        if st.button("🏢 View Department Analytics", use_container_width=True):
            st.session_state.page = "Department Analytics"
            st.rerun()

# =============================================================================
# PHYSICIAN ANALYTICS PAGE
# =============================================================================
elif page == "👨‍⚕️ Physician Analytics":
    st.markdown('<h1 class="main-header">👨‍⚕️ Physician Performance Analytics</h1>', unsafe_allow_html=True)
    
    # Filters
    st.sidebar.markdown("### 🔍 Filters")
    
    # Year filter
    years = sorted(dept_df['Year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        years,
        default=years
    )
    
    # Physician filter
    physicians = sorted(dept_df['Subject ID'].unique())
    selected_physician = st.sidebar.selectbox(
        "Select Physician (Optional)",
        ["All Physicians"] + physicians
    )
    
    # Filter data
    filtered_df = dept_df[dept_df['Year'].isin(selected_years)]
    if selected_physician != "All Physicians":
        filtered_df = filtered_df[filtered_df['Subject ID'] == selected_physician]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Evaluations",
            len(filtered_df)
        )
    
    with col2:
        avg_score = filtered_df['Response_Numeric'].mean()
        st.metric(
            "Average Score",
            f"{avg_score:.2f}/5"
        )
    
    with col3:
        always_pct = (filtered_df['Response'] == 'Always').sum() / len(filtered_df) * 100
        st.metric(
            "'Always' Rate",
            f"{always_pct:.1f}%"
        )
    
    with col4:
        unique_physicians = filtered_df['Subject ID'].nunique()
        st.metric(
            "Physicians",
            unique_physicians
        )
    
    st.markdown("---")
    
    # Average score trend by year
    st.subheader("📈 Average Score Trend Over Years")
    yearly_avg = filtered_df.groupby('Year')['Response_Numeric'].mean().reset_index()
    
    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=yearly_avg['Year'],
        y=yearly_avg['Response_Numeric'],
        mode='lines+markers',
        name='Average Score',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=10)
    ))
    fig_trend.update_layout(
        xaxis_title="Year",
        yaxis_title="Average Score",
        yaxis_range=[0, 5],
        height=400,
        hovermode='x unified'
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Two columns for additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Response Distribution")
        response_counts = filtered_df['Response'].value_counts()
        response_order = ['Always', 'Most of the time', 'Sometimes', 'Hardly ever', 'Never']
        response_counts = response_counts.reindex([r for r in response_order if r in response_counts.index])
        
        fig_responses = px.bar(
            x=response_counts.index,
            y=response_counts.values,
            labels={'x': 'Response', 'y': 'Count'},
            color=response_counts.values,
            color_continuous_scale='RdYlGn_r'
        )
        fig_responses.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_responses, use_container_width=True)
    
    with col2:
        st.subheader("👥 Evaluations by Rater Group")
        rater_group_counts = filtered_df['Raters Group'].value_counts()
        
        fig_raters = px.bar(
            x=rater_group_counts.values,
            y=rater_group_counts.index,
            orientation='h',
            labels={'x': 'Number of Evaluations', 'y': 'Rater Group'},
            color=rater_group_counts.values,
            color_continuous_scale='Blues'
        )
        fig_raters.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_raters, use_container_width=True)
    
    # Sentiment Analysis Section
    st.markdown("---")
    st.subheader("💬 Comments Sentiment Analysis")
    
    # Filter out null comments
    comments_df = filtered_df[filtered_df['Q2_Comments'].notna() & (filtered_df['Q2_Comments'] != '')]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Total Comments",
            len(comments_df)
        )
    
    with col2:
        positive_comments = len(comments_df[comments_df['Response_Numeric'] >= 4])
        st.metric(
            "Positive Context",
            positive_comments,
            help="Comments with score >= 4"
        )
    
    with col3:
        negative_comments = len(comments_df[comments_df['Response_Numeric'] <= 2])
        st.metric(
            "Needs Improvement",
            negative_comments,
            help="Comments with score <= 2"
        )
    
    # Comments by score
    if len(comments_df) > 0:
        st.subheader("📝 Comments by Performance Level")
        comments_by_score = comments_df.groupby('Response_Numeric').size().reset_index(name='Count')
        
        fig_comments = px.bar(
            comments_by_score,
            x='Response_Numeric',
            y='Count',
            labels={'Response_Numeric': 'Performance Score', 'Count': 'Number of Comments'},
            color='Count',
            color_continuous_scale='RdYlGn'
        )
        fig_comments.update_layout(height=350)
        st.plotly_chart(fig_comments, use_container_width=True)
        
        # Sample comments
        st.subheader("💡 Sample Comments")
        sample_comments = comments_df[['Subject ID', 'Raters Group', 'Response', 'Q2_Comments']].head(10)
        st.dataframe(sample_comments, use_container_width=True, hide_index=True)
    else:
        st.info("No comments available for the selected filters.")
    
    # Performance by question category
    st.markdown("---")
    st.subheader("📋 Top 10 Performance Questions")
    
    # Simplify question text for display
    filtered_df['Question_Short'] = filtered_df['Question'].str.split('_').str[-1].str[:80] + '...'
    question_avg = filtered_df.groupby('Question_Short')['Response_Numeric'].agg(['mean', 'count']).reset_index()
    question_avg = question_avg.sort_values('mean', ascending=False).head(10)
    
    fig_questions = px.bar(
        question_avg,
        y='Question_Short',
        x='mean',
        orientation='h',
        labels={'mean': 'Average Score', 'Question_Short': 'Question'},
        color='mean',
        color_continuous_scale='RdYlGn',
        hover_data={'count': True}
    )
    fig_questions.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_questions, use_container_width=True)

# =============================================================================
# DEPARTMENT ANALYTICS PAGE
# =============================================================================
elif page == "🏢 Department Analytics":
    st.markdown('<h1 class="main-header">🏢 Department Performance Analytics</h1>', unsafe_allow_html=True)
    
    # Get departments from physicians data and merge with evaluation data
    if 'Department' in physicians_df.columns:
        # Create a mapping of physician to department
        phys_dept_map = physicians_df[['Aubnetid', 'Department']].drop_duplicates()
        phys_dept_map.columns = ['Subject ID', 'Department']
        
        # Merge with evaluation data
        dept_analysis_df = dept_df.merge(phys_dept_map, on='Subject ID', how='left')
    else:
        st.warning("Department information not available in physician data.")
        dept_analysis_df = dept_df.copy()
        dept_analysis_df['Department'] = 'Unknown'
    
    # Filters
    st.sidebar.markdown("### 🔍 Filters")
    
    # Year filter
    years = sorted(dept_analysis_df['Year'].unique())
    selected_years = st.sidebar.multiselect(
        "Select Year(s)",
        years,
        default=years
    )
    
    # Department filter
    if 'Department' in dept_analysis_df.columns:
        departments = sorted(dept_analysis_df['Department'].dropna().unique())
        selected_department = st.sidebar.selectbox(
            "Select Department (Optional)",
            ["All Departments"] + departments
        )
    else:
        selected_department = "All Departments"
    
    # Filter data
    filtered_dept_df = dept_analysis_df[dept_analysis_df['Year'].isin(selected_years)]
    if selected_department != "All Departments":
        filtered_dept_df = filtered_dept_df[filtered_dept_df['Department'] == selected_department]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Evaluations",
            len(filtered_dept_df)
        )
    
    with col2:
        avg_score = filtered_dept_df['Response_Numeric'].mean()
        st.metric(
            "Average Score",
            f"{avg_score:.2f}/5"
        )
    
    with col3:
        unique_physicians = filtered_dept_df['Subject ID'].nunique()
        st.metric(
            "Physicians",
            unique_physicians
        )
    
    with col4:
        unique_depts = filtered_dept_df['Department'].nunique() if 'Department' in filtered_dept_df.columns else 1
        st.metric(
            "Departments",
            unique_depts
        )
    
    st.markdown("---")
    
    # Department comparison
    if 'Department' in filtered_dept_df.columns and selected_department == "All Departments":
        st.subheader("🏆 Department Performance Comparison")
        
        dept_stats = filtered_dept_df.groupby('Department').agg({
            'Response_Numeric': ['mean', 'count'],
            'Subject ID': 'nunique'
        }).reset_index()
        dept_stats.columns = ['Department', 'Avg_Score', 'Total_Evals', 'Num_Physicians']
        dept_stats = dept_stats.sort_values('Avg_Score', ascending=False)
        
        fig_dept_comparison = px.bar(
            dept_stats,
            x='Department',
            y='Avg_Score',
            color='Avg_Score',
            color_continuous_scale='RdYlGn',
            labels={'Avg_Score': 'Average Score', 'Department': 'Department'},
            hover_data=['Total_Evals', 'Num_Physicians']
        )
        fig_dept_comparison.update_layout(height=400, showlegend=False)
        fig_dept_comparison.update_xaxes(tickangle=45)
        st.plotly_chart(fig_dept_comparison, use_container_width=True)
    
    # Year-over-year performance
    st.subheader("📈 Performance Trend by Year")
    
    if selected_department != "All Departments" and 'Department' in filtered_dept_df.columns:
        yearly_dept_avg = filtered_dept_df.groupby('Year')['Response_Numeric'].mean().reset_index()
        
        fig_yearly = go.Figure()
        fig_yearly.add_trace(go.Scatter(
            x=yearly_dept_avg['Year'],
            y=yearly_dept_avg['Response_Numeric'],
            mode='lines+markers',
            name=selected_department,
            line=dict(width=3),
            marker=dict(size=10)
        ))
    else:
        yearly_dept_avg = filtered_dept_df.groupby(['Year', 'Department'])['Response_Numeric'].mean().reset_index()
        
        fig_yearly = px.line(
            yearly_dept_avg,
            x='Year',
            y='Response_Numeric',
            color='Department',
            markers=True,
            labels={'Response_Numeric': 'Average Score', 'Year': 'Year'}
        )
    
    fig_yearly.update_layout(height=400, yaxis_range=[0, 5])
    st.plotly_chart(fig_yearly, use_container_width=True)
    
    # Two columns for additional insights
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Evaluations by Rater Type")
        rater_type_stats = filtered_dept_df.groupby('Raters Group').agg({
            'Response_Numeric': 'mean',
            'Subject ID': 'count'
        }).reset_index()
        rater_type_stats.columns = ['Rater_Group', 'Avg_Score', 'Count']
        
        fig_rater_type = px.bar(
            rater_type_stats,
            x='Rater_Group',
            y='Avg_Score',
            color='Avg_Score',
            color_continuous_scale='Blues',
            labels={'Avg_Score': 'Average Score', 'Rater_Group': 'Rater Group'},
            hover_data=['Count']
        )
        fig_rater_type.update_layout(height=350, showlegend=False)
        fig_rater_type.update_xaxes(tickangle=45)
        st.plotly_chart(fig_rater_type, use_container_width=True)
    
    with col2:
        st.subheader("📊 Score Distribution")
        score_dist = filtered_dept_df['Response_Numeric'].value_counts().sort_index()
        
        fig_score_dist = px.bar(
            x=score_dist.index,
            y=score_dist.values,
            labels={'x': 'Score', 'y': 'Frequency'},
            color=score_dist.values,
            color_continuous_scale='RdYlGn'
        )
        fig_score_dist.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_score_dist, use_container_width=True)
    
    # Monthly evaluation trends
    st.markdown("---")
    st.subheader("📅 Monthly Evaluation Activity")
    
    monthly_data = filtered_dept_df.copy()
    monthly_data['Month'] = pd.to_datetime(monthly_data['Fillout Date (mm/dd/yy)']).dt.to_period('M')
    monthly_counts = monthly_data.groupby('Month').size().reset_index(name='Count')
    monthly_counts['Month'] = monthly_counts['Month'].astype(str)
    
    fig_monthly = px.line(
        monthly_counts,
        x='Month',
        y='Count',
        labels={'Count': 'Number of Evaluations', 'Month': 'Month'},
        markers=True
    )
    fig_monthly.update_layout(height=350)
    fig_monthly.update_xaxes(tickangle=45)
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Performance heatmap by rater group and year
    st.markdown("---")
    st.subheader("🔥 Performance Heatmap: Rater Group vs Year")
    
    heatmap_data = filtered_dept_df.groupby(['Raters Group', 'Year'])['Response_Numeric'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Raters Group', columns='Year', values='Response_Numeric')
    
    fig_heatmap = px.imshow(
        heatmap_pivot,
        labels=dict(x="Year", y="Rater Group", color="Avg Score"),
        color_continuous_scale='RdYlGn',
        aspect="auto"
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Detailed statistics table
    st.markdown("---")
    st.subheader("📋 Detailed Statistics")
    
    if 'Department' in filtered_dept_df.columns and selected_department == "All Departments":
        detailed_stats = filtered_dept_df.groupby('Department').agg({
            'Response_Numeric': ['mean', 'std', 'min', 'max'],
            'Subject ID': 'nunique',
            'Raters Group': 'count'
        }).reset_index()
        detailed_stats.columns = ['Department', 'Avg Score', 'Std Dev', 'Min Score', 'Max Score', 'Num Physicians', 'Total Evaluations']
        detailed_stats = detailed_stats.round(2)
        detailed_stats = detailed_stats.sort_values('Avg Score', ascending=False)
        
        st.dataframe(detailed_stats, use_container_width=True, hide_index=True)
    else:
        st.info("Select 'All Departments' to view comparative statistics.")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #888; padding: 20px;'>
        <p>AUBMC Physician Performance Dashboard | Data Analytics & Insights</p>
        <p>Last Updated: {}</p>
    </div>
""".format(datetime.now().strftime("%B %d, %Y")), unsafe_allow_html=True)
