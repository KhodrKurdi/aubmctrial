# AUBMC Physician Performance Dashboard

A comprehensive Streamlit dashboard for analyzing physician performance metrics and evaluation data at the American University of Beirut Medical Center (AUBMC).

## 📋 Features

### 1. **Overview Page**
- Key metrics summary (total physicians, evaluations, average scores, departments)
- Evaluations distribution by year
- Evaluations breakdown by rater group
- Quick navigation to detailed analytics

### 2. **Physician Analytics**
- **Interactive Filters:**
  - Filter by year(s)
  - Filter by individual physician
  
- **Key Metrics Cards:**
  - Total evaluations
  - Average performance score
  - "Always" response rate
  - Number of physicians analyzed

- **Visualizations:**
  - Line chart showing average score trends over years
  - Response distribution (Always, Most of the time, Sometimes, etc.)
  - Evaluations breakdown by rater group
  - Top 10 performance questions analysis
  
- **Sentiment Analysis:**
  - Total comments count
  - Positive context comments (score ≥ 4)
  - Comments needing improvement (score ≤ 2)
  - Comments distribution by performance score
  - Sample comments viewer

### 3. **Department Analytics**
- **Interactive Filters:**
  - Filter by year(s)
  - Filter by department
  
- **Key Metrics:**
  - Total evaluations
  - Average score by department
  - Number of physicians
  - Number of departments

- **Visualizations:**
  - Department performance comparison
  - Year-over-year performance trends
  - Evaluations by rater type
  - Score distribution
  - Monthly evaluation activity trends
  - Performance heatmap (Rater Group vs Year)
  - Detailed statistics table

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the files:**
   - `dashboard.py` - Main dashboard application
   - `requirements.txt` - Python dependencies
   - Place your data files in the same directory or update file paths

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data:**
   - Place these files in `/mnt/user-data/uploads/` or update paths in the code:
     - `Physicians_Indicators_Anonymized.csv`
     - `All_Departments_2023.csv`
     - `All_Departments_2024.csv`
     - `All_Departments_2025.csv`

4. **Run the dashboard:**
   ```bash
   streamlit run dashboard.py
   ```

5. **Access the dashboard:**
   - Open your browser and navigate to `http://localhost:8501`

## 📊 Data Requirements

### Physicians Indicators File
Expected columns:
- `Aubnetid` - Physician unique identifier
- `FiscalCycle` - Fiscal cycle (e.g., "Cycle 2022-2023")
- `Physician Name` - Physician name
- `Department` - Department name
- `ClinicVisits` - Number of clinic visits
- `ClinicWaitingTime` - Average waiting time
- `PatientComplaints` - Number of complaints
- `ContractEffectiveDate` - Contract start date

### Department Evaluation Files
Expected columns:
- `Subject ID` - Physician being evaluated
- `Raters Group` - Type of rater (Medical Staff, Hospital Staff, etc.)
- `Completed by` - Evaluator ID
- `Rater Name` - Evaluator name
- `Source` - Source of evaluation
- `Fillout Date (mm/dd/yy)` - Date of evaluation
- `Q2_Comments` - Textual comments
- `Question` - Evaluation question
- `Response` - Text response (Always, Most of the time, etc.)
- `Response_Numeric` - Numeric score (1-5)
- `Year` - Year of evaluation

## 🎯 Usage Guide

### Navigation
Use the sidebar to switch between three main views:
1. **📊 Overview** - High-level summary and quick navigation
2. **👨‍⚕️ Physician Analytics** - Detailed physician performance analysis
3. **🏢 Department Analytics** - Department-level insights and trends

### Filtering Data
All filters are available in the sidebar:
- **Year Filter:** Multi-select to analyze specific years or year ranges
- **Physician Filter:** Select individual physician or view all physicians
- **Department Filter:** Select specific department or view all departments

### Interactive Charts
- Hover over charts to see detailed values
- Click on legend items to show/hide data series
- Use chart controls (zoom, pan, download) in the top-right corner of each chart

## 📈 Key Insights Available

1. **Performance Trends:** Track how physician scores change over time
2. **Rater Perspectives:** Compare evaluations from different rater groups
3. **Department Comparison:** Identify top-performing departments
4. **Question Analysis:** Understand which competencies score highest/lowest
5. **Comment Sentiment:** Analyze qualitative feedback patterns
6. **Activity Patterns:** Monitor evaluation submission trends

## 🔧 Customization

### Updating File Paths
If your data files are in different locations, update the paths in the `load_data()` function:

```python
physicians = pd.read_csv('YOUR_PATH/Physicians_Indicators_Anonymized.csv')
dept_2023 = pd.read_csv('YOUR_PATH/All_Departments_2023.csv', low_memory=False)
dept_2024 = pd.read_csv('YOUR_PATH/All_Departments_2024.csv', low_memory=False)
dept_2025 = pd.read_csv('YOUR_PATH/All_Departments_2025.csv', low_memory=False)
```

### Changing Color Schemes
Update the color scales in plotly chart definitions:
- `color_continuous_scale='Blues'` - For blue gradients
- `color_continuous_scale='RdYlGn'` - For red-yellow-green gradients
- `color_continuous_scale='RdYlGn_r'` - For reversed gradients

### Adding Custom Metrics
Add new metrics in the metrics columns sections:
```python
with col5:
    st.metric("Your Metric", value, delta=change)
```

## 🐛 Troubleshooting

### Common Issues

1. **"File not found" error:**
   - Check that all CSV files are in the correct directory
   - Verify file names match exactly (case-sensitive)
   - Update file paths in the code if needed

2. **"Module not found" error:**
   - Run `pip install -r requirements.txt` again
   - Ensure you're using the correct Python environment

3. **Data loading issues:**
   - Check CSV file encoding (should be UTF-8)
   - Verify column names match expected format
   - Check for empty or corrupted files

4. **Dashboard not updating:**
   - Clear Streamlit cache: Click "Always rerun" or press 'C' in the dashboard
   - Restart the Streamlit server

## 📝 Notes

- The dashboard uses caching for improved performance
- Large datasets may take a few seconds to load initially
- All visualizations are interactive and exportable
- The dashboard is optimized for desktop/laptop viewing

## 🔒 Data Privacy

- All physician identifiers are anonymized
- No personal health information (PHI) is displayed
- Comments are shown in aggregate or sample format only

## 📧 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review Streamlit documentation: https://docs.streamlit.io
3. Review Plotly documentation: https://plotly.com/python/

---

**Version:** 1.0  
**Last Updated:** February 2026  
**Platform:** Streamlit + Plotly  
