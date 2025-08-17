import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os

# Configure page
st.set_page_config(
    page_title="Healthcare Workflow Optimization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'preprocessing_pipelines' not in st.session_state:
    st.session_state.preprocessing_pipelines = {}

def main():
    st.title("🏥 Healthcare Workflow Optimization Platform")
    st.markdown("### Multi-Dataset ML Analysis for Healthcare Operations")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("📂 Navigation")

        st.page_link("app.py", label="🏠 Home", icon="🏠")
        st.page_link("pages/1_Data_Upload.py", label="📤 Upload Data")
        st.page_link("pages/2_Model_Training.py", label="🧠 Train Model")
        st.page_link("pages/3_Model_Comparison.py", label="📊 Compare Models")
        st.page_link("pages/Insights_Dashboard.py", label="📈 Insights Dashboard")

        st.markdown("---")
        st.header("🧾 Current Session")
        st.write(f"📁 Datasets loaded: `{len(st.session_state.datasets)}`")
        st.write(f"🤖 Models trained: `{len(st.session_state.trained_models)}`")
    
    # Main content
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Available Datasets",
            value=len(st.session_state.datasets),
            help="Number of datasets currently loaded in the session"
        )
    
    with col2:
        st.metric(
            label="Trained Models",
            value=len(st.session_state.trained_models),
            help="Number of ML models trained in this session"
        )
    
    with col3:
        total_records = sum(len(df) for df in st.session_state.datasets.values())
        st.metric(
            label="Total Records",
            value=total_records,
            help="Total number of records across all datasets"
        )
    
    # Application overview
    st.markdown("---")
    st.header("Application Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Features")
        st.markdown("""
        - **Multi-Dataset Support**: Upload and analyze multiple healthcare datasets
        - **Advanced ML Models**: Random Forest, XGBoost, Linear Regression, SVR
        - **Comprehensive Preprocessing**: Automated feature engineering and data cleaning
        - **Model Comparison**: Side-by-side performance analysis
        - **Healthcare Metrics**: Domain-specific KPIs and insights
        - **Interactive Visualizations**: Dynamic charts and dashboards
        """)
    
    with col2:
        st.subheader("📊 Supported Analysis Types")
        st.markdown("""
        - **Wait Time Prediction**: ER and appointment scheduling optimization
        - **Resource Utilization**: Bed allocation and staff scheduling
        - **Patient Flow Analysis**: Workflow bottleneck identification
        - **Outcome Prediction**: Patient satisfaction and clinical outcomes
        - **Cost Optimization**: Resource allocation and efficiency metrics
        - **Cross-Dataset Correlation**: Multi-source data insights
        """)
    
    # Getting started guide
    st.markdown("---")
    st.header("🚀 Getting Started")
    
    if len(st.session_state.datasets) == 0:
        st.info("👆 Start by uploading your healthcare datasets using the **Upload Data** page in the sidebar.")
        
        st.subheader("Sample Dataset Formats Supported")
        
        # Show example data structure
        example_data = {
            'Visit ID': ['V001', 'V002', 'V003'],
            'Patient ID': ['P001', 'P002', 'P003'],
            'Hospital ID': ['H001', 'H001', 'H002'],
            'Visit Date': ['2024-01-01', '2024-01-01', '2024-01-02'],
            'Urgency Level': ['High', 'Medium', 'Low'],
            'Total Wait Time (min)': [45, 120, 30],
            'Nurse-to-Patient Ratio': [0.2, 0.3, 0.25],
            'Specialist Availability': [3, 2, 4],
            'Facility Size (Beds)': [200, 200, 150]
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **Supported file formats**: CSV, Excel (.xlsx, .xls)

        **Key columns typically expected**:
        - Date/time columns for temporal analysis
        - Categorical variables (urgency levels, regions, etc.)
        - Numerical metrics (wait times, ratios, capacities)
        - Outcome variables (satisfaction, clinical outcomes)
        """)
    else:
        st.success(f"✅ You have {len(st.session_state.datasets)} dataset(s) loaded. Continue with model training!")
        
        # Show loaded datasets summary
        st.subheader("Loaded Datasets Summary")
        for name, df in st.session_state.datasets.items():
            with st.expander(f"📋 {name} ({len(df)} records, {len(df.columns)} columns)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Column Names:**")
                    st.write(list(df.columns))
                with col2:
                    st.write("**Data Types:**")
                    st.write(df.dtypes.to_dict())
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 20px;'>"
        "Healthcare Workflow Optimization Platform | Built with Streamlit & ML"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
