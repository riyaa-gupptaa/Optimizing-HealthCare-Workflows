import streamlit as st
import pandas as pd
import numpy as np
from Utils.data_preprocessing import HealthcareDataPreprocessor
from Utils.visulaization import create_data_overview_plots
import io

st.set_page_config(
    page_title="Data Upload - Healthcare Optimization",
    page_icon="📁",
    layout="wide"
)

def main():
    st.title("📁 Data Upload & Preprocessing")
    st.markdown("Upload your healthcare datasets and perform automated preprocessing")
    
    # File upload section
    st.header("1. Upload Datasets")
    
    uploaded_files = st.file_uploader(
        "Choose CSV or Excel files",
        type=['csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
        help="Upload one or more healthcare datasets for analysis"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            dataset_name = uploaded_file.name.split('.')[0]
            
            try:
                # Read file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Successfully loaded {dataset_name}")
                
                # Show basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows", len(df))
                with col2:
                    st.metric("Columns", len(df.columns))
                with col3:
                    st.metric("Missing Values", df.isnull().sum().sum())
                for col in df.columns:
                    if df[col].dtype == 'O':
                        df[col] = df[col].astype(str)

                # Store in session state
                st.session_state.datasets[dataset_name] = df
                
            except Exception as e:
                st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
    
    # Dataset selection and preprocessing
    if st.session_state.datasets:
        st.header("2. Dataset Overview & Preprocessing")
        
        # Select dataset to work with
        selected_dataset = st.selectbox(
            "Select dataset for preprocessing",
            list(st.session_state.datasets.keys())
        )
        
        if selected_dataset:
            df = st.session_state.datasets[selected_dataset]
            
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔧 Preprocessing", "📈 Visualizations", "💾 Export"])
            
            with tab1:
                st.subheader(f"Dataset Overview: {selected_dataset}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Statistics:**")
                    st.dataframe(df.describe(), use_container_width=True)
                
                with col2:
                    st.write("**Data Types & Missing Values:**")
                    info_df = pd.DataFrame({
                        'Column': df.columns,
                        'Data Type': df.dtypes.astype(str),  # Convert to string to avoid PyArrow issues
                        'Missing Values': df.isnull().sum(),
                        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
                    })
                    st.dataframe(info_df, use_container_width=True)
                
                # Show sample data
                st.write("**Sample Data:**")
                st.dataframe(df.head(), use_container_width=True)
            
            with tab2:
                st.subheader("Automated Preprocessing")
                
                # Preprocessing options
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Preprocessing Options:**")
                    handle_missing = st.selectbox(
                        "Handle Missing Values",
                        ["mean/mode", "median/mode", "drop", "forward_fill"]
                    )
                    
                    normalize_numerical = st.checkbox("Normalize Numerical Features", value=True)
                    encode_categorical = st.checkbox("Encode Categorical Features", value=True)
                    engineer_features = st.checkbox("Feature Engineering", value=True)
                
                with col2:
                    st.write("**Target Variable Selection:**")
                    
                    # Auto-detect potential target variables
                    potential_targets = []
                    for col in df.columns:
                        if any(keyword in col.lower() for keyword in 
                               ['wait', 'time', 'outcome', 'satisfaction', 'cost', 'length', 'duration']):
                            potential_targets.append(col)
                    
                    target_column = st.selectbox(
                        "Select Target Variable",
                        ['None'] + list(df.columns),
                        index=1 if potential_targets and potential_targets[0] in df.columns else 0
                    )
                
                # Preprocessing button
                if st.button("🔧 Run Preprocessing", type="primary"):
                    with st.spinner("Processing data..."):
                        try:
                            preprocessor = HealthcareDataPreprocessor()
                            
                            # Configure preprocessor
                            preprocessor.set_options(
                                missing_strategy=handle_missing,
                                normalize=normalize_numerical,
                                encode_categorical=encode_categorical,
                                feature_engineering=engineer_features
                            )
                            
                            # Preprocess data
                            processed_df, pipeline = preprocessor.fit_transform(
                                df, target_column if target_column != 'None' else None
                            )
                            
                            # Store processed data and pipeline
                            st.session_state.datasets[f"{selected_dataset}_processed"] = processed_df
                            st.session_state.preprocessing_pipelines[selected_dataset] = pipeline
                            
                            st.success("✅ Preprocessing completed successfully!")
                            
                            # Show processing summary
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Original Features", len(df.columns))
                                st.metric("Processed Features", len(processed_df.columns))
                            with col2:
                                st.metric("Original Rows", len(df))
                                st.metric("Processed Rows", len(processed_df))
                            
                            # Show sample of processed data
                            st.write("**Processed Data Sample:**")
                            st.dataframe(processed_df.head(), use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"❌ Error during preprocessing: {str(e)}")
            
            with tab3:
                st.subheader("Data Visualizations")
                
                if len(df.select_dtypes(include=[np.number]).columns) > 0:
                    # Create visualizations
                    try:
                        figs = create_data_overview_plots(df)
                        
                        for title, fig in figs.items():
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate all visualizations: {str(e)}")
                        
                        # Fallback to basic plots
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**Correlation Matrix:**")
                                correlation_matrix = df[numeric_cols].corr()
                                st.dataframe(correlation_matrix)
                            
                            with col2:
                                st.write("**Distribution of Numerical Features:**")
                                selected_col = st.selectbox("Select column to plot", numeric_cols)
                                if selected_col:
                                    st.bar_chart(df[selected_col].value_counts().head(20))
                else:
                    st.info("No numerical columns found for visualization")
            
            with tab4:
                st.subheader("Export Processed Data")
                
                processed_datasets = [name for name in st.session_state.datasets.keys() 
                                    if name.endswith('_processed')]
                
                if processed_datasets:
                    export_dataset = st.selectbox(
                        "Select processed dataset to export",
                        processed_datasets
                    )
                    
                    if export_dataset:
                        export_df = st.session_state.datasets[export_dataset]
                        
                        # Convert to CSV for download
                        csv_buffer = io.StringIO()
                        export_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download Processed Data (CSV)",
                            data=csv_data,
                            file_name=f"{export_dataset}.csv",
                            mime="text/csv"
                        )
                        
                        st.info(f"Dataset ready for download: {len(export_df)} rows, {len(export_df.columns)} columns")
                else:
                    st.info("No processed datasets available. Run preprocessing first.")
    
    else:
        st.info("👆 Upload datasets to get started")

if __name__ == "__main__":
    main()
