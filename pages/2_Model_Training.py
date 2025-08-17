import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from Utils.models_utils import HealthcareMLModels
from Utils.Heathcare_metrics import calculate_healthcare_metrics
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Model Training - Healthcare Optimization",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("🤖 Model Training & Evaluation")
    st.markdown("Train and evaluate multiple ML models on your healthcare datasets")
    
    # Check if datasets are available
    if not st.session_state.datasets:
        st.warning("⚠️ No datasets available. Please upload data first.")
        st.stop()
    
    # Dataset selection
    st.header("1. Select Dataset & Target Variable")
    
    # Filter for processed datasets (preferred) or original datasets
    available_datasets = list(st.session_state.datasets.keys())
    processed_datasets = [name for name in available_datasets if name.endswith('_processed')]
    
    if processed_datasets:
        st.info("✅ Processed datasets found. These are recommended for model training.")
        dataset_options = processed_datasets + [name for name in available_datasets if not name.endswith('_processed')]
    else:
        st.warning("⚠️ No processed datasets found. Consider preprocessing your data first.")
        dataset_options = available_datasets
    
    selected_dataset = st.selectbox("Select dataset for training", dataset_options)
    
    if selected_dataset:
        df = st.session_state.datasets[selected_dataset]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Target variable selection
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Auto-suggest target variables
            suggested_targets = []
            for col in numeric_columns:
                if any(keyword in col.lower() for keyword in 
                       ['wait', 'time', 'outcome', 'satisfaction', 'cost', 'length', 'duration', 'score']):
                    suggested_targets.append(col)
            
            target_variable = st.selectbox(
                "Select Target Variable",
                numeric_columns,
                index=0 if suggested_targets and suggested_targets[0] in numeric_columns else 0,
                help="Choose the variable you want to predict"
            )
        
        with col2:
            # Feature selection
            available_features = [col for col in df.columns if col != target_variable]
            
            feature_selection_mode = st.radio(
                "Feature Selection",
                ["Use All Features", "Select Specific Features"]
            )
            
            if feature_selection_mode == "Select Specific Features":
                selected_features = st.multiselect(
                    "Select Features",
                    available_features,
                    default=available_features[:min(10, len(available_features))],
                    help="Choose the features to use for prediction"
                )
            else:
                selected_features = available_features
        
        # Model configuration
        st.header("2. Model Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Proportion of data to use for testing"
            )
            
            random_state = st.number_input(
                "Random State",
                value=42,
                help="Set for reproducible results"
            )
        
        with col2:
            cv_folds = st.slider(
                "Cross-Validation Folds",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of folds for cross-validation"
            )
            
            enable_tuning = st.checkbox(
                "Enable Hyperparameter Tuning",
                value=False,
                help="Optimize model parameters (takes longer)"
            )
        
        with col3:
            models_to_train = st.multiselect(
                "Select Models to Train",
                ["Random Forest", "XGBoost", "Linear Regression", "Support Vector Regression"],
                default=["Random Forest", "XGBoost", "Linear Regression", "Support Vector Regression"],
                help="Choose which models to train and compare"
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                scale_features = st.checkbox(
                    "Scale Features",
                    value=not selected_dataset.endswith('_processed'),
                    help="Standardize numerical features"
                )
                
                handle_outliers = st.checkbox(
                    "Remove Outliers",
                    value=False,
                    help="Remove statistical outliers from training data"
                )
            
            with col2:
                feature_importance = st.checkbox(
                    "Calculate Feature Importance",
                    value=True,
                    help="Analyze which features are most important"
                )
                
                residual_analysis = st.checkbox(
                    "Residual Analysis",
                    value=True,
                    help="Analyze model prediction errors"
                )
        
        # Training section
        st.header("3. Model Training & Results")
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if not selected_features:
                st.error("❌ Please select at least one feature for training")
                st.stop()
            
            if not models_to_train:
                st.error("❌ Please select at least one model to train")
                st.stop()
            
            with st.spinner("Training models... This may take a few minutes."):
                try:
                    # Prepare data
                    X = df[selected_features]
                    y = df[target_variable]
                    
                    # Remove rows with missing target values
                    mask = ~y.isnull()
                    X = X[mask]
                    y = y[mask]
                    
                    if len(X) == 0:
                        st.error("❌ No valid data remaining after removing missing values")
                        st.stop()
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state
                    )
                    
                    # Initialize ML models
                    ml_models = HealthcareMLModels()
                    
                    # Configure options
                    ml_models.set_options(
                        scale_features=scale_features,
                        handle_outliers=handle_outliers,
                        cv_folds=cv_folds,
                        random_state=random_state,
                        enable_tuning=enable_tuning
                    )
                    
                    # Train models
                    results = ml_models.train_multiple_models(
                        X_train, X_test, y_train, y_test,
                        models_to_train
                    )
                    
                    # Store results in session state
                    session_key = f"{selected_dataset}_{target_variable}"
                    st.session_state.trained_models[session_key] = {
                        'results': results,
                        'dataset': selected_dataset,
                        'target': target_variable,
                        'features': selected_features,
                        'timestamp': datetime.now()
                    }
                    
                    st.success("✅ Model training completed successfully!")
                    
                    # Display results
                    display_training_results(results, target_variable, feature_importance, residual_analysis)
                    
                except Exception as e:
                    st.error(f"❌ Error during model training: {str(e)}")
                    st.exception(e)
    
    # Show previously trained models
    if st.session_state.trained_models:
        st.header("4. Previously Trained Models")
        
        for session_key, model_data in st.session_state.trained_models.items():
            with st.expander(f"📊 {model_data['dataset']} → {model_data['target']} (Trained: {model_data['timestamp'].strftime('%Y-%m-%d %H:%M')})"):
                results = model_data['results']
                
                # Quick metrics summary
                col1, col2, col3, col4 = st.columns(4)
                
                best_model = min(results.keys(), key=lambda x: results[x]['test_mse'])
                
                with col1:
                    st.metric("Best Model", best_model)
                with col2:
                    st.metric("Best R² Score", f"{results[best_model]['test_r2']:.3f}")
                with col3:
                    st.metric("Best RMSE", f"{np.sqrt(results[best_model]['test_mse']):.3f}")
                with col4:
                    st.metric("Features Used", len(model_data['features']))

def display_training_results(results, target_variable, show_feature_importance, show_residual_analysis):
    """Display comprehensive training results"""
    
    # Performance metrics comparison
    st.subheader("📈 Model Performance Comparison")
    
    # Create comparison dataframe
    comparison_data = []
    for model_name, model_results in results.items():
        comparison_data.append({
            'Model': model_name,
            'Train R²': model_results['train_r2'],
            'Test R²': model_results['test_r2'],
            'Train RMSE': np.sqrt(model_results['train_mse']),
            'Test RMSE': np.sqrt(model_results['test_mse']),
            'Test MAE': model_results['test_mae'],
            'CV Score': model_results['cv_score_mean'],
            'CV Std': model_results['cv_score_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['CV Score'] = comparison_df['CV Score'].clip(lower=1e-5)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visual comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # R² Score comparison
        fig_r2 = px.bar(
            comparison_df,
            x='Model',
            y='Test R²',
            title='Model R² Score Comparison',
            color='Test R²',
            color_continuous_scale='viridis'
        )
        fig_r2.update_layout(height=400)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        # RMSE comparison
        fig_rmse = px.bar(
            comparison_df,
            x='Model',
            y='Test RMSE',
            title='Model RMSE Comparison',
            color='Test RMSE',
            color_continuous_scale='reds_r'
        )
        fig_rmse.update_layout(height=400)
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Cross-validation scores
    if all('cv_scores' in results[model] for model in results):
        st.subheader("🎯 Cross-Validation Analysis")
        
        cv_data = []
        for model_name, model_results in results.items():
            for i, score in enumerate(model_results['cv_scores']):
                cv_data.append({
                    'Model': model_name,
                    'Fold': i + 1,
                    'CV Score': score
                })
        
        cv_df = pd.DataFrame(cv_data)
        
        fig_cv = px.box(
            cv_df,
            x='Model',
            y='CV Score',
            title='Cross-Validation Score Distribution',
            color='Model'
        )
        st.plotly_chart(fig_cv, use_container_width=True)
    
    # Feature importance analysis
    if show_feature_importance:
        st.subheader("🔍 Feature Importance Analysis")
        
        # Show feature importance for models that support it
        importance_models = ['Random Forest', 'XGBoost']
        available_importance_models = [m for m in importance_models if m in results]
        
        if available_importance_models:
            for model_name in available_importance_models:
                if 'feature_importance' in results[model_name]:
                    importance_data = results[model_name]['feature_importance']
                    
                    # Create feature importance plot
                    fig_importance = px.bar(
                        x=list(importance_data.values()),
                        y=list(importance_data.keys()),
                        orientation='h',
                        title=f'{model_name} - Feature Importance',
                        labels={'x': 'Importance', 'y': 'Features'}
                    )
                    fig_importance.update_layout(height=max(400, len(importance_data) * 25))
                    st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Feature importance not available for selected models")
    
    # Healthcare-specific metrics
    st.subheader("🏥 Healthcare-Specific Analysis")
    
    best_model_name = min(results.keys(), key=lambda x: results[x]['test_mse'])
    best_results = results[best_model_name]
    
    if 'predictions' in best_results:
        healthcare_metrics = calculate_healthcare_metrics(
            best_results['y_true'],
            best_results['predictions'],
            target_variable
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Clinical Accuracy",
                f"{healthcare_metrics.get('clinical_accuracy', 0):.1%}",
                help="Percentage of predictions within clinically acceptable range"
            )
        
        with col2:
            st.metric(
                "Critical Case Detection",
                f"{healthcare_metrics.get('critical_detection_rate', 0):.1%}",
                help="Ability to correctly identify high-priority cases"
            )
        
        with col3:
            st.metric(
                "Resource Planning Accuracy",
                f"{healthcare_metrics.get('resource_accuracy', 0):.1%}",
                help="Accuracy for resource allocation decisions"
            )
    
    # Residual analysis
    if show_residual_analysis and best_model_name in results:
        st.subheader("📊 Residual Analysis")
        
        if 'predictions' in results[best_model_name]:
            y_true = results[best_model_name]['y_true']
            y_pred = results[best_model_name]['predictions']
            residuals = y_true - y_pred
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Residuals vs Predicted
                fig_residuals = px.scatter(
                    x=y_pred,
                    y=residuals,
                    title='Residuals vs Predicted Values',
                    labels={'x': f'Predicted {target_variable}', 'y': 'Residuals'},
                    trendline='ols'
                )
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residuals, use_container_width=True)
            
            with col2:
                # Actual vs Predicted
                fig_actual_pred = px.scatter(
                    x=y_true,
                    y=y_pred,
                    title='Actual vs Predicted Values',
                    labels={'x': f'Actual {target_variable}', 'y': f'Predicted {target_variable}'},
                    trendline='ols'
                )
                # Add perfect prediction line
                min_val = min(min(y_true), min(y_pred))
                max_val = max(max(y_true), max(y_pred))
                fig_actual_pred.add_shape(
                    type="line",
                    x0=min_val, y0=min_val,
                    x1=max_val, y1=max_val,
                    line=dict(color="red", dash="dash")
                )
                st.plotly_chart(fig_actual_pred, use_container_width=True)

if __name__ == "__main__":
    main()
