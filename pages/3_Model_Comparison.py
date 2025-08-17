import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Utils.Heathcare_metrics import generate_healthcare_recommendations
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Model Comparison - Healthcare Optimization",
    page_icon="⚖️",
    layout="wide"
)

def main():
    st.title("⚖️ Model Comparison & Analysis")
    st.markdown("Compare performance across multiple models and datasets")
    
    # Check if trained models are available
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available. Please train models first.")
        st.stop()
    
    # Model selection for comparison
    st.header("1. Select Models for Comparison")
    
    # List all available trained models
    model_sessions = list(st.session_state.trained_models.keys())
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_sessions = st.multiselect(
            "Select Model Sessions to Compare",
            model_sessions,
            default=model_sessions,
            help="Choose which training sessions to include in comparison"
        )
    
    with col2:
        comparison_metric = st.selectbox(
            "Primary Comparison Metric",
            ["Test R² Score", "Test RMSE", "Test MAE", "CV Score", "Training Time"],
            help="Metric to use for primary model ranking"
        )
    
    if not selected_sessions:
        st.info("👆 Select at least one model session to compare")
        st.stop()
    
    # Aggregate comparison data
    comparison_data = []
    all_model_results = {}
    
    for session_key in selected_sessions:
        session_data = st.session_state.trained_models[session_key]
        dataset_name = session_data['dataset']
        target_var = session_data['target']
        
        for model_name, model_results in session_data['results'].items():
            entry = {
                'Session': f"{dataset_name} → {target_var}",
                'Dataset': dataset_name,
                'Target': target_var,
                'Model': model_name,
                'Test R² Score': model_results['test_r2'],
                'Test RMSE': np.sqrt(model_results['test_mse']),
                'Test MAE': model_results['test_mae'],
                'CV Score': model_results['cv_score_mean'],
                'CV Std': model_results['cv_score_std'],
                'Training Time': model_results.get('training_time', 0),
                'Overfitting': model_results['train_r2'] - model_results['test_r2']
            }
            comparison_data.append(entry)
            
            # Store full results for detailed analysis
            all_model_results[f"{session_key}_{model_name}"] = {
                'session_data': session_data,
                'model_results': model_results,
                'display_name': f"{dataset_name} - {model_name}"
            }
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df['CV Score'] = comparison_df['CV Score'].clip(lower=1e-5)
    # Main comparison dashboard
    st.header("2. Performance Comparison Dashboard")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_overall_idx = comparison_df['Test R² Score'].idxmax()
        best_model = comparison_df.loc[best_overall_idx]
        st.metric(
            "Best Overall Model",
            f"{best_model['Model']}",
            f"R² = {best_model['Test R² Score']:.3f}"
        )
    
    with col2:
        avg_r2 = comparison_df['Test R² Score'].mean()
        st.metric(
            "Average R² Score",
            f"{avg_r2:.3f}",
            f"Std: {comparison_df['Test R² Score'].std():.3f}"
        )
    
    with col3:
        lowest_rmse_idx = comparison_df['Test RMSE'].idxmin()
        lowest_rmse_model = comparison_df.loc[lowest_rmse_idx]
        st.metric(
            "Lowest RMSE",
            f"{lowest_rmse_model['Test RMSE']:.3f}",
            f"({lowest_rmse_model['Model']})"
        )
    
    with col4:
        models_count = len(comparison_df)
        datasets_count = comparison_df['Dataset'].nunique()
        st.metric(
            "Models Compared",
            f"{models_count}",
            f"{datasets_count} datasets"
        )
    
    # Performance comparison table
    st.subheader("📊 Detailed Performance Comparison")
    
    # Sort by selected metric
    metric_column_map = {
        "Test R² Score": "Test R² Score",
        "Test RMSE": "Test RMSE", 
        "Test MAE": "Test MAE",
        "CV Score": "CV Score",
        "Training Time": "Training Time"
    }
    
    sort_column = metric_column_map[comparison_metric]
    ascending = sort_column in ["Test RMSE", "Test MAE", "Training Time"]  # Lower is better for these
    
    sorted_df = comparison_df.sort_values(sort_column, ascending=ascending)
    
    # Color-code the performance table
    def color_performance(val, column):
        if column in ['Test R² Score', 'CV Score']:
            # Higher is better
            if val >= 0.8:
                return 'background-color: #d4edda'  # Green
            elif val >= 0.6:
                return 'background-color: #fff3cd'  # Yellow
            else:
                return 'background-color: #f8d7da'  # Red
        elif column in ['Test RMSE', 'Test MAE']:
            # Lower is better (relative to dataset)
            col_median = comparison_df[column].median()
            if val <= col_median * 0.8:
                return 'background-color: #d4edda'  # Green
            elif val <= col_median * 1.2:
                return 'background-color: #fff3cd'  # Yellow
            else:
                return 'background-color: #f8d7da'  # Red
        return ''
    
    # Apply styling
    styled_df = sorted_df.style.format({
        'Test R² Score': '{:.3f}',
        'Test RMSE': '{:.3f}',
        'Test MAE': '{:.3f}',
        'CV Score': '{:.3f}',
        'CV Std': '{:.3f}',
        'Training Time': '{:.2f}s',
        'Overfitting': '{:.3f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Visualization section
    st.header("3. Performance Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Charts", "🎯 Model Rankings", "📊 Cross-Dataset Analysis", "🔍 Detailed Analysis"])
    
    with tab1:
        # Performance comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig_r2 = px.bar(
                comparison_df.sort_values('Test R² Score', ascending=False),
                x='Test R² Score',
                y='Model',
                color='Dataset',
                title='R² Score Comparison Across Models',
                orientation='h',
                height=max(400, len(comparison_df) * 30)
            )
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # RMSE comparison
            fig_rmse = px.bar(
                comparison_df.sort_values('Test RMSE'),
                x='Test RMSE',
                y='Model',
                color='Dataset',
                title='RMSE Comparison Across Models',
                orientation='h',
                height=max(400, len(comparison_df) * 30)
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Ensure CV Score is positive for scatter plot sizing
        comparison_df_plot = comparison_df.copy()
        comparison_df_plot['CV Score Abs'] = np.maximum(np.abs(comparison_df_plot['CV Score']), 0.1)
        
        # Model performance scatter plot
        fig_scatter = px.scatter(
            comparison_df_plot,
            x='Test RMSE',
            y='Test R² Score',
            color='Model',
            size='CV Score Abs',
            hover_data=['Dataset', 'Target', 'CV Std'],
            title='Model Performance Scatter Plot (Size = |CV Score|)',
            labels={'Test RMSE': 'Test RMSE (Lower is Better)', 'Test R² Score': 'Test R² Score (Higher is Better)'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        # Model rankings across different metrics
        st.subheader("🏆 Model Rankings")
        
        ranking_metrics = ['Test R² Score', 'Test RMSE', 'Test MAE', 'CV Score']
        ranking_data = []
        
        for metric in ranking_metrics:
            ascending = metric in ['Test RMSE', 'Test MAE']
            ranked = comparison_df.sort_values(metric, ascending=ascending).reset_index(drop=True)
            
            for idx, row in ranked.iterrows():
                ranking_data.append({
                    'Metric': metric,
                    'Rank': idx + 1,
                    'Model': row['Model'],
                    'Dataset': row['Dataset'],
                    'Value': row[metric],
                    'Model_Dataset': f"{row['Model']} ({row['Dataset']})"
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Average rank across all metrics
        avg_ranks = ranking_df.groupby('Model_Dataset')['Rank'].mean().sort_values()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Average Ranking Across All Metrics:**")
            avg_rank_df = avg_ranks.reset_index()
            avg_rank_df.columns = ['Model (Dataset)', 'Average Rank']
            avg_rank_df['Average Rank'] = avg_rank_df['Average Rank'].round(2)
            st.dataframe(avg_rank_df, use_container_width=True)
        
        with col2:
            # Ranking heatmap
            pivot_rankings = ranking_df.pivot_table(
                index='Model_Dataset',
                columns='Metric',
                values='Rank'
            )
            
            fig_heatmap = px.imshow(
                pivot_rankings.values,
                x=pivot_rankings.columns,
                y=pivot_rankings.index,
                color_continuous_scale='RdYlGn_r',
                title='Model Ranking Heatmap (1=Best)',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=max(400, len(pivot_rankings) * 40))
            st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with tab3:
        # Cross-dataset analysis
        st.subheader("🔄 Cross-Dataset Performance Analysis")
        
        if comparison_df['Dataset'].nunique() > 1:
            # Model performance consistency across datasets
            model_consistency = comparison_df.groupby('Model').agg({
                'Test R² Score': ['mean', 'std'],
                'Test RMSE': ['mean', 'std'],
                'CV Score': ['mean', 'std']
            }).round(3)
            
            model_consistency.columns = ['_'.join(col).strip() for col in model_consistency.columns]
            
            st.write("**Model Consistency Across Datasets:**")
            st.dataframe(model_consistency, use_container_width=True)
            
            # Violin plot for performance distribution
            fig_violin = px.violin(
                comparison_df,
                x='Model',
                y='Test R² Score',
                color='Model',
                title='R² Score Distribution Across Datasets',
                box=True
            )
            st.plotly_chart(fig_violin, use_container_width=True)
            
        else:
            st.info("Cross-dataset analysis requires multiple datasets")
    
    with tab4:
        # Detailed model analysis
        st.subheader("🔍 Detailed Model Analysis")
        
        # Select specific models for detailed comparison
        available_model_keys = list(all_model_results.keys())
        selected_models = st.multiselect(
            "Select Models for Detailed Analysis",
            available_model_keys,
            default=available_model_keys[:min(3, len(available_model_keys))],
            format_func=lambda x: all_model_results[x]['display_name']
        )
        
        if selected_models:
            # Model-specific metrics and analysis
            for model_key in selected_models:
                model_info = all_model_results[model_key]
                model_results = model_info['model_results']
                display_name = model_info['display_name']
                
                with st.expander(f"📊 {display_name} - Detailed Analysis"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("R² Score", f"{model_results['test_r2']:.3f}")
                        st.metric("RMSE", f"{np.sqrt(model_results['test_mse']):.3f}")
                    
                    with col2:
                        st.metric("MAE", f"{model_results['test_mae']:.3f}")
                        st.metric("CV Score", f"{model_results['cv_score_mean']:.3f}")
                    
                    with col3:
                        overfitting = model_results['train_r2'] - model_results['test_r2']
                        st.metric("Overfitting", f"{overfitting:.3f}")
                        st.metric("CV Std", f"{model_results['cv_score_std']:.3f}")
                    
                    # Feature importance if available
                    if 'feature_importance' in model_results and model_results['feature_importance']:
                        st.write("**Top 10 Important Features:**")
                        importance = model_results['feature_importance']
                        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
                        
                        if top_features:  # Only create chart if we have features
                            importance_df = pd.DataFrame([
                                {'Feature': k, 'Importance': v} for k, v in top_features.items()
                            ])
                            
                            fig_importance = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title=f'Feature Importance - {display_name}'
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                        else:
                            st.info("No feature importance data available for this model.")
    
    # Healthcare recommendations
    st.header("4. 🏥 Healthcare Implementation Recommendations")
    
    # Generate recommendations based on model performance
    best_model_data = comparison_df.loc[comparison_df['Test R² Score'].idxmax()]
    
    recommendations = generate_healthcare_recommendations(
        best_model_data,
        comparison_df
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Primary Recommendations")
        for rec in recommendations['primary']:
            st.success(f"✅ {rec}")
    
    with col2:
        st.subheader("⚠️ Considerations")
        for consideration in recommendations['considerations']:
            st.warning(f"⚠️ {consideration}")
    
    # Export comparison results
    st.header("5. 📤 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export comparison table
        csv_data = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Comparison Table",
            data=csv_data,
            file_name="model_comparison_results.csv",
            mime="text/csv"
        )
    
    with col2:
        # Export detailed results
        if st.button("📋 Generate Detailed Report"):
            detailed_report = generate_detailed_comparison_report(comparison_df, all_model_results)
            st.text_area(
                "Detailed Comparison Report",
                detailed_report,
                height=300
            )
    
    with col3:
        # Export recommendations
        recommendations_text = "\n".join([
            "HEALTHCARE ML MODEL RECOMMENDATIONS\n",
            "=" * 40,
            "\nPRIMARY RECOMMENDATIONS:",
            *[f"• {rec}" for rec in recommendations['primary']],
            "\nCONSIDERATIONS:",
            *[f"• {con}" for con in recommendations['considerations']]
        ])
        
        st.download_button(
            label="📋 Download Recommendations",
            data=recommendations_text,
            file_name="healthcare_recommendations.txt",
            mime="text/plain"
        )

def generate_detailed_comparison_report(comparison_df, all_model_results):
    """Generate a detailed text report of model comparison"""
    
    report_lines = [
        "HEALTHCARE WORKFLOW OPTIMIZATION - MODEL COMPARISON REPORT",
        "=" * 60,
        f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Models Compared: {len(comparison_df)}",
        f"Datasets Analyzed: {comparison_df['Dataset'].nunique()}",
        "",
        "PERFORMANCE SUMMARY:",
        "-" * 20
    ]
    
    # Best performing models
    best_r2 = comparison_df.loc[comparison_df['Test R² Score'].idxmax()]
    best_rmse = comparison_df.loc[comparison_df['Test RMSE'].idxmin()]
    
    report_lines.extend([
        f"Best R² Score: {best_r2['Model']} on {best_r2['Dataset']} (R² = {best_r2['Test R² Score']:.3f})",
        f"Lowest RMSE: {best_rmse['Model']} on {best_rmse['Dataset']} (RMSE = {best_rmse['Test RMSE']:.3f})",
        f"Average R² Score: {comparison_df['Test R² Score'].mean():.3f}",
        f"Average RMSE: {comparison_df['Test RMSE'].mean():.3f}",
        ""
    ])
    
    # Model-wise performance
    report_lines.extend([
        "DETAILED MODEL PERFORMANCE:",
        "-" * 30
    ])
    
    for _, row in comparison_df.iterrows():
        report_lines.extend([
            f"Model: {row['Model']} | Dataset: {row['Dataset']} | Target: {row['Target']}",
            f"  R² Score: {row['Test R² Score']:.3f} | RMSE: {row['Test RMSE']:.3f} | MAE: {row['Test MAE']:.3f}",
            f"  CV Score: {row['CV Score']:.3f} ± {row['CV Std']:.3f}",
            ""
        ])
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    main()
