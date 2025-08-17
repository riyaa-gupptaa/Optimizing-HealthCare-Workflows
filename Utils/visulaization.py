import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def create_data_overview_plots(df):
    """Create comprehensive data overview visualizations"""
    plots = {}
    
    # Get numerical and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # 1. Missing values heatmap
    if df.isnull().sum().sum() > 0:
        missing_data = df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) > 0:
            fig_missing = px.bar(
                x=missing_data.values,
                y=missing_data.index,
                orientation='h',
                title='Missing Values by Column',
                labels={'x': 'Number of Missing Values', 'y': 'Columns'}
            )
            plots['Missing Values Analysis'] = fig_missing
    
    # 2. Correlation matrix for numerical columns
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            color_continuous_scale='RdBu',
            aspect='auto',
            title='Feature Correlation Matrix'
        )
        fig_corr.update_layout(height=min(800, max(400, len(numeric_cols) * 30)))
        plots['Correlation Matrix'] = fig_corr
    
    # 3. Distribution plots for key numerical columns
    if len(numeric_cols) > 0:
        # Select up to 4 most important numerical columns
        important_numeric = numeric_cols[:4]
        
        fig_dist = make_subplots(
            rows=2, cols=2,
            subplot_titles=important_numeric,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, col in enumerate(important_numeric):
            row = (i // 2) + 1
            col_pos = (i % 2) + 1
            
            # Create histogram
            values = df[col].dropna()
            fig_dist.add_trace(
                go.Histogram(x=values, name=col, showlegend=False),
                row=row, col=col_pos
            )
        
        fig_dist.update_layout(
            title='Distribution of Key Numerical Features',
            height=600
        )
        plots['Numerical Distributions'] = fig_dist
    
    # 4. Categorical variable analysis
    if len(categorical_cols) > 0:
        # Select up to 4 categorical columns
        important_categorical = categorical_cols[:4]
        
        fig_cat = make_subplots(
            rows=2, cols=2,
            subplot_titles=important_categorical,
            specs=[[{"type": "xy"}, {"type": "xy"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        for i, col in enumerate(important_categorical):
            if col in df.columns:
                row = (i // 2) + 1
                col_pos = (i % 2) + 1
                
                value_counts = df[col].value_counts().head(10)  # Top 10 categories
                
                fig_cat.add_trace(
                    go.Bar(x=value_counts.index, y=value_counts.values, 
                          name=col, showlegend=False),
                    row=row, col=col_pos
                )
        
        fig_cat.update_layout(
            title='Categorical Feature Distribution',
            height=600
        )
        plots['Categorical Distributions'] = fig_cat
    
    # 5. Statistical summary visualization
    if len(numeric_cols) > 0:
        stats_data = df[numeric_cols].describe().T
        
        fig_stats = go.Figure()
        
        # Add box plot for each numerical column (normalized)
        for col in numeric_cols[:8]:  # Limit to 8 columns for readability
            values = df[col].dropna()
            # Normalize values for comparison
            normalized_values = (values - values.mean()) / values.std()
            
            fig_stats.add_trace(go.Box(
                y=normalized_values,
                name=col,
                boxpoints='outliers'
            ))
        
        fig_stats.update_layout(
            title='Statistical Distribution Summary (Normalized)',
            yaxis_title='Normalized Values',
            xaxis_title='Features'
        )
        plots['Statistical Summary'] = fig_stats
    
    return plots

def create_model_performance_plots(results_dict):
    """Create model performance comparison visualizations"""
    plots = {}
    
    # Prepare data for visualization
    model_data = []
    for model_name, results in results_dict.items():
        model_data.append({
            'Model': model_name,
            'Test R²': results['test_r2'],
            'Train R²': results['train_r2'],
            'Test RMSE': np.sqrt(results['test_mse']),
            'Train RMSE': np.sqrt(results['train_mse']),
            'Test MAE': results['test_mae'],
            'CV Score': results['cv_score_mean'],
            'CV Std': results['cv_score_std'],
            'Training Time': results.get('training_time', 0),
            'Overfitting': results['train_r2'] - results['test_r2']
        })
    
    df_models = pd.DataFrame(model_data)
    
    # 1. Performance comparison bar chart
    fig_performance = go.Figure()
    
    fig_performance.add_trace(go.Bar(
        name='Train R²',
        x=df_models['Model'],
        y=df_models['Train R²'],
        marker_color='lightblue'
    ))
    
    fig_performance.add_trace(go.Bar(
        name='Test R²',
        x=df_models['Model'],
        y=df_models['Test R²'],
        marker_color='darkblue'
    ))
    
    fig_performance.update_layout(
        title='Model Performance Comparison (R² Scores)',
        xaxis_title='Models',
        yaxis_title='R² Score',
        barmode='group'
    )
    plots['Performance Comparison'] = fig_performance
    
    # 2. Error metrics comparison
    fig_errors = make_subplots(
        rows=1, cols=2,
        subplot_titles=['RMSE Comparison', 'MAE Comparison']
    )
    
    fig_errors.add_trace(
        go.Bar(x=df_models['Model'], y=df_models['Test RMSE'], name='RMSE'),
        row=1, col=1
    )
    
    fig_errors.add_trace(
        go.Bar(x=df_models['Model'], y=df_models['Test MAE'], name='MAE'),
        row=1, col=2
    )
    
    fig_errors.update_layout(title='Error Metrics Comparison', height=400)
    plots['Error Metrics'] = fig_errors
    
    # 3. Cross-validation analysis
    if 'cv_scores' in list(results_dict.values())[0]:
        fig_cv = go.Figure()
        
        for model_name, results in results_dict.items():
            cv_scores = results.get('cv_scores', [])
            if len(cv_scores) > 0:
                fig_cv.add_trace(go.Box(
                    y=cv_scores,
                    name=model_name,
                    boxpoints='all'
                ))
        
        fig_cv.update_layout(
            title='Cross-Validation Score Distribution',
            yaxis_title='CV Score',
            xaxis_title='Models'
        )
        plots['Cross-Validation Analysis'] = fig_cv

    df_models['CV Score'] = df_models['CV Score'].clip(lower=1e-5)
    # 4. Performance vs Training Time scatter
    fig_scatter = px.scatter(
        df_models,
        x='Training Time',
        y='Test R²',
        size='CV Score',
        color='Model',
        title='Performance vs Training Time',
        labels={'Training Time': 'Training Time (seconds)', 'Test R²': 'Test R² Score'},
        hover_data=['Test RMSE', 'CV Std']
    )
    plots['Performance vs Time'] = fig_scatter
    
    df_models['CV Score'] = df_models['CV Score'].clip(lower=1e-5)
    # 5. Overfitting analysis
    fig_overfitting = px.scatter(
        df_models,
        x='Train R²',
        y='Test R²',
        color='Model',
        size='CV Score',
        title='Overfitting Analysis (Train vs Test Performance)',
        labels={'Train R²': 'Training R² Score', 'Test R²': 'Test R² Score'}
    )
    
    # Add perfect fit line
    max_r2 = max(df_models['Train R²'].max(), df_models['Test R²'].max())
    min_r2 = min(df_models['Train R²'].min(), df_models['Test R²'].min())
    fig_overfitting.add_shape(
        type="line",
        x0=min_r2, y0=min_r2,
        x1=max_r2, y1=max_r2,
        line=dict(color="red", dash="dash"),
    )
    plots['Overfitting Analysis'] = fig_overfitting
    
    return plots

def create_feature_importance_plots(results_dict):
    """Create feature importance visualizations"""
    plots = {}
    
    for model_name, results in results_dict.items():
        if 'feature_importance' in results and results['feature_importance']:
            importance = results['feature_importance']
            
            # Sort by importance
            sorted_importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            # Take top 15 features
            top_features = dict(list(sorted_importance.items())[:15])
            
            fig = px.bar(
                x=list(top_features.values()),
                y=list(top_features.keys()),
                orientation='h',
                title=f'Feature Importance - {model_name}',
                labels={'x': 'Importance Score', 'y': 'Features'}
            )
            fig.update_layout(height=max(400, len(top_features) * 25))
            
            plots[f'{model_name} - Feature Importance'] = fig
    
    return plots

def create_residual_analysis_plots(results_dict):
    """Create residual analysis plots for model evaluation"""
    plots = {}
    
    for model_name, results in results_dict.items():
        if 'predictions' in results and 'y_true' in results:
            y_true = results['y_true']
            y_pred = results['predictions']
            residuals = y_true - y_pred
            
            # Create subplot with multiple residual analyses
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Residuals vs Predicted',
                    'Actual vs Predicted',
                    'Residual Distribution',
                    'Q-Q Plot'
                ]
            )
            
            # 1. Residuals vs Predicted
            fig.add_trace(
                go.Scatter(x=y_pred, y=residuals, mode='markers', 
                          name='Residuals', showlegend=False),
                row=1, col=1
            )
            fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
            
            # 2. Actual vs Predicted
            fig.add_trace(
                go.Scatter(x=y_true, y=y_pred, mode='markers', 
                          name='Predictions', showlegend=False),
                row=1, col=2
            )
            # Perfect prediction line
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            fig.add_shape(
                type="line",
                x0=min_val, y0=min_val,
                x1=max_val, y1=max_val,
                line=dict(color="red", dash="dash"),
                row=1, col=2
            )
            
            # 3. Residual distribution
            fig.add_trace(
                go.Histogram(x=residuals, name='Residuals', showlegend=False),
                row=2, col=1
            )
            
            # 4. Q-Q plot
            sorted_residuals = np.sort(residuals)
            theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
            
            fig.add_trace(
                go.Scatter(x=theoretical_quantiles, y=sorted_residuals, 
                          mode='markers', name='Q-Q Plot', showlegend=False),
                row=2, col=2
            )
            # Q-Q reference line
            fig.add_shape(
                type="line",
                x0=min(theoretical_quantiles), y0=min(sorted_residuals),
                x1=max(theoretical_quantiles), y1=max(sorted_residuals),
                line=dict(color="red", dash="dash"),
                row=2, col=2
            )
            
            fig.update_layout(
                title=f'Residual Analysis - {model_name}',
                height=800
            )
            
            plots[f'{model_name} - Residual Analysis'] = fig
    
    return plots

def create_healthcare_specific_plots(df, target_variable):
    """Create healthcare domain-specific visualizations"""
    plots = {}
    
    # 1. Wait time analysis by urgency (if applicable)
    if 'Urgency Level' in df.columns and 'wait' in target_variable.lower():
        fig_urgency = px.box(
            df,
            x='Urgency Level',
            y=target_variable,
            title='Wait Time Distribution by Urgency Level',
            color='Urgency Level'
        )
        plots['Wait Time by Urgency'] = fig_urgency
    
    # 2. Regional performance analysis
    region_cols = [col for col in df.columns if 'region' in col.lower()]
    if region_cols:
        region_col = region_cols[0]
        fig_region = px.violin(
            df,
            x=region_col,
            y=target_variable,
            title=f'{target_variable} by {region_col}',
            box=True
        )
        plots[f'Performance by {region_col}'] = fig_region
    
    # 3. Time-based patterns
    time_cols = [col for col in df.columns if any(word in col.lower() 
                 for word in ['hour', 'time_of_day', 'day_of_week'])]
    
    for time_col in time_cols[:2]:  # Limit to 2 time columns
        if time_col in df.columns:
            avg_by_time = df.groupby(time_col)[target_variable].mean().reset_index()
            
            fig_time = px.bar(
                avg_by_time,
                x=time_col,
                y=target_variable,
                title=f'Average {target_variable} by {time_col}'
            )
            plots[f'Time Pattern - {time_col}'] = fig_time
    
    # 4. Capacity utilization analysis
    capacity_cols = [col for col in df.columns if any(word in col.lower() 
                     for word in ['bed', 'capacity', 'facility_size'])]
    staff_cols = [col for col in df.columns if any(word in col.lower() 
                  for word in ['nurse', 'staff', 'specialist'])]
    
    if capacity_cols and staff_cols:
        capacity_col = capacity_cols[0]
        staff_col = staff_cols[0]
        
        fig_capacity = px.scatter(
            df,
            x=capacity_col,
            y=staff_col,
            color=target_variable,
            title='Capacity vs Staffing Analysis',
            size=target_variable
        )
        plots['Capacity vs Staffing'] = fig_capacity
    
    # 5. Outcome correlation heatmap
    outcome_cols = [col for col in df.columns if any(word in col.lower() 
                    for word in ['satisfaction', 'outcome', 'quality', 'score'])]
    
    if len(outcome_cols) > 1:
        correlation_outcomes = df[outcome_cols + [target_variable]].corr()
        
        fig_outcome_corr = px.imshow(
            correlation_outcomes.values,
            x=correlation_outcomes.columns,
            y=correlation_outcomes.index,
            color_continuous_scale='RdBu',
            title='Healthcare Outcomes Correlation'
        )
        plots['Outcomes Correlation'] = fig_outcome_corr
    
    return plots

def create_insight_visualizations(insights_data):
    """Create visualizations for actionable insights"""
    plots = {}
    
    # 1. Performance improvement opportunities
    if 'improvement_opportunities' in insights_data:
        opportunities = insights_data['improvement_opportunities']
        
        fig_opportunities = px.bar(
            x=list(opportunities.values()),
            y=list(opportunities.keys()),
            orientation='h',
            title='Performance Improvement Opportunities',
            labels={'x': 'Potential Improvement (%)', 'y': 'Areas'}
        )
        plots['Improvement Opportunities'] = fig_opportunities
    
    # 2. Risk assessment visualization
    if 'risk_factors' in insights_data:
        risk_factors = insights_data['risk_factors']
        
        fig_risk = px.pie(
            values=list(risk_factors.values()),
            names=list(risk_factors.keys()),
            title='Risk Factor Distribution'
        )
        plots['Risk Assessment'] = fig_risk
    
    # 3. Cost-benefit analysis
    if 'cost_benefit' in insights_data:
        cost_benefit = insights_data['cost_benefit']
        
        scenarios = list(cost_benefit.keys())
        costs = [cost_benefit[s]['cost'] for s in scenarios]
        benefits = [cost_benefit[s]['benefit'] for s in scenarios]
        
        fig_cost_benefit = go.Figure()
        fig_cost_benefit.add_trace(go.Bar(name='Cost', x=scenarios, y=costs))
        fig_cost_benefit.add_trace(go.Bar(name='Benefit', x=scenarios, y=benefits))
        
        fig_cost_benefit.update_layout(
            title='Cost-Benefit Analysis by Scenario',
            barmode='group'
        )
        plots['Cost-Benefit Analysis'] = fig_cost_benefit
    
    return plots

def create_dashboard_summary_chart(summary_metrics):
    """Create a comprehensive dashboard summary chart"""
    
    # Create a dashboard-style summary
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            'Model Performance', 'Data Quality', 'Prediction Accuracy',
            'Processing Time', 'Coverage', 'Improvement Potential'
        ],
        specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # Model Performance
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=summary_metrics.get('model_performance', 0) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Model Performance (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [{'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 90}}
    ), row=1, col=1)
    
    # Data Quality
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=summary_metrics.get('data_quality', 0) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Data Quality (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "green"},
               'steps': [{'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 95}}
    ), row=1, col=2)
    
    # Prediction Accuracy
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=summary_metrics.get('prediction_accuracy', 0) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Prediction Accuracy (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "purple"},
               'steps': [{'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 90}}
    ), row=1, col=3)
    
    # Processing Time (lower is better, so invert scale)
    processing_score = max(0, 100 - summary_metrics.get('processing_time', 10))
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=processing_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Processing Efficiency (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "orange"},
               'steps': [{'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 85}}
    ), row=2, col=1)
    
    # Coverage
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=summary_metrics.get('coverage', 0) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Data Coverage (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "teal"},
               'steps': [{'range': [0, 70], 'color': "lightgray"},
                        {'range': [70, 90], 'color': "gray"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                           'thickness': 0.75, 'value': 95}}
    ), row=2, col=2)
    
    # Improvement Potential
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=summary_metrics.get('improvement_potential', 0) * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Improvement Potential (%)"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "red"},
               'steps': [{'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 60], 'color': "gray"}],
               'threshold': {'line': {'color': "green", 'width': 4},
                           'thickness': 0.75, 'value': 20}}
    ), row=2, col=3)
    
    fig.update_layout(
        title='Healthcare Workflow Optimization Dashboard',
        height=600
    )
    
    return fig

# Example usage
if __name__ == "__main__":
    # This would typically be called from the main application
    print("Visualization utilities loaded successfully")
