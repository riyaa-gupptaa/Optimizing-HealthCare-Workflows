import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from Utils.Heathcare_metrics import generate_workflow_insights, calculate_cost_impact
from Utils.visulaization import create_insight_visualizations
from datetime import datetime, timedelta
import json

st.set_page_config(
    page_title="Insights Dashboard - Healthcare Optimization",
    page_icon="💡",
    layout="wide"
)

def main():
    st.title("💡 Insights & Recommendations Dashboard")
    st.markdown("Actionable insights for healthcare workflow optimization")
    
    # Check if we have trained models and datasets
    if not st.session_state.trained_models:
        st.warning("⚠️ No trained models available. Please train models first to generate insights.")
        st.stop()
    
    if not st.session_state.datasets:
        st.warning("⚠️ No datasets available. Please upload data first.")
        st.stop()
    
    # Dashboard configuration
    st.header("🔧 Dashboard Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Select model session for insights
        model_sessions = list(st.session_state.trained_models.keys())
        selected_session = st.selectbox(
            "Select Model Session",
            model_sessions,
            help="Choose which trained model session to analyze"
        )
    
    with col2:
        # Select specific model
        session_data = st.session_state.trained_models[selected_session]
        available_models = list(session_data['results'].keys())
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            help="Choose specific model for detailed insights"
        )
    
    with col3:
        # Insight type selection
        insight_types = st.multiselect(
            "Insight Categories",
            ["Operational Efficiency", "Resource Optimization", "Patient Experience", "Cost Analysis", "Predictive Alerts"],
            default=["Operational Efficiency", "Resource Optimization", "Patient Experience"],
            help="Select types of insights to display"
        )
    
    # Get model and dataset information
    model_results = session_data['results'][selected_model]
    dataset_name = session_data['dataset']
    target_variable = session_data['target']
    original_dataset = st.session_state.datasets[dataset_name]
    
    # Main insights dashboard
    st.header("📊 Key Performance Insights")
    
    # KPI Overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        accuracy_pct = model_results['test_r2'] * 100
        st.metric(
            "Model Accuracy",
            f"{accuracy_pct:.1f}%",
            f"R² = {model_results['test_r2']:.3f}",
            help="Percentage of variance explained by the model"
        )
    
    with col2:
        rmse_val = np.sqrt(model_results['test_mse'])
        target_std = original_dataset[target_variable].std()
        relative_error = (rmse_val / target_std) * 100
        st.metric(
            "Prediction Error",
            f"{rmse_val:.1f}",
            f"{relative_error:.1f}% of std",
            help="Root Mean Square Error of predictions"
        )
    
    with col3:
        if 'predictions' in model_results:
            predictions = model_results['predictions']
            avg_prediction = np.mean(predictions)
            st.metric(
                f"Avg Predicted {target_variable}",
                f"{avg_prediction:.1f}",
                help="Average predicted value across test set"
            )
        else:
            st.metric("Avg Predicted Value", "N/A")
    
    with col4:
        data_quality = (1 - original_dataset.isnull().sum().sum() / (len(original_dataset) * len(original_dataset.columns))) * 100
        st.metric(
            "Data Quality",
            f"{data_quality:.1f}%",
            help="Percentage of non-missing values in dataset"
        )
    
    # Insights tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Operational Insights", 
        "📈 Predictive Analytics", 
        "💰 Cost Impact", 
        "🏥 Clinical Recommendations", 
        "📋 Action Plan"
    ])
    
    with tab1:
        display_operational_insights(
            original_dataset, model_results, target_variable, selected_model
        )
    
    with tab2:
        display_predictive_analytics(
            original_dataset, model_results, target_variable, selected_model
        )
    
    with tab3:
        display_cost_impact_analysis(
            original_dataset, model_results, target_variable, selected_model
        )
    
    with tab4:
        display_clinical_recommendations(
            original_dataset, model_results, target_variable, selected_model
        )
    
    with tab5:
        display_action_plan(
            original_dataset, model_results, target_variable, selected_model, session_data
        )

def display_operational_insights(dataset, model_results, target_variable, model_name):
    """Display operational efficiency insights"""
    
    st.subheader("🎯 Operational Efficiency Analysis")
    
    # Feature importance insights
    if 'feature_importance' in model_results:
        st.write("### 🔍 Key Performance Drivers")
        
        importance = model_results['feature_importance']
        top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top drivers chart
            fig_importance = px.bar(
                x=list(top_features.values()),
                y=list(top_features.keys()),
                orientation='h',
                title='Top 10 Performance Drivers',
                labels={'x': 'Importance Score', 'y': 'Factors'}
            )
            fig_importance.update_layout(height=400)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.write("**Key Insights:**")
            
            # Generate insights based on top features
            insights = []
            for feature, importance_score in list(top_features.items())[:5]:
                if 'nurse' in feature.lower() or 'staff' in feature.lower():
                    insights.append(f"👥 **Staffing Impact**: {feature} is a critical factor (importance: {importance_score:.3f})")
                elif 'bed' in feature.lower() or 'capacity' in feature.lower():
                    insights.append(f"🏥 **Capacity Management**: {feature} significantly affects outcomes (importance: {importance_score:.3f})")
                elif 'time' in feature.lower() or 'hour' in feature.lower():
                    insights.append(f"⏰ **Timing Factors**: {feature} shows strong influence (importance: {importance_score:.3f})")
                elif 'urgent' in feature.lower() or 'priority' in feature.lower():
                    insights.append(f"🚨 **Priority Management**: {feature} is crucial for optimization (importance: {importance_score:.3f})")
                else:
                    insights.append(f"📊 **Key Factor**: {feature} significantly impacts performance (importance: {importance_score:.3f})")
            
            for insight in insights:
                st.markdown(insight)
    
    # Performance distribution analysis
    if 'predictions' in model_results:
        st.write("### 📊 Performance Distribution Analysis")
        
        predictions = model_results['predictions']
        actual = model_results['y_true']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution comparison
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(x=actual, name='Actual', opacity=0.7, nbinsx=30))
            fig_dist.add_trace(go.Histogram(x=predictions, name='Predicted', opacity=0.7, nbinsx=30))
            fig_dist.update_layout(
                title='Actual vs Predicted Distribution',
                xaxis_title=target_variable,
                yaxis_title='Frequency',
                barmode='overlay'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Performance quartiles
            quartiles = np.percentile(predictions, [25, 50, 75])
            
            st.write("**Performance Quartiles:**")
            st.metric("25th Percentile", f"{quartiles[0]:.1f}")
            st.metric("Median (50th)", f"{quartiles[1]:.1f}")
            st.metric("75th Percentile", f"{quartiles[2]:.1f}")
            
            # Actionable insights
            if 'wait' in target_variable.lower() or 'time' in target_variable.lower():
                st.write("**🎯 Optimization Opportunities:**")
                if quartiles[2] - quartiles[0] > quartiles[1]:
                    st.warning("⚠️ High variability detected - standardize processes")
                if quartiles[1] > np.mean(predictions) * 1.2:
                    st.error("🚨 Average performance exceeds targets - immediate action needed")
                else:
                    st.success("✅ Performance within acceptable range")
    
    # Bottleneck identification
    st.write("### 🚫 Bottleneck Identification")
    
    # Analyze numerical columns for bottlenecks
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) > 1:
        bottleneck_analysis = {}
        
        for col in numeric_cols:
            if col != target_variable:
                correlation = dataset[col].corr(dataset[target_variable])
                std_ratio = dataset[col].std() / dataset[col].mean() if dataset[col].mean() != 0 else 0
                bottleneck_analysis[col] = {
                    'correlation': correlation,
                    'variability': std_ratio,
                    'bottleneck_score': abs(correlation) * std_ratio
                }
        
        # Sort by bottleneck score
        sorted_bottlenecks = sorted(
            bottleneck_analysis.items(),
            key=lambda x: x[1]['bottleneck_score'],
            reverse=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Bottleneck Factors:**")
            for i, (factor, scores) in enumerate(sorted_bottlenecks[:5]):
                if scores['bottleneck_score'] > 0.1:  # Threshold for significance
                    correlation_dir = "positively" if scores['correlation'] > 0 else "negatively"
                    st.write(f"{i+1}. **{factor}**: {correlation_dir} correlated ({scores['correlation']:.3f}), high variability")
        
        with col2:
            # Bottleneck score visualization
            bottleneck_df = pd.DataFrame([
                {'Factor': factor, 'Bottleneck Score': scores['bottleneck_score']}
                for factor, scores in sorted_bottlenecks[:8]
            ])
            
            fig_bottleneck = px.bar(
                bottleneck_df,
                x='Bottleneck Score',
                y='Factor',
                orientation='h',
                title='Bottleneck Analysis',
                color='Bottleneck Score',
                color_continuous_scale='reds'
            )
            st.plotly_chart(fig_bottleneck, use_container_width=True)

def display_predictive_analytics(dataset, model_results, target_variable, model_name):
    """Display predictive analytics insights"""
    
    st.subheader("📈 Predictive Analytics & Forecasting")
    
    # Model performance trends
    if 'cv_scores' in model_results:
        st.write("### 📊 Model Reliability Analysis")
        
        cv_scores = model_results['cv_scores']
        cv_mean = model_results['cv_score_mean']
        cv_std = model_results['cv_score_std']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cross-validation stability
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Scatter(
                x=list(range(1, len(cv_scores) + 1)),
                y=cv_scores,
                mode='lines+markers',
                name='CV Scores',
                line=dict(color='blue')
            ))
            fig_cv.add_hline(y=cv_mean, line_dash="dash", line_color="red", 
                           annotation_text=f"Mean: {cv_mean:.3f}")
            fig_cv.update_layout(
                title='Cross-Validation Stability',
                xaxis_title='Fold',
                yaxis_title='Score'
            )
            st.plotly_chart(fig_cv, use_container_width=True)
        
        with col2:
            st.write("**Reliability Metrics:**")
            reliability_score = 1 - (cv_std / cv_mean) if cv_mean != 0 else 0
            
            st.metric("Model Consistency", f"{reliability_score:.1%}")
            st.metric("CV Standard Deviation", f"{cv_std:.3f}")
            
            if reliability_score > 0.9:
                st.success("🟢 High reliability - suitable for production")
            elif reliability_score > 0.8:
                st.warning("🟡 Moderate reliability - monitor performance")
            else:
                st.error("🔴 Low reliability - requires improvement")
    
    # Prediction confidence intervals
    if 'predictions' in model_results:
        st.write("### 🎯 Prediction Confidence Analysis")
        
        predictions = model_results['predictions']
        actual = model_results['y_true']
        residuals = actual - predictions
        
        # Calculate confidence intervals
        residual_std = np.std(residuals)
        confidence_80 = 1.28 * residual_std  # 80% confidence interval
        confidence_95 = 1.96 * residual_std  # 95% confidence interval
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prediction Confidence Bands:**")
            st.metric("80% Confidence", f"± {confidence_80:.1f}")
            st.metric("95% Confidence", f"± {confidence_95:.1f}")
            
            # Accuracy within confidence bands
            within_80 = np.sum(np.abs(residuals) <= confidence_80) / len(residuals)
            within_95 = np.sum(np.abs(residuals) <= confidence_95) / len(residuals)
            
            st.metric("Actual 80% Accuracy", f"{within_80:.1%}")
            st.metric("Actual 95% Accuracy", f"{within_95:.1%}")
        
        with col2:
            # Residuals distribution
            fig_residuals = px.histogram(
                x=residuals,
                nbins=30,
                title='Prediction Error Distribution',
                labels={'x': 'Residuals (Actual - Predicted)', 'y': 'Frequency'}
            )
            fig_residuals.add_vline(x=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Scenario analysis
    st.write("### 🔮 Scenario Analysis")
    
    scenario_col1, scenario_col2 = st.columns(2)
    
    with scenario_col1:
        st.write("**'What-If' Scenarios:**")
        
        # Generate scenarios based on feature importance
        if 'feature_importance' in model_results:
            top_features = sorted(
                model_results['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            for feature, importance in top_features:
                if feature in dataset.columns:
                    current_value = dataset[feature].mean()
                    
                    # Simulate 10% improvement
                    improvement_10 = current_value * 1.1 if 'ratio' in feature.lower() else current_value * 0.9
                    potential_impact = importance * 0.1  # Rough estimate
                    
                    st.write(f"📊 **{feature}**:")
                    st.write(f"  Current: {current_value:.2f}")
                    st.write(f"  10% improvement could impact {target_variable} by ~{potential_impact:.2%}")
    
    with scenario_col2:
        st.write("**Risk Assessment:**")
        
        # Identify high-risk predictions
        if 'predictions' in model_results:
            predictions = model_results['predictions']
            
            # Define risk thresholds based on target variable
            if 'wait' in target_variable.lower() or 'time' in target_variable.lower():
                high_risk_threshold = np.percentile(predictions, 90)
                high_risk_count = np.sum(predictions > high_risk_threshold)
                
                st.metric("High-Risk Cases", f"{high_risk_count}")
                st.metric("Risk Threshold", f"{high_risk_threshold:.1f}")
                
                risk_percentage = (high_risk_count / len(predictions)) * 100
                if risk_percentage > 15:
                    st.error(f"🚨 High risk: {risk_percentage:.1f}% of cases exceed threshold")
                elif risk_percentage > 5:
                    st.warning(f"⚠️ Moderate risk: {risk_percentage:.1f}% of cases need attention")
                else:
                    st.success(f"✅ Low risk: Only {risk_percentage:.1f}% of cases exceed threshold")
    
    # Alert system recommendations
    st.write("### 🚨 Automated Alert Recommendations")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.write("**Recommended Alert Thresholds:**")
        
        if 'predictions' in model_results:
            predictions = model_results['predictions']
            
            # Critical threshold (95th percentile)
            critical_threshold = np.percentile(predictions, 95)
            warning_threshold = np.percentile(predictions, 85)
            
            st.error(f"🚨 **Critical Alert**: {target_variable} > {critical_threshold:.1f}")
            st.warning(f"⚠️ **Warning Alert**: {target_variable} > {warning_threshold:.1f}")
            st.info(f"ℹ️ **Information**: {target_variable} > {np.percentile(predictions, 75):.1f}")
    
    with alert_col2:
        st.write("**Implementation Guidelines:**")
        st.markdown("""
        - **Real-time monitoring**: Implement continuous prediction updates
        - **Escalation paths**: Define clear response procedures for each alert level
        - **Historical tracking**: Monitor alert frequency and response effectiveness
        - **Feedback loop**: Use alert outcomes to improve model accuracy
        """)

def display_cost_impact_analysis(dataset, model_results, target_variable, model_name):
    """Display cost impact and ROI analysis"""
    
    st.subheader("💰 Cost Impact & ROI Analysis")
    
    # Cost modeling section
    st.write("### 💵 Cost Modeling")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Configure Cost Parameters:**")
        
        # Cost per unit of target variable
        if 'wait' in target_variable.lower() or 'time' in target_variable.lower():
            unit_cost = st.number_input(
                "Cost per minute (USD)",
                min_value=0.0,
                value=2.5,
                step=0.1,
                help="Estimated cost per minute of wait time"
            )
            
            overtime_multiplier = st.number_input(
                "Overtime cost multiplier",
                min_value=1.0,
                value=1.5,
                step=0.1,
                help="Multiplier for costs beyond normal thresholds"
            )
        else:
            unit_cost = st.number_input(
                f"Cost per unit of {target_variable} (USD)",
                min_value=0.0,
                value=100.0,
                step=10.0
            )
            overtime_multiplier = 1.0
        
        improvement_cost = st.number_input(
            "Implementation cost (USD)",
            min_value=0.0,
            value=50000.0,
            step=1000.0,
            help="One-time cost to implement optimization recommendations"
        )
    
    with col2:
        st.write("**Current Cost Analysis:**")
        
        if 'predictions' in model_results:
            predictions = model_results['predictions']
            
            # Calculate current costs
            baseline_cost = np.mean(predictions) * unit_cost
            total_annual_cases = len(dataset) * 12  # Assuming monthly data
            annual_baseline_cost = baseline_cost * total_annual_cases
            
            st.metric("Average Cost per Case", f"${baseline_cost:.2f}")
            st.metric("Estimated Annual Cost", f"${annual_baseline_cost:,.0f}")
            
            # Calculate high-cost cases
            high_cost_threshold = np.percentile(predictions, 90)
            high_cost_cases = np.sum(predictions > high_cost_threshold)
            high_cost_total = np.sum(predictions[predictions > high_cost_threshold]) * unit_cost
            
            st.metric("High-Cost Cases", f"{high_cost_cases} ({high_cost_cases/len(predictions)*100:.1f}%)")
            st.metric("High-Cost Impact", f"${high_cost_total:.0f}")
    
    # ROI Analysis
    st.write("### 📈 Return on Investment Analysis")
    
    roi_col1, roi_col2 = st.columns(2)
    
    with roi_col1:
        st.write("**Improvement Scenarios:**")
        
        improvement_scenarios = [5, 10, 15, 20]  # Percentage improvements
        scenario_data = []
        
        if 'predictions' in model_results:
            current_avg = np.mean(predictions)
            
            for improvement_pct in improvement_scenarios:
                improved_avg = current_avg * (1 - improvement_pct/100)
                cost_savings_per_case = (current_avg - improved_avg) * unit_cost
                annual_savings = cost_savings_per_case * total_annual_cases
                roi = ((annual_savings - improvement_cost) / improvement_cost) * 100 if improvement_cost > 0 else 0
                payback_months = (improvement_cost / (annual_savings / 12)) if annual_savings > 0 else float('inf')
                
                scenario_data.append({
                    'Improvement': f"{improvement_pct}%",
                    'Annual Savings': annual_savings,
                    'ROI': roi,
                    'Payback (months)': min(payback_months, 60)  # Cap at 5 years
                })
            
            scenario_df = pd.DataFrame(scenario_data)
            st.dataframe(scenario_df.style.format({
                'Annual Savings': '${:,.0f}',
                'ROI': '{:.1f}%',
                'Payback (months)': '{:.1f}'
            }), use_container_width=True)
    
    with roi_col2:
        st.write("**ROI Visualization:**")
        
        if 'predictions' in model_results and len(scenario_data) > 0:
            fig_roi = px.bar(
                scenario_df,
                x='Improvement',
                y='ROI',
                title='ROI by Improvement Scenario',
                color='ROI',
                color_continuous_scale='greens'
            )
            fig_roi.add_hline(y=0, line_dash="dash", line_color="red")
            st.plotly_chart(fig_roi, use_container_width=True)
    
    # Cost breakdown analysis
    st.write("### 🔍 Cost Breakdown Analysis")
    
    breakdown_col1, breakdown_col2 = st.columns(2)
    
    with breakdown_col1:
        if 'feature_importance' in model_results:
            st.write("**Cost Drivers by Feature:**")
            
            # Estimate cost impact by feature importance
            importance = model_results['feature_importance']
            total_importance = sum(importance.values())
            
            cost_drivers = []
            for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                feature_cost_impact = (imp / total_importance) * annual_baseline_cost
                cost_drivers.append({
                    'Factor': feature,
                    'Cost Impact': feature_cost_impact,
                    'Percentage': (imp / total_importance) * 100
                })
            
            cost_driver_df = pd.DataFrame(cost_drivers)
            st.dataframe(cost_driver_df.style.format({
                'Cost Impact': '${:,.0f}',
                'Percentage': '{:.1f}%'
            }), use_container_width=True)
    
    with breakdown_col2:
        st.write("**Optimization Priority Matrix:**")
        
        if len(cost_drivers) > 0:
            # Create priority matrix based on cost impact and improvement feasibility
            priority_data = []
            for driver in cost_drivers:
                # Simplified feasibility score (in practice, this would be more sophisticated)
                feasibility = 0.8 if 'staff' in driver['Factor'].lower() or 'process' in driver['Factor'].lower() else 0.6
                priority_score = (driver['Cost Impact'] / 1000) * feasibility  # Normalize and weight by feasibility
                
                priority_data.append({
                    'Factor': driver['Factor'],
                    'Cost Impact': driver['Cost Impact'],
                    'Feasibility': feasibility,
                    'Priority Score': priority_score
                })
            
            priority_df = pd.DataFrame(priority_data)
            
            # Ensure priority scores are positive for plotly
            priority_df['Priority Score'] = np.maximum(priority_df['Priority Score'], 1)
            fig_priority = px.scatter(
                priority_df,
                x='Feasibility',
                y='Cost Impact',
                size='Priority Score',
                hover_data=['Factor'],
                title='Optimization Priority Matrix',
                labels={'Feasibility': 'Implementation Feasibility', 'Cost Impact': 'Annual Cost Impact ($)'}
            )
            st.plotly_chart(fig_priority, use_container_width=True)

def display_clinical_recommendations(dataset, model_results, target_variable, model_name):
    """Display clinical and operational recommendations"""
    
    st.subheader("🏥 Clinical & Operational Recommendations")
    
    # Evidence-based recommendations
    st.write("### 📋 Evidence-Based Recommendations")
    
    recommendations = []
    
    # Analyze feature importance for recommendations
    if 'feature_importance' in model_results:
        importance = model_results['feature_importance']
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for feature, imp in top_features:
            if 'nurse' in feature.lower() or 'staff' in feature.lower():
                recommendations.append({
                    'category': 'Staffing',
                    'priority': 'High',
                    'recommendation': f'Optimize {feature} - shows {imp:.3f} importance score',
                    'action': 'Review staffing ratios and implement dynamic scheduling',
                    'timeline': '2-4 weeks',
                    'evidence': f'Feature importance analysis shows {imp:.1%} contribution to outcomes'
                })
            elif 'bed' in feature.lower() or 'capacity' in feature.lower():
                recommendations.append({
                    'category': 'Capacity Management',
                    'priority': 'Medium',
                    'recommendation': f'Address {feature} bottlenecks',
                    'action': 'Implement flexible bed allocation and capacity monitoring',
                    'timeline': '4-8 weeks',
                    'evidence': f'Capacity factors account for {imp:.1%} of performance variation'
                })
            elif 'time' in feature.lower() or 'hour' in feature.lower():
                recommendations.append({
                    'category': 'Process Timing',
                    'priority': 'High',
                    'recommendation': f'Optimize timing patterns in {feature}',
                    'action': 'Analyze peak hours and implement load balancing',
                    'timeline': '1-2 weeks',
                    'evidence': f'Temporal factors show {imp:.1%} impact on outcomes'
                })
    
    # Display recommendations in organized format
    if recommendations:
        for category in ['Staffing', 'Capacity Management', 'Process Timing']:
            category_recs = [r for r in recommendations if r['category'] == category]
            if category_recs:
                st.write(f"#### {category}")
                for rec in category_recs:
                    with st.expander(f"{'🔴' if rec['priority'] == 'High' else '🟡'} {rec['recommendation']}"):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Action Required:** {rec['action']}")
                            st.write(f"**Timeline:** {rec['timeline']}")
                        with col2:
                            st.write(f"**Priority:** {rec['priority']}")
                            st.write(f"**Evidence:** {rec['evidence']}")
    
    # Quality improvement opportunities
    st.write("### 📊 Quality Improvement Opportunities")
    
    quality_col1, quality_col2 = st.columns(2)
    
    with quality_col1:
        st.write("**Process Improvements:**")
        
        # Analyze prediction accuracy for quality insights
        if 'predictions' in model_results:
            predictions = model_results['predictions']
            actual = model_results['y_true']
            accuracy = model_results['test_r2']
            
            if accuracy > 0.8:
                st.success("🟢 **High Model Accuracy** - Proceed with confidence")
                st.write("• Model predictions are reliable for decision-making")
                st.write("• Implement automated decision support systems")
            elif accuracy > 0.6:
                st.warning("🟡 **Moderate Accuracy** - Monitor and validate")
                st.write("• Use predictions as guidance with human oversight")
                st.write("• Collect additional data to improve accuracy")
            else:
                st.error("🔴 **Low Accuracy** - Requires improvement")
                st.write("• Focus on data quality and feature engineering")
                st.write("• Consider alternative modeling approaches")
        
        # Data quality recommendations
        missing_pct = dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))
        if missing_pct > 0.1:
            st.warning(f"⚠️ **Data Quality**: {missing_pct:.1%} missing values detected")
            st.write("• Implement data validation at source")
            st.write("• Train staff on data entry protocols")
    
    with quality_col2:
        st.write("**Clinical Guidelines:**")
        
        # Generate clinical guidelines based on target variable
        if 'wait' in target_variable.lower():
            st.write("**Wait Time Optimization:**")
            st.write("• Implement triage protocol standardization")
            st.write("• Establish fast-track pathways for low-acuity cases")
            st.write("• Monitor and address bottlenecks in real-time")
            
            if 'predictions' in model_results:
                avg_wait = np.mean(predictions)
                if avg_wait > 60:  # More than 1 hour
                    st.error("🚨 Average wait times exceed recommended guidelines")
                elif avg_wait > 30:
                    st.warning("⚠️ Wait times approaching concerning levels")
                else:
                    st.success("✅ Wait times within acceptable range")
        
        elif 'satisfaction' in target_variable.lower():
            st.write("**Patient Satisfaction Enhancement:**")
            st.write("• Improve communication protocols")
            st.write("• Implement patient feedback loops")
            st.write("• Focus on staff training and empathy")
    
    # Implementation roadmap
    st.write("### 🗺️ Implementation Roadmap")
    
    roadmap_phases = [
        {
            'phase': 'Phase 1: Immediate (1-2 weeks)',
            'actions': [
                'Implement high-priority recommendations',
                'Establish monitoring dashboards',
                'Train staff on new protocols'
            ],
            'success_metrics': [
                f'10% improvement in {target_variable}',
                'Reduced variation in outcomes',
                'Staff compliance > 90%'
            ]
        },
        {
            'phase': 'Phase 2: Short-term (1-3 months)',
            'actions': [
                'Deploy predictive alerts system',
                'Optimize resource allocation',
                'Refine prediction models'
            ],
            'success_metrics': [
                f'20% improvement in {target_variable}',
                'Reduced false alerts < 5%',
                'Model accuracy > 85%'
            ]
        },
        {
            'phase': 'Phase 3: Long-term (3-6 months)',
            'actions': [
                'Integrate with hospital systems',
                'Expand to additional departments',
                'Continuous improvement processes'
            ],
            'success_metrics': [
                f'30% improvement in {target_variable}',
                'System-wide adoption',
                'ROI > 200%'
            ]
        }
    ]
    
    for phase_info in roadmap_phases:
        with st.expander(phase_info['phase']):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Actions:**")
                for action in phase_info['actions']:
                    st.write(f"• {action}")
            with col2:
                st.write("**Success Metrics:**")
                for metric in phase_info['success_metrics']:
                    st.write(f"• {metric}")

def display_action_plan(dataset, model_results, target_variable, model_name, session_data):
    """Display comprehensive action plan"""
    
    st.subheader("📋 Comprehensive Action Plan")
    
    # Executive summary
    st.write("### 📄 Executive Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.write("**Current State:**")
        if 'predictions' in model_results:
            current_performance = np.mean(model_results['predictions'])
            model_accuracy = model_results['test_r2']
            
            st.metric("Current Average Performance", f"{current_performance:.1f}")
            st.metric("Prediction Accuracy", f"{model_accuracy:.1%}")
            
            # Performance assessment
            if model_accuracy > 0.8:
                st.success("✅ High confidence in recommendations")
            elif model_accuracy > 0.6:
                st.warning("⚠️ Moderate confidence - proceed with caution")
            else:
                st.error("❌ Low confidence - improve data quality first")
    
    with summary_col2:
        st.write("**Opportunity Assessment:**")
        
        # Calculate improvement potential
        if 'predictions' in model_results:
            predictions = model_results['predictions']
            
            # Potential improvement (difference between best quartile and average)
            best_quartile = np.percentile(predictions, 25)  # Assuming lower is better
            current_avg = np.mean(predictions)
            improvement_potential = ((current_avg - best_quartile) / current_avg) * 100
            
            st.metric("Improvement Potential", f"{max(0, improvement_potential):.1f}%")
            
            # Risk assessment
            high_risk_cases = np.sum(predictions > np.percentile(predictions, 90))
            risk_percentage = (high_risk_cases / len(predictions)) * 100
            
            st.metric("High-Risk Cases", f"{risk_percentage:.1f}%")
            
            if risk_percentage > 15:
                st.error("🚨 High risk - immediate action required")
            elif risk_percentage > 5:
                st.warning("⚠️ Moderate risk - monitor closely")
            else:
                st.success("✅ Low risk - maintain current practices")
    
    # Detailed action items
    st.write("### ✅ Detailed Action Items")
    
    # Create action items based on analysis
    action_items = []
    
    # High-priority actions based on feature importance
    if 'feature_importance' in model_results:
        importance = model_results['feature_importance']
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        for i, (feature, imp) in enumerate(top_features, 1):
            action_items.append({
                'priority': 'High',
                'action': f'Optimize {feature}',
                'description': f'Address top performance driver (importance: {imp:.3f})',
                'responsible': 'Operations Manager',
                'timeline': '2-4 weeks',
                'success_metric': f'10% improvement in {feature} metric',
                'status': 'Not Started'
            })
    
    # Data quality actions
    missing_pct = dataset.isnull().sum().sum() / (len(dataset) * len(dataset.columns))
    if missing_pct > 0.05:
        action_items.append({
            'priority': 'Medium',
            'action': 'Improve Data Quality',
            'description': f'Address {missing_pct:.1%} missing data rate',
            'responsible': 'IT Team + Clinical Staff',
            'timeline': '4-6 weeks',
            'success_metric': 'Reduce missing data to <2%',
            'status': 'Not Started'
        })
    
    # Model improvement actions
    if model_results['test_r2'] < 0.8:
        action_items.append({
            'priority': 'Medium',
            'action': 'Enhance Prediction Model',
            'description': f'Improve model accuracy from {model_results["test_r2"]:.1%}',
            'responsible': 'Data Science Team',
            'timeline': '6-8 weeks',
            'success_metric': 'Achieve >80% model accuracy',
            'status': 'Not Started'
        })
    
    # Display action items
    for i, item in enumerate(action_items):
        with st.expander(f"{'🔴' if item['priority'] == 'High' else '🟡'} Action {i+1}: {item['action']}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write(f"**Description:** {item['description']}")
                st.write(f"**Responsible:** {item['responsible']}")
            
            with col2:
                st.write(f"**Timeline:** {item['timeline']}")
                st.write(f"**Priority:** {item['priority']}")
            
            with col3:
                st.write(f"**Success Metric:** {item['success_metric']}")
                
                # Status selector
                new_status = st.selectbox(
                    "Status",
                    ["Not Started", "In Progress", "Completed", "Blocked"],
                    index=["Not Started", "In Progress", "Completed", "Blocked"].index(item['status']),
                    key=f"status_{i}"
                )
                item['status'] = new_status
    
    # Progress tracking
    st.write("### 📊 Progress Tracking")
    
    if action_items:
        progress_col1, progress_col2 = st.columns(2)
        
        with progress_col1:
            # Progress metrics
            total_actions = len(action_items)
            completed_actions = sum(1 for item in action_items if item['status'] == 'Completed')
            in_progress_actions = sum(1 for item in action_items if item['status'] == 'In Progress')
            
            completion_rate = (completed_actions / total_actions) * 100 if total_actions > 0 else 0
            
            st.metric("Completion Rate", f"{completion_rate:.1f}%")
            st.metric("Completed Actions", f"{completed_actions}/{total_actions}")
            st.metric("In Progress", in_progress_actions)
        
        with progress_col2:
            # Progress visualization
            status_counts = {}
            for item in action_items:
                status_counts[item['status']] = status_counts.get(item['status'], 0) + 1
            
            fig_progress = px.pie(
                values=list(status_counts.values()),
                names=list(status_counts.keys()),
                title='Action Item Status Distribution'
            )
            st.plotly_chart(fig_progress, use_container_width=True)
    
    # Export action plan
    st.write("### 📤 Export Action Plan")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Generate detailed report
        if st.button("📋 Generate Detailed Report"):
            report = generate_action_plan_report(
                dataset, model_results, target_variable, model_name, action_items
            )
            st.text_area(
                "Action Plan Report",
                report,
                height=400
            )
    
    with col2:
        # Export to CSV
        if action_items:
            action_df = pd.DataFrame(action_items)
            csv_data = action_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Action Items (CSV)",
                data=csv_data,
                file_name="healthcare_action_plan.csv",
                mime="text/csv"
            )
    
    with col3:
        # Save to session (for future reference)
        if st.button("💾 Save Action Plan"):
            session_key = f"action_plan_{datetime.now().strftime('%Y%m%d_%H%M')}"
            if 'action_plans' not in st.session_state:
                st.session_state.action_plans = {}
            
            st.session_state.action_plans[session_key] = {
                'action_items': action_items,
                'dataset': session_data['dataset'],
                'target': target_variable,
                'model': model_name,
                'timestamp': datetime.now()
            }
            
            st.success("✅ Action plan saved to session!")

def generate_action_plan_report(dataset, model_results, target_variable, model_name, action_items):
    """Generate a comprehensive action plan report"""
    
    report_lines = [
        "HEALTHCARE WORKFLOW OPTIMIZATION - ACTION PLAN",
        "=" * 50,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Target Variable: {target_variable}",
        f"Model Used: {model_name}",
        f"Model Accuracy: {model_results['test_r2']:.1%}",
        "",
        "EXECUTIVE SUMMARY:",
        "-" * 20
    ]
    
    if 'predictions' in model_results:
        current_avg = np.mean(model_results['predictions'])
        best_quartile = np.percentile(model_results['predictions'], 25)
        improvement_potential = max(0, ((current_avg - best_quartile) / current_avg) * 100)
        
        report_lines.extend([
            f"Current Average Performance: {current_avg:.2f}",
            f"Improvement Potential: {improvement_potential:.1f}%",
            f"Model Reliability: {model_results['test_r2']:.1%}",
            ""
        ])
    
    report_lines.extend([
        "ACTION ITEMS:",
        "-" * 15
    ])
    
    for i, item in enumerate(action_items, 1):
        report_lines.extend([
            f"{i}. {item['action']} [{item['priority']} Priority]",
            f"   Description: {item['description']}",
            f"   Responsible: {item['responsible']}",
            f"   Timeline: {item['timeline']}",
            f"   Success Metric: {item['success_metric']}",
            f"   Status: {item['status']}",
            ""
        ])
    
    if 'feature_importance' in model_results:
        report_lines.extend([
            "KEY PERFORMANCE DRIVERS:",
            "-" * 25
        ])
        
        importance = model_results['feature_importance']
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for feature, imp in top_features:
            report_lines.append(f"• {feature}: {imp:.3f} importance score")
        
        report_lines.append("")
    
    report_lines.extend([
        "IMPLEMENTATION RECOMMENDATIONS:",
        "-" * 30,
        "1. Focus on high-priority actions first",
        "2. Establish regular monitoring and review cycles",
        "3. Collect feedback from implementation teams",
        "4. Track success metrics continuously",
        "5. Adjust approach based on initial results",
        "",
        "For technical support or questions, contact the Data Science Team."
    ])
    
    return "\n".join(report_lines)

if __name__ == "__main__":
    main()
