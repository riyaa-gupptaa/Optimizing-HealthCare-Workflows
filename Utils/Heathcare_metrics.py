import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_healthcare_metrics(y_true, y_pred, target_variable):
    """
    Calculate healthcare-specific performance metrics
    """
    metrics = {}
    
    # Basic regression metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    # Healthcare-specific metrics
    
    # 1. Clinical Accuracy (within acceptable range)
    if 'wait' in target_variable.lower() or 'time' in target_variable.lower():
        # For wait times, acceptable range is typically ±20% or ±15 minutes (whichever is smaller)
        acceptable_range = np.minimum(np.abs(y_true) * 0.2, 15)
        within_acceptable = np.abs(y_true - y_pred) <= acceptable_range
        clinical_accuracy = np.mean(within_acceptable)
    else:
        # For other metrics, use ±10% as acceptable range
        acceptable_range = np.abs(y_true) * 0.1
        within_acceptable = np.abs(y_true - y_pred) <= acceptable_range
        clinical_accuracy = np.mean(within_acceptable)
    
    # 2. Critical Case Detection (for high-priority cases)
    if 'wait' in target_variable.lower():
        # Critical cases are those with very high wait times (top 10%)
        critical_threshold = np.percentile(y_true, 90)
        critical_cases_actual = y_true >= critical_threshold
        critical_cases_predicted = y_pred >= critical_threshold
        
        if np.sum(critical_cases_actual) > 0:
            # Sensitivity for critical cases
            critical_detection_rate = np.sum(critical_cases_actual & critical_cases_predicted) / np.sum(critical_cases_actual)
        else:
            critical_detection_rate = 1.0
    else:
        # For other metrics, use different approach
        critical_threshold = np.percentile(y_true, 90)
        critical_cases_actual = y_true >= critical_threshold
        critical_cases_predicted = y_pred >= (critical_threshold * 0.9)  # Slightly lower threshold for predictions
        
        if np.sum(critical_cases_actual) > 0:
            critical_detection_rate = np.sum(critical_cases_actual & critical_cases_predicted) / np.sum(critical_cases_actual)
        else:
            critical_detection_rate = 1.0
    
    # 3. Resource Planning Accuracy
    # Measure how well the model predicts resource needs
    # Group predictions into quartiles and measure accuracy within groups
    quartiles = np.percentile(y_true, [25, 50, 75])
    
    resource_accuracy_scores = []
    for i in range(4):
        if i == 0:
            mask = y_true <= quartiles[0]
        elif i == 1:
            mask = (y_true > quartiles[0]) & (y_true <= quartiles[1])
        elif i == 2:
            mask = (y_true > quartiles[1]) & (y_true <= quartiles[2])
        else:
            mask = y_true > quartiles[2]
        
        if np.sum(mask) > 0:
            quartile_mae = np.mean(np.abs(y_true[mask] - y_pred[mask]))
            quartile_mean = np.mean(y_true[mask])
            if quartile_mean > 0:
                quartile_accuracy = 1 - (quartile_mae / quartile_mean)
                resource_accuracy_scores.append(max(0, quartile_accuracy))
    
    resource_accuracy = np.mean(resource_accuracy_scores) if resource_accuracy_scores else 0
    
    # 4. Efficiency Impact Score
    # Measure potential operational efficiency gains
    if 'wait' in target_variable.lower():
        # For wait times, calculate potential time savings
        potential_savings = np.maximum(0, y_true - y_pred)
        efficiency_impact = np.mean(potential_savings) / np.mean(y_true)
    else:
        # For other metrics, calculate improvement potential
        improvement_potential = np.abs(y_true - y_pred) / np.mean(y_true)
        efficiency_impact = 1 - np.mean(improvement_potential)
    
    # 5. Consistency Score
    # Measure prediction consistency across different subgroups
    residuals = y_true - y_pred
    consistency_score = 1 - (np.std(residuals) / np.mean(np.abs(residuals))) if np.mean(np.abs(residuals)) > 0 else 0
    consistency_score = max(0, min(1, consistency_score))
    
    # Compile all metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'clinical_accuracy': clinical_accuracy,
        'critical_detection_rate': critical_detection_rate,
        'resource_accuracy': resource_accuracy,
        'efficiency_impact': efficiency_impact,
        'consistency_score': consistency_score,
        'mean_absolute_percentage_error': np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true != 0) else np.inf
    }
    
    return metrics

def generate_workflow_insights(dataset, model_results, target_variable):
    """
    Generate actionable workflow optimization insights
    """
    insights = {
        'operational_insights': [],
        'resource_optimization': [],
        'quality_improvements': [],
        'cost_reduction_opportunities': [],
        'risk_mitigation': []
    }
    
    # Analyze feature importance for insights
    if 'feature_importance' in model_results:
        importance = model_results['feature_importance']
        top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for feature, importance_score in top_features:
            feature_lower = feature.lower()
            
            # Staffing insights
            if any(word in feature_lower for word in ['nurse', 'staff', 'specialist']):
                insights['operational_insights'].append({
                    'category': 'Staffing',
                    'insight': f'{feature} is a critical performance driver (importance: {importance_score:.3f})',
                    'recommendation': 'Optimize staffing levels and schedules based on demand patterns',
                    'impact': 'High',
                    'timeline': '2-4 weeks'
                })
                
                insights['resource_optimization'].append({
                    'category': 'Human Resources',
                    'opportunity': f'Staff allocation optimization via {feature}',
                    'potential_improvement': f'{importance_score * 100:.1f}% of performance variation',
                    'implementation_cost': 'Medium',
                    'roi_timeframe': '1-3 months'
                })
            
            # Capacity insights
            elif any(word in feature_lower for word in ['bed', 'capacity', 'facility']):
                insights['operational_insights'].append({
                    'category': 'Capacity Management',
                    'insight': f'{feature} significantly impacts {target_variable}',
                    'recommendation': 'Implement dynamic capacity allocation and monitoring',
                    'impact': 'Medium',
                    'timeline': '4-8 weeks'
                })
                
                insights['resource_optimization'].append({
                    'category': 'Physical Resources',
                    'opportunity': f'Capacity optimization through {feature}',
                    'potential_improvement': f'{importance_score * 100:.1f}% contribution to outcomes',
                    'implementation_cost': 'High',
                    'roi_timeframe': '3-6 months'
                })
            
            # Time-based insights
            elif any(word in feature_lower for word in ['time', 'hour', 'day']):
                insights['operational_insights'].append({
                    'category': 'Process Timing',
                    'insight': f'Temporal patterns in {feature} drive performance variations',
                    'recommendation': 'Implement time-based workflow optimization',
                    'impact': 'High',
                    'timeline': '1-2 weeks'
                })
    
    # Analyze predictions for quality insights
    if 'predictions' in model_results:
        predictions = model_results['predictions']
        y_true = model_results['y_true']
        
        # Quality improvement opportunities
        prediction_accuracy = model_results.get('test_r2', 0)
        
        if prediction_accuracy < 0.8:
            insights['quality_improvements'].append({
                'area': 'Model Accuracy',
                'current_state': f'{prediction_accuracy:.1%} accuracy',
                'target_state': '>80% accuracy',
                'actions': ['Collect additional data', 'Feature engineering', 'Model ensemble'],
                'priority': 'High'
            })
        
        # Identify high-variability areas
        residuals = y_true - predictions
        high_error_threshold = np.percentile(np.abs(residuals), 90)
        high_error_cases = np.abs(residuals) > high_error_threshold
        
        if np.sum(high_error_cases) > len(predictions) * 0.1:
            insights['quality_improvements'].append({
                'area': 'Prediction Consistency',
                'current_state': f'{np.sum(high_error_cases)/len(predictions)*100:.1f}% high-error cases',
                'target_state': '<5% high-error cases',
                'actions': ['Identify outlier patterns', 'Improve data quality', 'Specialized models'],
                'priority': 'Medium'
            })
    
    # Cost reduction opportunities
    if 'wait' in target_variable.lower():
        # Wait time specific cost analysis
        if 'predictions' in model_results:
            avg_wait = np.mean(model_results['predictions'])
            
            insights['cost_reduction_opportunities'].append({
                'opportunity': 'Wait Time Reduction',
                'current_cost': f'Average wait time: {avg_wait:.1f} minutes',
                'savings_potential': 'High - reduces patient dissatisfaction and operational costs',
                'implementation': ['Process streamlining', 'Predictive scheduling', 'Resource optimization'],
                'estimated_savings': '15-25% reduction in operational inefficiencies'
            })
    
    # Risk mitigation strategies
    if 'predictions' in model_results:
        predictions = model_results['predictions']
        
        # Identify high-risk predictions
        risk_threshold = np.percentile(predictions, 95)
        high_risk_cases = np.sum(predictions > risk_threshold)
        risk_percentage = (high_risk_cases / len(predictions)) * 100
        
        if risk_percentage > 10:
            insights['risk_mitigation'].append({
                'risk': 'High-Risk Case Volume',
                'description': f'{risk_percentage:.1f}% of cases exceed risk threshold',
                'mitigation_strategies': [
                    'Implement early warning systems',
                    'Develop escalation protocols',
                    'Increase monitoring frequency'
                ],
                'priority': 'Critical'
            })
    
    return insights

def calculate_cost_impact(predictions, target_variable, cost_parameters=None):
    """
    Calculate cost impact and ROI projections
    """
    if cost_parameters is None:
        # Default cost parameters
        if 'wait' in target_variable.lower():
            cost_parameters = {
                'cost_per_unit': 2.5,  # Cost per minute
                'overtime_multiplier': 1.5,
                'annual_volume': len(predictions) * 52  # Assuming weekly data
            }
        else:
            cost_parameters = {
                'cost_per_unit': 100,  # Generic cost per unit
                'overtime_multiplier': 1.2,
                'annual_volume': len(predictions) * 12  # Assuming monthly data
            }
    
    cost_analysis = {}
    
    # Current cost calculation
    baseline_cost_per_case = np.mean(predictions) * cost_parameters['cost_per_unit']
    annual_baseline_cost = baseline_cost_per_case * cost_parameters['annual_volume']
    
    # High-cost cases
    high_cost_threshold = np.percentile(predictions, 80)
    high_cost_cases = predictions > high_cost_threshold
    high_cost_total = np.sum(predictions[high_cost_cases]) * cost_parameters['cost_per_unit']
    
    # Improvement scenarios
    improvement_scenarios = {}
    for improvement_pct in [5, 10, 15, 20]:
        improved_predictions = predictions * (1 - improvement_pct/100)
        improved_cost_per_case = np.mean(improved_predictions) * cost_parameters['cost_per_unit']
        annual_improved_cost = improved_cost_per_case * cost_parameters['annual_volume']
        annual_savings = annual_baseline_cost - annual_improved_cost
        
        improvement_scenarios[f'{improvement_pct}%'] = {
            'annual_savings': annual_savings,
            'cost_per_case_reduction': baseline_cost_per_case - improved_cost_per_case,
            'roi_percentage': improvement_pct * 2,  # Simplified ROI calculation
            'payback_period_months': max(1, 12 / (improvement_pct / 5))  # Simplified payback
        }
    
    cost_analysis = {
        'baseline_cost_per_case': baseline_cost_per_case,
        'annual_baseline_cost': annual_baseline_cost,
        'high_cost_cases_count': np.sum(high_cost_cases),
        'high_cost_total_impact': high_cost_total,
        'improvement_scenarios': improvement_scenarios,
        'cost_drivers': {
            'high_variability': np.std(predictions) / np.mean(predictions),
            'outlier_impact': len(predictions[predictions > np.percentile(predictions, 95)]) / len(predictions)
        }
    }
    
    return cost_analysis

def generate_healthcare_recommendations(best_model_data, comparison_df):
    """
    Generate comprehensive healthcare implementation recommendations
    """
    recommendations = {
        'primary': [],
        'considerations': [],
        'implementation_steps': [],
        'success_metrics': []
    }
    
    # Primary recommendations based on best model performance
    best_r2 = best_model_data['Test R² Score']
    best_model = best_model_data['Model']
    
    if best_r2 > 0.8:
        recommendations['primary'].append(
            f"Deploy {best_model} for production use - high accuracy ({best_r2:.1%}) indicates reliable predictions"
        )
        recommendations['primary'].append(
            "Implement real-time prediction pipeline for operational decision support"
        )
    elif best_r2 > 0.6:
        recommendations['primary'].append(
            f"Use {best_model} for decision support with human oversight - moderate accuracy requires validation"
        )
        recommendations['primary'].append(
            "Focus on data quality improvements to enhance model performance"
        )
    else:
        recommendations['primary'].append(
            "Improve data collection and feature engineering before deployment"
        )
        recommendations['primary'].append(
            "Consider ensemble methods or alternative modeling approaches"
        )
    
    # Model-specific recommendations
    if best_model == 'Random Forest':
        recommendations['primary'].append(
            "Leverage feature importance insights for operational improvements"
        )
    elif best_model == 'XGBoost':
        recommendations['primary'].append(
            "Implement gradient boosting pipeline for complex pattern recognition"
        )
    elif best_model == 'Linear Regression':
        recommendations['primary'].append(
            "Focus on linear relationships and interpretable predictions"
        )
    
    # Considerations based on model comparison
    model_count = len(comparison_df)
    performance_std = comparison_df['Test R² Score'].std()
    
    if performance_std > 0.1:
        recommendations['considerations'].append(
            "High variability between models suggests data quality or feature engineering issues"
        )
    
    if model_count > 1:
        best_3_models = comparison_df.head(3)['Model'].tolist()
        recommendations['considerations'].append(
            f"Consider ensemble of top models: {', '.join(best_3_models)}"
        )
    
    # Overfitting analysis
    avg_overfitting = comparison_df['Overfitting'].mean()
    if avg_overfitting > 0.1:
        recommendations['considerations'].append(
            "Models show signs of overfitting - implement cross-validation and regularization"
        )
    
    # Training time considerations
    max_training_time = comparison_df['Training Time'].max()
    if max_training_time > 60:  # More than 1 minute
        recommendations['considerations'].append(
            "Some models have long training times - consider computational resources for retraining"
        )
    
    # Implementation steps
    recommendations['implementation_steps'] = [
        {
            'phase': 'Phase 1: Pilot (2-4 weeks)',
            'actions': [
                'Deploy best model in test environment',
                'Validate predictions with historical data',
                'Train staff on new system'
            ]
        },
        {
            'phase': 'Phase 2: Limited Deployment (4-8 weeks)',
            'actions': [
                'Implement in single department/unit',
                'Monitor performance metrics',
                'Collect user feedback'
            ]
        },
        {
            'phase': 'Phase 3: Full Deployment (2-3 months)',
            'actions': [
                'Scale to entire organization',
                'Integrate with existing systems',
                'Establish monitoring and maintenance protocols'
            ]
        }
    ]
    
    # Success metrics
    recommendations['success_metrics'] = [
        f'Achieve >85% prediction accuracy',
        f'Reduce {comparison_df.iloc[0]["Target"]} by 15-25%',
        'Improve operational efficiency by 20%',
        'Maintain model performance over time',
        'User adoption rate >90%'
    ]
    
    return recommendations

def evaluate_model_fairness(predictions, actual, sensitive_attributes=None):
    """
    Evaluate model fairness across different groups (if sensitive attributes available)
    """
    fairness_metrics = {}
    
    if sensitive_attributes is None:
        fairness_metrics['note'] = 'No sensitive attributes provided for fairness analysis'
        return fairness_metrics
    
    for attr_name, attr_values in sensitive_attributes.items():
        attr_fairness = {}
        unique_groups = np.unique(attr_values)
        
        group_performances = {}
        for group in unique_groups:
            group_mask = attr_values == group
            if np.sum(group_mask) > 0:
                group_pred = predictions[group_mask]
                group_actual = actual[group_mask]
                
                group_mae = np.mean(np.abs(group_actual - group_pred))
                group_rmse = np.sqrt(np.mean((group_actual - group_pred) ** 2))
                
                group_performances[group] = {
                    'mae': group_mae,
                    'rmse': group_rmse,
                    'count': np.sum(group_mask)
                }
        
        # Calculate fairness metrics
        maes = [perf['mae'] for perf in group_performances.values()]
        rmses = [perf['rmse'] for perf in group_performances.values()]
        
        attr_fairness['group_performances'] = group_performances
        attr_fairness['mae_ratio'] = max(maes) / min(maes) if min(maes) > 0 else float('inf')
        attr_fairness['rmse_ratio'] = max(rmses) / min(rmses) if min(rmses) > 0 else float('inf')
        attr_fairness['fairness_score'] = 1 / max(attr_fairness['mae_ratio'], attr_fairness['rmse_ratio'])
        
        fairness_metrics[attr_name] = attr_fairness
    
    return fairness_metrics

def generate_alert_thresholds(predictions, target_variable, percentiles=[75, 85, 95]):
    """
    Generate intelligent alert thresholds based on prediction distribution
    """
    thresholds = {}
    
    # Calculate percentile-based thresholds
    threshold_values = np.percentile(predictions, percentiles)
    
    for i, percentile in enumerate(percentiles):
        threshold_value = threshold_values[i]
        
        if percentile <= 80:
            alert_level = 'Information'
            color = 'blue'
        elif percentile <= 90:
            alert_level = 'Warning'
            color = 'orange'
        else:
            alert_level = 'Critical'
            color = 'red'
        
        # Calculate expected alert frequency
        alert_frequency = (100 - percentile) / 100
        
        thresholds[alert_level] = {
            'threshold': threshold_value,
            'percentile': percentile,
            'expected_frequency': alert_frequency,
            'description': f'{alert_level} alert when {target_variable} exceeds {threshold_value:.1f}',
            'color': color,
            'response_time': '5 minutes' if alert_level == 'Critical' else '30 minutes'
        }
    
    return thresholds

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate healthcare predictions
    y_true = np.random.exponential(30, n_samples)  # Wait times
    y_pred = y_true + np.random.normal(0, 5, n_samples)  # Predictions with some error
    
    # Calculate healthcare metrics
    metrics = calculate_healthcare_metrics(y_true, y_pred, 'Total Wait Time (min)')
    
    print("Healthcare Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    # Generate alert thresholds
    thresholds = generate_alert_thresholds(y_pred, 'Total Wait Time (min)')
    
    print("\nAlert Thresholds:")
    for level, info in thresholds.items():
        print(f"{level}: {info['threshold']:.1f} (Expected frequency: {info['expected_frequency']:.1%})")
