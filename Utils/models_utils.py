import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import time
import warnings
warnings.filterwarnings('ignore')

class HealthcareMLModels:
    """
    Comprehensive ML model training and evaluation for healthcare datasets
    """
    
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        self.scalers = {}
        self.options = {
            'scale_features': True,
            'handle_outliers': False,
            'cv_folds': 5,
            'random_state': 42,
            'enable_tuning': False,
            'n_jobs': -1
        }
        
        # Initialize model configurations
        self._initialize_models()
    
    def set_options(self, **kwargs):
        """Set training options"""
        self.options.update(kwargs)
        self._initialize_models()  # Reinitialize with new options
    
    def _initialize_models(self):
        """Initialize ML models with default parameters"""
        random_state = self.options.get('random_state', 42)
        n_jobs = self.options.get('n_jobs', -1)
        
        self.models = {
            'Random Forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=n_jobs
            ),
            'XGBoost': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=n_jobs,
                verbosity=0
            ),
            'Linear Regression': LinearRegression(n_jobs=n_jobs),
            'Support Vector Regression': SVR(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                epsilon=0.1
            )
        }
        
        # Parameter grids for hyperparameter tuning
        self.param_grids = {
            'Random Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'Linear Regression': {},
            'Support Vector Regression': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'epsilon': [0.01, 0.1, 0.2]
            }
        }
    
    def _remove_outliers(self, X, y, method='iqr', threshold=1.5):
        """Remove outliers from training data"""
        if not self.options.get('handle_outliers', False):
            return X, y
        
        outlier_indices = set()
        
        # Check numerical columns for outliers
        for col in X.select_dtypes(include=[np.number]).columns:
            if method == 'iqr':
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)].index
            elif method == 'zscore':
                z_scores = np.abs((X[col] - X[col].mean()) / X[col].std())
                outliers = X[z_scores > threshold].index
            
            outlier_indices.update(outliers)
        
        # Remove outliers
        if len(outlier_indices) > 0 and len(outlier_indices) < len(X) * 0.1:  # Remove only if < 10%
            clean_indices = X.index.difference(list(outlier_indices))
            return X.loc[clean_indices], y.loc[clean_indices]
        
        return X, y
    
    def _scale_features(self, X_train, X_test):
        """Scale features if requested"""
        if not self.options.get('scale_features', True):
            return X_train, X_test,None
        
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        return X_train_scaled, X_test_scaled, scaler
    
    def train_single_model(self, model_name, X_train, X_test, y_train, y_test):
        """Train a single model and return comprehensive results"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        start_time = time.time()
        
        # Get model
        model = self.models[model_name]
        
        # Handle outliers
        X_train_clean, y_train_clean = self._remove_outliers(X_train, y_train)
        
        # Scale features for models that benefit from scaling
        if model_name in ['Support Vector Regression', 'Linear Regression'] or self.options.get('scale_features', True):
            X_train_scaled, X_test_scaled, scaler = self._scale_features(X_train_clean, X_test)
            self.scalers[model_name] = scaler
        else:
            X_train_scaled, X_test_scaled = X_train_clean, X_test
            self.scalers[model_name] = None
        
        # Hyperparameter tuning
        if self.options.get('enable_tuning', False) and model_name in self.param_grids and self.param_grids[model_name]:
            try:
                grid_search = GridSearchCV(
                    model,
                    self.param_grids[model_name],
                    cv=min(self.options.get('cv_folds', 5), len(X_train_scaled)),
                    scoring='r2',
                    n_jobs=self.options.get('n_jobs', -1),
                    verbose=0
                )
                grid_search.fit(X_train_scaled, y_train_clean)
                model = grid_search.best_estimator_
                best_params = grid_search.best_params_
            except Exception as e:
                print(f"Grid search failed for {model_name}: {e}")
                best_params = {}
        else:
            best_params = {}
        
        # Train model
        model.fit(X_train_scaled, y_train_clean)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        train_mse = mean_squared_error(y_train_clean, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train_clean, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train_clean,
                cv=min(self.options.get('cv_folds', 5), len(X_train_scaled)),
                scoring='r2',
                n_jobs=self.options.get('n_jobs', -1)
            )
            cv_score_mean = cv_scores.mean()
            cv_score_std = cv_scores.std()
        except Exception as e:
            print(f"Cross-validation failed for {model_name}: {e}")
            cv_scores = np.array([test_r2])
            cv_score_mean = test_r2
            cv_score_std = 0
        
        # Feature importance (for tree-based models)
        feature_importance = {}
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, model.feature_importances_))
        elif hasattr(model, 'coef_') and len(model.coef_) == len(X_train.columns):
            # For linear models, use absolute coefficients
            feature_importance = dict(zip(X_train.columns, np.abs(model.coef_)))
        
        training_time = time.time() - start_time
        
        # Store trained model
        self.trained_models[model_name] = {
            'model': model,
            'scaler': self.scalers[model_name],
            'feature_names': X_train.columns.tolist()
        }
        
        # Return comprehensive results
        results = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_mae': test_mae,
            'cv_scores': cv_scores,
            'cv_score_mean': cv_score_mean,
            'cv_score_std': cv_score_std,
            'feature_importance': feature_importance,
            'best_params': best_params,
            'training_time': training_time,
            'predictions': y_test_pred,
            'y_true': y_test
        }
        
        return results
    
    def train_multiple_models(self, X_train, X_test, y_train, y_test, model_names=None):
        """Train multiple models and return comparison results"""
        if model_names is None:
            model_names = list(self.models.keys())
        
        results = {}
        
        for model_name in model_names:
            if model_name in self.models:
                try:
                    print(f"Training {model_name}...")
                    model_results = self.train_single_model(model_name, X_train, X_test, y_train, y_test)
                    results[model_name] = model_results
                    print(f"✅ {model_name} completed - R²: {model_results['test_r2']:.3f}")
                except Exception as e:
                    print(f"❌ Error training {model_name}: {e}")
                    continue
        
        return results
    
    def predict_with_model(self, model_name, X_new):
        """Make predictions with a trained model"""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model_info = self.trained_models[model_name]
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Ensure feature order matches training
        X_new = X_new[model_info['feature_names']]
        
        # Scale if necessary
        if scaler is not None:
            X_new_scaled = pd.DataFrame(
                scaler.transform(X_new),
                columns=X_new.columns,
                index=X_new.index
            )
        else:
            X_new_scaled = X_new
        
        predictions = model.predict(X_new_scaled)
        return predictions
    
    def get_model_comparison_summary(self, results):
        """Generate a summary comparison of model results"""
        if not results:
            return pd.DataFrame()
        
        summary_data = []
        for model_name, model_results in results.items():
            summary_data.append({
                'Model': model_name,
                'Test R²': model_results['test_r2'],
                'Test RMSE': np.sqrt(model_results['test_mse']),
                'Test MAE': model_results['test_mae'],
                'CV Score (mean)': model_results['cv_score_mean'],
                'CV Score (std)': model_results['cv_score_std'],
                'Training Time (s)': model_results['training_time'],
                'Overfitting': model_results['train_r2'] - model_results['test_r2']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test R²', ascending=False)
        
        return summary_df
    
    def evaluate_model_stability(self, model_name, X, y, n_runs=5):
        """Evaluate model stability across multiple runs"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        stability_results = {
            'r2_scores': [],
            'rmse_scores': [],
            'training_times': []
        }
        
        for run in range(n_runs):
            # Create different train/test splits
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=run
            )
            
            # Train model
            results = self.train_single_model(model_name, X_train, X_test, y_train, y_test)
            
            stability_results['r2_scores'].append(results['test_r2'])
            stability_results['rmse_scores'].append(np.sqrt(results['test_mse']))
            stability_results['training_times'].append(results['training_time'])
        
        # Calculate stability metrics
        stability_summary = {
            'r2_mean': np.mean(stability_results['r2_scores']),
            'r2_std': np.std(stability_results['r2_scores']),
            'rmse_mean': np.mean(stability_results['rmse_scores']),
            'rmse_std': np.std(stability_results['rmse_scores']),
            'time_mean': np.mean(stability_results['training_times']),
            'time_std': np.std(stability_results['training_times']),
            'consistency_score': 1 - (np.std(stability_results['r2_scores']) / np.mean(stability_results['r2_scores']))
        }
        
        return stability_summary, stability_results

def create_ensemble_model(trained_models, weights=None):
    """Create an ensemble model from multiple trained models"""
    if not trained_models:
        raise ValueError("No trained models provided")
    
    if weights is None:
        # Equal weights
        weights = {name: 1.0 / len(trained_models) for name in trained_models.keys()}
    
    class EnsembleModel:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
        
        def predict(self, X):
            predictions = []
            for model_name, model_info in self.models.items():
                model = model_info['model']
                scaler = model_info['scaler']
                
                # Prepare features
                X_model = X[model_info['feature_names']]
                
                if scaler is not None:
                    X_scaled = pd.DataFrame(
                        scaler.transform(X_model),
                        columns=X_model.columns,
                        index=X_model.index
                    )
                else:
                    X_scaled = X_model
                
                pred = model.predict(X_scaled)
                predictions.append(pred * self.weights[model_name])
            
            return np.sum(predictions, axis=0)
    
    return EnsembleModel(trained_models, weights)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    from Utils.data_preprocessing import create_sample_healthcare_data
    
    # Generate sample data
    sample_data = create_sample_healthcare_data(1000)
    
    # Prepare features and target
    target_col = 'Total Wait Time (min)'
    feature_cols = [col for col in sample_data.columns if col != target_col and 'ID' not in col]
    
    X = sample_data[feature_cols].select_dtypes(include=[np.number])
    y = sample_data[target_col]
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize ML models
    ml_models = HealthcareMLModels()
    
    # Train models
    results = ml_models.train_multiple_models(X_train, X_test, y_train, y_test)
    
    # Get comparison summary
    summary = ml_models.get_model_comparison_summary(results)
    print("\nModel Comparison Summary:")
    print(summary)
    
    # Test ensemble
    ensemble = create_ensemble_model(ml_models.trained_models)
    ensemble_pred = ensemble.predict(X_test)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    print(f"\nEnsemble R² Score: {ensemble_r2:.3f}")
