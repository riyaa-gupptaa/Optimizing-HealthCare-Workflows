import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class HealthcareDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for healthcare datasets
    """
    
    def __init__(self):
        self.preprocessor = None
        self.feature_names = None
        self.target_column = None
        self.options = {
            'missing_strategy': 'mean/mode',
            'normalize': True,
            'encode_categorical': True,
            'feature_engineering': True,
            'outlier_detection': True,
            'feature_selection': False,
            'max_features': None
        }
    
    def set_options(self, **kwargs):
        """Set preprocessing options"""
        self.options.update(kwargs)
    
    def detect_column_types(self, df):
        """Automatically detect numerical and categorical columns"""
        numerical_cols = []
        categorical_cols = []
        datetime_cols = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (few unique values)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    categorical_cols.append(col)
                else:
                    numerical_cols.append(col)
            elif df[col].dtype == 'object':
                # Try to convert to datetime
                try:
                    pd.to_datetime(df[col].dropna().iloc[:100])
                    datetime_cols.append(col)
                except:
                    categorical_cols.append(col)
            elif df[col].dtype == 'datetime64[ns]':
                datetime_cols.append(col)
            else:
                categorical_cols.append(col)
        
        return numerical_cols, categorical_cols, datetime_cols
    
    def engineer_features(self, df):
        """Create healthcare-specific engineered features"""
        df_eng = df.copy()
        
        # Date/time feature engineering
        date_columns = df_eng.select_dtypes(include=['datetime64']).columns
        for col in date_columns:
            if col in df_eng.columns:
                df_eng[f'{col}_hour'] = df_eng[col].dt.hour
                df_eng[f'{col}_day_of_week'] = df_eng[col].dt.dayofweek
                df_eng[f'{col}_month'] = df_eng[col].dt.month
                df_eng[f'{col}_quarter'] = df_eng[col].dt.quarter
                df_eng[f'{col}_is_weekend'] = df_eng[col].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Drop original datetime column
                df_eng = df_eng.drop(columns=[col])
        
        # Healthcare-specific feature engineering
        numerical_cols = df_eng.select_dtypes(include=[np.number]).columns
        
        # Ratio features
        if 'Nurse-to-Patient Ratio' in df_eng.columns and 'Specialist Availability' in df_eng.columns:
            df_eng['Staff_Efficiency_Ratio'] = df_eng['Nurse-to-Patient Ratio'] / (df_eng['Specialist Availability'] + 1e-6)
        
        if 'Facility Size (Beds)' in df_eng.columns and 'Specialist Availability' in df_eng.columns:
            df_eng['Beds_per_Specialist'] = df_eng['Facility Size (Beds)'] / (df_eng['Specialist Availability'] + 1e-6)
        
        # Time-based features
        time_columns = [col for col in df_eng.columns if 'time' in col.lower() and col in numerical_cols]
        if len(time_columns) >= 2:
            for i, col1 in enumerate(time_columns):
                for col2 in time_columns[i+1:]:
                    df_eng[f'{col1}_to_{col2}_ratio'] = df_eng[col1] / (df_eng[col2] + 1e-6)
        
        # Capacity utilization features
        if 'Facility Size (Beds)' in df_eng.columns:
            # Assuming patient volume can be estimated from other features
            capacity_related = [col for col in numerical_cols if any(word in col.lower() 
                               for word in ['patient', 'visit', 'case', 'admission'])]
            if capacity_related:
                df_eng['Capacity_Utilization'] = df_eng[capacity_related[0]] / df_eng['Facility Size (Beds)']
        
        # Urgency-based features
        if 'Urgency Level' in df_eng.columns:
            urgency_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
            df_eng['Urgency_Numeric'] = df_eng['Urgency Level'].map(urgency_mapping).fillna(2)
        
        # Region-based features (if exists)
        region_cols = [col for col in df_eng.columns if 'region' in col.lower()]
        if region_cols:
            # Create region complexity score based on average performance
            for col in region_cols:
                if self.target_column and self.target_column in df_eng.columns:
                    region_stats = df_eng.groupby(col)[self.target_column].agg(['mean', 'std']).fillna(0)
                    df_eng[f'{col}_complexity'] = df_eng[col].map(region_stats['std'])
        
        return df_eng
    
    def detect_outliers(self, df, method='iqr', threshold=1.5):
        """Detect outliers in numerical columns"""
        outlier_indices = set()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > threshold].index
            
            outlier_indices.update(outliers)
        
        return list(outlier_indices)
    
    def fit_transform(self, df, target_column=None):
        """Fit preprocessing pipeline and transform data"""
        self.target_column = target_column
        df_processed = df.copy()
        
        # Remove ID-like columns
        id_columns = []
        for col in df_processed.columns:
            if any(keyword in col.lower() for keyword in ['id', 'name']) and col != target_column:
                id_columns.append(col)
        
        df_processed = df_processed.drop(columns=id_columns, errors='ignore')
        
        # Feature engineering
        if self.options['feature_engineering']:
            df_processed = self.engineer_features(df_processed)
        
        # Detect column types
        numerical_cols, categorical_cols, datetime_cols = self.detect_column_types(df_processed)
        
        # Remove target column from features
        if target_column and target_column in numerical_cols:
            numerical_cols.remove(target_column)
        if target_column and target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        # Outlier detection and removal
        if self.options['outlier_detection'] and len(numerical_cols) > 0:
            outlier_indices = self.detect_outliers(df_processed[numerical_cols])
            if len(outlier_indices) > 0 and len(outlier_indices) < len(df_processed) * 0.1:  # Remove only if < 10%
                df_processed = df_processed.drop(outlier_indices)
        
        # Create preprocessing pipeline
        transformers = []
        
        # Numerical preprocessing
        if numerical_cols:
            if self.options['missing_strategy'] == 'mean/mode':
                num_strategy = 'mean'
            elif self.options['missing_strategy'] == 'median/mode':
                num_strategy = 'median'
            else:
                num_strategy = 'mean'
            
            num_pipeline = [('imputer', SimpleImputer(strategy=num_strategy))]
            
            if self.options['normalize']:
                num_pipeline.append(('scaler', StandardScaler()))
            
            transformers.append(('num', Pipeline(num_pipeline), numerical_cols))
        
        # Categorical preprocessing
        if categorical_cols and self.options['encode_categorical']:
            if self.options['missing_strategy'] == 'mean/mode':
                cat_strategy = 'most_frequent'
            else:
                cat_strategy = 'most_frequent'
            
            cat_pipeline = [
                ('imputer', SimpleImputer(strategy=cat_strategy)),
                ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]
            
            transformers.append(('cat', Pipeline(cat_pipeline), categorical_cols))
        
        # Create and fit preprocessor
        if transformers:
            self.preprocessor = ColumnTransformer(transformers, remainder='drop')
            
            # Prepare feature matrix
            feature_columns = numerical_cols + categorical_cols
            X = df_processed[feature_columns]
            
            # Fit and transform
            X_transformed = self.preprocessor.fit_transform(X)
            
            # Get feature names
            self.feature_names = self._get_feature_names(numerical_cols, categorical_cols)
            
            # Create final dataframe
            result_df = pd.DataFrame(X_transformed, columns=self.feature_names)
            
            # Add target column back if specified
            if target_column and target_column in df_processed.columns:
                target_data = df_processed[target_column].reset_index(drop=True)
                if len(target_data) == len(result_df):
                    result_df[target_column] = target_data.values
            
            # Feature selection if requested
            if self.options['feature_selection'] and target_column and target_column in result_df.columns:
                result_df = self._apply_feature_selection(result_df, target_column)
            
            return result_df, self.preprocessor
        else:
            # No transformation needed
            return df_processed, None
    
    def _get_feature_names(self, numerical_cols, categorical_cols):
        """Get feature names after transformation"""
        feature_names = []
        
        # Add numerical column names
        feature_names.extend(numerical_cols)
        
        # Add one-hot encoded categorical feature names
        if hasattr(self.preprocessor, 'named_transformers_') and 'cat' in self.preprocessor.named_transformers_:
            cat_transformer = self.preprocessor.named_transformers_['cat']
            if hasattr(cat_transformer, 'named_steps') and 'onehot' in cat_transformer.named_steps:
                onehot_encoder = cat_transformer.named_steps['onehot']
                if hasattr(onehot_encoder, 'get_feature_names_out'):
                    try:
                        cat_feature_names = onehot_encoder.get_feature_names_out(categorical_cols)
                        feature_names.extend(cat_feature_names)
                    except:
                        # Fallback for older sklearn versions
                        for i, col in enumerate(categorical_cols):
                            n_categories = len(onehot_encoder.categories_[i])
                            for j in range(n_categories - 1):  # -1 because of drop='first'
                                feature_names.append(f"{col}_{j}")
        
        return feature_names
    
    def _apply_feature_selection(self, df, target_column):
        """Apply feature selection to reduce dimensionality"""
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Determine number of features to select
        max_features = self.options.get('max_features', min(50, len(X.columns)))
        k = min(max_features, len(X.columns))
        
        # Apply SelectKBest
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create result dataframe
        result_df = pd.DataFrame(X_selected, columns=selected_features)
        result_df[target_column] = y.values
        
        return result_df
    
    def transform(self, df):
        """Transform new data using fitted preprocessor"""
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        df_processed = df.copy()
        
        # Apply same preprocessing steps
        if self.options['feature_engineering']:
            df_processed = self.engineer_features(df_processed)
        
        # Remove ID columns and target column
        id_columns = [col for col in df_processed.columns 
                     if any(keyword in col.lower() for keyword in ['id', 'name'])]
        df_processed = df_processed.drop(columns=id_columns, errors='ignore')
        
        if self.target_column and self.target_column in df_processed.columns:
            df_processed = df_processed.drop(columns=[self.target_column])
        
        # Transform using fitted preprocessor
        X_transformed = self.preprocessor.transform(df_processed)
        
        # Create result dataframe
        result_df = pd.DataFrame(X_transformed, columns=self.feature_names)
        
        return result_df

def create_sample_healthcare_data(n_samples=1000):
    """Create sample healthcare data for testing"""
    np.random.seed(42)
    
    # Generate synthetic healthcare data
    data = {
        'Visit ID': [f'V{i:05d}' for i in range(n_samples)],
        'Patient ID': [f'P{i:05d}' for i in range(n_samples)],
        'Hospital ID': np.random.choice(['H001', 'H002', 'H003', 'H004'], n_samples),
        'Visit Date': pd.date_range('2023-01-01', periods=n_samples, freq='H'),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'Day of Week': np.random.choice(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], n_samples),
        'Season': np.random.choice(['Spring', 'Summer', 'Fall', 'Winter'], n_samples),
        'Time of Day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples),
        'Urgency Level': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.4, 0.4, 0.2]),
        'Nurse-to-Patient Ratio': np.random.normal(0.25, 0.05, n_samples),
        'Specialist Availability': np.random.randint(1, 8, n_samples),
        'Facility Size (Beds)': np.random.choice([150, 200, 300, 500], n_samples),
        'Time to Registration (min)': np.random.exponential(5, n_samples),
        'Time to Triage (min)': np.random.exponential(10, n_samples),
        'Time to Medical Professional (min)': np.random.exponential(20, n_samples),
    }
    
    # Create target variable (Total Wait Time)
    urgency_multiplier = {'Low': 1.2, 'Medium': 1.0, 'High': 0.7}
    base_wait_time = (
        data['Time to Registration (min)'] + 
        data['Time to Triage (min)'] + 
        data['Time to Medical Professional (min)']
    )
    
    # Add complexity based on other factors
    complexity_factor = (
        (1 / np.array(data['Nurse-to-Patient Ratio'])) * 0.1 +
        (1 / np.array(data['Specialist Availability'])) * 0.2 +
        np.random.normal(1, 0.2, n_samples)
    )
    
    urgency_factor = np.array([urgency_multiplier[u] for u in data['Urgency Level']])
    
    data['Total Wait Time (min)'] = base_wait_time * complexity_factor * urgency_factor
    
    # Add some outcome variables
    data['Patient Satisfaction'] = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.2, 0.5, 0.3])
    data['Patient Outcome'] = np.random.choice(['Discharged', 'Admitted', 'Transferred'], n_samples, p=[0.7, 0.25, 0.05])
    
    return pd.DataFrame(data)

# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_data = create_sample_healthcare_data(1000)
    
    # Initialize preprocessor
    preprocessor = HealthcareDataPreprocessor()
    
    # Set options
    preprocessor.set_options(
        missing_strategy='mean/mode',
        normalize=True,
        encode_categorical=True,
        feature_engineering=True,
        outlier_detection=True
    )
    
    # Preprocess data
    processed_data, pipeline = preprocessor.fit_transform(sample_data, 'Total Wait Time (min)')
    
    print(f"Original shape: {sample_data.shape}")
    print(f"Processed shape: {processed_data.shape}")
    print(f"Features created: {len(processed_data.columns)}")
    print("\nProcessed columns:")
    print(processed_data.columns.tolist())
