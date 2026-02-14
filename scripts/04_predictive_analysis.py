"""
04 - Predictive Analysis
Forecast future capacity needs and performance issues
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os


class PredictiveAnalyzer:
    """Predict future AVD capacity and performance"""
    
    def __init__(self):
        self.data_dir = "data"
        self.models = {}
    
    def load_data(self):
        """Load data with anomalies"""
        path = os.path.join(self.data_dir, "data_with_anomalies.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"✓ Loaded data: {df.shape}")
            return df
        else:
            print("✗ Data not found. Run 03_detection.py first.")
            return None
    
    def prepare_features(self, df):
        """Engineer features for prediction"""
        df = df.copy()
        
        # Time-based features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Lag features
        for col in ['TotalConnections', 'SuccessRate', 'ActiveSessions']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1)
                df[f'{col}_lag24'] = df[col].shift(24)  # Same hour yesterday
        
        # Rolling statistics
        for col in ['TotalConnections', 'ActiveSessions']:
            if col in df.columns:
                df[f'{col}_rolling_mean'] = df[col].rolling(window=6, min_periods=1).mean()
                df[f'{col}_rolling_std'] = df[col].rolling(window=6, min_periods=1).std()
        
        return df.dropna()
    
    def train_capacity_model(self, df):
        """Train model to predict session capacity needs"""
        if 'ActiveSessions' not in df.columns:
            print("⚠ ActiveSessions not available")
            return None
        
        # Prepare data
        feature_cols = [col for col in df.columns if col not in [
            'ActiveSessions', 'IsAnomaly', 'AnomalyScore',
            'MaxSessionLimit', 'UtilizationPct'
        ]]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols]
        y = df['ActiveSessions']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Capacity Model Trained")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.3f}")
        
        self.models['capacity'] = model
        return model, X_test, y_test, y_pred
    
    def train_connection_success_model(self, df):
        """Train model to predict connection success rate"""
        if 'SuccessRate' not in df.columns:
            print("⚠ SuccessRate not available")
            return None
        
        # Prepare data
        feature_cols = [col for col in df.columns if col not in [
            'SuccessRate', 'IsAnomaly', 'AnomalyScore', 'TotalConnections'
        ]]
        feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols]
        y = df['SuccessRate']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\n✓ Connection Success Model Trained")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.3f}")
        
        self.models['connection_success'] = model
        return model, X_test, y_test, y_pred
    
    def forecast_next_24h(self, df, model_name='capacity'):
        """Forecast next 24 hours"""
        if model_name not in self.models:
            print(f"⚠ Model '{model_name}' not trained")
            return None
        
        # Use last known data point as baseline
        last_row = df.iloc[-1:].copy()
        
        predictions = []
        for hour_ahead in range(1, 25):
            # Update time features
            last_row['hour'] = (last_row['hour'] + 1) % 24
            
            # Predict
            pred = self.models[model_name].predict(last_row)
            predictions.append(pred[0])
        
        return predictions
    
    def visualize_predictions(self, y_test, y_pred, title):
        """Visualize prediction results"""
        plt.figure(figsize=(12, 6))
        
        plt.plot(y_test.index, y_test.values, label='Actual', alpha=0.7)
        plt.plot(y_test.index, y_pred, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.data_dir, f"{title.replace(' ', '_').lower()}.png")
        plt.savefig(output_path, dpi=300)
        print(f"✓ Saved: {output_path}")
    
    def run(self):
        """Run complete predictive analysis"""
        print("Starting predictive analysis...\n")
        
        # Load data
        df = self.load_data()
        if df is None:
            return
        
        # Prepare features
        df = self.prepare_features(df)
        
        # Train models
        capacity_results = self.train_capacity_model(df)
        if capacity_results:
            model, X_test, y_test, y_pred = capacity_results
            self.visualize_predictions(y_test, y_pred, "Capacity Prediction")
        
        connection_results = self.train_connection_success_model(df)
        if connection_results:
            model, X_test, y_test, y_pred = connection_results
            self.visualize_predictions(y_test, y_pred, "Connection Success Prediction")
        
        # Forecast
        if 'capacity' in self.models:
            forecast = self.forecast_next_24h(df, 'capacity')
            print(f"\n📈 24-hour capacity forecast:")
            print(f"  Min: {min(forecast):.0f} sessions")
            print(f"  Max: {max(forecast):.0f} sessions")
            print(f"  Avg: {np.mean(forecast):.0f} sessions")
        
        print("\n✓ Predictive analysis complete!")


if __name__ == "__main__":
    analyzer = PredictiveAnalyzer()
    analyzer.run()
