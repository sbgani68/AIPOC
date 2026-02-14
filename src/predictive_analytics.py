"""
Predictive Analytics Module
Machine learning-based predictions for capacity planning and failure forecasting
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


@dataclass
class CapacityPrediction:
    """Capacity forecast"""
    predicted_sessions: int
    predicted_cpu: float
    predicted_memory: float
    capacity_alert: bool
    alert_reason: str
    recommended_hosts: int
    confidence: float
    forecast_timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['forecast_timestamp'] = self.forecast_timestamp.isoformat()
        return data


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    is_anomaly: bool
    anomaly_score: float
    affected_metrics: List[str]
    description: str
    timestamp: datetime
    
    def to_dict(self):
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class PredictiveAnalytics:
    """Machine learning-based predictive analytics for AVD"""
    
    def __init__(self):
        """Initialize ML models"""
        self.cpu_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.memory_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.session_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_features(self, historical_data: List[Dict]) -> pd.DataFrame:
        """
        Prepare features from historical data
        
        Args:
            historical_data: List of historical metric snapshots
        
        Returns:
            DataFrame with engineered features
        """
        if not historical_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(historical_data)
        
        # Convert timestamp if present
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 17)).astype(int)
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Rolling statistics
        for col in ['avg_cpu_usage', 'avg_memory_usage', 'total_sessions']:
            if col in df.columns:
                df[f'{col}_ma_3'] = df[col].rolling(window=3, min_periods=1).mean()
                df[f'{col}_ma_6'] = df[col].rolling(window=6, min_periods=1).mean()
                df[f'{col}_std_3'] = df[col].rolling(window=3, min_periods=1).std().fillna(0)
        
        # Lag features
        for col in ['avg_cpu_usage', 'avg_memory_usage', 'total_sessions']:
            if col in df.columns:
                df[f'{col}_lag1'] = df[col].shift(1).fillna(df[col].mean())
                df[f'{col}_lag2'] = df[col].shift(2).fillna(df[col].mean())
        
        # Fill any remaining NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
        
        return df
    
    def train_models(self, historical_data: List[Dict]) -> bool:
        """
        Train predictive models on historical data
        
        Args:
            historical_data: List of historical pool metrics
        
        Returns:
            True if training successful
        """
        try:
            if len(historical_data) < 10:
                logger.warning("Insufficient data for training (minimum 10 samples required)")
                return False
            
            df = self.prepare_features(historical_data)
            
            # Select feature columns (exclude target and timestamp)
            feature_cols = [col for col in df.columns if col not in [
                'avg_cpu_usage', 'avg_memory_usage', 'total_sessions', 'timestamp',
                'pool_name', 'critical_errors', 'warnings'
            ]]
            
            if not feature_cols:
                logger.error("No valid features found for training")
                return False
            
            X = df[feature_cols].values
            
            # Train individual models
            if 'avg_cpu_usage' in df.columns:
                y_cpu = df['avg_cpu_usage'].values
                self.cpu_model.fit(X, y_cpu)
            
            if 'avg_memory_usage' in df.columns:
                y_memory = df['avg_memory_usage'].values
                self.memory_model.fit(X, y_memory)
            
            if 'total_sessions' in df.columns:
                y_sessions = df['total_sessions'].values
                self.session_model.fit(X, y_sessions)
            
            # Train anomaly detector
            self.anomaly_detector.fit(X)
            self.scaler.fit(X)
            
            self.is_trained = True
            logger.info(f"Models trained successfully on {len(historical_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    def predict_capacity(self, current_metrics: Dict, 
                        hours_ahead: int = 4) -> CapacityPrediction:
        """
        Predict future capacity needs
        
        Args:
            current_metrics: Current pool metrics
            hours_ahead: Hours to forecast ahead
        
        Returns:
            Capacity prediction
        """
        if not self.is_trained:
            return self._get_fallback_prediction(current_metrics)
        
        try:
            # Prepare features from current metrics
            df = self.prepare_features([current_metrics])
            feature_cols = [col for col in df.columns if col not in [
                'avg_cpu_usage', 'avg_memory_usage', 'total_sessions', 'timestamp',
                'pool_name', 'critical_errors', 'warnings'
            ]]
            
            X = df[feature_cols].values
            
            # Make predictions
            pred_cpu = float(self.cpu_model.predict(X)[0])
            pred_memory = float(self.memory_model.predict(X)[0])
            pred_sessions = int(self.session_model.predict(X)[0])
            
            # Adjust predictions based on time ahead
            growth_factor = 1 + (hours_ahead * 0.05)  # 5% growth per hour
            pred_sessions = int(pred_sessions * growth_factor)
            pred_cpu = min(pred_cpu * growth_factor, 100.0)
            pred_memory = min(pred_memory * growth_factor, 100.0)
            
            # Capacity alerting logic
            capacity_alert = False
            alert_reason = ""
            recommended_hosts = current_metrics.get('total_hosts', 0)
            
            if pred_cpu > 80 or pred_memory > 85:
                capacity_alert = True
                alert_reason = "High resource utilization predicted"
                recommended_hosts = int(recommended_hosts * 1.3)
            
            if pred_sessions > (current_metrics.get('available_capacity', 100) * 0.9):
                capacity_alert = True
                alert_reason = "Session capacity threshold approaching"
                recommended_hosts = max(recommended_hosts, int(pred_sessions / 10) + 1)
            
            # Calculate confidence based on model performance
            confidence = 0.75  # Default confidence
            
            return CapacityPrediction(
                predicted_sessions=pred_sessions,
                predicted_cpu=pred_cpu,
                predicted_memory=pred_memory,
                capacity_alert=capacity_alert,
                alert_reason=alert_reason,
                recommended_hosts=recommended_hosts,
                confidence=confidence,
                forecast_timestamp=datetime.utcnow() + timedelta(hours=hours_ahead)
            )
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return self._get_fallback_prediction(current_metrics)
    
    def detect_anomalies(self, current_metrics: Dict) -> AnomalyDetection:
        """
        Detect anomalies in current metrics
        
        Args:
            current_metrics: Current pool metrics
        
        Returns:
            Anomaly detection result
        """
        if not self.is_trained:
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_score=0.0,
                affected_metrics=[],
                description="Anomaly detection not available - insufficient training data",
                timestamp=datetime.utcnow()
            )
        
        try:
            # Prepare features
            df = self.prepare_features([current_metrics])
            feature_cols = [col for col in df.columns if col not in [
                'avg_cpu_usage', 'avg_memory_usage', 'total_sessions', 'timestamp',
                'pool_name', 'critical_errors', 'warnings'
            ]]
            
            X = df[feature_cols].values
            X_scaled = self.scaler.transform(X)
            
            # Detect anomalies
            prediction = self.anomaly_detector.predict(X_scaled)[0]
            anomaly_score = -self.anomaly_detector.score_samples(X_scaled)[0]
            
            is_anomaly = (prediction == -1)
            
            # Identify affected metrics
            affected_metrics = []
            if current_metrics.get('avg_cpu_usage', 0) > 90:
                affected_metrics.append('CPU Usage')
            if current_metrics.get('avg_memory_usage', 0) > 90:
                affected_metrics.append('Memory Usage')
            if current_metrics.get('critical_errors', 0) > 10:
                affected_metrics.append('Critical Errors')
            
            description = "Normal operation"
            if is_anomaly:
                if affected_metrics:
                    description = f"Anomaly detected in: {', '.join(affected_metrics)}"
                else:
                    description = "Unusual pattern detected in metrics"
            
            return AnomalyDetection(
                is_anomaly=is_anomaly,
                anomaly_score=float(anomaly_score),
                affected_metrics=affected_metrics,
                description=description,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return AnomalyDetection(
                is_anomaly=False,
                anomaly_score=0.0,
                affected_metrics=[],
                description=f"Anomaly detection failed: {str(e)}",
                timestamp=datetime.utcnow()
            )
    
    def _get_fallback_prediction(self, current_metrics: Dict) -> CapacityPrediction:
        """Fallback prediction when models aren't trained"""
        return CapacityPrediction(
            predicted_sessions=current_metrics.get('total_sessions', 0),
            predicted_cpu=current_metrics.get('avg_cpu_usage', 0.0),
            predicted_memory=current_metrics.get('avg_memory_usage', 0.0),
            capacity_alert=False,
            alert_reason="Predictive models not trained - using current values",
            recommended_hosts=current_metrics.get('total_hosts', 0),
            confidence=0.3,
            forecast_timestamp=datetime.utcnow()
        )
