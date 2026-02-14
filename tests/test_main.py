"""
Tests for AVD AI Monitoring PoC
"""
import pytest
import asyncio
from datetime import datetime

from src.ai_analysis import AIAnalysisEngine, RootCauseAnalysis
from src.predictive_analytics import PredictiveAnalytics
from src.sample_data import (
    generate_mock_host_health,
    generate_mock_pool_metrics,
    generate_historical_data
)


class TestAIAnalysis:
    """Tests for AI analysis engine"""
    
    def test_root_cause_analysis_creation(self):
        """Test creating RootCauseAnalysis object"""
        analysis = RootCauseAnalysis(
            issue_summary="Test issue",
            root_cause="Test cause",
            severity="High",
            impact="Test impact",
            recommended_actions=["Action 1", "Action 2"],
            prevention_tips=["Tip 1"],
            confidence_score=0.85
        )
        
        assert analysis.issue_summary == "Test issue"
        assert analysis.severity == "High"
        assert len(analysis.recommended_actions) == 2
        assert 0 <= analysis.confidence_score <= 1.0
    
    def test_analysis_to_dict(self):
        """Test converting analysis to dictionary"""
        analysis = RootCauseAnalysis(
            issue_summary="Test",
            root_cause="Cause",
            severity="Medium",
            impact="Impact",
            recommended_actions=[],
            prevention_tips=[],
            confidence_score=0.7
        )
        
        data = analysis.to_dict()
        assert isinstance(data, dict)
        assert 'timestamp' in data
        assert data['severity'] == "Medium"


class TestPredictiveAnalytics:
    """Tests for predictive analytics"""
    
    def test_predictor_initialization(self):
        """Test predictor initialization"""
        predictor = PredictiveAnalytics()
        assert predictor is not None
        assert predictor.is_trained == False
    
    def test_feature_preparation(self):
        """Test feature engineering"""
        predictor = PredictiveAnalytics()
        historical_data = generate_historical_data(days=3, samples_per_day=10)
        
        df = predictor.prepare_features(historical_data)
        
        assert len(df) == 30  # 3 days * 10 samples
        assert 'avg_cpu_usage' in df.columns
        assert 'total_sessions' in df.columns
    
    def test_model_training(self):
        """Test model training with sufficient data"""
        predictor = PredictiveAnalytics()
        historical_data = generate_historical_data(days=2, samples_per_day=10)
        
        success = predictor.train_models(historical_data)
        
        assert success == True
        assert predictor.is_trained == True
    
    def test_model_training_insufficient_data(self):
        """Test model training with insufficient data"""
        predictor = PredictiveAnalytics()
        historical_data = generate_historical_data(days=1, samples_per_day=5)
        
        success = predictor.train_models(historical_data)
        
        assert success == False
        assert predictor.is_trained == False
    
    def test_capacity_prediction_untrained(self):
        """Test capacity prediction without training"""
        predictor = PredictiveAnalytics()
        current_metrics = generate_mock_pool_metrics()
        
        prediction = predictor.predict_capacity(current_metrics)
        
        assert prediction is not None
        assert prediction.confidence < 0.5  # Low confidence when untrained
    
    def test_capacity_prediction_trained(self):
        """Test capacity prediction with trained model"""
        predictor = PredictiveAnalytics()
        historical_data = generate_historical_data(days=3, samples_per_day=10)
        predictor.train_models(historical_data)
        
        current_metrics = generate_mock_pool_metrics()
        prediction = predictor.predict_capacity(current_metrics, hours_ahead=4)
        
        assert prediction is not None
        assert prediction.predicted_sessions >= 0
        assert 0 <= prediction.predicted_cpu <= 100
        assert 0 <= prediction.predicted_memory <= 100
        assert prediction.confidence > 0.5
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        predictor = PredictiveAnalytics()
        historical_data = generate_historical_data(days=3, samples_per_day=10)
        predictor.train_models(historical_data)
        
        current_metrics = generate_mock_pool_metrics()
        anomaly_result = predictor.detect_anomalies(current_metrics)
        
        assert anomaly_result is not None
        assert isinstance(anomaly_result.is_anomaly, bool)
        assert anomaly_result.anomaly_score >= 0


class TestSampleData:
    """Tests for sample data generation"""
    
    def test_mock_host_health(self):
        """Test mock host health generation"""
        health = generate_mock_host_health("avd-host-01", "Available")
        
        assert health['host_name'] == "avd-host-01"
        assert health['status'] == "Available"
        assert 0 <= health['cpu_usage'] <= 100
        assert 0 <= health['memory_usage'] <= 100
        assert isinstance(health['errors'], list)
        assert isinstance(health['warnings'], list)
    
    def test_mock_pool_metrics(self):
        """Test mock pool metrics generation"""
        metrics = generate_mock_pool_metrics("test-pool", num_hosts=10)
        
        assert metrics['pool_name'] == "test-pool"
        assert metrics['total_hosts'] == 10
        assert metrics['healthy_hosts'] <= 10
        assert metrics['unhealthy_hosts'] >= 0
        assert 'timestamp' in metrics
    
    def test_historical_data_generation(self):
        """Test historical data generation"""
        historical = generate_historical_data(days=5, samples_per_day=12)
        
        assert len(historical) == 60  # 5 days * 12 samples
        assert all('timestamp' in record for record in historical)
        assert all('avg_cpu_usage' in record for record in historical)


@pytest.mark.asyncio
async def test_async_placeholder():
    """Placeholder async test"""
    await asyncio.sleep(0.01)
    assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

