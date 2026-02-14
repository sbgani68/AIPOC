"""
FastAPI Web Interface for AVD AI Monitoring
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
from datetime import datetime
import logging
from contextlib import asynccontextmanager

from src.azure_avd_monitor import AVDMonitor, HostPoolMetrics, HostHealth
from src.ai_analysis import AIAnalysisEngine, RootCauseAnalysis
from src.predictive_analytics import PredictiveAnalytics, CapacityPrediction, AnomalyDetection
from config import settings

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
avd_monitor: Optional[AVDMonitor] = None
ai_engine: Optional[AIAnalysisEngine] = None
predictor: Optional[PredictiveAnalytics] = None
historical_data: List[Dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global avd_monitor, ai_engine, predictor
    
    logger.info("Initializing AVD AI Monitoring System...")
    
    # Initialize Azure AVD Monitor
    if all([settings.azure_subscription_id, settings.azure_tenant_id,
            settings.azure_client_id, settings.azure_client_secret]):
        avd_monitor = AVDMonitor(
            subscription_id=settings.azure_subscription_id,
            tenant_id=settings.azure_tenant_id,
            client_id=settings.azure_client_id,
            client_secret=settings.azure_client_secret,
            workspace_id=settings.avd_workspace_id
        )
        logger.info("✓ Azure AVD Monitor initialized")
    else:
        logger.warning("⚠ Azure credentials not configured - monitor disabled")
    
    # Initialize AI Engine
    if settings.ai_provider == "openai" and settings.openai_api_key:
        ai_engine = AIAnalysisEngine(
            provider="openai",
            openai_key=settings.openai_api_key,
            openai_model=settings.openai_model
        )
        logger.info("✓ OpenAI Analysis Engine initialized")
    elif settings.ai_provider == "anthropic" and settings.anthropic_api_key:
        ai_engine = AIAnalysisEngine(
            provider="anthropic",
            anthropic_key=settings.anthropic_api_key,
            anthropic_model=settings.anthropic_model
        )
        logger.info("✓ Anthropic Analysis Engine initialized")
    else:
        logger.warning("⚠ AI provider not configured - analysis disabled")
    
    # Initialize Predictive Analytics
    if settings.enable_predictive_analytics:
        predictor = PredictiveAnalytics()
        logger.info("✓ Predictive Analytics initialized")
    
    logger.info("🚀 AVD AI Monitoring System ready!")
    
    yield
    
    logger.info("Shutting down AVD AI Monitoring System...")


# Create FastAPI app
app = FastAPI(
    title="AVD AI Monitoring & Diagnostics",
    description="Intelligent Azure Virtual Desktop monitoring with AI-powered analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "AVD AI Monitoring & Diagnostics",
        "version": "1.0.0",
        "status": "operational",
        "features": {
            "azure_monitoring": avd_monitor is not None,
            "ai_analysis": ai_engine is not None,
            "predictive_analytics": predictor is not None
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "avd_monitor": "enabled" if avd_monitor else "disabled",
            "ai_engine": "enabled" if ai_engine else "disabled",
            "predictor": "enabled" if predictor else "disabled"
        }
    }


@app.get("/api/v1/pool/metrics")
async def get_pool_metrics():
    """Get current host pool metrics"""
    if not avd_monitor:
        raise HTTPException(status_code=503, detail="AVD Monitor not configured")
    
    if not all([settings.avd_resource_group, settings.avd_host_pool_name]):
        raise HTTPException(status_code=400, detail="AVD configuration incomplete")
    
    try:
        pool_metrics, host_healths = await avd_monitor.get_pool_health(
            settings.avd_resource_group,
            settings.avd_host_pool_name
        )
        
        return {
            "pool_metrics": pool_metrics.to_dict(),
            "host_count": len(host_healths),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching pool metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/pool/hosts")
async def get_all_hosts():
    """Get health status for all hosts in the pool"""
    if not avd_monitor:
        raise HTTPException(status_code=503, detail="AVD Monitor not configured")
    
    try:
        pool_metrics, host_healths = await avd_monitor.get_pool_health(
            settings.avd_resource_group,
            settings.avd_host_pool_name
        )
        
        return {
            "hosts": [h.to_dict() for h in host_healths],
            "total_hosts": len(host_healths),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error fetching hosts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/host/{host_name}/health")
async def get_host_health(host_name: str):
    """Get detailed health status for a specific host"""
    if not avd_monitor:
        raise HTTPException(status_code=503, detail="AVD Monitor not configured")
    
    try:
        health = await avd_monitor.get_host_health(
            settings.avd_resource_group,
            settings.avd_host_pool_name,
            host_name
        )
        
        return health.to_dict()
    except Exception as e:
        logger.error(f"Error fetching host health: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/host/{host_name}/analyze")
async def analyze_host(host_name: str):
    """Get AI-powered analysis for a specific host"""
    if not avd_monitor:
        raise HTTPException(status_code=503, detail="AVD Monitor not configured")
    
    if not ai_engine:
        raise HTTPException(status_code=503, detail="AI Engine not configured")
    
    try:
        # Get host health data
        health = await avd_monitor.get_host_health(
            settings.avd_resource_group,
            settings.avd_host_pool_name,
            host_name
        )
        
        # Perform AI analysis
        analysis = await ai_engine.analyze_host_health(host_name, health.to_dict())
        
        return {
            "host_name": host_name,
            "health_status": health.status,
            "analysis": analysis.to_dict(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing host: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/pool/analyze")
async def analyze_pool():
    """Get comprehensive pool analysis with AI insights"""
    if not avd_monitor or not ai_engine:
        raise HTTPException(status_code=503, detail="Required services not configured")
    
    try:
        # Get pool health
        pool_metrics, host_healths = await avd_monitor.get_pool_health(
            settings.avd_resource_group,
            settings.avd_host_pool_name
        )
        
        # Store for historical trending
        historical_data.append(pool_metrics.to_dict())
        if len(historical_data) > 100:  # Keep last 100 snapshots
            historical_data.pop(0)
        
        # Analyze trends
        trend_analysis = await ai_engine.analyze_pool_trends(
            pool_metrics.to_dict(),
            historical_data
        )
        
        return {
            "pool_metrics": pool_metrics.to_dict(),
            "trend_analysis": trend_analysis,
            "unhealthy_hosts": [h.to_dict() for h in host_healths if h.status != 'Available'],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error analyzing pool: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/predict/capacity")
async def predict_capacity(hours_ahead: int = 4):
    """Predict future capacity needs"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictive analytics not enabled")
    
    if not avd_monitor:
        raise HTTPException(status_code=503, detail="AVD Monitor not configured")
    
    try:
        # Train model if we have enough data and not already trained
        if len(historical_data) >= 10 and not predictor.is_trained:
            predictor.train_models(historical_data)
        
        # Get current metrics
        pool_metrics, _ = await avd_monitor.get_pool_health(
            settings.avd_resource_group,
            settings.avd_host_pool_name
        )
        
        # Make prediction
        prediction = predictor.predict_capacity(pool_metrics.to_dict(), hours_ahead)
        
        return {
            "current_metrics": pool_metrics.to_dict(),
            "prediction": prediction.to_dict(),
            "model_trained": predictor.is_trained,
            "training_samples": len(historical_data)
        }
    except Exception as e:
        logger.error(f"Error predicting capacity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/detect/anomalies")
async def detect_anomalies():
    """Detect anomalies in current metrics"""
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictive analytics not enabled")
    
    if not avd_monitor:
        raise HTTPException(status_code=503, detail="AVD Monitor not configured")
    
    try:
        # Train model if needed
        if len(historical_data) >= 10 and not predictor.is_trained:
            predictor.train_models(historical_data)
        
        # Get current metrics
        pool_metrics, _ = await avd_monitor.get_pool_health(
            settings.avd_resource_group,
            settings.avd_host_pool_name
        )
        
        # Detect anomalies
        anomaly_result = predictor.detect_anomalies(pool_metrics.to_dict())
        
        return {
            "anomaly_detection": anomaly_result.to_dict(),
            "current_metrics": pool_metrics.to_dict()
        }
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dashboard")
async def get_dashboard():
    """Get comprehensive dashboard data"""
    if not avd_monitor:
        raise HTTPException(status_code=503, detail="AVD Monitor not configured")
    
    try:
        # Get pool data
        pool_metrics, host_healths = await avd_monitor.get_pool_health(
            settings.avd_resource_group,
            settings.avd_host_pool_name
        )
        
        # Store historical data
        historical_data.append(pool_metrics.to_dict())
        if len(historical_data) > 100:
            historical_data.pop(0)
        
        response = {
            "pool_metrics": pool_metrics.to_dict(),
            "hosts": [h.to_dict() for h in host_healths],
            "alerts": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add AI analysis if available
        if ai_engine:
            trend_analysis = await ai_engine.analyze_pool_trends(
                pool_metrics.to_dict(),
                historical_data[-10:]  # Last 10 snapshots
            )
            response["trend_analysis"] = trend_analysis
        
        # Add predictions if available
        if predictor and len(historical_data) >= 10:
            if not predictor.is_trained:
                predictor.train_models(historical_data)
            
            prediction = predictor.predict_capacity(pool_metrics.to_dict())
            anomaly = predictor.detect_anomalies(pool_metrics.to_dict())
            
            response["capacity_forecast"] = prediction.to_dict()
            response["anomaly_detection"] = anomaly.to_dict()
            
            if prediction.capacity_alert:
                response["alerts"].append({
                    "type": "capacity",
                    "severity": "warning",
                    "message": prediction.alert_reason
                })
            
            if anomaly.is_anomaly:
                response["alerts"].append({
                    "type": "anomaly",
                    "severity": "critical",
                    "message": anomaly.description
                })
        
        return response
        
    except Exception as e:
        logger.error(f"Error fetching dashboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
