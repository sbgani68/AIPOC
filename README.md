# AI-Powered AVD Host Pool Monitoring & Diagnostics

An intelligent Azure Virtual Desktop (AVD) monitoring system that uses AI to provide real-time diagnostics, root cause analysis, and predictive insights for host pool health.

# AI-Powered AVD Host Pool Monitoring & Diagnostics

An intelligent Azure Virtual Desktop (AVD) monitoring system that uses AI to provide real-time diagnostics, root cause analysis, and predictive insights for host pool health.

## 🌟 Features

### 1. **Azure AVD Monitoring**
- Real-time host pool metrics collection
- Individual session host health tracking
- Performance metrics (CPU, Memory, Disk usage)
- Error and warning event aggregation
- Session count and capacity monitoring

### 2. **AI-Powered Root Cause Analysis**
- Intelligent error pattern recognition
- Human-readable explanations using GPT-4 or Claude
- Automated severity assessment
- Recommended actions and prevention tips
- Confidence scoring for reliability

### 3. **Predictive Analytics**
- Machine learning-based capacity forecasting
- Anomaly detection in metrics
- Resource utilization predictions
- Proactive alerting for capacity issues
- Historical trend analysis

### 4. **REST API Interface**
- FastAPI-based web service
- Interactive API documentation
- Real-time dashboard endpoint
- Comprehensive health monitoring
- Easy integration with other tools

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Web Interface                     │
│              (REST API + Interactive Docs)                   │
└─────────────────┬──────────────┬────────────────────────────┘
                  │              │
        ┌─────────▼────┐  ┌──────▼──────────┐
        │ Azure AVD    │  │  AI Analysis    │
        │ Monitor      │  │  Engine         │
        │              │  │  (GPT-4/Claude) │
        └──────┬───────┘  └─────────────────┘
               │
        ┌──────▼──────────────────────┐
        │ Predictive Analytics        │
        │ (Random Forest + Isolation  │
        │  Forest ML Models)          │
        └─────────────────────────────┘
```

## 📋 Prerequisites

- Python 3.9 or higher ✅ (Installed: 3.12.10)
- Visual Studio Code with Python extension
- GitHub Copilot (optional but recommended)
- Azure subscription with AVD host pool
- OpenAI API key or Anthropic API key

## 🚀 Quick Start

### 1. Environment Setup (Already Complete ✅)

The virtual environment is already set up with all dependencies:

```powershell
# Activate the environment
.\venv\Scripts\Activate.ps1

# Verify installation
python -c "import openai, anthropic, pandas, sklearn; print('✓ All packages installed')"
```

### 2. Configure Azure & AI Credentials

Create a `.env` file in the root directory:

```env
# Azure Credentials
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret

# Azure AVD Configuration
AVD_RESOURCE_GROUP=your_resource_group
AVD_HOST_POOL_NAME=your_host_pool_name
AVD_WORKSPACE_ID=your_log_analytics_workspace_id

# AI API Keys (choose one or both)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# AI Model Configuration
AI_PROVIDER=anthropic  # Options: openai, anthropic
OPENAI_MODEL=gpt-4
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
ENABLE_PREDICTIVE_ANALYTICS=true
CACHE_TTL_SECONDS=300
```

### 3. Run the Application

**Option A: Interactive Demo**
```powershell
python main.py
# Select option 1 for demo
```

**Option B: Start API Server**
```powershell
# Method 1: Through main.py
python main.py
# Select option 2

# Method 2: Direct uvicorn
uvicorn api:app --reload

# Access API documentation
# Open browser: http://localhost:8000/docs
```

### 4. Test the API

```powershell
# Get pool health metrics
curl http://localhost:8000/api/v1/pool/metrics

# Get comprehensive dashboard
curl http://localhost:8000/api/v1/dashboard

# Analyze specific host
curl -X POST http://localhost:8000/api/v1/host/avd-host-01/analyze

# Get capacity predictions
curl http://localhost:8000/api/v1/predict/capacity?hours_ahead=4

# Detect anomalies
curl http://localhost:8000/api/v1/detect/anomalies
```

## 📚 API Endpoints

### Core Monitoring
- `GET /` - API information
- `GET /health` - Service health check
- `GET /api/v1/pool/metrics` - Host pool metrics
- `GET /api/v1/pool/hosts` - All host health status
- `GET /api/v1/host/{host_name}/health` - Single host details

### AI Analysis
- `POST /api/v1/host/{host_name}/analyze` - AI root cause analysis
- `POST /api/v1/pool/analyze` - Pool-wide trend analysis

### Predictive Analytics
- `GET /api/v1/predict/capacity` - Capacity forecasting
- `GET /api/v1/detect/anomalies` - Anomaly detection

### Dashboard
- `GET /api/v1/dashboard` - Comprehensive dashboard data

## 🧪 Running Tests

```powershell
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_main.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📊 Sample Output

### AI Root Cause Analysis Example

```json
{
  "host_name": "avd-host-01",
  "analysis": {
    "issue_summary": "Host experiencing high memory utilization with FSLogix profile attachment failures",
    "root_cause": "The session host is running out of available memory (85% utilization) causing FSLogix Profile Container attachments to fail...",
    "severity": "High",
    "impact": "Users unable to access personalized profiles, ~30% login failures",
    "recommended_actions": [
      "Immediately restart the affected session host to clear memory",
      "Increase VM memory to at least 16GB",
      "Investigate running processes for memory leak source"
    ],
    "prevention_tips": [
      "Implement automated memory monitoring with alerts at 75% threshold",
      "Configure FSLogix Cloud Cache for redundancy",
      "Enable Application Insights for proactive detection"
    ],
    "confidence_score": 0.92
  }
}
```

### Capacity Prediction Example

```json
{
  "prediction": {
    "predicted_sessions": 65,
    "predicted_cpu": 78.5,
    "predicted_memory": 82.0,
    "capacity_alert": true,
    "alert_reason": "High resource utilization predicted",
    "recommended_hosts": 7,
    "confidence": 0.85,
    "forecast_timestamp": "2026-02-14T18:00:00Z"
  }
}
```

## 🔧 Development Workflow

### Using GitHub Copilot

1. **Code Generation**: Use Copilot to generate function implementations
2. **Documentation**: Let Copilot write docstrings and comments
3. **Testing**: Generate test cases with Copilot suggestions
4. **Refactoring**: Get improvement suggestions

### Project Structure

```
AIPOC/
├── src/                          # Source code
│   ├── azure_avd_monitor.py     # Azure AVD monitoring
│   ├── ai_analysis.py            # AI analysis engine
│   ├── predictive_analytics.py   # ML predictions
│   └── sample_data.py            # Mock data for testing
├── tests/                        # Test files
│   └── test_main.py             # Unit tests
├── api.py                        # FastAPI application
├── main.py                       # Main entry point
├── config.py                     # Configuration management
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (create this)
└── README.md                     # This file
```

## 🎯 Use Cases

1. **Real-time Monitoring**: Track AVD host pool health in real-time
2. **Troubleshooting**: Get AI-powered insights when issues occur
3. **Capacity Planning**: Predict future resource needs
4. **Proactive Alerts**: Detect anomalies before they impact users
5. **Documentation**: Auto-generate incident reports with AI analysis

## 🛠️ Customization

### Adding Custom Metrics

Edit `src/azure_avd_monitor.py`:
```python
async def query_custom_metrics(self, host_name: str) -> Dict:
    query = """
    YourCustomTable
    | where Computer == "{host_name}"
    | your custom KQL query
    """
    # Add your implementation
```

### Custom AI Prompts

Edit `src/ai_analysis.py` - `_build_analysis_prompt()` method to customize analysis focus.

### ML Model Tuning

Edit `src/predictive_analytics.py`:
```python
self.cpu_model = RandomForestRegressor(
    n_estimators=200,  # Increase for better accuracy
    max_depth=10,      # Adjust based on data
    random_state=42
)
```

## 📝 Next Steps

1. ✅ Set up `.env` with your credentials
2. ✅ Run the demo to verify connectivity
3. ✅ Start collecting historical data (run API server)
4. ✅ Train ML models (requires 10+ data samples)
5. ✅ Build custom dashboards using API data
6. ✅ Set up automated monitoring and alerting

## 🔒 Security Notes

- Never commit `.env` file to version control
- Use Azure Key Vault for production secrets
- Implement proper authentication for API endpoints
- Follow Azure RBAC best practices for service principal

## 📖 Additional Resources

- [Azure Virtual Desktop Documentation](https://docs.microsoft.com/azure/virtual-desktop/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Anthropic Claude Documentation](https://docs.anthropic.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - feel free to use this for your projects!
