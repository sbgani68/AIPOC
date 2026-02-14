# Quick Start Guide - AVD AI Monitoring PoC

## 🚀 Run Without Azure (Demo Mode)

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run tests to see functionality
pytest -v

# Or use the demo script
.\demo.ps1
```

## 🔧 Run With Azure Credentials

### Step 1: Create .env file
Copy and fill in your values:
```env
AZURE_SUBSCRIPTION_ID=your_subscription_id
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
AVD_RESOURCE_GROUP=your_resource_group
AVD_HOST_POOL_NAME=your_host_pool_name
AVD_WORKSPACE_ID=your_log_analytics_workspace_id
ANTHROPIC_API_KEY=your_anthropic_key_here
AI_PROVIDER=anthropic
```

### Step 2: Run the Application
```powershell
python main.py
# Choose option 1 for interactive demo
# Choose option 2 to start API server
```

## 📊 API Server Mode

```powershell
# Start server
uvicorn api:app --reload

# Access documentation
Start-Process http://localhost:8000/docs
```

### Try These Endpoints

```powershell
# Health check
curl http://localhost:8000/health

# Dashboard (requires Azure config)
curl http://localhost:8000/api/v1/dashboard

# Capacity prediction (requires historical data)
curl http://localhost:8000/api/v1/predict/capacity
```

## 🧪 Running Tests

```powershell
# All tests
pytest

# Verbose output
pytest -v

# With coverage
pytest --cov=src tests/
```

## 📝 Test AI Analysis (No Azure Required)

Create a test script `test_ai.py`:

```python
import asyncio
from src.ai_analysis import AIAnalysisEngine
from src.sample_data import generate_mock_host_health

async def test():
    # Initialize AI engine (requires API key in .env)
    ai = AIAnalysisEngine(provider="anthropic")
    
    # Generate mock data
    health_data = generate_mock_host_health("test-host", "Unavailable")
    
    # Get AI analysis
    analysis = await ai.analyze_host_health("test-host", health_data)
    
    print(f"Issue: {analysis.issue_summary}")
    print(f"Severity: {analysis.severity}")
    print(f"Actions: {analysis.recommended_actions}")

asyncio.run(test())
```

Run it:
```powershell
python test_ai.py
```

## 💡 Common Issues

### Python not found
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

### Module not found
```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Azure authentication errors
- Verify service principal has Reader role on AVD resources
- Check Log Analytics workspace permissions
- Ensure all credentials in .env are correct

## 🎯 What to Expect

- ✅ **Without Azure**: Tests run, sample data generation works
- ✅ **With Azure + No AI Key**: Monitoring works, no AI analysis
- ✅ **With Azure + AI Key**: Full functionality including predictions
