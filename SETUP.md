# AVD AI PoC - Setup Guide

## Environment Configuration

Your PoC is pre-configured for:

**Host Pool:** `jswvdprd_ARTHUR_02`
- **Resource Group:** rg-jswvd-prod-arthur
- **Location:** North Europe
- **Type:** Pooled (BreadthFirst)
- **Max Sessions:** 16
- **Subscription:** NextGen Virtual Desktop prod

**Log Analytics Workspace:** `lw-jswvd-prod`
- **Workspace ID:** 33e031af-7d03-4590-a96f-a4458980e7b2
- **Resource Group:** rg-jswvd-prod-log-analytics
- **Retention:** 30 days

**AI Provider:** Local models via Ollama (no cloud API needed)

---

## Quick Start

### 1. Install Ollama (Local AI Models)

Download and install Ollama from: https://ollama.ai

```powershell
# After installation, pull the recommended model
ollama pull llama3.1:70b

# Or use a smaller/faster model
ollama pull llama3.1:8b
ollama pull mistral
```

**Verify Ollama is running:**
```powershell
curl http://localhost:11434
# Should return: "Ollama is running"
```

### 2. Install Python Dependencies

```powershell
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Test Azure Connection

```powershell
python scripts/01_data_collection.py
```

This collects comprehensive AVD logs from the **last 30 days** including:
- Network metrics (RTT, jitter, TCP fallback)
- Session host health (RDAgent, performance)
- Capacity (logon failures, queues)
- FSLogix (Event IDs 26/27/28/1012/1013, VHD errors)
- Disk pressure (C: drive, temp folders)
- Session hygiene (disconnected sessions)
- Client posture (versions, Teams optimization)

---

## Run Full Pipeline

Execute scripts in order:

```powershell
# 1. Collect data from Azure
python scripts/01_data_collection.py

# 2. Process and aggregate
python scripts/02_preprocess_aggregate.py

# 3. Detect anomalies with ML
python scripts/03_detection.py

# 4. Predictive forecasting
python scripts/04_predictive_analysis.py

# 5. AI root cause analysis (uses local Ollama)
python scripts/05_ai_root_cause.py

# 6. Generate remediation scripts
python scripts/06_remediation.py

# 7. Launch interactive dashboard
streamlit run scripts/07_dashboard.py
```

---

## Local AI Model Options

### Recommended Models (via Ollama)

| Model | Size | Speed | Quality | Command |
|-------|------|-------|---------|---------|
| llama3.1:70b | ~40GB | Slow | Excellent | `ollama pull llama3.1:70b` |
| llama3.1:8b | ~4.7GB | Fast | Good | `ollama pull llama3.1:8b` |
| mistral | ~4.1GB | Fast | Good | `ollama pull mistral` |
| phi3 | ~2.3GB | Very Fast | Decent | `ollama pull phi3` |

### Using LM Studio Instead

If you prefer LM Studio:

1. Download LM Studio: https://lmstudio.ai
2. Load a model (e.g., Llama 3.1, Mistral)
3. Start the local server (port 1234)
4. Update `.env`:
```env
AI_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_MODEL=your-model-name
```

### Using Cloud APIs (Optional)

If you prefer cloud APIs, update `.env`:

```env
# For Anthropic Claude
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-xxxxx

# OR for OpenAI GPT-4
AI_PROVIDER=openai
OPENAI_API_KEY=sk-xxxxx
```

---

## Authentication

The scripts use `DefaultAzureCredential`:
1. Environment variables
2. Managed Identity
3. **Visual Studio Code** ✓ (you're already signed in)
4. Azure CLI
5. Azure PowerShell
6. Interactive browser

**You're already authenticated** via VS Code Azure extensions, so Azure access should work automatically.

---

## Data Sources

Available from **lw-jswvd-prod** workspace:

- `WVDConnections` - Connection events and diagnostics
- `Perf` - CPU, memory, disk performance
- `WVDAgentHealthStatus` - Session host health
- `Event` - Windows events (FSLogix profiles)
- `Update` - Windows update status
- `WVDCheckpoints` - Client connection checkpoints

---

## Expected Outputs

### Data Files (in `data/` folder)
- `network_logs.csv` - Connection logs
- `sessionhost_logs.csv` - Performance metrics
- `capacity_logs.csv` - Session capacity
- `fslogix_logs.csv` - Profile events
- `disk_logs.csv` - Disk performance
- `hygiene_logs.csv` - Update compliance
- `client_logs.csv` - Client diagnostics
- `master_dataset.csv` - Aggregated metrics
- `data_with_anomalies.csv` - ML-annotated data
- `anomalies_only.csv` - Detected issues

### AI Analysis
- `ai_root_cause_analyses.json` - Structured AI insights
- `root_cause_report.txt` - Human-readable report

### Remediation
- `remediation/remediation_*.ps1` - PowerShell fix scripts
- `remediation/remediation_plan.json` - Action plan
- `remediation/REMEDIATION_README.txt` - Guide

### Visualizations
- `anomalies_visualization.png` - Anomaly charts
- `capacity_prediction.png` - Forecast plots
- `connection_success_prediction.png` - Success trends

---

## Troubleshooting

### Ollama Connection Error

```
✗ Cannot connect to ollama at http://localhost:11434
```

**Fix:**
```powershell
# Check if Ollama is running
curl http://localhost:11434

# Restart Ollama service or app
# Pull the model if not already downloaded
ollama pull llama3.1:70b
```

### Azure Authentication Error

```powershell
# Re-authenticate with Azure CLI
az login --tenant e11fd634-26b5-47f4-8b8c-908e466e9bdf
az account set --subscription 0fd85432-3d22-4f35-a7d6-b9d505d46f78
```

### No Data Returned from Azure

- Check diagnostic settings on the host pool (must send to lw-jswvd-prod)
- Verify Log Analytics workspace has data (Azure Portal → lw-jswvd-prod → Logs)
- Adjust timespan in `01_data_collection.py` if needed (default: 24 hours)

### Missing Python Packages

```powershell
pip install --upgrade -r requirements.txt
```

---

## Customization

### Adjust Anomaly Sensitivity

Edit `scripts/03_detection.py`:
```python
self.detector = IsolationForest(
    contamination=0.05,  # Lower = fewer anomalies (default: 0.1)
    random_state=42
)
```

### Change Timespan

Edit `scripts/01_data_collection.py`:
```python
self.timespan = timedelta(days=30)  # Default is 30 days
# Change to: timedelta(days=7) for 1 week
# Or: timedelta(days=90) for 3 months
```

### Add Custom KQL Queries

Edit `scripts/01_data_collection.py` and add:
```python
def collect_custom_logs(self):
    query = """
    YourTable
    | where TimeGenerated > ago(24h)
    | project TimeGenerated, Field1, Field2
    """
    return self._execute_query(query, "custom_logs.csv")
```

---

## Performance Tips

1. **Use smaller AI model for testing** - `llama3.1:8b` is much faster than 70b
2. **Limit data collection** - Reduce timespan if workspace has lots of data
3. **GPU acceleration** - Ollama runs faster with NVIDIA GPU
4. **Reduce anomaly analysis** - Lower `top_n` in script 05 (default: 5)

---

## Next Steps

1. ✅ **Verify Ollama is running** - `curl http://localhost:11434`
2. ✅ **Test data collection** - `python scripts/01_data_collection.py`
3. ✅ **Run anomaly detection** - Execute scripts 02 and 03
4. ✅ **Try AI analysis** - Run script 05 with local model
5. ✅ **View dashboard** - `streamlit run scripts/07_dashboard.py`

---

## Resources

- [Ollama Documentation](https://github.com/ollama/ollama)
- [AVD Monitoring](https://learn.microsoft.com/azure/virtual-desktop/insights)
- [Log Analytics KQL](https://learn.microsoft.com/azure/azure-monitor/logs/log-query-overview)
- [Llama 3.1 Model Card](https://github.com/meta-llama/llama-models)
