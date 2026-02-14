# AVD AI PoC - Azure Virtual Desktop Intelligent Monitoring

AI-powered monitoring, anomaly detection, and automated remediation for Azure Virtual Desktop environments.

## 🎯 Overview

This PoC demonstrates an end-to-end AI-powered monitoring solution for Azure Virtual Desktop (AVD) that:
- Collects comprehensive logs from AVD workspaces
- Detects anomalies using machine learning
- Provides AI-powered root cause analysis
- Generates predictive insights
- Creates automated remediation scripts
- Visualizes everything in an interactive dashboard

## 📁 Project Structure

```
avd_ai_poc/
├─ data/                          # Data storage
│  ├─ network_logs.csv           # Connection logs
│  ├─ sessionhost_logs.csv       # Performance metrics
│  ├─ capacity_logs.csv          # Session capacity
│  ├─ fslogix_logs.csv           # Profile logs
│  ├─ disk_logs.csv              # Disk performance
│  ├─ hygiene_logs.csv           # Update status
│  └─ client_logs.csv            # Client diagnostics
├─ scripts/                       # Analysis scripts
│  ├─ 01_data_collection.py      # Collect from Azure
│  ├─ 02_preprocess_aggregate.py # Clean and aggregate
│  ├─ 03_detection.py            # Anomaly detection
│  ├─ 04_predictive_analysis.py  # ML forecasting
│  ├─ 05_ai_root_cause.py        # AI analysis
│  ├─ 06_remediation.py          # Auto-remediation
│  └─ 07_dashboard.py            # Streamlit dashboard
├─ requirements.txt               # Python dependencies
└─ README.md                      # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file:

```env
# Azure credentials
AVD_WORKSPACE_ID=your_log_analytics_workspace_id

# AI API keys
ANTHROPIC_API_KEY=your_anthropic_key
# OR
OPENAI_API_KEY=your_openai_key
```

### 3. Run the Pipeline

Execute scripts in order:

```powershell
# Step 1: Collect data from Azure
python scripts/01_data_collection.py

# Step 2: Preprocess and aggregate
python scripts/02_preprocess_aggregate.py

# Step 3: Detect anomalies
python scripts/03_detection.py

# Step 4: Predictive analysis
python scripts/04_predictive_analysis.py

# Step 5: AI root cause analysis
python scripts/05_ai_root_cause.py

# Step 6: Generate remediation scripts
python scripts/06_remediation.py

# Step 7: Launch dashboard
streamlit run scripts/07_dashboard.py
```

## 📊 Features

### 1. Data Collection (Script 01)
- Network connectivity logs
- Session host performance metrics
- Capacity and utilization data
- FSLogix profile events
- Disk performance
- Windows update status
- Client connection diagnostics

### 2. Data Processing (Script 02)
- Time-series aggregation
- Feature engineering
- Missing data handling
- Master dataset creation

### 3. Anomaly Detection (Script 03)
- Isolation Forest ML algorithm
- Anomaly scoring
- Feature importance analysis
- Visual anomaly reports

### 4. Predictive Analysis (Script 04)
- Capacity forecasting (24hr ahead)
- Connection success prediction
- Random Forest models
- Performance trend analysis

### 5. AI Root Cause Analysis (Script 05)
- GPT-4 or Claude powered analysis
- Detailed root cause explanations
- User impact assessment
- Actionable recommendations
- Prevention measures

### 6. Automated Remediation (Script 06)
- PowerShell remediation scripts
- Capacity scaling automation
- Connection issue fixes
- FSLogix profile repairs
- Performance optimization

### 7. Interactive Dashboard (Script 07)
- Real-time metrics visualization
- Anomaly timeline
- AI analysis display
- Capacity heatmaps
- Filterable date ranges

## 🔧 Configuration

### Azure Log Analytics Queries

The PoC uses KQL queries to fetch:
- `WVDConnections` - Connection events
- `Perf` - Performance counters
- `WVDAgentHealthStatus` - Host health
- `Event` - Windows events
- `Update` - Update compliance
- `WVDCheckpoints` - Client diagnostics

### ML Model Parameters

Configure in scripts:
- Anomaly contamination rate: `0.1` (10%)
- Random Forest estimators: `100`
- Forecast horizon: `24 hours`

### AI Provider

Choose in script 05:
```python
analyzer = AIRootCauseAnalyzer(provider='anthropic')  # or 'openai'
```

## 📈 Expected Outputs

### Data Files
- `master_dataset.csv` - Aggregated metrics
- `data_with_anomalies.csv` - Annotated data
- `anomalies_only.csv` - Detected anomalies
- `ai_root_cause_analyses.json` - AI insights
- `remediation_plan.json` - Remediation details

### Visualizations
- `anomalies_visualization.png` - Anomaly charts
- `capacity_prediction.png` - Forecast plots
- `connection_success_prediction.png` - Success rate trends

### Reports
- `root_cause_report.txt` - Human-readable analysis
- `REMEDIATION_README.txt` - Remediation guide
- `remediation_*.ps1` - PowerShell scripts

## 🎯 Use Cases

1. **Proactive Monitoring** - Detect issues before users report them
2. **Capacity Planning** - Forecast resource needs
3. **Root Cause Analysis** - Understand why failures occur
4. **Automated Response** - Generate fix scripts automatically
5. **Executive Reporting** - Visual dashboards for stakeholders

## 🛠️ Customization

### Add New Log Source

Edit `01_data_collection.py`:
```python
def collect_custom_logs(self):
    query = """
    YourCustomTable
    | where TimeGenerated > ago(24h)
    | project TimeGenerated, CustomField1, CustomField2
    """
    return self._execute_query(query, "custom_logs.csv")
```

### Adjust Anomaly Sensitivity

Edit `03_detection.py`:
```python
self.detector = IsolationForest(
    contamination=0.05,  # Lower = fewer anomalies
    random_state=42
)
```

### Custom AI Prompts

Edit `05_ai_root_cause.py` - modify `build_analysis_prompt()` method.

## 📝 Requirements

- Python 3.9+
- Azure subscription with AVD
- Log Analytics workspace
- OpenAI or Anthropic API key
- Contributor access to AVD resources

## 🔒 Security Notes

- Never commit `.env` file
- Use managed identities when possible
- Limit API key permissions
- Review generated scripts before execution

## 📚 Additional Resources

- [AVD Documentation](https://docs.microsoft.com/azure/virtual-desktop/)
- [Log Analytics KQL](https://docs.microsoft.com/azure/azure-monitor/logs/log-query-overview)
- [Anthropic Claude](https://docs.anthropic.com)
- [OpenAI API](https://platform.openai.com/docs)

## 🤝 Contributing

This is a PoC template. Customize for your environment!

## 📄 License

MIT License
