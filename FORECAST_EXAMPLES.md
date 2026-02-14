# Predictive Analysis Examples

## Overview
The predictive analysis script uses **Facebook Prophet** to forecast AVD metrics 24 hours into the future, identifying potential issues before they occur.

---

## 1. RTT (Round Trip Time) Forecast

### Example Code
```python
from prophet import Prophet

# Prepare RTT time series
df_ts = df[['TimeGenerated','RTT']].rename(columns={'TimeGenerated':'ds','RTT':'y'})
model = Prophet()
model.fit(df_ts)

# Predict next 24 hours
future = model.make_future_dataframe(periods=24, freq='h')
forecast = model.predict(future)
```

### Implementation in Script
```python
def forecast_rtt_trend(self, df):
    # Aggregates RTT across all hosts per hour
    # Forecasts next 24 hours
    # Identifies critical periods (RTT > 150ms)
    # Identifies warning periods (RTT > 100ms)
```

### Example Output
```
📡 Forecasting RTT Trend...

  Current avg RTT (last 24h): 85.3 ms
  Forecast avg RTT (next 24h): 92.7 ms
  Forecast max RTT (next 24h): 165.2 ms
  🚨 CRITICAL: 3 hours with RTT > 150ms
  ⚠ WARNING: 8 hours with RTT > 100ms
```

### High-Risk Detection
- **Critical (RTT > 150ms)**: Network degradation, likely user complaints
- **Warning (RTT > 100ms)**: Degraded user experience
- **Action**: Pre-emptively investigate network path, bandwidth, or regional issues

---

## 2. Disk Usage Forecast

### Example Code
```python
# Per-host disk forecasting
for host in hosts:
    df_host = df[df['SessionHostName'] == host]
    df_ts = df_host[['Hour','disk_free']].rename(columns={'Hour':'ds','disk_free':'y'})
    
    model = Prophet(changepoint_prior_scale=0.1)
    model.fit(df_ts)
    
    future = model.make_future_dataframe(periods=24, freq='h')
    forecast = model.predict(future)
    
    # Check if forecast drops below threshold
    min_forecast = forecast['yhat'].min()
    if min_forecast < 15:
        print(f"CRITICAL: {host} will run out of disk space!")
```

### Implementation in Script
```python
def forecast_disk_usage(self, df):
    # Forecasts disk free % per host
    # Identifies hosts that will drop below 15% (Critical)
    # Identifies hosts that will drop below 20% (Warning)
    # Returns list of at-risk hosts with severity
```

### Example Output
```
💾 Forecasting Disk Usage...

  🚨 jswvdprd-vm-001: CRITICAL - Disk forecast min: 12.3% (current: 18.5%)
  ⚠ jswvdprd-vm-007: WARNING - Disk forecast min: 18.7% (current: 23.2%)
  
  Found 2 hosts at disk risk
```

### High-Risk Detection
- **Critical (< 15% free)**: Imminent disk full, sessions may fail
- **Warning (< 20% free)**: Low disk space, monitor closely
- **Action**: Clean temp files, expand disk, or drain sessions from host
- **Predictive value**: Identifies issues **before** they impact users

---

## 3. Login Queue Forecast

### Example Code
```python
# Aggregate login queue across all hosts
df_ts = df.groupby('Hour')['queued_users'].sum().reset_index()
df_ts = df_ts.rename(columns={'Hour':'ds','queued_users':'y'})

model = Prophet(changepoint_prior_scale=0.15, daily_seasonality=True)
model.fit(df_ts)

future = model.make_future_dataframe(periods=24, freq='h')
forecast = model.predict(future)

# Identify peak login times
peak_time = forecast.loc[forecast['yhat'].idxmax(), 'ds']
peak_queue = forecast['yhat'].max()
```

### Implementation in Script
```python
def forecast_login_queue(self, df):
    # Aggregates queue length across all hosts
    # Forecasts next 24 hours
    # Identifies high queue periods (> 10 users)
    # Predicts peak login times
```

### Example Output
```
⏳ Forecasting Login Queue Trends...

  Current avg queue (last 24h): 2.3 users
  Forecast avg queue (next 24h): 4.7 users
  Forecast max queue (next 24h): 18.5 users
  ⚠ WARNING: 5 hours with queue > 10 users
  Peak queue expected at: 2026-02-15 09:00:00
```

### High-Risk Detection
- **High queue (> 10 users)**: Capacity bottleneck expected
- **Peak time prediction**: Identify when to scale out capacity
- **Action**: Add session hosts before peak hours, adjust load balancing

---

## 4. High-Risk Host Identification

### Example Output
```
🚨 Identifying high-risk hosts...

Found 8 high-risk hosts:

1. 🚨 jswvdprd-vm-001 (Risk Score: 125)
   • 🚨 Critical disk space (min: 12.3%)
   • 🚨 Critical utilization (max: 94.2%)
   • ⚠ High RTT (max: 132ms)

2. ⚠ jswvdprd-vm-007 (Risk Score: 75)
   • ⚠ High utilization (max: 87.5%, avg: 78.3%)
   • ⚠ High disconnect events (max: 42.1)
   • ⚠ Low disk space (min: 18.7%)

3. 🚨 jswvdprd-vm-012 (Risk Score: 85)
   • 🚨 Critical RTT (max: 165ms)
   • ⚠ High utilization (max: 82.1%)
```

### Risk Scoring System
| Risk Factor | Score | Severity |
|-------------|-------|----------|
| Disk < 15% | +50 | Critical |
| Disk < 20% | +30 | Warning |
| Utilization > 90% | +40 | Critical |
| Utilization > 80% | +25 | Warning |
| RTT > 150ms | +35 | Critical |
| RTT > 100ms | +20 | Warning |
| Low success rate | +20 | Warning |
| High disconnects | +15 | Warning |

**Total Risk Score:**
- **≥ 50**: Critical - Immediate action required
- **< 50**: Warning - Monitor closely

---

## 5. Forecast Visualization

### Files Generated
```
data/
├── predictive_forecasts.json          # All forecast data in JSON format
├── forecast_jswvdprd-vm-001.png       # Per-host charts (top 3 risks)
├── forecast_jswvdprd-vm-007.png
└── forecast_jswvdprd-vm-012.png
```

### Chart Structure
Each visualization includes:
- **Historical data** (blue line): Last 30 days of actual values
- **Forecast** (red line): Next 24 hours prediction
- **Confidence interval** (red shaded): 95% confidence bounds
- **Threshold lines** (dotted): Warning/Critical thresholds

---

## 6. Using Forecasts for Proactive Action

### Example Workflow

#### Morning: Review Forecasts
```bash
python scripts/04_predictive_analysis.py
```

#### Identify Critical Hosts
```python
# From output: jswvdprd-vm-001 has critical disk forecast (12.3%)
```

#### Take Action Before Impact
```powershell
# Drain sessions from at-risk host
New-AzWvdUserSession -HostPoolName jswvdprd_ARTHUR_02 `
    -ResourceGroupName rg-jswvd-prod-arthur `
    -SessionHostName jswvdprd-vm-001 `
    -DenyNewConnections $true

# Clean temp files
Invoke-AzVMRunCommand -ResourceGroupName rg-jswvd-prod-arthur `
    -VMName jswvdprd-vm-001 `
    -CommandId 'RunPowerShellScript' `
    -ScriptString 'Remove-Item C:\Windows\Temp\* -Recurse -Force'
```

#### Verify Forecast Prevented Issue
```bash
# Next day: Check if disk issue was avoided
python scripts/01_data_collection.py --hours 24
```

---

## 7. Prophet Model Configuration

### Seasonality Settings
```python
# Daily patterns (login peaks, backups)
daily_seasonality=True

# Weekly patterns (Monday login surges)
weekly_seasonality=True  # Enable for longer datasets

# Yearly patterns (not relevant for AVD)
yearly_seasonality=False
```

### Changepoint Sensitivity
```python
# RTT (network is stable, low sensitivity)
changepoint_prior_scale=0.05

# Disk (gradual decline, medium sensitivity)
changepoint_prior_scale=0.1

# Queue (sudden spikes, higher sensitivity)
changepoint_prior_scale=0.15
```

---

## 8. Integration with AI Root Cause Analysis

### Workflow
```
1. Predictive Analysis (04) → Identifies future high-risk hosts
2. AI Root Cause (05) → Analyzes WHY these hosts are at risk
3. Remediation (06) → Generates scripts to prevent predicted issues
```

### Example
```json
{
  "host": "jswvdprd-vm-001",
  "predicted_issue": "Disk will drop to 12.3% in 8 hours",
  "ai_root_cause": "FSLogix profile disk growth due to OneDrive cache",
  "recommended_action": "Clear OneDrive cache, expand VHD, or migrate profiles"
}
```

---

## 9. Metrics Forecasted

| Metric | Granularity | Threshold |
|--------|-------------|-----------|
| RTT | Aggregate | >150ms Critical, >100ms Warning |
| Disk Free % | Per-host | <15% Critical, <20% Warning |
| Login Queue | Aggregate | >10 users |
| Utilization % | Per-host | >90% Critical, >80% Warning |
| Disconnect Events | Per-host | >30 events |
| Success Rate | Per-host | <95% |

---

## 10. Best Practices

✅ **Run daily at 6 AM** - Get forecasts before business hours
✅ **Review critical hosts first** - Risk score ≥ 50
✅ **Act on disk warnings immediately** - Disk issues escalate quickly
✅ **Track forecast accuracy** - Compare predictions vs. actual outcomes
✅ **Adjust thresholds** - Tune based on your environment
✅ **Automate remediation** - Link forecasts to auto-scaling, cleanup scripts

❌ **Don't ignore warnings** - Warnings become critical issues
❌ **Don't forecast too far** - 24 hours is optimal, 48+ hours unreliable
❌ **Don't skip visualization** - Charts reveal patterns missed in numbers

---

## Summary

The predictive analysis provides **24-hour advance warning** of:
- 📡 Network degradation (RTT spikes)
- 💾 Disk space exhaustion (per-host)
- ⏳ Capacity bottlenecks (login queues)
- 🎯 High-risk hosts (composite scoring)

This enables **proactive management** instead of reactive firefighting!
