# AVD Detection Rules - Test Results

## Detection Categories

### 1. Network / UX Issues
**Detection Rules:**
- **Warning:** RTT > 100ms OR TCP fallback > 3
- **Critical:** RTT > 150ms OR TCP fallback > 5

**Metrics Collected:**
- Average RTT per session
- TCP fallback count
- Regional gateway mismatch

---

### 2. Session Host Issues
**Detection Rules:**
- **Warning:** Heartbeat gap > 3 min OR RDAgent errors detected
- **Critical:** Heartbeat gap > 5 min

**Metrics Collected:**
- RDAgent heartbeat gaps
- Event IDs: 1001, 1002, 1003

---

### 3. Capacity Issues
**Detection Rules:**
- **Warning:** Utilization > 75% OR any login failures
- **Critical:** Utilization > 90%

**Metrics Collected:**
- Session utilization percentage
- Login queue length
- Login failures count

---

### 4. FSLogix Issues
**Detection Rules:**
- **Warning:** FSLogix errors >= 1
- **Critical:** FSLogix errors >= 3

**Metrics Collected:**
- Event IDs: 26, 27, 28, 1012, 1013
- VHD attach errors
- Profile IO latency

---

### 5. Disk Pressure Issues
**Detection Rules:**
- **Warning:** Disk free < 20% OR temp/cache > 3GB
- **Critical:** Disk free < 15% OR temp/cache > 5GB

**Metrics Collected:**
- C: drive % free space
- Temp folder size
- Profile cache size

---

### 6. Hygiene Issues
**Detection Rules:**
- **Warning:** Disconnected sessions > 20 min
- **Critical:** Disconnected sessions > 30 min

**Metrics Collected:**
- Disconnected session duration
- Number of disconnected sessions per host

---

### 7. Client Posture Issues
**Detection Rules:**
- **Warning:** Client version < 2.0
- **Info:** Teams optimization disabled

**Metrics Collected:**
- Client version
- Teams optimization status

---

## ML Anomaly Detection

In addition to rule-based detection, the system uses **Isolation Forest** to detect multi-metric anomalies that may not trigger individual thresholds but represent unusual patterns across:

- RTT
- Heartbeat gaps
- Queue length
- FSLogix errors
- Disk free space
- Temp/cache size
- Disconnected sessions

The model uses these canonical features (with column mapping/fallbacks):
- RTT
- HeartbeatGap
- QueuedUsers
- FSLogixErrors
- DiskFreePercent
- TempCacheGB
- DisconnectedSessions

Prediction output:
- `anomaly = -1` → high-risk/anomaly
- `anomaly = 1` → normal
- `IsAnomaly = (anomaly == -1)` for pipeline compatibility

**Contamination rate:** 5% (configurable)

**Strict mode:** Enabled by default. ML anomaly detection runs only when all 7 canonical metrics are available.

---

## Severity Levels

1. **Critical** - Immediate action required
2. **Warning** - Monitor closely, action may be needed
3. **Info** - Informational, low priority
4. **None** - No issues detected

---

## Output Files

1. **hosts_with_issues.csv** - All time windows with detected issues
2. **data_with_anomalies.csv** - Full dataset with ML anomaly flags
3. **anomalies_only.csv** - Only ML-detected anomalies
4. **issue_report.txt** - Human-readable issue summary
5. **anomalies_visualization.png** - Charts and graphs

---

## Next Steps After Detection

For each issue type, the system will:
1. Log the issue with timestamp and severity
2. Generate suggested remediation actions
3. Feed to AI for root cause analysis
4. Create automated remediation scripts where safe
