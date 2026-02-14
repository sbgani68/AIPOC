# Step 5b: Suggested Actions Mapping

## Overview

The **Suggested Actions Mapping** module categorizes detected AVD issues into 7 standard categories and maps each to specific remediation actions. This provides a structured, consistent approach to issue resolution.

## Issue Categories & Actions

### 1️⃣ Network/UX

**Description:** Network connectivity and user experience issues  
**Severity:** High  
**Approval Required:** Yes  
**Auto-Remediate:** No

**Suggested Actions:**
1. Send Teams alert to network/AVD operations team
2. Propose RFC (Request for Change) if UDR/proxy misroute detected
3. Review network latency metrics and trace routes
4. Verify WVD service endpoint connectivity
5. Check Azure Virtual Network and NSG configurations

**When Triggered:** 
- Gateway/proxy issues
- High network latency
- Connection routing problems
- Service endpoint failures

---

### 2️⃣ Session Host

**Description:** Session host agent or service issues  
**Severity:** High  
**Approval Required:** No  
**Auto-Remediate:** Yes

**Suggested Actions:**
1. **Step 1:** Drain host (prevent new sessions)
2. **Step 2:** Restart RD Agent and RD Agent Boot Loader services
3. **Step 3:** Run health probe to verify service status
4. **Step 4:** Undrain host (allow new sessions)
5. **Step 5:** Monitor for 15 minutes post-remediation

**When Triggered:**
- RD Agent heartbeat gaps
- Service failures (Event IDs 1001/1002/1003)
- Host unavailability
- Agent communication errors

---

### 3️⃣ Capacity

**Description:** Host pool capacity constraints  
**Severity:** Medium  
**Approval Required:** Yes  
**Auto-Remediate:** No

**Suggested Actions:**
1. Recommend scaling out per auto-scaling policy
2. Calculate required additional session hosts
3. Submit scaling request for approval
4. Provision new session hosts (post-approval)
5. Update load balancing configuration
6. Verify capacity headroom after scaling

**When Triggered:**
- High session host utilization
- Users queued for login
- Session limit reached
- Capacity forecast warnings

---

### 4️⃣ FSLogix

**Description:** FSLogix profile container issues  
**Severity:** Medium  
**Approval Required:** No  
**Auto-Remediate:** Yes

**Suggested Actions:**
1. Run FSLogix maintenance and compaction scripts
2. Check profile disk space and file share capacity
3. Nudge users to sign out (send notifications)
4. Clear orphaned profile locks
5. Verify FSLogix services are running on session hosts
6. Review profile disk performance metrics

**When Triggered:**
- Profile attach errors (Event IDs 26/27/28)
- VHD corruption issues
- Profile IO latency
- Profile loading failures

---

### 5️⃣ Disk Pressure

**Description:** Session host disk space issues  
**Severity:** Critical  
**Approval Required:** No  
**Auto-Remediate:** Yes

**Suggested Actions:**
1. **Step 1:** Drain host immediately (prevent new logins)
2. **Step 2:** Run disk cleanup on temp directories
3. **Step 3:** Clear profile cache and old user data
4. **Step 4:** Verify disk space threshold met (>20% free)
5. **Step 5:** Undrain host or keep offline if persistent
6. Schedule host rebuild if recurring issue

**When Triggered:**
- C: drive space < 20%
- Temp folder size excessive
- Profile cache pressure
- Disk queue length high

---

### 6️⃣ Hygiene

**Description:** Session management and resource cleanup  
**Severity:** Low  
**Approval Required:** No  
**Auto-Remediate:** Yes

**Suggested Actions:**
1. Log off long disconnected sessions (>24 hours)
2. Send notifications to affected users before logoff
3. Clear idle sessions consuming resources
4. Update session timeout policies if needed
5. Generate session hygiene report for admin review

**When Triggered:**
- Excessive disconnected sessions
- Sessions idle >24 hours
- Resource consumption by orphaned sessions

---

### 7️⃣ Client Posture

**Description:** Client-side configuration or performance issues  
**Severity:** Low  
**Approval Required:** No  
**Auto-Remediate:** No

**Suggested Actions:**
1. Send client update instructions to affected users
2. Provide client optimization guidance (RDP settings)
3. Recommend specific client version if outdated
4. Send network optimization tips (WiFi, VPN, etc.)
5. Create knowledge base article for common issues
6. Track client remediation acknowledgment

**When Triggered:**
- Outdated RDP client versions
- Suboptimal client configurations
- Teams optimization disabled
- TCP fallback (UDP not available)

---

## Usage

### Run the Script

```powershell
python scripts/05b_suggested_actions.py
```

### Input Requirements

- **ai_root_cause_analyses.json** - Output from step 05 (AI Root Cause Analysis)
- **anomalies_only.csv** - Output from step 03 (Detection)

### Generated Outputs

1. **suggested_actions_plan.json**
   - Structured JSON with all categorized issues and actions
   - Machine-readable format for automation

2. **SUGGESTED_ACTIONS_REPORT.txt**
   - Human-readable report with detailed action plans
   - Organized by issue with severity and remediation steps

3. **ACTION_MAPPINGS_REFERENCE.txt**
   - Reference document showing all category definitions
   - Standard operating procedure for each issue type

---

## Integration with Remediation Pipeline

The suggested actions feed into Step 6 (Automated Remediation):

```
Step 3: Detection → Step 5: AI Analysis → Step 5b: Action Mapping → Step 6: Remediation
```

**Auto-Remediation Path:**
- Session Host issues → Automated drain/restart/undrain
- FSLogix issues → Automated maintenance scripts
- Disk Pressure → Automated cleanup
- Hygiene → Automated session logoff

**Approval-Required Path:**
- Network/UX → Teams alert + Manual RFC review
- Capacity → Scaling recommendation + Approval workflow

---

## Customization

Edit `scripts/05b_suggested_actions.py` to customize:

### Add New Category

```python
self.action_mappings["New Category"] = {
    "category": "New Category",
    "description": "Description of issue type",
    "actions": [
        "Action 1",
        "Action 2",
        "Action 3"
    ],
    "severity": "Medium",
    "approval_required": False,
    "auto_remediate": True
}
```

### Modify Categorization Logic

Edit the `categorize_issue()` method to adjust keyword matching:

```python
def categorize_issue(self, analysis: Dict, anomaly_data: pd.Series = None) -> str:
    root_cause = analysis.get('root_cause', '').lower()
    
    # Add custom logic
    if 'custom_keyword' in root_cause:
        return "Custom Category"
    
    # ... existing logic
```

---

## Example Output

### Action Plan Summary

```
Total Issues:      5
Auto-Remediate:    3
Approval Required: 2

Issues by Category:
  • Session Host       : 2
  • Capacity          : 1
  • FSLogix           : 1
  • Network/UX        : 1
```

### Sample Action Item

```
ISSUE #1: Session Host
--------------------------------------------------------------------------------
Severity:         High
Timestamp:        2026-02-11 13:00:00+00:00
Auto-Remediate:   Yes
Approval Needed:  No

Root Cause:
  Connection success degradation detected on host/session path.

User Impact:
  More failed or retried user connection attempts are likely.

Suggested Actions:
  1. Step 1: Drain host (prevent new sessions)
  2. Step 2: Restart RD Agent and RD Agent Boot Loader services
  3. Step 3: Run health probe to verify service status
  4. Step 4: Undrain host (allow new sessions)
  5. Step 5: Monitor for 15 minutes post-remediation

AI Recommendations:
  • Validate host pool session distribution and drain overloaded hosts.
  • Review recent gateway/session connection failures and retry trends.
  • Check FSLogix/profile and disconnection events in corresponding timeframe.

Monitoring:
  • Track per-host success rate and disconnect spikes hourly.
  • Track FSLogix error count and remediation completion status.
```

---

## Best Practices

### 1. Review Before Auto-Remediation
- Always review suggested actions before enabling auto-remediation
- Test remediation scripts in non-production first
- Monitor impact of automated actions

### 2. Approval Workflows
- Implement proper approval processes for capacity scaling
- Document RFC procedures for network changes
- Track approval responses and timing

### 3. User Communications
- Use clear, non-technical language in user notifications
- Provide estimated impact duration
- Include contact information for support

### 4. Monitoring
- Track remediation success rates
- Monitor for recurring issues on same hosts
- Alert on remediation failures

### 5. Documentation
- Maintain knowledge base articles for common issues
- Document escalation procedures
- Keep runbooks updated with latest remediation steps

---

## Troubleshooting

### No Issues Categorized

**Problem:** All issues show as "Session Host" default  
**Solution:** Check `categorize_issue()` keyword matching logic

### Missing AI Analyses

**Problem:** `ai_root_cause_analyses.json` not found  
**Solution:** Run `python scripts/05_ai_root_cause.py` first

### Incorrect Category Assignment

**Problem:** Issue categorized incorrectly  
**Solution:** Review root cause text and adjust keyword matching

---

## Future Enhancements

- [ ] Integration with Azure Logic Apps for automated workflows
- [ ] Teams/Slack webhook notifications
- [ ] ServiceNow ticket creation for approval processes
- [ ] Real-time monitoring dashboard integration
- [ ] Machine learning-based categorization (vs. rule-based)
- [ ] Historical remediation success rate tracking
- [ ] User feedback loop on action effectiveness

---

## Related Documentation

- [Main README](README.md)
- [Detection Documentation](TEST_DETECTION.md)
- [Forecasting Examples](FORECAST_EXAMPLES.md)
- [Setup Guide](SETUP.md)
