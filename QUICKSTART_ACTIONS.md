# Quick Start: Suggested Actions Mapping

## What Was Implemented

✅ **Script Created:** `scripts/05b_suggested_actions.py`  
✅ **Documentation:** `SUGGESTED_ACTIONS.md`  
✅ **README Updated:** Added Step 5b to pipeline  
✅ **Syntax Validated:** Script ready for execution

## 7 Issue Categories Implemented

| Category | Actions | Auto-Remediate | Approval |
|----------|---------|----------------|----------|
| **Network/UX** | Teams alert; RFC for misroute | ❌ | ✅ |
| **Session Host** | Drain → restart agent → undrain | ✅ | ❌ |
| **Capacity** | Scaling recommendation | ❌ | ✅ |
| **FSLogix** | Maintenance; user notifications | ✅ | ❌ |
| **Disk Pressure** | Drain → cleanup → prevent login | ✅ | ❌ |
| **Hygiene** | Log off idle sessions | ✅ | ❌ |
| **Client Posture** | Client update instructions | ❌ | ❌ |

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                     ISSUE DETECTION PIPELINE                     │
└─────────────────────────────────────────────────────────────────┘

Step 3: Detection → Step 5: AI Analysis → Step 5b: Action Mapping → Step 6: Remediation
     (anomalies)      (root causes)         (categorize & map)       (execute scripts)
```

### Processing Flow

1. **Load AI Analyses** from `data/ai_root_cause_analyses.json`
2. **Categorize Each Issue** using keyword matching on root cause
3. **Map to Actions** from predefined action templates
4. **Generate Outputs:**
   - `suggested_actions_plan.json` - Machine-readable action plan
   - `SUGGESTED_ACTIONS_REPORT.txt` - Human-readable detailed report
   - `ACTION_MAPPINGS_REFERENCE.txt` - Category reference guide

## Running the Script

### Prerequisites

```powershell
# 1. Install dependencies (if not already done)
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Ensure previous steps completed
# - Step 3: Detection must have run (creates anomalies_only.csv)
# - Step 5: AI Analysis must have run (creates ai_root_cause_analyses.json)
```

### Execute

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the suggested actions mapper
python scripts/05b_suggested_actions.py
```

## Expected Output

```
================================================================================
Step 5: Suggested Actions Mapping
================================================================================

✓ Loaded 5 AI analyses
✓ Loaded 250 anomalies

Categorizing issues and mapping actions...

✓ Issue #1: Session Host (High)
✓ Issue #2: Session Host (High)
✓ Issue #3: Capacity (Medium)
✓ Issue #4: Network/UX (High)
✓ Issue #5: FSLogix (Medium)

✓ Action plan saved: data/suggested_actions_plan.json
✓ Report saved: data/SUGGESTED_ACTIONS_REPORT.txt
✓ Reference document saved: data/ACTION_MAPPINGS_REFERENCE.txt

================================================================================
SUMMARY
================================================================================
Total Issues:      5
Auto-Remediate:    3
Approval Required: 2

Issues by Category:
  • Session Host       : 2
  • Capacity          : 1
  • FSLogix           : 1
  • Network/UX        : 1

✓ Suggested actions mapping complete!
================================================================================
```

## Generated Files

### 1. suggested_actions_plan.json
**Purpose:** Machine-readable action plan for automation  
**Format:** JSON array with categorized issues and mapped actions  
**Use Case:** Integrate with Azure Logic Apps, ServiceNow, etc.

### 2. SUGGESTED_ACTIONS_REPORT.txt
**Purpose:** Human-readable detailed action plan  
**Format:** Structured text report  
**Use Case:** Review by operations team, approval workflows

### 3. ACTION_MAPPINGS_REFERENCE.txt
**Purpose:** Standard operating procedures reference  
**Format:** Category-by-category action definitions  
**Use Case:** Training, process documentation

## Integration with Step 6 (Remediation)

The remediation script (`06_remediation.py`) can be enhanced to use the action plan:

```python
# In 06_remediation.py - future enhancement
import json

# Load suggested actions
with open('data/suggested_actions_plan.json') as f:
    action_plan = json.load(f)

# Execute auto-remediation items
for item in action_plan['action_items']:
    if item['auto_remediate'] and not item['approval_required']:
        # Execute remediation based on category
        execute_remediation(item['category'], item['suggested_actions'])
```

## Customization Examples

### Add New Category

Edit `scripts/05b_suggested_actions.py`:

```python
"Database": {
    "category": "Database",
    "description": "Database connectivity or performance issues",
    "actions": [
        "Check database connection strings",
        "Verify SQL Server availability",
        "Review connection pool settings"
    ],
    "severity": "High",
    "approval_required": False,
    "auto_remediate": False
}
```

### Modify Action Steps

```python
"Session Host": {
    # ... existing config ...
    "actions": [
        "Step 1: Notify operations team via Teams",  # Added
        "Step 2: Drain host (prevent new sessions)",
        "Step 3: Restart RD Agent services",
        "Step 4: Run automated health checks",  # Modified
        "Step 5: Undrain host if healthy",
        "Step 6: Create incident ticket if issues persist"  # Added
    ]
}
```

### Adjust Categorization

```python
def categorize_issue(self, analysis: Dict, anomaly_data: pd.Series = None) -> str:
    root_cause = analysis.get('root_cause', '').lower()
    severity = analysis.get('severity', '').lower()
    
    # Prioritize Critical severity → Disk Pressure
    if severity == 'critical':
        return "Disk Pressure"
    
    # Your custom logic here
    if 'custom_keyword' in root_cause:
        return "Custom Category"
    
    # ... rest of logic
```

## Next Steps

After running Step 5b:

1. **Review** `SUGGESTED_ACTIONS_REPORT.txt` for accuracy
2. **Validate** categorization matches issue types
3. **Customize** action mappings if needed
4. **Run Step 6** (Remediation) to generate PowerShell scripts
5. **Integrate** with approval workflows for capacity/network changes
6. **Monitor** remediation effectiveness in Step 7 (Dashboard)

## Approval Workflow Integration

For issues requiring approval (Network/UX, Capacity):

```powershell
# Example: Create approval request from action plan
$actionPlan = Get-Content data/suggested_actions_plan.json | ConvertFrom-Json

$approvalItems = $actionPlan.action_items | Where-Object { $_.approval_required }

foreach ($item in $approvalItems) {
    # Send to approval system (e.g., ServiceNow, Azure Logic Apps)
    Send-ApprovalRequest -Category $item.category `
                        -Severity $item.severity `
                        -Actions $item.suggested_actions `
                        -Impact $item.user_impact
}
```

## Troubleshooting

### Script Fails to Run

**Error:** `ModuleNotFoundError: No module named 'pandas'`  
**Solution:** 
```powershell
.\venv\Scripts\Activate.ps1
pip install pandas
```

### No Analyses Found

**Error:** `No AI analyses found`  
**Solution:** Run Step 5 first:
```powershell
python scripts/05_ai_root_cause.py
```

### All Issues Same Category

**Issue:** Everything categorized as "Session Host"  
**Cause:** Root cause text doesn't match keywords  
**Solution:** Review categorization logic or customize keywords

## Support

For detailed information, see:
- 📄 [SUGGESTED_ACTIONS.md](SUGGESTED_ACTIONS.md) - Full documentation
- 📄 [README.md](README.md) - Main project overview
- 📄 [SETUP.md](SETUP.md) - Setup and configuration

---

**Status:** ✅ Implementation Complete  
**Version:** 1.0  
**Date:** February 14, 2026
