"""
06 - Automated Remediation
Generate and execute remediation scripts based on AI recommendations
"""
import json
import os
from datetime import datetime
import pandas as pd


class AutomatedRemediation:
    """Generate remediation actions from AI analysis"""
    
    def __init__(self):
        self.data_dir = "data"
        self.remediation_dir = os.path.join(self.data_dir, "remediation")
        os.makedirs(self.remediation_dir, exist_ok=True)
    
    def load_ai_analyses(self):
        """Load AI root cause analyses"""
        path = os.path.join(self.data_dir, "ai_root_cause_analyses.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                analyses = json.load(f)
            print(f"✓ Loaded {len(analyses)} AI analyses")
            return analyses
        else:
            print("✗ No AI analyses found. Run 05_ai_root_cause.py first.")
            return []

    def load_hosts_with_issues(self):
        """Load hosts/issues dataset from detection stage"""
        path = os.path.join(self.data_dir, "hosts_with_issues.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ Loaded {len(df)} hosts-with-issues rows")
            return df
        print("✗ No hosts_with_issues.csv found. Run 03_detection.py first.")
        return pd.DataFrame()

    def _is_true(self, value):
        """Normalize mixed bool/string/int truthy values"""
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    def drain_host(self, host):
        print(f"[ACTION] Draining host {host}...")

    def cleanup_disk(self, host_path):
        print(f"[ACTION] Cleaning temp/profile cache on {host_path}...")

    def prevent_login(self, host):
        print(f"[ACTION] Preventing login to {host} until remediated")

    def restart_agent_health_probe_undrain(self, host):
        print(f"[ACTION] Restart RDAgentBootLoader, health probe, undrain {host}")

    def run_issue_based_remediation(self, hosts_with_issues):
        """Step 6 example: apply automated actions for disk/session-host issues"""
        if hosts_with_issues.empty:
            print("ℹ No hosts_with_issues data available for action execution.")
            return

        print("\nApplying issue-based automated remediation actions...\n")

        handled = 0
        for _, row in hosts_with_issues.iterrows():
            host = row.get('Computer') or row.get('SessionHostName') or "UnknownHost"
            disk_issue = self._is_true(row.get('disk_issue', False))
            session_host_issue = self._is_true(row.get('session_host_issue', False))

            if disk_issue:
                self.drain_host(host)
                self.cleanup_disk(f"\\\\{host}\\C$")
                self.prevent_login(host)
                handled += 1

            if session_host_issue:
                self.drain_host(host)
                self.restart_agent_health_probe_undrain(host)
                handled += 1

        print(f"\n✓ Issue-based actions applied for {handled} host-issue event(s)")
    
    def generate_capacity_remediation(self, analysis):
        """Generate capacity scaling script"""
        script = f"""# AVD Capacity Remediation Script
# Generated: {datetime.now().isoformat()}
# Severity: {analysis.get('severity')}

# Azure CLI commands to scale host pool

$ResourceGroup = "YOUR_RESOURCE_GROUP"
$HostPoolName = "YOUR_HOST_POOL"

Write-Host "Scaling AVD Host Pool..." -ForegroundColor Yellow

# Check current capacity
az desktopvirtualization hostpool show `
    --resource-group $ResourceGroup `
    --name $HostPoolName

# Recommended Actions:
"""
        for action in analysis.get('recommended_actions', []):
            script += f"# - {action}\n"
        
        script += """
# Example: Add new session hosts
# Update max session limit if needed
az desktopvirtualization hostpool update `
    --resource-group $ResourceGroup `
    --name $HostPoolName `
    --max-session-limit 20

Write-Host "✓ Remediation complete" -ForegroundColor Green
"""
        return script
    
    def generate_connection_remediation(self, analysis):
        """Generate connection issue remediation"""
        script = f"""# AVD Connection Issue Remediation
# Generated: {datetime.now().isoformat()}
# Severity: {analysis.get('severity')}

# PowerShell script to diagnose and fix connection issues

Write-Host "Diagnosing AVD connection issues..." -ForegroundColor Yellow

# Check RD Gateway health
Test-NetConnection -ComputerName "rdgateway.wvd.microsoft.com" -Port 443

# Recommended Actions:
"""
        for action in analysis.get('recommended_actions', []):
            script += f"# - {action}\n"
        
        script += """
# Check network connectivity
Get-NetRoute | Where-Object {$_.DestinationPrefix -eq '0.0.0.0/0'}

# Verify DNS resolution
Resolve-DnsName "rdgateway.wvd.microsoft.com"

# Restart AVD services if needed
# Restart-Service -Name RDAgent, RDAgentBootLoader -Force

Write-Host "✓ Diagnostics complete" -ForegroundColor Green
"""
        return script
    
    def generate_fslogix_remediation(self, analysis):
        """Generate FSLogix profile remediation"""
        script = f"""# FSLogix Profile Remediation
# Generated: {datetime.now().isoformat()}
# Severity: {analysis.get('severity')}

# Remediation for FSLogix profile issues

Write-Host "Remediating FSLogix profile issues..." -ForegroundColor Yellow

# Recommended Actions:
"""
        for action in analysis.get('recommended_actions', []):
            script += f"# - {action}\n"
        
        script += """
# Check FSLogix services
Get-Service frxsvc, frxdrv | Select-Object Name, Status, StartType

# Check profile disk space
$ProfilePath = "\\\\fileserver\\profiles$"
Get-ChildItem $ProfilePath -Directory | ForEach-Object {
    $Size = (Get-ChildItem $_.FullName -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
    [PSCustomObject]@{
        User = $_.Name
        SizeGB = [math]::Round($Size, 2)
    }
} | Sort-Object SizeGB -Descending | Format-Table

# Clear temp profiles if needed
# Remove-Item "C:\\Users\\*.TEMP*" -Recurse -Force

Write-Host "✓ FSLogix remediation complete" -ForegroundColor Green
"""
        return script
    
    def generate_performance_remediation(self, analysis):
        """Generate performance optimization script"""
        script = f"""# AVD Performance Optimization
# Generated: {datetime.now().isoformat()}
# Severity: {analysis.get('severity')}

# Performance optimization for session hosts

Write-Host "Optimizing session host performance..." -ForegroundColor Yellow

# Recommended Actions:
"""
        for action in analysis.get('recommended_actions', []):
            script += f"# - {action}\n"
        
        script += """
# Check resource utilization
Get-Counter -Counter `
    "\\Processor(_Total)\\% Processor Time", `
    "\\Memory\\Available MBytes", `
    "\\LogicalDisk(C:)\\% Free Space"

# Optimize Windows services (example)
# Set-Service -Name "SysMain" -StartupType Disabled -ErrorAction SilentlyContinue

# Clear temp files
# Remove-Item -Path "$env:TEMP\\*" -Recurse -Force -ErrorAction SilentlyContinue

# Restart session host if needed
# Restart-Computer -Force

Write-Host "✓ Performance optimization complete" -ForegroundColor Green
"""
        return script
    
    def generate_monitoring_alert(self, analysis):
        """Generate Azure Monitor alert rule"""
        alert = {
            "name": f"AVD-{analysis.get('severity')}-Alert",
            "description": analysis.get('root_cause', '')[:200],
            "severity": analysis.get('severity'),
            "recommended_actions": analysis.get('recommended_actions', []),
            "monitoring": analysis.get('monitoring_recommendations', [])
        }
        return alert
    
    def create_remediation_plan(self, analyses):
        """Create remediation plan for all analyses"""
        plan = {
            "generated_at": datetime.now().isoformat(),
            "total_issues": len(analyses),
            "remediations": []
        }
        
        for i, analysis in enumerate(analyses, 1):
            severity = analysis.get('severity', 'Medium')
            root_cause = analysis.get('root_cause', '')
            
            # Determine remediation type
            remediation_type = "general"
            if "capacity" in root_cause.lower() or "session" in root_cause.lower():
                remediation_type = "capacity"
            elif "connection" in root_cause.lower() or "network" in root_cause.lower():
                remediation_type = "connection"
            elif "fslogix" in root_cause.lower() or "profile" in root_cause.lower():
                remediation_type = "fslogix"
            elif "cpu" in root_cause.lower() or "memory" in root_cause.lower():
                remediation_type = "performance"
            
            # Generate script
            if remediation_type == "capacity":
                script = self.generate_capacity_remediation(analysis)
            elif remediation_type == "connection":
                script = self.generate_connection_remediation(analysis)
            elif remediation_type == "fslogix":
                script = self.generate_fslogix_remediation(analysis)
            else:
                script = self.generate_performance_remediation(analysis)
            
            # Save script
            script_path = os.path.join(
                self.remediation_dir,
                f"remediation_{i}_{remediation_type}.ps1"
            )
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(script)
            
            # Generate alert
            alert = self.generate_monitoring_alert(analysis)
            
            plan["remediations"].append({
                "issue_number": i,
                "severity": severity,
                "type": remediation_type,
                "script_path": script_path,
                "alert_rule": alert
            })
            
            print(f"✓ Created remediation #{i}: {remediation_type} ({severity})")
        
        return plan
    
    def save_remediation_plan(self, plan):
        """Save remediation plan"""
        plan_path = os.path.join(self.remediation_dir, "remediation_plan.json")
        with open(plan_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, indent=2)
        
        print(f"\n✓ Remediation plan saved: {plan_path}")
        
        # Create summary
        summary_path = os.path.join(self.remediation_dir, "REMEDIATION_README.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("AVD AUTOMATED REMEDIATION PLAN\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Generated: {plan['generated_at']}\n")
            f.write(f"Total Issues: {plan['total_issues']}\n\n")
            
            f.write("REMEDIATION SCRIPTS:\n")
            f.write("-" * 70 + "\n")
            
            for rem in plan['remediations']:
                f.write(f"\n{rem['issue_number']}. {rem['type'].upper()} (Severity: {rem['severity']})\n")
                f.write(f"   Script: {os.path.basename(rem['script_path'])}\n")
                f.write(f"   Actions:\n")
                for action in rem['alert_rule']['recommended_actions']:
                    f.write(f"     • {action}\n")
        
        print(f"✓ Summary saved: {summary_path}")
    
    def run(self):
        """Run automated remediation generation"""
        print("Starting automated remediation generation...\n")
        
        # Load AI analyses
        analyses = self.load_ai_analyses()
        
        if not analyses:
            return
        
        # Create remediation plan
        plan = self.create_remediation_plan(analyses)
        
        # Save plan
        self.save_remediation_plan(plan)

        # Step 6 automated remediation example (disk pressure / session host)
        hosts_with_issues = self.load_hosts_with_issues()
        self.run_issue_based_remediation(hosts_with_issues)
        
        print(f"\n✓ Generated {len(plan['remediations'])} remediation scripts!")
        print(f"✓ Location: {self.remediation_dir}")


if __name__ == "__main__":
    remediation = AutomatedRemediation()
    remediation.run()
