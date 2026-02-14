"""
05b - Suggested Actions Mapping
Map detected issues to specific remediation actions based on issue category
"""
import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List


class SuggestedActionsMapper:
    """Map detected issues to actionable remediation steps"""
    
    def __init__(self):
        self.data_dir = "data"
        
        # Define action mappings for each issue category
        self.action_mappings = {
            "Network/UX": {
                "category": "Network/UX",
                "description": "Network connectivity and user experience issues",
                "actions": [
                    "Send Teams alert to network/AVD operations team",
                    "Propose RFC (Request for Change) if UDR/proxy misroute detected",
                    "Review network latency metrics and trace routes",
                    "Verify WVD service endpoint connectivity",
                    "Check Azure Virtual Network and NSG configurations"
                ],
                "severity": "High",
                "approval_required": True,
                "auto_remediate": False
            },
            
            "Session Host": {
                "category": "Session Host",
                "description": "Session host agent or service issues",
                "actions": [
                    "Step 1: Drain host (prevent new sessions)",
                    "Step 2: Restart RD Agent and RD Agent Boot Loader services",
                    "Step 3: Run health probe to verify service status",
                    "Step 4: Undrain host (allow new sessions)",
                    "Step 5: Monitor for 15 minutes post-remediation"
                ],
                "severity": "High",
                "approval_required": False,
                "auto_remediate": True
            },
            
            "Capacity": {
                "category": "Capacity",
                "description": "Host pool capacity constraints",
                "actions": [
                    "Recommend scaling out per auto-scaling policy",
                    "Calculate required additional session hosts",
                    "Submit scaling request for approval",
                    "Provision new session hosts (post-approval)",
                    "Update load balancing configuration",
                    "Verify capacity headroom after scaling"
                ],
                "severity": "Medium",
                "approval_required": True,
                "auto_remediate": False
            },
            
            "FSLogix": {
                "category": "FSLogix",
                "description": "FSLogix profile container issues",
                "actions": [
                    "Run FSLogix maintenance and compaction scripts",
                    "Check profile disk space and file share capacity",
                    "Nudge users to sign out (send notifications)",
                    "Clear orphaned profile locks",
                    "Verify FSLogix services are running on session hosts",
                    "Review profile disk performance metrics"
                ],
                "severity": "Medium",
                "approval_required": False,
                "auto_remediate": True
            },
            
            "Disk Pressure": {
                "category": "Disk Pressure",
                "description": "Session host disk space issues",
                "actions": [
                    "Step 1: Drain host immediately (prevent new logins)",
                    "Step 2: Run disk cleanup on temp directories",
                    "Step 3: Clear profile cache and old user data",
                    "Step 4: Verify disk space threshold met (>20% free)",
                    "Step 5: Undrain host or keep offline if persistent",
                    "Schedule host rebuild if recurring issue"
                ],
                "severity": "Critical",
                "approval_required": False,
                "auto_remediate": True
            },
            
            "Hygiene": {
                "category": "Hygiene",
                "description": "Session management and resource cleanup",
                "actions": [
                    "Log off long disconnected sessions (>24 hours)",
                    "Send notifications to affected users before logoff",
                    "Clear idle sessions consuming resources",
                    "Update session timeout policies if needed",
                    "Generate session hygiene report for admin review"
                ],
                "severity": "Low",
                "approval_required": False,
                "auto_remediate": True
            },
            
            "Client Posture": {
                "category": "Client Posture",
                "description": "Client-side configuration or performance issues",
                "actions": [
                    "Send client update instructions to affected users",
                    "Provide client optimization guidance (RDP settings)",
                    "Recommend specific client version if outdated",
                    "Send network optimization tips (WiFi, VPN, etc.)",
                    "Create knowledge base article for common issues",
                    "Track client remediation acknowledgment"
                ],
                "severity": "Low",
                "approval_required": False,
                "auto_remediate": False
            }
        }
    
    def load_ai_analyses(self) -> List[Dict]:
        """Load AI root cause analyses"""
        path = os.path.join(self.data_dir, "ai_root_cause_analyses.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                analyses = json.load(f)
            print(f"✓ Loaded {len(analyses)} AI analyses")
            return analyses
        else:
            print("⚠ No AI analyses found. Run 05_ai_root_cause.py first.")
            return []
    
    def load_anomalies(self) -> pd.DataFrame:
        """Load detected anomalies"""
        path = os.path.join(self.data_dir, "anomalies_only.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"✓ Loaded {len(df)} anomalies")
            return df
        else:
            print("⚠ No anomalies found. Run 03_detection.py first.")
            return pd.DataFrame()
    
    def categorize_issue(self, analysis: Dict, anomaly_data: pd.Series = None) -> str:
        """Categorize an issue based on root cause and metrics"""
        root_cause = analysis.get('root_cause', '').lower()
        
        # Network/UX issues
        if any(keyword in root_cause for keyword in ['network', 'latency', 'connectivity', 'gateway', 'proxy', 'udr', 'routing']):
            return "Network/UX"
        
        # Session Host issues
        if any(keyword in root_cause for keyword in ['session host', 'agent', 'rdagent', 'host down', 'host unavailable']):
            return "Session Host"
        
        # Capacity issues
        if any(keyword in root_cause for keyword in ['capacity', 'overload', 'max session', 'scaling', 'session limit']):
            return "Capacity"
        
        # FSLogix issues
        if any(keyword in root_cause for keyword in ['fslogix', 'profile', 'vhd', 'container']):
            return "FSLogix"
        
        # Disk Pressure
        if any(keyword in root_cause for keyword in ['disk', 'storage', 'space', 'full disk']):
            return "Disk Pressure"
        
        # Hygiene issues
        if any(keyword in root_cause for keyword in ['disconnect', 'idle', 'stale session', 'logoff']):
            return "Hygiene"
        
        # Client Posture
        if any(keyword in root_cause for keyword in ['client', 'rdp client', 'user device', 'endpoint']):
            return "Client Posture"
        
        # Default to Network/UX for connection issues
        if 'connection' in root_cause:
            return "Network/UX"
        
        return "Session Host"  # Default category
    
    def create_action_plan(self, analyses: List[Dict], anomalies: pd.DataFrame) -> Dict:
        """Create comprehensive action plan with categorized issues"""
        
        action_plan = {
            "generated_at": datetime.now().isoformat(),
            "total_issues": len(analyses),
            "issues_by_category": {},
            "action_items": []
        }
        
        # Initialize category counters
        for category in self.action_mappings.keys():
            action_plan["issues_by_category"][category] = 0
        
        # Process each analysis
        for i, analysis in enumerate(analyses, 1):
            # Categorize the issue
            category = self.categorize_issue(analysis)
            action_plan["issues_by_category"][category] += 1
            
            # Get action mapping
            action_mapping = self.action_mappings[category]
            
            # Create action item
            action_item = {
                "issue_id": i,
                "timestamp": analysis.get('timestamp', datetime.now().isoformat()),
                "category": category,
                "severity": analysis.get('severity', action_mapping['severity']),
                "root_cause": analysis.get('root_cause', 'Unknown'),
                "user_impact": analysis.get('user_impact', 'Potential user experience degradation'),
                "suggested_actions": action_mapping['actions'],
                "approval_required": action_mapping['approval_required'],
                "auto_remediate": action_mapping['auto_remediate'],
                "ai_recommendations": analysis.get('recommended_actions', []),
                "monitoring": analysis.get('monitoring_recommendations', [])
            }
            
            action_plan["action_items"].append(action_item)
            
            print(f"✓ Issue #{i}: {category} ({action_item['severity']})")
        
        return action_plan
    
    def save_action_plan(self, action_plan: Dict):
        """Save action plan to JSON and generate readable report"""
        
        # Save JSON
        json_path = os.path.join(self.data_dir, "suggested_actions_plan.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(action_plan, f, indent=2)
        print(f"\n✓ Action plan saved: {json_path}")
        
        # Generate readable report
        report_path = os.path.join(self.data_dir, "SUGGESTED_ACTIONS_REPORT.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AVD SUGGESTED ACTIONS REPORT\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {action_plan['generated_at']}\n")
            f.write(f"Total Issues: {action_plan['total_issues']}\n\n")
            
            # Summary by category
            f.write("ISSUES BY CATEGORY:\n")
            f.write("-" * 80 + "\n")
            for category, count in action_plan['issues_by_category'].items():
                if count > 0:
                    f.write(f"  {category:20s}: {count:3d} issue(s)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
            
            # Detailed action items
            for item in action_plan['action_items']:
                f.write(f"ISSUE #{item['issue_id']}: {item['category']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"Severity:         {item['severity']}\n")
                f.write(f"Timestamp:        {item['timestamp']}\n")
                f.write(f"Auto-Remediate:   {'Yes' if item['auto_remediate'] else 'No'}\n")
                f.write(f"Approval Needed:  {'Yes' if item['approval_required'] else 'No'}\n\n")
                
                f.write(f"Root Cause:\n  {item['root_cause']}\n\n")
                f.write(f"User Impact:\n  {item['user_impact']}\n\n")
                
                f.write("Suggested Actions:\n")
                for i, action in enumerate(item['suggested_actions'], 1):
                    f.write(f"  {i}. {action}\n")
                
                if item['ai_recommendations']:
                    f.write("\nAI Recommendations:\n")
                    for rec in item['ai_recommendations']:
                        f.write(f"  • {rec}\n")
                
                if item['monitoring']:
                    f.write("\nMonitoring:\n")
                    for mon in item['monitoring']:
                        f.write(f"  • {mon}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        print(f"✓ Report saved: {report_path}")
    
    def generate_action_mapping_reference(self):
        """Generate reference document of all action mappings"""
        
        ref_path = os.path.join(self.data_dir, "ACTION_MAPPINGS_REFERENCE.txt")
        with open(ref_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AVD ISSUE CATEGORY → SUGGESTED ACTIONS MAPPING\n")
            f.write("=" * 80 + "\n\n")
            f.write("This document defines the standard remediation actions for each issue category.\n\n")
            
            for category, mapping in self.action_mappings.items():
                f.write("=" * 80 + "\n")
                f.write(f"CATEGORY: {category}\n")
                f.write("=" * 80 + "\n")
                f.write(f"Description:      {mapping['description']}\n")
                f.write(f"Default Severity: {mapping['severity']}\n")
                f.write(f"Approval Needed:  {'Yes' if mapping['approval_required'] else 'No'}\n")
                f.write(f"Auto-Remediate:   {'Yes' if mapping['auto_remediate'] else 'No'}\n\n")
                
                f.write("Suggested Actions:\n")
                for i, action in enumerate(mapping['actions'], 1):
                    f.write(f"  {i}. {action}\n")
                
                f.write("\n\n")
        
        print(f"✓ Reference document saved: {ref_path}")
    
    def run(self):
        """Run suggested actions mapper"""
        print("=" * 80)
        print("Step 5: Suggested Actions Mapping")
        print("=" * 80 + "\n")
        
        # Load data
        analyses = self.load_ai_analyses()
        anomalies = self.load_anomalies()
        
        if not analyses:
            print("\n⚠ No analyses to process. Please run previous steps first.")
            return
        
        # Create action plan
        print("\nCategorizing issues and mapping actions...\n")
        action_plan = self.create_action_plan(analyses, anomalies)
        
        # Save outputs
        print()
        self.save_action_plan(action_plan)
        self.generate_action_mapping_reference()
        
        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Total Issues:      {action_plan['total_issues']}")
        print(f"Auto-Remediate:    {sum(1 for item in action_plan['action_items'] if item['auto_remediate'])}")
        print(f"Approval Required: {sum(1 for item in action_plan['action_items'] if item['approval_required'])}")
        print("\nIssues by Category:")
        for category, count in action_plan['issues_by_category'].items():
            if count > 0:
                print(f"  • {category:20s}: {count}")
        
        print("\n✓ Suggested actions mapping complete!")
        print("=" * 80)


if __name__ == "__main__":
    mapper = SuggestedActionsMapper()
    mapper.run()
