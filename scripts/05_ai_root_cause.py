"""
05 - AI Root Cause Analysis
Use local GPT/Claude models (Ollama) to analyze anomalies and provide recommendations
"""
import pandas as pd
import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()


class AIRootCauseAnalyzer:
    """AI-powered root cause analysis using local models"""
    
    def __init__(self, provider=None):
        self.data_dir = "data"
        self.provider = provider or os.getenv('AI_PROVIDER', 'ollama')
        
        if self.provider == 'ollama':
            self.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
            self.model = os.getenv('OLLAMA_MODEL', 'llama3.1:70b')
            print(f"Using Ollama model: {self.model}")
        elif self.provider == 'lmstudio':
            self.base_url = os.getenv('LMSTUDIO_BASE_URL', 'http://localhost:1234/v1')
            self.model = os.getenv('LMSTUDIO_MODEL', 'local-model')
            print(f"Using LM Studio model: {self.model}")
        elif self.provider == 'anthropic':
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.model = "claude-3-5-sonnet-20241022"
            print("Using Anthropic Claude API")
        else:
            import openai
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.model = "gpt-4"
            print("Using OpenAI GPT-4 API")
    
    def load_anomalies(self):
        """Load detected anomalies"""
        path = os.path.join(self.data_dir, "anomalies_only.csv")
        if os.path.exists(path):
            df= pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"✓ Loaded {len(df)} anomalies")
            return df
        else:
            print("✗ No anomalies found. Run 03_detection.py first.")
            return None
    
    def build_analysis_prompt(self, anomaly_row):
        """Build prompt for AI analysis"""
        prompt = f"""You are an expert Azure Virtual Desktop (AVD) administrator and troubleshooter.

Analyze this anomalous state detected in an AVD environment:

**Timestamp:** {anomaly_row.name}
**Anomaly Score:** {anomaly_row.get('AnomalyScore', 'N/A'):.3f}

**Metrics at time of anomaly:**
"""
        # Add metrics
        for col, value in anomaly_row.items():
            if col not in ['IsAnomaly', 'AnomalyScore'] and pd.notna(value):
                prompt += f"- {col}: {value}\n"
        
        prompt += """

Provide a comprehensive analysis in JSON format:
{
    "root_cause": "Detailed explanation of the most likely root cause",
    "severity": "Critical|High|Medium|Low",
    "user_impact": "Description of impact on end users",
    "technical_details": "Technical explanation of what's happening",
    "recommended_actions": [
        "Immediate action 1",
        "Immediate action 2",
        "Follow-up action 1"
    ],
    "prevention_measures": [
        "Prevention measure 1",
        "Prevention measure 2"
    ],
    "monitoring_recommendations": [
        "What to monitor 1",
        "What to monitor 2"
    ]
}

Focus on AVD-specific issues like:
- Connection gateway problems
- Session host capacity/performance 
- FSLogix profile issues
- Network latency/bandwidth
- Resource exhaustion
- Authentication failures
"""
        return prompt
    
    def analyze_with_ai(self, prompt):
        """Get AI analysis from local or cloud models"""
        try:
            if self.provider == 'ollama':
                # Use Ollama local models
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "num_predict": 2000
                        }
                    },
                    timeout=120
                )
                response.raise_for_status()
                return response.json()['response']
            
            elif self.provider == 'lmstudio':
                # Use LM Studio OpenAI-compatible API
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are an AVD expert."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.3,
                        "max_tokens": 2000
                    },
                    timeout=120
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            else:
                import openai
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an AVD expert."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=2000
                )
                return response.choices[0].message.content
        
        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to {self.provider} at {self.base_url}")
            print(f"  Make sure {self.provider} is running locally")
            return None
        except Exception as e:
            print(f"✗ AI analysis error: {e}")
            return None
    
    def parse_analysis(self, response_text):
        """Parse AI response"""
        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)
                return analysis
            else:
                return {"error": "Could not parse JSON response"}
        except Exception as e:
            return {"error": f"Parse error: {str(e)}"}

    def build_fallback_analysis(self, anomaly_row):
        """Build deterministic fallback analysis when AI endpoint is unavailable."""
        anomaly_score = float(anomaly_row.get('AnomalyScore', 0) or 0)
        success_rate = anomaly_row.get('network_SuccessRate', anomaly_row.get('SuccessRate'))
        utilization = anomaly_row.get('capacity_UtilizationPct', anomaly_row.get('UtilizationPct'))
        disconnect_events = anomaly_row.get('hygiene_DisconnectEventCount')
        fslogix_errors = anomaly_row.get('fslogix_ErrorCount')

        root_cause = "Multi-signal anomaly across host metrics with likely transient load or connection instability."
        severity = "Medium"
        user_impact = "Intermittent user experience degradation is possible."

        if pd.notna(utilization) and float(utilization) >= 90:
            root_cause = "Host pool capacity pressure detected (high session utilization)."
            severity = "High"
            user_impact = "Users may face slow sessions, delayed logons, or connection throttling."
        elif pd.notna(success_rate) and float(success_rate) < 0.95:
            root_cause = "Connection success degradation detected on host/session path."
            severity = "High"
            user_impact = "More failed or retried user connection attempts are likely."
        elif pd.notna(disconnect_events) and float(disconnect_events) > 20:
            root_cause = "Elevated disconnection pattern detected from session/checkpoint signals."
            severity = "Medium"
            user_impact = "Users may experience abrupt disconnects and reconnect interruptions."
        elif pd.notna(fslogix_errors) and float(fslogix_errors) > 0:
            root_cause = "FSLogix-related profile errors detected in anomaly window."
            severity = "Medium"
            user_impact = "Profile load times and sign-in experience may degrade for affected users."

        if anomaly_score >= 0.75:
            severity = "Critical" if severity == "High" else "High"

        return {
            "root_cause": root_cause,
            "severity": severity,
            "user_impact": user_impact,
            "technical_details": "Fallback rules-based analysis generated from anomaly metrics due unavailable AI endpoint.",
            "recommended_actions": [
                "Validate host pool session distribution and drain overloaded hosts.",
                "Review recent gateway/session connection failures and retry trends.",
                "Check FSLogix/profile and disconnection events in corresponding timeframe."
            ],
            "prevention_measures": [
                "Set host-level alert thresholds for utilization, success rate, and disconnect counts.",
                "Maintain proactive capacity headroom and scheduled health checks."
            ],
            "monitoring_recommendations": [
                "Track per-host success rate and disconnect spikes hourly.",
                "Track FSLogix error count and remediation completion status."
            ]
        }
    
    def analyze_top_anomalies(self, df, top_n=5):
        """Analyze top N anomalies"""
        if df is None or len(df) == 0:
            print("No anomalies to analyze")
            return []
        
        # Sort by anomaly score
        df_sorted = df.sort_values('AnomalyScore', ascending=False)
        
        analyses = []
        print(f"\nAnalyzing top {top_n} anomalies...\n")
        
        for i, (idx, row) in enumerate(df_sorted.head(top_n).iterrows(), 1):
            print(f"📊 Analyzing anomaly #{i} ({idx})...")
            
            # Build prompt
            prompt = self.build_analysis_prompt(row)
            
            # Get AI analysis
            response = self.analyze_with_ai(prompt)
            
            if response:
                analysis = self.parse_analysis(response)
            else:
                analysis = self.build_fallback_analysis(row)
                analysis['analysis_mode'] = 'fallback'

            analysis['timestamp'] = str(idx)
            analysis['anomaly_score'] = row.get('AnomalyScore', 0)
            analyses.append(analysis)
            
            # Display summary
            print(f"  ✓ Severity: {analysis.get('severity', 'Unknown')}")
            print(f"  ✓ Root Cause: {analysis.get('root_cause', 'Unknown')[:80]}...")
            if analysis.get('analysis_mode') == 'fallback':
                print("  ✓ Mode: Fallback (rules-based)")
            print()
        
        return analyses
    
    def save_analyses(self, analyses):
        """Save AI analyses"""
        if not analyses:
            return
        
        output_path = os.path.join(self.data_dir, "ai_root_cause_analyses.json")
        with open(output_path, 'w') as f:
            json.dump(analyses, f, indent=2)
        
        print(f"\n✓ Saved {len(analyses)} analyses to {output_path}")
        
        # Create summary report
        report_path = os.path.join(self.data_dir, "root_cause_report.txt")
        with open(report_path, 'w') as f:
            f.write("AVD ANOMALY ROOT CAUSE ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            for i, analysis in enumerate(analyses, 1):
                f.write(f"ANOMALY #{i}\n")
                f.write(f"Timestamp: {analysis.get('timestamp')}\n")
                f.write(f"Severity: {analysis.get('severity')}\n")
                f.write(f"Anomaly Score: {analysis.get('anomaly_score', 0):.3f}\n\n")
                
                f.write(f"ROOT CAUSE:\n{analysis.get('root_cause', 'N/A')}\n\n")
                f.write(f"USER IMPACT:\n{analysis.get('user_impact', 'N/A')}\n\n")
                
                f.write("RECOMMENDED ACTIONS:\n")
                for action in analysis.get('recommended_actions', []):
                    f.write(f"  • {action}\n")
                f.write("\n")
                
                f.write("PREVENTION MEASURES:\n")
                for measure in analysis.get('prevention_measures', []):
                    f.write(f"  • {measure}\n")
                f.write("\n" + "-" * 70 + "\n\n")
        
        print(f"✓ Saved summary report to {report_path}")
    
    def run(self):
        """Run AI root cause analysis"""
        print("Starting AI root cause analysis...\n")
        
        # Load anomalies
        df = self.load_anomalies()
        
        # Analyze top anomalies
        analyses = self.analyze_top_anomalies(df, top_n=5)
        
        # Save results
        self.save_analyses(analyses)
        
        print("\n✓ AI root cause analysis complete!")


if __name__ == "__main__":
    # Use provider from .env file (defaults to 'ollama' for local models)
    analyzer = AIRootCauseAnalyzer()
    analyzer.run()
