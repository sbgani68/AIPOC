"""
05 - AI Root Cause Analysis
Use GPT/Claude to analyze anomalies and provide recommendations
"""
import pandas as pd
import os
from dotenv import load_dotenv
from anthropic import Anthropic
import openai
import json

load_dotenv()


class AIRootCauseAnalyzer:
    """AI-powered root cause analysis"""
    
    def __init__(self, provider='anthropic'):
        self.data_dir = "data"
        self.provider = provider
        
        if provider == 'anthropic':
            self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            self.model = "claude-3-5-sonnet-20241022"
        else:
            openai.api_key = os.getenv('OPENAI_API_KEY')
            self.model = "gpt-4"
    
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
        """Get AI analysis"""
        try:
            if self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            else:
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
                analysis['timestamp'] = str(idx)
                analysis['anomaly_score'] = row.get('AnomalyScore', 0)
                analyses.append(analysis)
                
                # Display summary
                print(f"  ✓ Severity: {analysis.get('severity', 'Unknown')}")
                print(f"  ✓ Root Cause: {analysis.get('root_cause', 'Unknown')[:80]}...")
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
    analyzer = AIRootCauseAnalyzer(provider='anthropic')
    analyzer.run()
