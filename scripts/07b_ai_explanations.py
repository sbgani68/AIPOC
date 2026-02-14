"""
07b - AI Explanations (Local LLM)
Generate host-level root cause explanations and remediation suggestions
from detected host issues using local LLM providers.
"""
import json
import os
from datetime import datetime

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


class AIHostExplanations:
    """Host-level AI explanations for detected issues."""

    ISSUE_COLUMNS = [
        'network_issue',
        'session_host_issue',
        'capacity_issue',
        'fslogix_issue',
        'disk_issue',
        'hygiene_issue',
        'client_issue'
    ]

    METRIC_COLUMNS = [
        'avg_rtt',
        'tcp_fallback_count',
        'heartbeat_gap',
        'disk_free',
        'temp_profile_cache',
        'queued_users'
    ]

    def __init__(self, provider=None):
        self.data_dir = "data"
        self.provider = provider or os.getenv('AI_PROVIDER', 'ollama')
        self.ai_unavailable = False

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

    def load_hosts_with_issues(self):
        """Load hosts_with_issues dataset and keep rows with active issues."""
        path = os.path.join(self.data_dir, "hosts_with_issues.csv")
        if not os.path.exists(path):
            print("✗ No hosts_with_issues.csv found. Run 03_detection.py first.")
            return pd.DataFrame()

        df = pd.read_csv(path)
        for issue_col in self.ISSUE_COLUMNS:
            if issue_col not in df.columns:
                df[issue_col] = False

        if 'has_issue' in df.columns:
            issue_df = df[df['has_issue'].astype(str).str.lower().isin(['true', '1'])].copy()
        else:
            issue_df = df[df[self.ISSUE_COLUMNS].any(axis=1)].copy()

        print(f"✓ Loaded {len(df)} rows; {len(issue_df)} with issues")
        return issue_df

    def _clean_value(self, value):
        if pd.isna(value):
            return "N/A"
        return value

    def _issue_flags_dict(self, row):
        return {
            col: bool(row.get(col, False)) if pd.notna(row.get(col, False)) else False
            for col in self.ISSUE_COLUMNS
        }

    def _host_name(self, row):
        return row.get('Computer') or row.get('SessionHostName') or 'UnknownHost'

    def build_prompt(self, host, row):
        """Build prompt using requested format for local Claude/GPT models."""
        issue_dict = self._issue_flags_dict(row)

        prompt = f"""
Host: {host}
Detected Issues: {issue_dict}
Metrics:
- RTT: {self._clean_value(row.get('avg_rtt'))}
- TCP fallback: {self._clean_value(row.get('tcp_fallback_count'))}
- Heartbeat gap: {self._clean_value(row.get('heartbeat_gap'))}
- Disk free: {self._clean_value(row.get('disk_free'))}
- Temp/Profile cache: {self._clean_value(row.get('temp_profile_cache'))}
- Queued users: {self._clean_value(row.get('queued_users'))}
Explain the likely root causes and suggest remediation actions for IT operations.
"""

        instruction = """
Return JSON only with this schema:
{
  "root_cause_summary": "brief summary",
  "likely_root_causes": ["cause1", "cause2"],
  "severity": "Critical|High|Medium|Low",
  "remediation_actions": ["action1", "action2", "action3"],
  "operational_notes": ["note1", "note2"]
}
"""
        return prompt + "\n" + instruction

    def analyze_with_ai(self, prompt):
        """Get AI output from configured provider."""
        if self.ai_unavailable:
            return None

        try:
            if self.provider == 'ollama':
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 1200
                        }
                    },
                    timeout=8
                )
                response.raise_for_status()
                return response.json()['response']

            if self.provider == 'lmstudio':
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are an expert AVD operations engineer."},
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.2,
                        "max_tokens": 1200
                    },
                    timeout=8
                )
                response.raise_for_status()
                return response.json()['choices'][0]['message']['content']

            if self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1200,
                    temperature=0.2,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            import openai
            response = openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert AVD operations engineer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=1200
            )
            return response.choices[0].message.content

        except requests.exceptions.ConnectionError:
            print(f"✗ Cannot connect to {self.provider} at {self.base_url}")
            print("  Make sure local model service is running")
            self.ai_unavailable = True
            return None
        except Exception as exc:
            print(f"✗ AI explanation error: {exc}")
            return None

    def parse_json_response(self, response_text):
        """Parse JSON payload from model response text."""
        if not response_text:
            return None
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response_text[json_start:json_end])
            return None
        except Exception:
            return None

    def fallback_explanation(self, row):
        """Deterministic fallback explanation when AI endpoint is unavailable."""
        causes = []
        actions = []
        notes = []
        severity = "Low"

        if bool(row.get('disk_issue', False)):
            causes.append("Low disk free space or profile cache pressure on session host.")
            actions.extend([
                "Drain host to avoid new sessions.",
                "Clean temp/profile cache and reclaim space.",
                "Prevent login until disk free threshold is restored."
            ])
            severity = "Critical"

        if bool(row.get('session_host_issue', False)):
            causes.append("Session host agent/heartbeat instability detected.")
            actions.extend([
                "Drain host before service operations.",
                "Restart RDAgentBootLoader and run health probe.",
                "Undrain host only after healthy state confirmation."
            ])
            if severity != "Critical":
                severity = "High"

        if bool(row.get('network_issue', False)):
            causes.append("Network path degradation (RTT/TCP fallback) may impact user sessions.")
            actions.append("Check routing/proxy and gateway path consistency.")
            if severity not in ["Critical", "High"]:
                severity = "Medium"

        if bool(row.get('capacity_issue', False)):
            causes.append("Capacity pressure indicated by queued users/session load.")
            actions.append("Recommend scale-out per policy with approval.")
            if severity not in ["Critical", "High"]:
                severity = "Medium"

        if bool(row.get('fslogix_issue', False)):
            causes.append("FSLogix profile failures/errors detected.")
            actions.append("Run FSLogix maintenance and ask users to sign out.")
            if severity not in ["Critical", "High"]:
                severity = "Medium"

        if bool(row.get('hygiene_issue', False)):
            causes.append("Session hygiene issue (long disconnected sessions).")
            actions.append("Log off long disconnected sessions and notify users.")

        if bool(row.get('client_issue', False)):
            causes.append("Client posture issue (version/config optimization).")
            actions.append("Send client update and optimization instructions.")

        if not causes:
            causes = ["No explicit issue flags were set; review host telemetry window."]
            actions = ["Continue monitoring and verify metric collection integrity."]

        notes.append("Fallback explanation generated due unavailable AI endpoint.")

        return {
            "root_cause_summary": causes[0],
            "likely_root_causes": causes,
            "severity": severity,
            "remediation_actions": actions,
            "operational_notes": notes
        }

    def explain_hosts(self, hosts_df, top_n=20):
        """Generate explanations for top host issue rows."""
        if hosts_df.empty:
            print("No issue rows to explain.")
            return []

        work_df = hosts_df.copy()
        if 'AnomalyScore' in work_df.columns:
            work_df = work_df.sort_values('AnomalyScore', ascending=False)

        results = []
        for i, (_, row) in enumerate(work_df.head(top_n).iterrows(), 1):
            host = self._host_name(row)
            print(f"🤖 Explaining host issue #{i}: {host}")

            prompt = self.build_prompt(host, row)
            response = self.analyze_with_ai(prompt)
            parsed = self.parse_json_response(response) if response else None

            if not parsed:
                parsed = self.fallback_explanation(row)

            results.append({
                "host": host,
                "timestamp": row.get('Hour', datetime.now().isoformat()),
                "detected_issues": self._issue_flags_dict(row),
                "metrics": {
                    key: self._clean_value(row.get(key))
                    for key in self.METRIC_COLUMNS
                },
                "analysis": parsed
            })

        return results

    def save_results(self, results):
        """Save explanations to json and text summary."""
        output_json = os.path.join(self.data_dir, "ai_host_explanations.json")
        with open(output_json, 'w', encoding='utf-8') as file_handle:
            json.dump(results, file_handle, indent=2)
        print(f"\n✓ Host explanations saved: {output_json}")

        output_txt = os.path.join(self.data_dir, "AI_HOST_EXPLANATIONS_REPORT.txt")
        with open(output_txt, 'w', encoding='utf-8') as file_handle:
            file_handle.write("AI HOST EXPLANATIONS REPORT\n")
            file_handle.write("=" * 80 + "\n\n")
            file_handle.write(f"Generated: {datetime.now().isoformat()}\n")
            file_handle.write(f"Total Hosts Explained: {len(results)}\n\n")

            for idx, item in enumerate(results, 1):
                analysis = item.get('analysis', {})
                file_handle.write(f"{idx}. Host: {item.get('host')}\n")
                file_handle.write(f"   Time: {item.get('timestamp')}\n")
                file_handle.write(f"   Severity: {analysis.get('severity', 'N/A')}\n")
                file_handle.write(f"   Root Cause: {analysis.get('root_cause_summary', 'N/A')}\n")
                file_handle.write("   Actions:\n")
                for action in analysis.get('remediation_actions', []):
                    file_handle.write(f"     - {action}\n")
                file_handle.write("\n")

        print(f"✓ Host explanation report saved: {output_txt}")

    def run(self):
        """Run Step 7 AI explanations."""
        print("Starting Step 7: AI Explanations (Local LLM)...\n")

        hosts_df = self.load_hosts_with_issues()
        if hosts_df.empty:
            return

        results = self.explain_hosts(hosts_df, top_n=20)
        self.save_results(results)

        print(f"\n✓ Generated {len(results)} host AI explanation(s)")


if __name__ == "__main__":
    explainer = AIHostExplanations()
    explainer.run()
