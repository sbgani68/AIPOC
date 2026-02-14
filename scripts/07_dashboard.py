"""
07 - Interactive Dashboard
Streamlit dashboard for visualizing AVD health and insights
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime
from urllib.parse import quote

import requests


class AVDDashboard:
    """Interactive Streamlit dashboard"""
    
    def __init__(self):
        self.data_dir = "data"
        st.set_page_config(
            page_title="AVD AI Monitoring Dashboard",
            page_icon="🖥️",
            layout="wide"
        )
    
    def load_data(self):
        """Load all data files"""
        data = {}
        
        # Load master dataset
        master_path = os.path.join(self.data_dir, "data_with_anomalies.csv")
        if os.path.exists(master_path):
            data['master'] = pd.read_csv(master_path, index_col=0, parse_dates=True)
        
        # Load anomalies
        anomaly_path = os.path.join(self.data_dir, "anomalies_only.csv")
        if os.path.exists(anomaly_path):
            data['anomalies'] = pd.read_csv(anomaly_path, index_col=0, parse_dates=True)
        
        # Load AI analyses
        ai_path = os.path.join(self.data_dir, "ai_root_cause_analyses.json")
        if os.path.exists(ai_path):
            with open(ai_path, 'r') as f:
                data['ai_analyses'] = json.load(f)

        # Load host-level issues
        hosts_path = os.path.join(self.data_dir, "hosts_with_issues.csv")
        if os.path.exists(hosts_path):
            data['hosts_with_issues'] = pd.read_csv(hosts_path)

        # Load predictive risks
        forecast_path = os.path.join(self.data_dir, "predictive_forecasts.json")
        if os.path.exists(forecast_path):
            with open(forecast_path, 'r', encoding='utf-8') as f:
                data['predictive_forecasts'] = json.load(f)

        # Load AI host explanations/actions
        host_expl_path = os.path.join(self.data_dir, "ai_host_explanations.json")
        if os.path.exists(host_expl_path):
            with open(host_expl_path, 'r', encoding='utf-8') as f:
                data['ai_host_explanations'] = json.load(f)
        
        return data

    @staticmethod
    def _is_true(value):
        if isinstance(value, bool):
            return value
        if pd.isna(value):
            return False
        if isinstance(value, (int, float)):
            return value != 0
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    @staticmethod
    def _host_col(df):
        if 'Computer' in df.columns:
            return 'Computer'
        if 'SessionHostName' in df.columns:
            return 'SessionHostName'
        return None

    def _issue_columns(self, df):
        issue_cols = [
            'network_issue',
            'session_host_issue',
            'capacity_issue',
            'fslogix_issue',
            'disk_issue',
            'hygiene_issue',
            'client_issue'
        ]
        return [c for c in issue_cols if c in df.columns]

    def _default_actions_from_flags(self, row):
        actions = []
        if row.get('network_issue', False):
            actions.append("Teams alert; propose RFC for UDR/proxy misroute")
        if row.get('session_host_issue', False):
            actions.append("Drain host → restart agent → health probe → undrain")
        if row.get('capacity_issue', False):
            actions.append("Recommend scale-out per policy (approval)")
        if row.get('fslogix_issue', False):
            actions.append("Run FSLogix maintenance; nudge users to sign out")
        if row.get('disk_issue', False):
            actions.append("Drain host → cleanup temp/profile cache → prevent login")
        if row.get('hygiene_issue', False):
            actions.append("Log off long disconnected sessions; notify users")
        if row.get('client_issue', False):
            actions.append("Send client update/optimization instructions")
        return actions

    def _build_host_actions_lookup(self, ai_host_explanations):
        lookup = {}
        for item in ai_host_explanations or []:
            host = item.get('host')
            if not host:
                continue
            analysis = item.get('analysis', {})
            actions = analysis.get('remediation_actions', [])
            severity = analysis.get('severity', 'Unknown')
            if host not in lookup:
                lookup[host] = {
                    'actions': actions,
                    'severity': severity
                }
        return lookup

    def build_host_summary(self, hosts_df, forecasts=None, ai_host_explanations=None):
        """Build Host | Detected Issues | Predictive Risk | Suggested Actions view."""
        if hosts_df is None or hosts_df.empty:
            return pd.DataFrame(), pd.DataFrame()

        df = hosts_df.copy()
        host_col = self._host_col(df)
        if not host_col:
            return pd.DataFrame(), pd.DataFrame()

        issue_cols = self._issue_columns(df)
        for col in issue_cols:
            df[col] = df[col].apply(self._is_true)

        if 'Hour' in df.columns:
            df['Hour'] = pd.to_datetime(df['Hour'], errors='coerce', utc=True)
            df = df.sort_values('Hour')

        latest = df.groupby(host_col, as_index=False).tail(1).copy()

        # Aggregate issues by host (any true)
        issue_agg = df.groupby(host_col)[issue_cols].max().reset_index() if issue_cols else latest[[host_col]].copy()
        host_df = latest.merge(issue_agg, on=host_col, how='left', suffixes=('', '_agg'))

        # Predictive risk lookup
        risk_lookup = {}
        for item in (forecasts or {}).get('high_risk_hosts', []):
            risk_lookup[item.get('host')] = {
                'risk_score': item.get('risk_score', 0),
                'severity': item.get('severity', 'Low'),
                'risk_factors': item.get('risk_factors', [])
            }

        action_lookup = self._build_host_actions_lookup(ai_host_explanations)

        rows = []
        for _, row in host_df.iterrows():
            host = row.get(host_col)
            issue_flags = {col: bool(row.get(col, False)) for col in issue_cols}
            detected_issues = [name.replace('_issue', '').replace('_', ' ').title() for name, val in issue_flags.items() if val]

            risk_info = risk_lookup.get(host, {'risk_score': 0, 'severity': 'Low', 'risk_factors': []})
            actions = action_lookup.get(host, {}).get('actions', [])
            ai_severity = action_lookup.get(host, {}).get('severity', 'Unknown')
            if not actions:
                actions = self._default_actions_from_flags(issue_flags)

            high_impact = (
                risk_info.get('severity') in ['Critical', 'High']
                or int(risk_info.get('risk_score', 0) or 0) >= 30
                or issue_flags.get('disk_issue', False)
                or issue_flags.get('session_host_issue', False)
            )

            rows.append({
                'Host': host,
                'Detected Issues': ", ".join(detected_issues) if detected_issues else "None",
                'Predictive Risk': f"{risk_info.get('severity', 'Low')} ({risk_info.get('risk_score', 0)})",
                'Suggested Actions': " | ".join(actions[:4]) if actions else "Monitor",
                'Risk Score': int(risk_info.get('risk_score', 0) or 0),
                'Risk Severity': risk_info.get('severity', 'Low'),
                'AI Severity': ai_severity,
                'Risk Factors': "; ".join(risk_info.get('risk_factors', [])),
                'High Impact': high_impact
            })

        summary_df = pd.DataFrame(rows).sort_values(['High Impact', 'Risk Score'], ascending=[False, False])
        high_impact_df = summary_df[summary_df['High Impact'] == True].copy()
        return summary_df, high_impact_df
    
    def render_kpis(self, df):
        """Render key performance indicators"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'SuccessRate' in df.columns:
                avg_success = df['SuccessRate'].mean() * 100
                st.metric(
                    "Avg Connection Success",
                    f"{avg_success:.1f}%",
                    delta=f"{(avg_success - 95):.1f}%" if avg_success < 95 else None
                )
        
        with col2:
            if 'ActiveSessions' in df.columns:
                current_sessions = df['ActiveSessions'].iloc[-1]
                max_sessions = df['ActiveSessions'].max()
                st.metric(
                    "Current Sessions",
                    f"{current_sessions:.0f}",
                    delta=f"Max: {max_sessions:.0f}"
                )
        
        with col3:
            if 'IsAnomaly' in df.columns:
                anomaly_count = df['IsAnomaly'].sum()
                anomaly_rate = df['IsAnomaly'].mean() * 100
                st.metric(
                    "Anomalies Detected",
                    f"{anomaly_count}",
                    delta=f"{anomaly_rate:.1f}% of time"
                )
        
        with col4:
            if 'UtilizationPct' in df.columns:
                avg_util = df['UtilizationPct'].mean()
                st.metric(
                    "Avg Capacity Utilization",
                    f"{avg_util:.1f}%",
                    delta="Healthy" if avg_util < 80 else "⚠️ High"
                )
    
    def render_timeline(self, df):
        """Render timeline chart"""
        st.subheader("📊 Metrics Timeline")
        
        # Select metric to display
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        numeric_cols = [col for col in numeric_cols if col not in ['IsAnomaly', 'AnomalyScore']]
        
        selected_metric = st.selectbox("Select Metric", numeric_cols)
        
        # Create figure
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[selected_metric],
            mode='lines',
            name=selected_metric,
            line=dict(color='blue')
        ))
        
        # Highlight anomalies
        if 'IsAnomaly' in df.columns:
            anomalies = df[df['IsAnomaly'] == True]
            fig.add_trace(go.Scatter(
                x=anomalies.index,
                y=anomalies[selected_metric],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=10, symbol='x')
            ))
        
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title=selected_metric,
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')
    
    def render_anomaly_details(self, anomalies, ai_analyses):
        """Render anomaly analysis details"""
        st.subheader("🔍 Anomaly Analysis")
        
        if anomalies is None or len(anomalies) == 0:
            st.success("✓ No anomalies detected!")
            return
        
        # Show anomaly distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                anomalies,
                x='AnomalyScore',
                nbins=20,
                title="Anomaly Score Distribution"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            # Anomaly counts over time
            anomaly_counts = anomalies.groupby(anomalies.index.date).size()
            fig = px.bar(
                x=anomaly_counts.index,
                y=anomaly_counts.values,
                title="Anomalies by Day",
                labels={'x': 'Date', 'y': 'Count'}
            )
            st.plotly_chart(fig, width='stretch')
        
        # Show AI analyses if available
        if ai_analyses:
            st.subheader("🤖 AI Root Cause Analysis")
            
            for i, analysis in enumerate(ai_analyses, 1):
                with st.expander(f"Anomaly #{i} - {analysis.get('severity', 'Unknown')} Severity"):
                    st.write(f"**Timestamp:** {analysis.get('timestamp')}")
                    st.write(f"**Anomaly Score:** {analysis.get('anomaly_score', 0):.3f}")
                    
                    st.write("**Root Cause:**")
                    st.write(analysis.get('root_cause', 'N/A'))
                    
                    st.write("**User Impact:**")
                    st.write(analysis.get('user_impact', 'N/A'))
                    
                    st.write("**Recommended Actions:**")
                    for action in analysis.get('recommended_actions', []):
                        st.write(f"- {action}")
                    
                    st.write("**Prevention Measures:**")
                    for measure in analysis.get('prevention_measures', []):
                        st.write(f"- {measure}")
    
    def render_capacity_analysis(self, df):
        """Render capacity analysis"""
        st.subheader("📈 Capacity Analysis")
        
        if 'ActiveSessions' not in df.columns:
            st.warning("Session capacity data not available")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Session trend
            fig = px.line(
                df,
                y='ActiveSessions',
                title="Active Sessions Over Time"
            )
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            if 'UtilizationPct' in df.columns:
                # Utilization heatmap by day/hour
                df_copy = df.copy()
                df_copy['Day'] = df_copy.index.day_name()
                df_copy['Hour'] = df_copy.index.hour
                
                pivot = df_copy.pivot_table(
                    values='UtilizationPct',
                    index='Day',
                    columns='Hour',
                    aggfunc='mean'
                )
                
                fig = px.imshow(
                    pivot,
                    title="Utilization Heatmap (Day × Hour)",
                    color_continuous_scale='RdYlGn_r',
                    aspect='auto'
                )
                st.plotly_chart(fig, width='stretch')

    def render_step8_host_actions(self, summary_df, enable_severity_colors=True):
        """Step 8 table: Host | Detected Issues | Predictive Risk | Suggested Actions"""
        st.subheader("🧭 Step 8: Host Risk & Actions")

        if summary_df is None or summary_df.empty:
            st.info("No host issue data available for Step 8 table.")
            return

        display_df = summary_df[
            ['Host', 'Detected Issues', 'Predictive Risk', 'Risk Severity', 'Suggested Actions', 'High Impact']
        ].copy()

        severity_colors = {
            'Critical': '#ffebee',
            'High': '#fff3e0',
            'Warning': '#fff8e1',
            'Medium': '#fffde7',
            'Low': '#e8f5e9'
        }

        def color_by_severity(row):
            severity = str(row.get('Risk Severity', 'Low'))
            color = severity_colors.get(severity, '#ffffff')
            return [f'background-color: {color}'] * len(row)

        if enable_severity_colors:
            styled = display_df.style.apply(color_by_severity, axis=1)
            st.dataframe(styled, width='stretch', hide_index=True)
            st.caption("Severity colors: Critical=red tint, High=orange tint, Medium/Warning=yellow tint, Low=green tint")
        else:
            st.dataframe(display_df, width='stretch', hide_index=True)

    def render_step8_metric_graphs(self, hosts_df):
        """Graphs for Disk %, RTT, Queue, FSLogix errors"""
        st.subheader("📉 Step 8: Operational Graphs")

        if hosts_df is None or hosts_df.empty or 'Hour' not in hosts_df.columns:
            st.info("Insufficient host metric data for Step 8 graphs.")
            return

        graph_df = hosts_df.copy()
        graph_df['Hour'] = pd.to_datetime(graph_df['Hour'], errors='coerce', utc=True)
        graph_df = graph_df.dropna(subset=['Hour'])

        # Normalize expected metric columns
        metric_map = {
            'Disk %': ['disk_free', 'DiskFreePercent'],
            'RTT': ['avg_rtt', 'RTT'],
            'Queue': ['queued_users', 'QueuedUsers'],
            'FSLogix Errors': ['fslogix_errors', 'FSLogixErrors']
        }

        plot_cols = {}
        for display_name, candidates in metric_map.items():
            chosen = next((c for c in candidates if c in graph_df.columns), None)
            if chosen:
                plot_cols[display_name] = chosen

        if not plot_cols:
            st.info("Required metric columns were not found in hosts_with_issues.csv")
            return

        host_col = self._host_col(graph_df)
        if host_col:
            selected_host = st.selectbox("Select Host for Step 8 Graphs", sorted(graph_df[host_col].dropna().unique()))
            graph_df = graph_df[graph_df[host_col] == selected_host]

        c1, c2 = st.columns(2)
        containers = [c1, c2, c1, c2]

        for idx, (label, col_name) in enumerate(plot_cols.items()):
            metric_series = graph_df[['Hour', col_name]].dropna()
            with containers[idx % len(containers)]:
                if metric_series.empty:
                    st.warning(f"No {label} data")
                    continue
                fig = px.line(metric_series, x='Hour', y=col_name, title=f"{label} Over Time")
                fig.update_layout(height=320)
                st.plotly_chart(fig, width='stretch')

    def render_alerts_notifications(self, high_impact_df, enable_severity_colors=True):
        """Alerts/Notifications: Teams or Outlook for high-impact hosts."""
        st.subheader("🚨 Step 8: Alerts & Notifications")

        if high_impact_df is None or high_impact_df.empty:
            st.success("No high-impact hosts detected at this time.")
            return

        st.warning(f"High-impact hosts detected: {len(high_impact_df)}")
        alert_df = high_impact_df[['Host', 'Detected Issues', 'Predictive Risk', 'Risk Severity', 'Risk Factors']].copy()

        severity_colors = {
            'Critical': '#ffebee',
            'High': '#fff3e0',
            'Warning': '#fff8e1',
            'Medium': '#fffde7',
            'Low': '#e8f5e9'
        }

        def color_by_severity(row):
            severity = str(row.get('Risk Severity', 'Low'))
            color = severity_colors.get(severity, '#ffffff')
            return [f'background-color: {color}'] * len(row)

        if enable_severity_colors:
            st.dataframe(alert_df.style.apply(color_by_severity, axis=1), width='stretch', hide_index=True)
        else:
            st.dataframe(alert_df, width='stretch', hide_index=True)

        with st.expander("Teams Notification"):
            teams_webhook = st.text_input("Teams Incoming Webhook URL (optional)", value="")

            if st.button("Generate Teams Payload"):
                payload = {
                    "title": "AVD High-Impact Host Alert",
                    "generated_at": datetime.now().isoformat(),
                    "hosts": high_impact_df[['Host', 'Detected Issues', 'Predictive Risk', 'Suggested Actions']].to_dict('records')
                }
                alerts_dir = os.path.join(self.data_dir, "alerts")
                os.makedirs(alerts_dir, exist_ok=True)
                payload_path = os.path.join(alerts_dir, "teams_alert_payload.json")
                with open(payload_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2)
                st.success(f"Teams payload generated: {payload_path}")

            if teams_webhook and st.button("Send Teams Alert"):
                summary_lines = [
                    f"• {r['Host']} | {r['Predictive Risk']} | {r['Detected Issues']}"
                    for _, r in high_impact_df.head(15).iterrows()
                ]
                text = "AVD High-Impact Hosts Detected\n" + "\n".join(summary_lines)
                body = {
                    "text": text
                }
                try:
                    response = requests.post(teams_webhook, json=body, timeout=10)
                    if 200 <= response.status_code < 300:
                        st.success("Teams alert sent successfully.")
                    else:
                        st.error(f"Teams alert failed: HTTP {response.status_code}")
                except Exception as exc:
                    st.error(f"Failed to send Teams alert: {exc}")

        with st.expander("Outlook Notification"):
            recipients = st.text_input("Outlook recipients (semicolon separated)", value="")
            subject = "AVD High-Impact Host Alert"
            body_lines = [
                "High-impact hosts detected:",
                ""
            ]
            for _, row in high_impact_df.iterrows():
                body_lines.append(f"- {row['Host']} | {row['Predictive Risk']} | {row['Detected Issues']}")

            body = "\n".join(body_lines)
            mailto = f"mailto:{quote(recipients)}?subject={quote(subject)}&body={quote(body)}"

            st.markdown(f"[Open Outlook Draft]({mailto})")

            if st.button("Generate Outlook Email Template"):
                alerts_dir = os.path.join(self.data_dir, "alerts")
                os.makedirs(alerts_dir, exist_ok=True)
                template_path = os.path.join(alerts_dir, "outlook_alert_email.txt")
                with open(template_path, 'w', encoding='utf-8') as f:
                    f.write(f"To: {recipients}\n")
                    f.write(f"Subject: {subject}\n\n")
                    f.write(body)
                    f.write("\n\nSuggested actions are available in the Step 8 host table.")
                st.success(f"Outlook template generated: {template_path}")
    
    def run(self):
        """Run the dashboard"""
        # Header
        st.title("🖥️ AVD AI Monitoring Dashboard")
        st.markdown("Real-time monitoring and AI-powered insights for Azure Virtual Desktop")
        
        # Load data
        data = self.load_data()
        
        if 'master' not in data:
            st.error("⚠️ No data available. Please run data collection scripts first.")
            st.stop()
        
        df = data['master']
        anomalies = data.get('anomalies')
        ai_analyses = data.get('ai_analyses', [])
        hosts_with_issues = data.get('hosts_with_issues', pd.DataFrame())
        predictive_forecasts = data.get('predictive_forecasts', {})
        ai_host_explanations = data.get('ai_host_explanations', [])

        summary_df, high_impact_df = self.build_host_summary(
            hosts_with_issues,
            forecasts=predictive_forecasts,
            ai_host_explanations=ai_host_explanations
        )
        
        # Sidebar
        with st.sidebar:
            st.header("Dashboard Controls")
            
            # Date range filter
            st.subheader("Date Range")
            date_range = st.date_input(
                "Select range",
                value=(df.index.min().date(), df.index.max().date())
            )
            
            # Filter data
            if len(date_range) == 2:
                start_date, end_date = date_range
                df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]

            st.subheader("Step 8 Display")
            enable_severity_colors = st.toggle("Enable severity colors", value=True)
            
            st.markdown("---")
            st.markdown(f"**Data Points:** {len(df)}")
            st.markdown(f"**Date Range:** {df.index.min()} to {df.index.max()}")
        
        # Main content
        self.render_kpis(df)
        st.markdown("---")
        self.render_timeline(df)
        st.markdown("---")
        self.render_anomaly_details(anomalies, ai_analyses)
        st.markdown("---")
        self.render_capacity_analysis(df)
        st.markdown("---")
        self.render_step8_host_actions(summary_df, enable_severity_colors=enable_severity_colors)
        st.markdown("---")
        self.render_step8_metric_graphs(hosts_with_issues)
        st.markdown("---")
        self.render_alerts_notifications(high_impact_df, enable_severity_colors=enable_severity_colors)
        
        # Footer
        st.markdown("---")
        st.markdown("*Dashboard generated by AVD AI PoC | Data refreshed: " + 
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")


if __name__ == "__main__":
    dashboard = AVDDashboard()
    dashboard.run()
