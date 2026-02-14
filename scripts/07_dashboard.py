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
        
        return data
    
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
        
        st.plotly_chart(fig, use_container_width=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Anomaly counts over time
            anomaly_counts = anomalies.groupby(anomalies.index.date).size()
            fig = px.bar(
                x=anomaly_counts.index,
                y=anomaly_counts.values,
                title="Anomalies by Day",
                labels={'x': 'Date', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
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
            st.plotly_chart(fig, use_container_width=True)
        
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
                st.plotly_chart(fig, use_container_width=True)
    
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
        
        # Footer
        st.markdown("---")
        st.markdown("*Dashboard generated by AVD AI PoC | Data refreshed: " + 
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "*")


if __name__ == "__main__":
    dashboard = AVDDashboard()
    dashboard.run()
