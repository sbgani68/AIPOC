"""
03 - Anomaly Detection
Detect anomalies in AVD metrics using ML
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os
from packaging.version import Version


class AnomalyDetector:
    """Detect anomalies in AVD metrics"""
    
    def __init__(self):
        self.data_dir = "data"
        self.strict_canonical_features = True
        self.scaler = StandardScaler()
        self.detector = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100
        )
        
        # Detection thresholds
        self.thresholds = {
            'rtt_warning': 100,
            'rtt_critical': 150,
            'tcp_fallback_warning': 3,
            'tcp_fallback_critical': 5,
            'heartbeat_gap_warning': 3,
            'heartbeat_gap_critical': 5,
            'queue_warning': 5,
            'queue_critical': 10,
            'login_failures_warning': 1,
            'disk_free_warning': 20,
            'disk_free_critical': 15,
            'temp_cache_warning': 3,
            'temp_cache_critical': 5,
            'disconnected_warning': 20,
            'disconnected_critical': 30,
            'fslogix_errors_warning': 1,
            'fslogix_errors_critical': 3
        }
    
    def load_master_dataset(self):
        """Load preprocessed master dataset"""
        path = os.path.join(self.data_dir, "master_dataset.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'Hour' in df.columns:
                df['Hour'] = pd.to_datetime(df['Hour'], errors='coerce')
            print(f"✓ Loaded master dataset: {df.shape}")
            return df
        else:
            print("✗ Master dataset not found. Run 02_preprocess_aggregate.py first.")
            return None

    def _first_available_series(self, df, column_candidates, fill_value=np.nan):
        for column_name in column_candidates:
            if column_name in df.columns:
                return df[column_name]
        return pd.Series([fill_value] * len(df), index=df.index)

    def _safe_version_lt(self, value, target="2.0.0"):
        try:
            return Version(str(value)) < Version(target)
        except Exception:
            return False

    def detect_rule_based_issues(self, df):
        """Apply rule-based host issue detection."""
        if 'SessionHostName' not in df.columns:
            print("⚠ SessionHostName column missing; skipping rule-based host detection")
            return pd.DataFrame()

        agg = df.copy()

        agg['avg_rtt'] = pd.to_numeric(
            self._first_available_series(agg, ['avg_rtt', 'network_avg_rtt', 'network_AvgRttMs', 'network_RoundTripTimeMs']),
            errors='coerce'
        )
        agg['tcp_fallback_count'] = pd.to_numeric(
            self._first_available_series(agg, ['tcp_fallback_count', 'network_tcp_fallback_count', 'network_TcpFallbackCount']),
            errors='coerce'
        ).fillna(0)

        agg['heartbeat_gap'] = pd.to_numeric(
            self._first_available_series(agg, ['heartbeat_gap', 'sessionhost_heartbeat_gap', 'sessionhost_HeartbeatGapMinutes']),
            errors='coerce'
        )
        event_ids_series = self._first_available_series(agg, ['EventIDs', 'sessionhost_EventIDs'], fill_value='')

        agg['queued_users'] = pd.to_numeric(
            self._first_available_series(agg, ['queued_users', 'capacity_queued_users', 'capacity_QueuedUsers']),
            errors='coerce'
        ).fillna(0)
        agg['login_failures'] = pd.to_numeric(
            self._first_available_series(agg, ['login_failures', 'capacity_login_failures', 'capacity_LoginFailures']),
            errors='coerce'
        ).fillna(0)

        agg['fslogix_errors'] = pd.to_numeric(
            self._first_available_series(agg, ['fslogix_errors', 'fslogix_ErrorCount']),
            errors='coerce'
        ).fillna(0)

        agg['disk_free'] = pd.to_numeric(
            self._first_available_series(agg, ['disk_free', 'disk_% Free Space', 'disk_PercentFree']),
            errors='coerce'
        )
        agg['temp_profile_cache'] = pd.to_numeric(
            self._first_available_series(agg, ['temp_profile_cache', 'disk_temp_profile_cache']),
            errors='coerce'
        ).fillna(0)

        agg['disconnected_sessions'] = pd.to_numeric(
            self._first_available_series(agg, ['disconnected_sessions', 'hygiene_DisconnectEventCount']),
            errors='coerce'
        ).fillna(0)

        agg['client_version'] = self._first_available_series(
            agg,
            ['client_version', 'network_client_version', 'network_ClientVersionMin'],
            fill_value=''
        ).astype(str)

        agg['network_issue'] = (agg['avg_rtt'] > 150) | (agg['tcp_fallback_count'] > 5)

        event_issue = event_ids_series.astype(str).str.contains(r'\b(?:1001|1002|1003)\b', regex=True, na=False)
        agg['session_host_issue'] = (agg['heartbeat_gap'] > 5) | event_issue

        agg['capacity_issue'] = (agg['queued_users'] > 10) | (agg['login_failures'] > 0)
        agg['fslogix_issue'] = agg['fslogix_errors'] > 0
        agg['disk_issue'] = (agg['disk_free'] < 15) | (agg['temp_profile_cache'] > 5)
        agg['hygiene_issue'] = agg['disconnected_sessions'] > 30
        agg['client_issue'] = agg['client_version'].apply(lambda value: self._safe_version_lt(value, '2.0.0'))

        issue_columns = [
            'network_issue', 'session_host_issue', 'capacity_issue',
            'fslogix_issue', 'disk_issue', 'hygiene_issue', 'client_issue'
        ]
        agg['has_issue'] = agg[issue_columns].any(axis=1)

        hosts_with_issues = agg[agg['has_issue']].copy()
        hosts_with_issues = hosts_with_issues.sort_values(['SessionHostName', 'Hour'] if 'Hour' in hosts_with_issues.columns else ['SessionHostName'])

        output_path = os.path.join(self.data_dir, 'hosts_with_issues.csv')
        hosts_with_issues.to_csv(output_path, index=False)

        print("\nHosts with issues detected:")
        if hosts_with_issues.empty:
            print("  None")
        else:
            issue_count_by_host = hosts_with_issues.groupby('SessionHostName')['has_issue'].count().sort_values(ascending=False)
            print(issue_count_by_host.head(20).to_string())
        print(f"✓ Saved: {output_path}")

        return hosts_with_issues
    
    def detect_network_issues(self, df):
        """Detect network and UX issues - RTT, TCP fallback, regional mismatch"""
        df['network_issue'] = False
        df['network_severity'] = 'None'
        df['network_details'] = ''
        
        if 'SuccessRate' in df.columns:
            # High RTT
            high_rtt = df.get('TotalConnections', 0) > 0
            if 'avg_rtt' in df.columns:
                critical_rtt = df['avg_rtt'] > self.thresholds['rtt_critical']
                warning_rtt = (df['avg_rtt'] > self.thresholds['rtt_warning']) & ~critical_rtt
                
                df.loc[critical_rtt, 'network_issue'] = True
                df.loc[critical_rtt, 'network_severity'] = 'Critical'
                df.loc[critical_rtt, 'network_details'] = 'Critical RTT: ' + df['avg_rtt'].astype(str) + 'ms'
                
                df.loc[warning_rtt & ~df['network_issue'], 'network_issue'] = True
                df.loc[warning_rtt & ~df['network_issue'], 'network_severity'] = 'Warning'
                df.loc[warning_rtt & ~df['network_issue'], 'network_details'] = 'High RTT: ' + df['avg_rtt'].astype(str) + 'ms'
            
            # TCP fallback
            if 'tcp_fallback_count' in df.columns:
                critical_tcp = df['tcp_fallback_count'] > self.thresholds['tcp_fallback_critical']
                warning_tcp = (df['tcp_fallback_count'] > self.thresholds['tcp_fallback_warning']) & ~critical_tcp
                
                df.loc[critical_tcp, 'network_issue'] = True
                df.loc[critical_tcp & (df['network_severity'] != 'Critical'), 'network_severity'] = 'Critical'
                df.loc[critical_tcp, 'network_details'] += ' | TCP fallback: ' + df['tcp_fallback_count'].astype(str)
                
                df.loc[warning_tcp & (df['network_severity'] == 'None'), 'network_issue'] = True
                df.loc[warning_tcp & (df['network_severity'] == 'None'), 'network_severity'] = 'Warning'
                df.loc[warning_tcp & (df['network_severity'] == 'None'), 'network_details'] += ' | TCP fallback: ' + df['tcp_fallback_count'].astype(str)
        
        return df
    
    def detect_session_host_issues(self, df):
        """Detect session host issues - RDAgent heartbeat gaps, Event IDs"""
        df['session_host_issue'] = False
        df['session_host_severity'] = 'None'
        df['session_host_details'] = ''
        
        # Heartbeat gaps
        if 'heartbeat_gap' in df.columns:
            critical_hb = df['heartbeat_gap'] > self.thresholds['heartbeat_gap_critical']
            warning_hb = (df['heartbeat_gap'] > self.thresholds['heartbeat_gap_warning']) & ~critical_hb
            
            df.loc[critical_hb, 'session_host_issue'] = True
            df.loc[critical_hb, 'session_host_severity'] = 'Critical'
            df.loc[critical_hb, 'session_host_details'] = 'Heartbeat gap: ' + df['heartbeat_gap'].astype(str) + 'min'
            
            df.loc[warning_hb, 'session_host_issue'] = True
            df.loc[warning_hb, 'session_host_severity'] = 'Warning'
            df.loc[warning_hb, 'session_host_details'] = 'Heartbeat gap: ' + df['heartbeat_gap'].astype(str) + 'min'
        
        # Event IDs (1001/1002/1003)
        if 'rdagent_errors' in df.columns:
            has_errors = df['rdagent_errors'] > 0
            df.loc[has_errors, 'session_host_issue'] = True
            df.loc[has_errors & (df['session_host_severity'] == 'None'), 'session_host_severity'] = 'Warning'
            df.loc[has_errors, 'session_host_details'] += ' | RDAgent errors: ' + df['rdagent_errors'].astype(str)
        
        return df
    
    def detect_capacity_issues(self, df):
        """Detect capacity issues - Login queue, failures"""
        df['capacity_issue'] = False
        df['capacity_severity'] = 'None'
        df['capacity_details'] = ''
        
        # Queued users
        if 'ActiveSessions' in df.columns and 'MaxSessionLimit' in df.columns:
            utilization = (df['ActiveSessions'] / df['MaxSessionLimit'] * 100).fillna(0)
            
            critical_queue = utilization > 90
            warning_queue = (utilization > 75) & ~critical_queue
            
            df.loc[critical_queue, 'capacity_issue'] = True
            df.loc[critical_queue, 'capacity_severity'] = 'Critical'
            df.loc[critical_queue, 'capacity_details'] = 'Utilization: ' + utilization.astype(str) + '%'
            
            df.loc[warning_queue, 'capacity_issue'] = True
            df.loc[warning_queue, 'capacity_severity'] = 'Warning'
            df.loc[warning_queue, 'capacity_details'] = 'Utilization: ' + utilization.astype(str) + '%'
        
        # Login failures
        if 'login_failures' in df.columns:
            has_failures = df['login_failures'] > self.thresholds['login_failures_warning']
            df.loc[has_failures, 'capacity_issue'] = True
            df.loc[has_failures & (df['capacity_severity'] == 'None'), 'capacity_severity'] = 'Warning'
            df.loc[has_failures, 'capacity_details'] += ' | Login failures: ' + df['login_failures'].astype(str)
        
        return df
    
    def detect_fslogix_issues(self, df):
        """Detect FSLogix issues - Event IDs, VHD attach errors"""
        df['fslogix_issue'] = False
        df['fslogix_severity'] = 'None'
        df['fslogix_details'] = ''
        
        if 'TotalEvents' in df.columns and 'ErrorCount' in df.columns:
            critical_errors = df['ErrorCount'] >= self.thresholds['fslogix_errors_critical']
            warning_errors = (df['ErrorCount'] >= self.thresholds['fslogix_errors_warning']) & ~critical_errors
            
            df.loc[critical_errors, 'fslogix_issue'] = True
            df.loc[critical_errors, 'fslogix_severity'] = 'Critical'
            df.loc[critical_errors, 'fslogix_details'] = 'FSLogix errors: ' + df['ErrorCount'].astype(str)
            
            df.loc[warning_errors, 'fslogix_issue'] = True
            df.loc[warning_errors, 'fslogix_severity'] = 'Warning'
            df.loc[warning_errors, 'fslogix_details'] = 'FSLogix errors: ' + df['ErrorCount'].astype(str)
        
        return df
    
    def detect_disk_issues(self, df):
        """Detect disk pressure - Low disk free, high temp/cache"""
        df['disk_issue'] = False
        df['disk_severity'] = 'None'
        df['disk_details'] = ''
        
        # Disk free space
        if 'disk_free_pct' in df.columns:
            critical_disk = df['disk_free_pct'] < self.thresholds['disk_free_critical']
            warning_disk = (df['disk_free_pct'] < self.thresholds['disk_free_warning']) & ~critical_disk
            
            df.loc[critical_disk, 'disk_issue'] = True
            df.loc[critical_disk, 'disk_severity'] = 'Critical'
            df.loc[critical_disk, 'disk_details'] = 'Disk free: ' + df['disk_free_pct'].astype(str) + '%'
            
            df.loc[warning_disk, 'disk_issue'] = True
            df.loc[warning_disk, 'disk_severity'] = 'Warning'
            df.loc[warning_disk, 'disk_details'] = 'Disk free: ' + df['disk_free_pct'].astype(str) + '%'
        
        # Temp/cache pressure
        if 'temp_cache_gb' in df.columns:
            critical_cache = df['temp_cache_gb'] > self.thresholds['temp_cache_critical']
            warning_cache = (df['temp_cache_gb'] > self.thresholds['temp_cache_warning']) & ~critical_cache
            
            df.loc[critical_cache, 'disk_issue'] = True
            df.loc[critical_cache & (df['disk_severity'] != 'Critical'), 'disk_severity'] = 'Critical'
            df.loc[critical_cache, 'disk_details'] += ' | Temp/cache: ' + df['temp_cache_gb'].astype(str) + 'GB'
            
            df.loc[warning_cache & (df['disk_severity'] == 'None'), 'disk_issue'] = True
            df.loc[warning_cache & (df['disk_severity'] == 'None'), 'disk_severity'] = 'Warning'
            df.loc[warning_cache & (df['disk_severity'] == 'None'), 'disk_details'] += ' | Temp/cache: ' + df['temp_cache_gb'].astype(str) + 'GB'
        
        return df
    
    def detect_hygiene_issues(self, df):
        """Detect session hygiene issues - Disconnected sessions"""
        df['hygiene_issue'] = False
        df['hygiene_severity'] = 'None'
        df['hygiene_details'] = ''
        
        if 'disconnected_duration' in df.columns:
            critical_disconnected = df['disconnected_duration'] > self.thresholds['disconnected_critical']
            warning_disconnected = (df['disconnected_duration'] > self.thresholds['disconnected_warning']) & ~critical_disconnected
            
            df.loc[critical_disconnected, 'hygiene_issue'] = True
            df.loc[critical_disconnected, 'hygiene_severity'] = 'Critical'
            df.loc[critical_disconnected, 'hygiene_details'] = 'Disconnected: ' + df['disconnected_duration'].astype(str) + 'min'
            
            df.loc[warning_disconnected, 'hygiene_issue'] = True
            df.loc[warning_disconnected, 'hygiene_severity'] = 'Warning'
            df.loc[warning_disconnected, 'hygiene_details'] = 'Disconnected: ' + df['disconnected_duration'].astype(str) + 'min'
        
        return df
    
    def detect_client_issues(self, df):
        """Detect client posture issues - Outdated clients, Teams optimization"""
        df['client_issue'] = False
        df['client_severity'] = 'None'
        df['client_details'] = ''
        
        if 'client_version' in df.columns:
            # Simplified version check
            outdated = df['client_version'].astype(str) < '2.0'
            
            df.loc[outdated, 'client_issue'] = True
            df.loc[outdated, 'client_severity'] = 'Warning'
            df.loc[outdated, 'client_details'] = 'Client version: ' + df['client_version'].astype(str)
        
        if 'teams_optimized' in df.columns:
            not_optimized = df['teams_optimized'] == False
            
            df.loc[not_optimized, 'client_issue'] = True
            df.loc[not_optimized & (df['client_severity'] == 'None'), 'client_severity'] = 'Info'
            df.loc[not_optimized, 'client_details'] += ' | Teams not optimized'
        
        return df
    
    def detect_anomalies(self, df):
        """Detect anomalies using Isolation Forest on core multi-metric risk features"""
        feature_map = {
            'RTT': ['RTT', 'avg_rtt', 'network_avg_rtt', 'network_AvgRttMs', 'network_RoundTripTimeMs'],
            'HeartbeatGap': ['HeartbeatGap', 'heartbeat_gap', 'sessionhost_heartbeat_gap', 'sessionhost_HeartbeatGapMinutes'],
            'QueuedUsers': ['QueuedUsers', 'queued_users', 'capacity_queued_users', 'capacity_QueuedUsers'],
            'FSLogixErrors': ['FSLogixErrors', 'fslogix_errors', 'ErrorCount', 'fslogix_ErrorCount', 'fslogix_TotalEvents'],
            'DiskFreePercent': ['DiskFreePercent', 'disk_free_pct', 'disk_free', 'disk_% Free Space', 'disk_PercentFree'],
            'TempCacheGB': ['TempCacheGB', 'temp_cache_gb', 'temp_profile_cache', 'disk_temp_profile_cache'],
            'DisconnectedSessions': [
                'DisconnectedSessions',
                'disconnected_sessions',
                'hygiene_DisconnectEventCount',
                'disconnected_duration',
                'hygiene_AvgDisconnectedSessionDurationMinutes',
                'hygiene_MaxDisconnectedSessionDurationMinutes'
            ]
        }

        feature_frame = pd.DataFrame(index=df.index)
        for canonical_name, candidates in feature_map.items():
            feature_frame[canonical_name] = pd.to_numeric(
                self._first_available_series(df, candidates),
                errors='coerce'
            )

        available_features = [
            column_name
            for column_name in feature_frame.columns
            if feature_frame[column_name].notna().any()
        ]

        missing_canonical_features = [
            column_name
            for column_name in feature_map.keys()
            if column_name not in available_features
        ]

        if self.strict_canonical_features and missing_canonical_features:
            print("⚠ Strict anomaly mode is enabled. Missing canonical features:")
            print(f"  {', '.join(missing_canonical_features)}")
            print("  ML anomaly detection skipped until all 7 canonical metrics are available.")
            df['anomaly'] = 1
            df['IsAnomaly'] = False
            df['AnomalyScore'] = 0
            return df

        if not available_features:
            print("⚠ No anomaly features available for Isolation Forest")
            df['anomaly'] = 1
            df['IsAnomaly'] = False
            df['AnomalyScore'] = 0
            return df

        # If strict mode is disabled and only a small subset of core features is present,
        # enrich with additional numeric telemetry.
        if (not self.strict_canonical_features) and len(available_features) < 4:
            excluded_numeric = {
                'anomaly', 'IsAnomaly', 'AnomalyScore',
                'network_issue', 'session_host_issue', 'capacity_issue',
                'fslogix_issue', 'disk_issue', 'hygiene_issue', 'client_issue'
            }
            numeric_candidates = [
                column_name
                for column_name in df.select_dtypes(include=[np.number]).columns
                if column_name not in excluded_numeric and column_name not in available_features
            ]
            for column_name in numeric_candidates:
                if df[column_name].notna().any():
                    feature_frame[column_name] = pd.to_numeric(df[column_name], errors='coerce')
                    available_features.append(column_name)

        X = feature_frame[available_features].copy()
        X = X.fillna(X.mean())
        X = X.fillna(0)

        X_scaled = self.scaler.fit_transform(X)

        self.detector = IsolationForest(
            contamination=0.05,
            random_state=42,
            n_estimators=100
        )
        predictions = self.detector.fit_predict(X_scaled)
        anomaly_scores = self.detector.score_samples(X_scaled)

        df['anomaly'] = predictions
        df['IsAnomaly'] = df['anomaly'] == -1
        df['AnomalyScore'] = -anomaly_scores

        print("\n✓ ML Anomaly detection complete (Isolation Forest)")
        print(f"  Features used: {', '.join(available_features)}")
        print(f"  Contamination: 5%")
        print(f"  Total ML anomalies (-1): {(df['anomaly'] == -1).sum()}")
        print(f"  ML Anomaly rate: {df['IsAnomaly'].mean():.2%}")

        return df
    
    def identify_anomalous_features(self, df):
        """Identify which features contribute to anomalies"""
        anomalies = df[df['IsAnomaly'] == True]
        
        if len(anomalies) == 0:
            print("No anomalies detected")
            return None
        
        # Compare anomalous vs normal distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            col for col in numeric_cols
            if col not in ['anomaly', 'IsAnomaly', 'AnomalyScore']
            and not str(col).startswith('SourceHas')
        ]
        
        feature_importance = {}
        for col in numeric_cols:
            normal_mean = df[~df['IsAnomaly']][col].mean()
            anomaly_mean = anomalies[col].mean()

            if pd.notna(normal_mean) and pd.notna(anomaly_mean) and normal_mean != 0:
                pct_diff = abs((anomaly_mean - normal_mean) / normal_mean * 100)
                feature_importance[col] = pct_diff
        
        # Sort by importance
        feature_importance = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
        
        print("\n📊 Top anomalous features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5], 1):
            print(f"  {i}. {feature}: {importance:.1f}% difference")
        
        return feature_importance
    
    def visualize_anomalies(self, df):
        """Create visualization of anomalies"""
        plt.figure(figsize=(15, 8))
        
        # Plot 1: Timeline of anomalies
        plt.subplot(2, 2, 1)
        timeline_series = df['IsAnomaly'].astype(int)
        if 'Hour' in df.columns:
            timeline_series.index = pd.to_datetime(df['Hour'], errors='coerce')
        timeline_series.plot()
        plt.title('Anomalies Over Time')
        plt.ylabel('Is Anomaly')
        
        # Plot 2: Anomaly score distribution
        plt.subplot(2, 2, 2)
        plt.hist(df['AnomalyScore'], bins=50, edgecolor='black')
        plt.title('Anomaly Score Distribution')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Frequency')
        
        # Plot 3: Anomalies by key metrics
        plt.subplot(2, 2, 3)
        success_col = 'SuccessRate' if 'SuccessRate' in df.columns else 'network_SuccessRate'
        if success_col in df.columns:
            plt.scatter(
                range(len(df)),
                df[success_col],
                c=df['IsAnomaly'],
                cmap='RdYlGn',
                alpha=0.6
            )
            plt.title('Connection Success Rate (Red = Anomaly)')
            plt.ylabel('Success Rate')
        
        # Plot 4: CPU usage if available
        plt.subplot(2, 2, 4)
        utilization_col = 'UtilizationPct' if 'UtilizationPct' in df.columns else 'capacity_UtilizationPct'
        if utilization_col in df.columns:
            plt.scatter(
                range(len(df)),
                df[utilization_col],
                c=df['IsAnomaly'],
                cmap='RdYlGn',
                alpha=0.6
            )
            plt.title('Capacity Utilization (Red = Anomaly)')
            plt.ylabel('Utilization %')
        
        plt.tight_layout()
        output_path = os.path.join(self.data_dir, "anomalies_visualization.png")
        plt.savefig(output_path, dpi=300)
        print(f"\n✓ Visualization saved: {output_path}")
        
    def save_anomalies(self, df):
        """Save detected anomalies"""
        # Save full dataset with anomalies
        output_path = os.path.join(self.data_dir, "data_with_anomalies.csv")
        df.to_csv(output_path, index=False)
        print(f"✓ Saved: {output_path}")
        
        # Save only anomalous records
        anomalies = df[df['IsAnomaly'] == True]
        anomaly_path = os.path.join(self.data_dir, "anomalies_only.csv")
        anomalies.to_csv(anomaly_path, index=False)
        print(f"✓ Saved: {anomaly_path}")
        
        return anomalies
    
    def run(self):
        """Run complete anomaly detection pipeline"""
        print("Starting anomaly detection...\n")
        
        # Load data
        df = self.load_master_dataset()
        if df is None:
            return
        
        # Detect anomalies
        df = self.detect_anomalies(df)

        # Rule-based host detection
        self.detect_rule_based_issues(df)
        
        # Identify important features
        self.identify_anomalous_features(df)
        
        # Visualize
        self.visualize_anomalies(df)
        
        # Save results
        anomalies = self.save_anomalies(df)
        
        print("\n✓ Anomaly detection complete!")
        return df, anomalies


if __name__ == "__main__":
    detector = AnomalyDetector()
    df, anomalies = detector.run()
