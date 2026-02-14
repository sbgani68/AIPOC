"""
02 - Preprocess and Aggregate Data
Cleans, transforms, and aggregates data for analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os


class DataPreprocessor:
    """Preprocess and aggregate AVD logs"""
    
    def __init__(self):
        self.data_dir = "data"

    def _empty_host_hour_df(self):
        return pd.DataFrame(columns=["Hour", "SessionHostName"])

    def _normalize_host(self, series):
        return (
            series.astype(str)
            .str.strip()
            .replace({"": "unknown", "nan": "unknown", "None": "unknown", "<>": "unknown"})
        )

    def _coalesce_numeric(self, df, candidates, default_value=np.nan):
        result = pd.Series([np.nan] * len(df), index=df.index)
        for column_name in candidates:
            if column_name in df.columns:
                candidate = pd.to_numeric(df[column_name], errors='coerce')
                result = result.combine_first(candidate)
        if pd.isna(default_value):
            return result
        return result.fillna(default_value)
    
    def load_logs(self, filename):
        """Load CSV log file"""
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ Loaded {filename}: {len(df)} rows")
            return df
        else:
            print(f"✗ File not found: {filename}")
            return pd.DataFrame()
    
    def preprocess_network_logs(self):
        """Process network connectivity logs"""
        df = self.load_logs("network_logs.csv")
        if df.empty:
            return self._empty_host_hour_df()
        
        required_columns = {"TimeGenerated", "SessionHostName", "State", "CorrelationId", "UserName"}
        if not required_columns.issubset(set(df.columns)):
            return self._empty_host_hour_df()

        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'], errors='coerce')
        df = df.dropna(subset=['TimeGenerated'])
        df['SessionHostName'] = self._normalize_host(df['SessionHostName'])
        df['ConnectionSuccess'] = (df['State'].astype(str).str.lower() == 'completed').astype(int)
        df['RoundTripTimeMs'] = pd.to_numeric(df.get('RoundTripTimeMs'), errors='coerce')
        df['StateLower'] = df['State'].astype(str).str.lower()
        df['IsStarted'] = (df['StateLower'] == 'started').astype(int)
        df['IsConnectedOrCompleted'] = df['StateLower'].isin(['connected', 'completed']).astype(int)
        df['Hour'] = df['TimeGenerated'].dt.floor('h')

        hourly = (
            df.groupby(['Hour', 'SessionHostName'], as_index=False)
            .agg(
                TotalConnections=('CorrelationId', 'count'),
                SuccessRate=('ConnectionSuccess', 'mean'),
                UniqueUsers=('UserName', 'nunique'),
                AvgRttMs=('RoundTripTimeMs', 'mean'),
                StartedCount=('IsStarted', 'sum'),
                ConnectedOrCompletedCount=('IsConnectedOrCompleted', 'sum'),
            )
        )

        hourly['QueuedUsers'] = (hourly['StartedCount'] - hourly['ConnectedOrCompletedCount']).clip(lower=0)
        hourly = hourly.drop(columns=['StartedCount', 'ConnectedOrCompletedCount'])

        return hourly
    
    def preprocess_sessionhost_logs(self):
        """Process session host performance"""
        df = self.load_logs("sessionhost_logs.csv")
        if df.empty:
            return self._empty_host_hour_df()

        required_columns = {"TimeGenerated", "Computer", "CounterName", "CounterValue"}
        if not required_columns.issubset(set(df.columns)):
            return self._empty_host_hour_df()

        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'], errors='coerce')
        df = df.dropna(subset=['TimeGenerated'])
        df['Computer'] = self._normalize_host(df['Computer'])

        pivot = df.pivot_table(
            index=['TimeGenerated', 'Computer'],
            columns='CounterName',
            values='CounterValue',
            aggfunc='mean'
        ).reset_index()

        if pivot.empty:
            return self._empty_host_hour_df()

        pivot['Hour'] = pivot['TimeGenerated'].dt.floor('h')
        host_time_order = pivot[['TimeGenerated', 'Computer']].sort_values(['Computer', 'TimeGenerated']).copy()
        host_time_order['HeartbeatGap'] = (
            host_time_order.groupby('Computer')['TimeGenerated']
            .diff()
            .dt.total_seconds()
            .div(60)
        )
        heartbeat_hourly = (
            host_time_order
            .assign(Hour=host_time_order['TimeGenerated'].dt.floor('h'))
            .groupby(['Hour', 'Computer'], as_index=False)['HeartbeatGap']
            .max()
        )

        aggregations = {}
        if '% Processor Time' in pivot.columns:
            aggregations['% Processor Time'] = 'mean'
        if 'Available MBytes' in pivot.columns:
            aggregations['Available MBytes'] = 'mean'
        if '% Free Space' in pivot.columns:
            aggregations['% Free Space'] = 'mean'

        if not aggregations:
            if heartbeat_hourly.empty:
                return self._empty_host_hour_df()
            heartbeat_hourly = heartbeat_hourly.rename(columns={'Computer': 'SessionHostName'})
            return heartbeat_hourly

        hourly = pivot.groupby(['Hour', 'Computer'], as_index=False).agg(aggregations)
        hourly = hourly.merge(heartbeat_hourly, on=['Hour', 'Computer'], how='outer')
        hourly = hourly.rename(columns={'Computer': 'SessionHostName'})

        return hourly
    
    def preprocess_capacity_logs(self):
        """Process capacity data"""
        df = self.load_logs("capacity_logs.csv")
        if df.empty:
            return self._empty_host_hour_df()

        required_columns = {"TimeGenerated", "SessionHostName", "ActiveSessions", "MaxSessionLimit", "Status"}
        if not required_columns.issubset(set(df.columns)):
            return self._empty_host_hour_df()

        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'], errors='coerce')
        df = df.dropna(subset=['TimeGenerated'])
        df['SessionHostName'] = self._normalize_host(df['SessionHostName'])
        df['ActiveSessions'] = pd.to_numeric(df['ActiveSessions'], errors='coerce')
        df['MaxSessionLimit'] = pd.to_numeric(df['MaxSessionLimit'], errors='coerce')
        df['UtilizationPct'] = (df['ActiveSessions'] / df['MaxSessionLimit'] * 100)
        df['IsHealthy'] = (df['Status'].astype(str).str.lower() == 'available').astype(int)
        df['Hour'] = df['TimeGenerated'].dt.floor('h')

        hourly = (
            df.groupby(['Hour', 'SessionHostName'], as_index=False)
            .agg(
                ActiveSessions=('ActiveSessions', 'mean'),
                MaxSessionLimit=('MaxSessionLimit', 'max'),
                UtilizationPct=('UtilizationPct', 'mean'),
                IsHealthy=('IsHealthy', 'mean'),
            )
        )

        return hourly
    
    def preprocess_fslogix_logs(self):
        """Process FSLogix events"""
        df = self.load_logs("fslogix_logs.csv")
        if df.empty:
            return self._empty_host_hour_df()

        required_columns = {"TimeGenerated", "Computer", "EventLevelName"}
        if not required_columns.issubset(set(df.columns)):
            return self._empty_host_hour_df()

        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'], errors='coerce')
        df = df.dropna(subset=['TimeGenerated'])
        df['SessionHostName'] = self._normalize_host(df['Computer'])
        df['IsError'] = (df['EventLevelName'].astype(str).str.lower() == 'error').astype(int)
        df['Hour'] = df['TimeGenerated'].dt.floor('h')

        hourly = (
            df.groupby(['Hour', 'SessionHostName'], as_index=False)
            .agg(
                TotalEvents=('EventLevelName', 'count'),
                ErrorCount=('IsError', 'sum'),
            )
        )

        return hourly
    
    def preprocess_disk_logs(self):
        """Process disk performance"""
        df = self.load_logs("disk_logs.csv")
        if df.empty:
            return self._empty_host_hour_df()

        required_columns = {"TimeGenerated", "Computer", "CounterName", "CounterValue"}
        if not required_columns.issubset(set(df.columns)):
            return self._empty_host_hour_df()

        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'], errors='coerce')
        df = df.dropna(subset=['TimeGenerated'])
        df['Computer'] = self._normalize_host(df['Computer'])

        pivot = df.pivot_table(
            index=['TimeGenerated', 'Computer'],
            columns='CounterName',
            values='CounterValue',
            aggfunc='mean'
        ).reset_index()

        if pivot.empty:
            return self._empty_host_hour_df()

        pivot['Hour'] = pivot['TimeGenerated'].dt.floor('h')

        if '% Free Space' in pivot.columns:
            pivot['DiskFreePercent'] = pd.to_numeric(pivot['% Free Space'], errors='coerce')
        elif '% Free Space C:' in pivot.columns:
            pivot['DiskFreePercent'] = pd.to_numeric(pivot['% Free Space C:'], errors='coerce')

        temp_candidates = [
            column_name
            for column_name in pivot.columns
            if isinstance(column_name, str) and (
                'temp' in column_name.lower() or 'cache' in column_name.lower()
            )
        ]
        if temp_candidates:
            pivot['TempCacheGB'] = (
                pd.to_numeric(pivot[temp_candidates[0]], errors='coerce') / 1024
            )

        numeric_columns = [column_name for column_name in pivot.columns if column_name not in ['TimeGenerated', 'Computer', 'Hour']]
        if not numeric_columns:
            return self._empty_host_hour_df()

        hourly = pivot.groupby(['Hour', 'Computer'], as_index=False)[numeric_columns].mean()
        hourly = hourly.rename(columns={'Computer': 'SessionHostName'})

        return hourly

    def preprocess_hygiene_logs(self):
        """Process hygiene summary data (already aggregated per host/hour)."""
        df = self.load_logs("hygiene_summary_hourly.csv")
        if df.empty:
            return self._empty_host_hour_df()

        required_columns = {"Hour", "SessionHostName"}
        if not required_columns.issubset(set(df.columns)):
            return self._empty_host_hour_df()

        df['Hour'] = pd.to_datetime(df['Hour'], errors='coerce')
        df = df.dropna(subset=['Hour'])
        df['SessionHostName'] = self._normalize_host(df['SessionHostName'])
        return df
    
    def create_master_dataset(self):
        """Combine all preprocessed data"""
        print("\nPreprocessing all datasets...\n")
        
        network = self.preprocess_network_logs()
        sessionhost = self.preprocess_sessionhost_logs()
        capacity = self.preprocess_capacity_logs()
        fslogix = self.preprocess_fslogix_logs()
        disk = self.preprocess_disk_logs()
        hygiene = self.preprocess_hygiene_logs()
        
        datasets = [
            ("network", network),
            ("sessionhost", sessionhost),
            ("capacity", capacity),
            ("fslogix", fslogix),
            ("disk", disk),
            ("hygiene", hygiene),
        ]
        
        non_empty_datasets = []
        for dataset_name, dataset_df in datasets:
            if not dataset_df.empty:
                prefixed_df = dataset_df.copy()
                metric_columns = [
                    column_name
                    for column_name in prefixed_df.columns
                    if column_name not in ["Hour", "SessionHostName"]
                ]
                prefixed_df = prefixed_df.rename(columns={column_name: f"{dataset_name}_{column_name}" for column_name in metric_columns})
                non_empty_datasets.append(prefixed_df)
        
        if not non_empty_datasets:
            master = self._empty_host_hour_df()
        else:
            master = non_empty_datasets[0]
            for dataset_df in non_empty_datasets[1:]:
                master = master.merge(dataset_df, on=['Hour', 'SessionHostName'], how='outer')

        master = master.sort_values(['Hour', 'SessionHostName']).reset_index(drop=True)

        master['RTT'] = self._coalesce_numeric(
            master,
            ['network_AvgRttMs', 'network_RoundTripTimeMs'],
            default_value=50.0
        )
        master['HeartbeatGap'] = self._coalesce_numeric(
            master,
            ['sessionhost_HeartbeatGap'],
            default_value=0.0
        )
        master['QueuedUsers'] = self._coalesce_numeric(
            master,
            ['network_QueuedUsers', 'capacity_QueuedUsers'],
            default_value=np.nan
        )
        if 'QueuedUsers' in master.columns:
            fallback_queue = (
                pd.to_numeric(master.get('network_TotalConnections'), errors='coerce')
                * (1 - pd.to_numeric(master.get('network_SuccessRate'), errors='coerce'))
            )
            master['QueuedUsers'] = pd.to_numeric(master['QueuedUsers'], errors='coerce').combine_first(fallback_queue)
            master['QueuedUsers'] = master['QueuedUsers'].fillna(0).clip(lower=0)

        master['FSLogixErrors'] = self._coalesce_numeric(
            master,
            ['fslogix_ErrorCount'],
            default_value=0.0
        )
        master['DiskFreePercent'] = self._coalesce_numeric(
            master,
            ['disk_DiskFreePercent', 'disk_% Free Space', 'sessionhost_% Free Space'],
            default_value=50.0
        )
        master['TempCacheGB'] = self._coalesce_numeric(
            master,
            ['disk_TempCacheGB', 'disk_temp_cache_gb'],
            default_value=0.0
        )
        master['DisconnectedSessions'] = self._coalesce_numeric(
            master,
            ['hygiene_DisconnectEventCount', 'hygiene_AvgDisconnectedSessionDurationMinutes'],
            default_value=0.0
        )

        master['SourceHasRTT'] = master[['network_AvgRttMs']].notna().any(axis=1).astype(int) if 'network_AvgRttMs' in master.columns else 0
        master['SourceHasHeartbeatGap'] = master[['sessionhost_HeartbeatGap']].notna().any(axis=1).astype(int) if 'sessionhost_HeartbeatGap' in master.columns else 0
        master['SourceHasQueuedUsers'] = master[['network_QueuedUsers']].notna().any(axis=1).astype(int) if 'network_QueuedUsers' in master.columns else 0
        master['SourceHasDisk'] = master[['disk_DiskFreePercent']].notna().any(axis=1).astype(int) if 'disk_DiskFreePercent' in master.columns else 0

        # Save master dataset
        output_path = os.path.join(self.data_dir, "master_dataset.csv")
        master.to_csv(output_path, index=False)
        print(f"\n✓ Master dataset saved: {output_path}")
        print(f"  Shape: {master.shape}")
        print(f"  Columns: {list(master.columns)}")
        
        return master


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    master = preprocessor.create_master_dataset()
    print("\n✓ Preprocessing complete!")
