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
            return df
        
        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'])
        df['ConnectionSuccess'] = (df['State'] == 'Completed').astype(int)
        
        # Aggregate by hour
        hourly = df.set_index('TimeGenerated').resample('H').agg({
            'CorrelationId': 'count',
            'ConnectionSuccess': 'mean',
            'UserName': 'nunique'
        })
        hourly.columns = ['TotalConnections', 'SuccessRate', 'UniqueUsers']
        
        return hourly
    
    def preprocess_sessionhost_logs(self):
        """Process session host performance"""
        df = self.load_logs("sessionhost_logs.csv")
        if df.empty:
            return df
        
        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'])
        
        # Pivot to wide format
        pivot = df.pivot_table(
            index=['TimeGenerated', 'Computer'],
            columns='CounterName',
            values='CounterValue',
            aggfunc='mean'
        ).reset_index()
        
        # Aggregate by host and hour
        pivot['Hour'] = pivot['TimeGenerated'].dt.floor('H')
        hourly = pivot.groupby(['Hour', 'Computer']).agg({
            '% Processor Time': 'mean',
            'Available MBytes': 'mean',
            '% Free Space': 'mean'
        }).reset_index()
        
        return hourly
    
    def preprocess_capacity_logs(self):
        """Process capacity data"""
        df = self.load_logs("capacity_logs.csv")
        if df.empty:
            return df
        
        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'])
        df['UtilizationPct'] = (df['ActiveSessions'] / df['MaxSessionLimit'] * 100)
        df['IsHealthy'] = (df['Status'] == 'Available').astype(int)
        
        # Aggregate
        hourly = df.set_index('TimeGenerated').resample('H').agg({
            'ActiveSessions': 'sum',
            'MaxSessionLimit': 'sum',
            'UtilizationPct': 'mean',
            'IsHealthy': 'mean'
        })
        
        return hourly
    
    def preprocess_fslogix_logs(self):
        """Process FSLogix events"""
        df = self.load_logs("fslogix_logs.csv")
        if df.empty:
            return df
        
        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'])
        df['IsError'] = (df['EventLevelName'] == 'Error').astype(int)
        
        # Count errors by hour
        hourly = df.set_index('TimeGenerated').resample('H').agg({
            'EventID': 'count',
            'IsError': 'sum'
        })
        hourly.columns = ['TotalEvents', 'ErrorCount']
        
        return hourly
    
    def preprocess_disk_logs(self):
        """Process disk performance"""
        df = self.load_logs("disk_logs.csv")
        if df.empty:
            return df
        
        df['TimeGenerated'] = pd.to_datetime(df['TimeGenerated'])
        
        # Pivot and aggregate
        pivot = df.pivot_table(
            index=['TimeGenerated', 'Computer'],
            columns='CounterName',
            values='CounterValue',
            aggfunc='mean'
        ).reset_index()
        
        pivot['Hour'] = pivot['TimeGenerated'].dt.floor('H')
        hourly = pivot.groupby(['Hour', 'Computer']).mean()
        
        return hourly
    
    def create_master_dataset(self):
        """Combine all preprocessed data"""
        print("\nPreprocessing all datasets...\n")
        
        network = self.preprocess_network_logs()
        sessionhost = self.preprocess_sessionhost_logs()
        capacity = self.preprocess_capacity_logs()
        fslogix = self.preprocess_fslogix_logs()
        disk = self.preprocess_disk_logs()
        
        # Merge datasets on time
        master = network
        if not capacity.empty:
            master = master.join(capacity, how='outer', rsuffix='_capacity')
        if not fslogix.empty:
            master = master.join(fslogix, how='outer', rsuffix='_fslogix')
        
        # Save master dataset
        output_path = os.path.join(self.data_dir, "master_dataset.csv")
        master.to_csv(output_path)
        print(f"\n✓ Master dataset saved: {output_path}")
        print(f"  Shape: {master.shape}")
        print(f"  Columns: {list(master.columns)}")
        
        return master


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    master = preprocessor.create_master_dataset()
    print("\n✓ Preprocessing complete!")
