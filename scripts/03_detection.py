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


class AnomalyDetector:
    """Detect anomalies in AVD metrics"""
    
    def __init__(self):
        self.data_dir = "data"
        self.scaler = StandardScaler()
        self.detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
    
    def load_master_dataset(self):
        """Load preprocessed master dataset"""
        path = os.path.join(self.data_dir, "master_dataset.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"✓ Loaded master dataset: {df.shape}")
            return df
        else:
            print("✗ Master dataset not found. Run 02_preprocess_aggregate.py first.")
            return None
    
    def detect_anomalies(self, df):
        """Detect anomalies using Isolation Forest"""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Detect anomalies
        predictions = self.detector.fit_predict(X_scaled)
        anomaly_scores = self.detector.score_samples(X_scaled)
        
        # Add results to dataframe
        df['IsAnomaly'] = predictions == -1
        df['AnomalyScore'] = -anomaly_scores  # Higher score = more anomalous
        
        print(f"\n✓ Anomaly detection complete")
        print(f"  Total anomalies: {df['IsAnomaly'].sum()}")
        print(f"  Anomaly rate: {df['IsAnomaly'].mean():.2%}")
        
        return df
    
    def identify_anomalous_features(self, df):
        """Identify which features contribute to anomalies"""
        anomalies = df[df['IsAnomaly'] == True]
        
        if len(anomalies) == 0:
            print("No anomalies detected")
            return None
        
        # Compare anomalous vs normal distributions
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['IsAnomaly', 'AnomalyScore']]
        
        feature_importance = {}
        for col in numeric_cols:
            normal_mean = df[~df['IsAnomaly']][col].mean()
            anomaly_mean = anomalies[col].mean()
            
            if normal_mean != 0:
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
        df['IsAnomaly'].astype(int).plot()
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
        if 'SuccessRate' in df.columns:
            plt.scatter(
                df.index,
                df['SuccessRate'],
                c=df['IsAnomaly'],
                cmap='RdYlGn',
                alpha=0.6
            )
            plt.title('Connection Success Rate (Red = Anomaly)')
            plt.ylabel('Success Rate')
        
        # Plot 4: CPU usage if available
        plt.subplot(2, 2, 4)
        if 'UtilizationPct' in df.columns:
            plt.scatter(
                df.index,
                df['UtilizationPct'],
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
        df.to_csv(output_path)
        print(f"✓ Saved: {output_path}")
        
        # Save only anomalous records
        anomalies = df[df['IsAnomaly'] == True]
        anomaly_path = os.path.join(self.data_dir, "anomalies_only.csv")
        anomalies.to_csv(anomaly_path)
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
