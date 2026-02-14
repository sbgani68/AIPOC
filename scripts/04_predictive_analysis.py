"""
04 - Predictive Analysis
Forecast future capacity needs and performance issues using Prophet
"""
import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import json
import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class PredictiveAnalyzer:
    """Predict future AVD capacity and performance using Prophet"""
    
    def __init__(self):
        self.data_dir = "data"
        self.forecast_hours = 24
        self.high_risk_threshold = {
            'utilization': 80,  # % utilization
            'disconnect_events': 30,
            'success_rate_low': 0.95
        }
    
    def load_data(self):
        """Load data with anomalies"""
        path = os.path.join(self.data_dir, "data_with_anomalies.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['Hour'])
            df = df.sort_values('Hour')
            # Remove timezone for Prophet compatibility
            if df['Hour'].dt.tz is not None:
                df['Hour'] = df['Hour'].dt.tz_localize(None)
            print(f"✓ Loaded data: {df.shape}")
            print(f"  Date range: {df['Hour'].min()} to {df['Hour'].max()}")
            print(f"  Unique hosts: {df['SessionHostName'].nunique()}")
            return df
        else:
            print("✗ Data not found. Run 03_detection.py first.")
            return None
    
    def forecast_metric_per_host(self, df, host, metric_col, metric_name):
        """
        Forecast a single metric for a single host using Prophet
        
        Args:
            df: DataFrame with Hour and metric columns
            host: SessionHostName to forecast
            metric_col: Column name of metric to forecast
            metric_name: Display name for metric
        
        Returns:
            forecast DataFrame with ds, yhat, yhat_lower, yhat_upper
        """
        # Filter to host and prepare time series
        host_df = df[df['SessionHostName'] == host][['Hour', metric_col]].copy()
        host_df = host_df.dropna(subset=[metric_col])
        
        if len(host_df) < 10:
            print(f"  ⚠ Insufficient data for {host}: {metric_name} (only {len(host_df)} points)")
            return None
        
        # Prepare for Prophet (requires 'ds' and 'y' columns)
        prophet_df = host_df.rename(columns={'Hour': 'ds', metric_col: 'y'})
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
            interval_width=0.95,
            changepoint_prior_scale=0.05
        )
        
        try:
            model.fit(prophet_df)
            
            # Create future dataframe for next 24 hours
            future = model.make_future_dataframe(periods=self.forecast_hours, freq='h')
            forecast = model.predict(future)
            
            # Return only future predictions
            forecast_future = forecast[forecast['ds'] > prophet_df['ds'].max()].copy()
            
            return forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        except Exception as e:
            print(f"  ✗ Prophet failed for {host}: {metric_name} - {str(e)}")
            return None
    
    def forecast_rtt_trend(self, df):
        """
        Forecast RTT (Round Trip Time) for next 24 hours
        Returns aggregate RTT forecast across all hosts
        """
        print("\n📡 Forecasting RTT Trend...\n")
        
        # Find RTT column
        rtt_col = None
        for col in ['RTT', 'network_AvgRttMs', 'network_avg_rtt', 'network_RoundTripTimeMs', 'avg_rtt']:
            if col in df.columns:
                rtt_col = col
                break
        
        if rtt_col is None:
            print("  ⚠ No RTT column found")
            return None
        
        # Prepare aggregate RTT time series (average across all hosts per hour)
        df_ts = df.groupby('Hour')[rtt_col].mean().reset_index()
        df_ts = df_ts.rename(columns={'Hour': 'ds', rtt_col: 'y'})
        df_ts = df_ts.dropna()
        
        if len(df_ts) < 10:
            print("  ⚠ Insufficient RTT data (<10 points)")
            return None
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.05
        )
        model.fit(df_ts)
        
        # Forecast next 24 hours
        future = model.make_future_dataframe(periods=self.forecast_hours, freq='h')
        forecast = model.predict(future)
        
        # Get future predictions only
        forecast_future = forecast[forecast['ds'] > df_ts['ds'].max()].copy()
        
        # Analyze results
        avg_current_rtt = df_ts['y'].tail(24).mean()
        avg_forecast_rtt = forecast_future['yhat'].mean()
        max_forecast_rtt = forecast_future['yhat'].max()
        
        print(f"  Current avg RTT (last 24h): {avg_current_rtt:.1f} ms")
        print(f"  Forecast avg RTT (next 24h): {avg_forecast_rtt:.1f} ms")
        print(f"  Forecast max RTT (next 24h): {max_forecast_rtt:.1f} ms")
        
        # Identify high-risk periods
        high_risk_periods = forecast_future[forecast_future['yhat'] > 150]
        warning_periods = forecast_future[(forecast_future['yhat'] > 100) & (forecast_future['yhat'] <= 150)]
        
        if len(high_risk_periods) > 0:
            print(f"  🚨 CRITICAL: {len(high_risk_periods)} hours with RTT > 150ms")
        if len(warning_periods) > 0:
            print(f"  ⚠ WARNING: {len(warning_periods)} hours with RTT > 100ms")
        
        return forecast_future
    
    def forecast_disk_usage(self, df):
        """
        Forecast disk usage for next 24 hours
        Returns per-host disk forecasts to identify hosts at risk
        """
        print("\n💾 Forecasting Disk Usage...\n")
        
        # Find disk column
        disk_col = None
        for col in ['DiskFreePercent', 'disk_DiskFreePercent', 'disk_free', 'disk_% Free Space']:
            if col in df.columns:
                disk_col = col
                break
        
        if disk_col is None:
            print("  ⚠ No disk free space column found")
            return None
        
        hosts = df['SessionHostName'].unique()
        disk_forecasts = {}
        critical_hosts = []
        
        for host in hosts:
            if pd.isna(host) or host == 'unknown':
                continue
            
            # Prepare host-specific disk trend
            host_df = df[df['SessionHostName'] == host][['Hour', disk_col]].dropna()
            
            if len(host_df) < 10:
                continue
            
            disk_ts = host_df.rename(columns={'Hour': 'ds', disk_col: 'y'})
            
            # Train model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=False,
                yearly_seasonality=False,
                changepoint_prior_scale=0.1
            )
            
            try:
                model.fit(disk_ts)
                future = model.make_future_dataframe(periods=self.forecast_hours, freq='h')
                forecast = model.predict(future)
                forecast_future = forecast[forecast['ds'] > disk_ts['ds'].max()].copy()
                
                disk_forecasts[host] = forecast_future
                
                # Check if disk will drop below critical threshold
                min_forecast = forecast_future['yhat'].min()
                current_disk = disk_ts['y'].iloc[-1]
                
                if min_forecast < 15:
                    critical_hosts.append({
                        'host': host,
                        'current_disk': current_disk,
                        'forecast_min': min_forecast,
                        'severity': 'Critical'
                    })
                    print(f"  🚨 {host}: CRITICAL - Disk forecast min: {min_forecast:.1f}% (current: {current_disk:.1f}%)")
                elif min_forecast < 20:
                    critical_hosts.append({
                        'host': host,
                        'current_disk': current_disk,
                        'forecast_min': min_forecast,
                        'severity': 'Warning'
                    })
                    print(f"  ⚠ {host}: WARNING - Disk forecast min: {min_forecast:.1f}% (current: {current_disk:.1f}%)")
            
            except Exception as e:
                continue
        
        if critical_hosts:
            print(f"\n  Found {len(critical_hosts)} hosts at disk risk")
        else:
            print(f"\n  ✓ No hosts at disk risk in next 24 hours")
        
        return {'forecasts': disk_forecasts, 'critical_hosts': critical_hosts}
    
    def forecast_login_queue(self, df):
        """
        Forecast login queue length for next 24 hours
        Identifies peak login times and capacity bottlenecks
        """
        print("\n⏳ Forecasting Login Queue Trends...\n")
        
        # Find queue column
        queue_col = None
        for col in ['QueuedUsers', 'network_QueuedUsers', 'capacity_queued_users', 'QueueLength', 'queued_users']:
            if col in df.columns:
                queue_col = col
                break
        
        if queue_col is None:
            print("  ⚠ No login queue column found")
            return None
        
        # Aggregate queue across all hosts per hour
        df_ts = df.groupby('Hour')[queue_col].sum().reset_index()
        df_ts = df_ts.rename(columns={'Hour': 'ds', queue_col: 'y'})
        df_ts = df_ts.dropna()
        
        if len(df_ts) < 10 or df_ts['y'].sum() == 0:
            print("  ⚠ Insufficient queue data or no queuing events")
            return None
        
        # Train Prophet model
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=False,
            yearly_seasonality=False,
            changepoint_prior_scale=0.15
        )
        model.fit(df_ts)
        
        # Forecast next 24 hours
        future = model.make_future_dataframe(periods=self.forecast_hours, freq='h')
        forecast = model.predict(future)
        forecast_future = forecast[forecast['ds'] > df_ts['ds'].max()].copy()
        
        # Analyze results
        current_avg_queue = df_ts['y'].tail(24).mean()
        forecast_avg_queue = forecast_future['yhat'].mean()
        forecast_max_queue = forecast_future['yhat'].max()
        
        print(f"  Current avg queue (last 24h): {current_avg_queue:.1f} users")
        print(f"  Forecast avg queue (next 24h): {forecast_avg_queue:.1f} users")
        print(f"  Forecast max queue (next 24h): {forecast_max_queue:.1f} users")
        
        # Identify high queue periods
        high_queue_periods = forecast_future[forecast_future['yhat'] > 10]
        
        if len(high_queue_periods) > 0:
            print(f"  ⚠ WARNING: {len(high_queue_periods)} hours with queue > 10 users")
            peak_time = forecast_future.loc[forecast_future['yhat'].idxmax(), 'ds']
            print(f"  Peak queue expected at: {peak_time}")
        
        return forecast_future
    
    def forecast_all_hosts(self, df):
        """
        Forecast key metrics for all hosts
        
        Returns:
            Dictionary: {host: {metric: forecast_df}}
        """
        print("\n🔮 Forecasting metrics per host...\n")
        
        # Define metrics to forecast
        metrics = {
            'network_SuccessRate': 'Success Rate',
            'capacity_UtilizationPct': 'Utilization %',
            'capacity_ActiveSessions': 'Active Sessions',
            'hygiene_DisconnectEventCount': 'Disconnect Events',
            'network_TotalConnections': 'Total Connections',
            'RTT': 'RTT (ms)',
            'DiskFreePercent': 'Disk Free %',
            'QueuedUsers': 'Queued Users'
        }
        
        # Available metrics in dataset
        available_metrics = {
            col: name for col, name in metrics.items() 
            if col in df.columns
        }
        
        if not available_metrics:
            print("✗ No forecast metrics available")
            return {}
        
        print(f"  Forecasting: {', '.join(available_metrics.values())}\n")
        
        # Forecast per host
        hosts = df['SessionHostName'].unique()
        all_forecasts = {}
        
        for host in hosts:
            if pd.isna(host) or host == 'unknown':
                continue
                
            host_forecasts = {}
            print(f"📊 Host: {host}")
            
            for metric_col, metric_name in available_metrics.items():
                forecast = self.forecast_metric_per_host(df, host, metric_col, metric_name)
                if forecast is not None:
                    host_forecasts[metric_col] = forecast
                    avg_forecast = forecast['yhat'].mean()
                    print(f"  ✓ {metric_name}: Avg forecast = {avg_forecast:.2f}")
            
            if host_forecasts:
                all_forecasts[host] = host_forecasts
            print()
        
        return all_forecasts
    
    def identify_high_risk_hosts(self, all_forecasts, disk_results=None):
        """
        Analyze forecasts to identify high-risk hosts
        
        Args:
            all_forecasts: {host: {metric: forecast_df}}
            disk_results: Output from forecast_disk_usage()
        
        Returns:
            List of dicts with high-risk host details
        """
        print("🚨 Identifying high-risk hosts...\n")
        
        high_risk_hosts = []
        
        for host, forecasts in all_forecasts.items():
            risk_factors = []
            risk_score = 0
            
            # Check utilization forecast
            if 'capacity_UtilizationPct' in forecasts:
                util_forecast = forecasts['capacity_UtilizationPct']
                max_util = util_forecast['yhat'].max()
                avg_util = util_forecast['yhat'].mean()
                
                if max_util > 90:
                    risk_factors.append(f"🚨 Critical utilization (max: {max_util:.1f}%)")
                    risk_score += 40
                elif max_util > self.high_risk_threshold['utilization']:
                    risk_factors.append(f"⚠ High utilization (max: {max_util:.1f}%, avg: {avg_util:.1f}%)")
                    risk_score += 25
            
            # Check RTT forecast
            if 'RTT' in forecasts:
                rtt_forecast = forecasts['RTT']
                max_rtt = rtt_forecast['yhat'].max()
                avg_rtt = rtt_forecast['yhat'].mean()
                
                if max_rtt > 150:
                    risk_factors.append(f"🚨 Critical RTT (max: {max_rtt:.0f}ms)")
                    risk_score += 35
                elif max_rtt > 100:
                    risk_factors.append(f"⚠ High RTT (max: {max_rtt:.0f}ms)")
                    risk_score += 20
            
            # Check disk forecast
            if 'DiskFreePercent' in forecasts:
                disk_forecast = forecasts['DiskFreePercent']
                min_disk = disk_forecast['yhat'].min()
                
                if min_disk < 15:
                    risk_factors.append(f"🚨 Critical disk space (min: {min_disk:.1f}%)")
                    risk_score += 50
                elif min_disk < 20:
                    risk_factors.append(f"⚠ Low disk space (min: {min_disk:.1f}%)")
                    risk_score += 30
            
            # Check disconnect events
            if 'hygiene_DisconnectEventCount' in forecasts:
                disconnect_forecast = forecasts['hygiene_DisconnectEventCount']
                max_disconnects = disconnect_forecast['yhat'].max()
                
                if max_disconnects > self.high_risk_threshold['disconnect_events']:
                    risk_factors.append(f"⚠ High disconnect events (max: {max_disconnects:.1f})")
                    risk_score += 15
            
            # Check success rate
            if 'network_SuccessRate' in forecasts:
                success_forecast = forecasts['network_SuccessRate']
                min_success = success_forecast['yhat'].min()
                
                if min_success < self.high_risk_threshold['success_rate_low']:
                    risk_factors.append(f"⚠ Low success rate (min: {min_success:.2%})")
                    risk_score += 20
            
            # If any risk factors, add to high-risk list
            if risk_factors:
                high_risk_hosts.append({
                    'host': host,
                    'risk_factors': risk_factors,
                    'risk_count': len(risk_factors),
                    'risk_score': risk_score,
                    'severity': 'Critical' if risk_score >= 50 else 'Warning'
                })
        
        # Add disk-specific critical hosts
        if disk_results and disk_results.get('critical_hosts'):
            for disk_host in disk_results['critical_hosts']:
                # Check if host already in list
                existing = next((h for h in high_risk_hosts if h['host'] == disk_host['host']), None)
                if not existing:
                    high_risk_hosts.append({
                        'host': disk_host['host'],
                        'risk_factors': [f"Disk forecast min: {disk_host['forecast_min']:.1f}%"],
                        'risk_count': 1,
                        'risk_score': 50 if disk_host['severity'] == 'Critical' else 30,
                        'severity': disk_host['severity']
                    })
        
        # Sort by risk score (descending)
        high_risk_hosts = sorted(high_risk_hosts, key=lambda x: x['risk_score'], reverse=True)
        
        # Display results
        if high_risk_hosts:
            print(f"Found {len(high_risk_hosts)} high-risk hosts:\n")
            for i, host_info in enumerate(high_risk_hosts[:10], 1):
                severity_icon = "🚨" if host_info['severity'] == 'Critical' else "⚠"
                print(f"{i}. {severity_icon} {host_info['host']} (Risk Score: {host_info['risk_score']})")
                for factor in host_info['risk_factors']:
                    print(f"   • {factor}")
                print()
        else:
            print("✓ No high-risk hosts identified in next 24 hours")
        
        return high_risk_hosts
    
    def save_forecasts(self, all_forecasts, high_risk_hosts):
        """Save forecast results to JSON"""
        output = {
            'forecast_hours': self.forecast_hours,
            'high_risk_hosts': high_risk_hosts,
            'forecasts': {}
        }
        
        # Convert forecasts to serializable format
        for host, forecasts in all_forecasts.items():
            output['forecasts'][host] = {}
            for metric, forecast_df in forecasts.items():
                output['forecasts'][host][metric] = {
                    'timestamps': forecast_df['ds'].astype(str).tolist(),
                    'forecast': forecast_df['yhat'].tolist(),
                    'lower_bound': forecast_df['yhat_lower'].tolist(),
                    'upper_bound': forecast_df['yhat_upper'].tolist()
                }
        
        output_path = os.path.join(self.data_dir, "predictive_forecasts.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Forecasts saved: {output_path}")
    
    def visualize_top_risks(self, df, all_forecasts, high_risk_hosts):
        """Visualize forecasts for top 3 high-risk hosts"""
        if not high_risk_hosts:
            print("\n  No high-risk visualizations (no risks detected)")
            return
        
        top_hosts = [h['host'] for h in high_risk_hosts[:3]]
        
        for host in top_hosts:
            if host not in all_forecasts:
                continue
            
            forecasts = all_forecasts[host]
            
            # Create subplot for each metric
            num_metrics = len(forecasts)
            if num_metrics == 0:
                continue
            
            fig, axes = plt.subplots(num_metrics, 1, figsize=(14, 4 * num_metrics))
            if num_metrics == 1:
                axes = [axes]
            
            fig.suptitle(f'24-Hour Forecast: {host}', fontsize=16, fontweight='bold')
            
            for idx, (metric_col, forecast_df) in enumerate(forecasts.items()):
                ax = axes[idx]
                
                # Historical data
                hist_data = df[df['SessionHostName'] == host][['Hour', metric_col]].dropna()
                
                # Plot historical
                ax.plot(hist_data['Hour'], hist_data[metric_col], 
                       label='Historical', color='blue', alpha=0.6, linewidth=2)

                # Plot forecast
                ax.plot(forecast_df['ds'], forecast_df['yhat'], 
                       label='Forecast', color='red', linewidth=2)

                # Confidence interval
                ax.fill_between(
                    forecast_df['ds'],
                    forecast_df['yhat_lower'],
                    forecast_df['yhat_upper'],
                    alpha=0.2, color='red', label='95% Confidence'
                )

                # Formatting
                metric_name = metric_col.replace('_', ' ').title()
                ax.set_title(metric_name, fontsize=12, fontweight='bold')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save figure
            safe_host = host.replace('/', '_').replace('\\', '_').replace('.', '_')
            output_path = os.path.join(self.data_dir, f"forecast_{safe_host}.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Saved visualization: {output_path}")
    
    def run(self):
        """Run complete predictive analysis"""
        print("="*80)
        print("STEP 4: PREDICTIVE ANALYSIS (Prophet)")
        print("="*80)
        
        # Load data
        df = self.load_data()
        if df is None:
            return

        # Aggregate forecasts
        print("\n" + "="*80)
        print("AGGREGATE TREND FORECASTS")
        print("="*80)
        rtt_forecast = self.forecast_rtt_trend(df)
        disk_results = self.forecast_disk_usage(df)
        queue_forecast = self.forecast_login_queue(df)
        
        # Forecast all hosts
        print("\n" + "="*80)
        print("PER-HOST METRIC FORECASTS")
        print("="*80)
        all_forecasts = self.forecast_all_hosts(df)
        
        if not all_forecasts:
            print("\n✗ No forecasts generated")
            return
        
        # Identify high-risk hosts
        print("\n" + "="*80)
        print("HIGH-RISK HOST IDENTIFICATION")
        print("="*80)
        high_risk_hosts = self.identify_high_risk_hosts(all_forecasts, disk_results)
        
        # Save results
        self.save_forecasts(all_forecasts, high_risk_hosts)
        
        # Visualize top risks
        print("\n📊 Generating visualizations...")
        self.visualize_top_risks(df, all_forecasts, high_risk_hosts)
        
        print("\n" + "="*80)
        if rtt_forecast is not None:
            print(f"✓ RTT forecast: {len(rtt_forecast)} hours predicted")
        if disk_results and disk_results.get('critical_hosts'):
            print(f"🚨 Disk pressure: {len(disk_results['critical_hosts'])} hosts at risk")
        if queue_forecast is not None:
            print(f"⏳ Login queue: Peak forecast {queue_forecast['yhat'].max():.0f} users")
        print(f"📊 Per-host forecasts: {len(all_forecasts)} hosts")
        print(f"🎯 High-risk hosts: {len(high_risk_hosts)}")
        print("✓ Predictive analysis complete!")
        print("="*80)


if __name__ == "__main__":
    analyzer = PredictiveAnalyzer()
    analyzer.run()

