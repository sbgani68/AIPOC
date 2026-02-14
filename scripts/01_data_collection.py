"""
01 - Data Collection Script
Collects logs from Azure Virtual Desktop environment
"""
import os
from datetime import datetime, timedelta
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class AVDDataCollector:
    """Collect AVD logs from Azure Monitor"""
    
    def __init__(self):
        self.credential = DefaultAzureCredential()
        self.logs_client = LogsQueryClient(self.credential)
        self.workspace_id = os.getenv("AVD_WORKSPACE_ID")
        self.timespan = timedelta(hours=24)
    
    def collect_network_logs(self):
        """Collect network connectivity logs"""
        query = """
        WVDConnections
        | where TimeGenerated > ago(24h)
        | project TimeGenerated, UserName, CorrelationId, State, 
                  ClientOS, ClientType, ClientVersion, ResourceAlias
        | order by TimeGenerated desc
        """
        return self._execute_query(query, "network_logs.csv")
    
    def collect_sessionhost_logs(self):
        """Collect session host performance logs"""
        query = """
        Perf
        | where TimeGenerated > ago(24h)
        | where ObjectName in ("Processor", "Memory", "LogicalDisk")
        | where CounterName in ("% Processor Time", "Available MBytes", "% Free Space")
        | project TimeGenerated, Computer, ObjectName, CounterName, 
                  CounterValue, InstanceName
        """
        return self._execute_query(query, "sessionhost_logs.csv")
    
    def collect_capacity_logs(self):
        """Collect capacity and session data"""
        query = """
        WVDAgentHealthStatus
        | where TimeGenerated > ago(24h)
        | project TimeGenerated, SessionHostName, Status, 
                  LastHeartBeat, ActiveSessions, MaxSessionLimit
        """
        return self._execute_query(query, "capacity_logs.csv")
    
    def collect_fslogix_logs(self):
        """Collect FSLogix profile logs"""
        query = """
        Event
        | where TimeGenerated > ago(24h)
        | where Source == "FSLogix-Apps"
        | project TimeGenerated, Computer, EventID, EventLevelName, 
                  RenderedDescription
        """
        return self._execute_query(query, "fslogix_logs.csv")
    
    def collect_disk_logs(self):
        """Collect disk performance logs"""
        query = """
        Perf
        | where TimeGenerated > ago(24h)
        | where ObjectName == "LogicalDisk"
        | where CounterName in ("% Free Space", "Avg. Disk sec/Read", 
                                "Avg. Disk sec/Write", "Current Disk Queue Length")
        | project TimeGenerated, Computer, InstanceName, CounterName, CounterValue
        """
        return self._execute_query(query, "disk_logs.csv")
    
    def collect_hygiene_logs(self):
        """Collect update and health status logs"""
        query = """
        Update
        | where TimeGenerated > ago(24h)
        | where Classification in ("Critical Updates", "Security Updates")
        | project TimeGenerated, Computer, Title, Classification, 
                  UpdateState, ApprovalSource
        """
        return self._execute_query(query, "hygiene_logs.csv")
    
    def collect_client_logs(self):
        """Collect client connection diagnostics"""
        query = """
        WVDCheckpoints
        | where TimeGenerated > ago(24h)
        | project TimeGenerated, Name, Parameters, Source, UserName
        """
        return self._execute_query(query, "client_logs.csv")
    
    def _execute_query(self, query, output_file):
        """Execute KQL query and save to CSV"""
        try:
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=self.timespan
            )
            
            if response.status == LogsQueryStatus.SUCCESS:
                # Convert to DataFrame
                table = response.tables[0]
                df = pd.DataFrame(
                    data=table.rows,
                    columns=[col.name for col in table.columns]
                )
                
                # Save to CSV
                output_path = os.path.join("data", output_file)
                df.to_csv(output_path, index=False)
                print(f"✓ Saved {len(df)} rows to {output_file}")
                return df
            else:
                print(f"✗ Query failed for {output_file}")
                return None
                
        except Exception as e:
            print(f"✗ Error collecting {output_file}: {e}")
            return None
    
    def collect_all(self):
        """Collect all logs"""
        print("Starting AVD data collection...\n")
        
        self.collect_network_logs()
        self.collect_sessionhost_logs()
        self.collect_capacity_logs()
        self.collect_fslogix_logs()
        self.collect_disk_logs()
        self.collect_hygiene_logs()
        self.collect_client_logs()
        
        print("\n✓ Data collection complete!")


if __name__ == "__main__":
    collector = AVDDataCollector()
    collector.collect_all()
