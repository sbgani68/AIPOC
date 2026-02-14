"""
01 - Data Collection Script
Collects logs from Azure Virtual Desktop environment
"""
import os
from datetime import datetime, timezone
from datetime import timedelta

import pandas as pd
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
from dotenv import load_dotenv

load_dotenv()


class AVDDataCollector:
    """Collect AVD logs from Azure Monitor"""

    def __init__(self):
        self.workspace_id = os.getenv("AVD_WORKSPACE_ID")
        if not self.workspace_id:
            raise ValueError("AVD_WORKSPACE_ID is not set in environment variables.")

        self.host_pool_name = os.getenv("AVD_HOSTPOOL_NAME", "jswvdprd_ARTHUR_02")
        self.lookback_days = int(os.getenv("AVD_LOOKBACK_DAYS", "30"))
        self.chunk_hours = int(os.getenv("AVD_CHUNK_HOURS", "24"))

        self.credential = DefaultAzureCredential()
        self.logs_client = LogsQueryClient(self.credential, connection_verify=False)
        self.timespan = timedelta(days=30)
        self.data_dir = "data"
        os.makedirs(self.data_dir, exist_ok=True)

    def _hostpool_filter_kql(self) -> str:
        escaped_hostpool = self.host_pool_name.replace('"', '\\"')
        return f"""
        | extend _HostPoolName = tostring(column_ifexists("HostPoolName", ""))
        | extend _ResourceId = tostring(column_ifexists("_ResourceId", ""))
        | where _HostPoolName =~ "{escaped_hostpool}"
            or _ResourceId has "/hostpools/{escaped_hostpool}"
            or _ResourceId has "/hostPools/{escaped_hostpool}"
        """

    def _response_to_dataframe(self, response) -> pd.DataFrame:
        tables = []
        if hasattr(response, "tables") and response.tables:
            tables = response.tables
        elif hasattr(response, "partial_data") and response.partial_data and hasattr(response.partial_data, "tables"):
            tables = response.partial_data.tables

        if not tables:
            return pd.DataFrame()

        table = tables[0]
        column_names = [col.name if hasattr(col, "name") else str(col) for col in table.columns]
        return pd.DataFrame(table.rows, columns=column_names)

    def _execute_query(self, query: str, output_file: str, expected_columns=None) -> pd.DataFrame:
        """Execute KQL query in chunks and save result to CSV."""
        print(f"  → Querying {output_file} in {self.chunk_hours}h chunks ...")
        output_path = os.path.join(self.data_dir, output_file)
        range_end = datetime.now(timezone.utc)
        range_start = range_end - timedelta(days=self.lookback_days)
        chunk_start = range_start
        frames = []

        chunk_number = 0
        while chunk_start < range_end:
            chunk_number += 1
            chunk_end = min(chunk_start + timedelta(hours=self.chunk_hours), range_end)
            print(f"    • Chunk {chunk_number}: {chunk_start.isoformat()} → {chunk_end.isoformat()}")

            try:
                response = self.logs_client.query_workspace(
                    workspace_id=self.workspace_id,
                    query=query,
                    timespan=(chunk_start, chunk_end),
                )

                if response.status == LogsQueryStatus.SUCCESS:
                    chunk_df = self._response_to_dataframe(response)
                    if not chunk_df.empty:
                        frames.append(chunk_df)
                elif response.status == LogsQueryStatus.PARTIAL:
                    chunk_df = self._response_to_dataframe(response)
                    if not chunk_df.empty:
                        frames.append(chunk_df)
                    print(f"    ⚠ Partial result for {output_file} chunk {chunk_number}: {response.partial_error}")
                else:
                    print(f"    ⚠ Failed chunk {chunk_number} for {output_file}")
            except Exception as exc:
                print(f"    ⚠ Error in chunk {chunk_number} for {output_file}: {exc}")

            chunk_start = chunk_end

        try:
            if frames:
                df = pd.concat(frames, ignore_index=True)
            else:
                df = pd.DataFrame(columns=expected_columns or [])

            if "TimeGenerated" in df.columns:
                df["TimeGenerated"] = pd.to_datetime(df["TimeGenerated"], errors="coerce")
                df = df.sort_values("TimeGenerated", ascending=False)

            df = df.drop_duplicates()
            df.to_csv(output_path, index=False)
            print(f"    ✓ Saved {output_file}: {len(df)} rows")
            return df
        except Exception as exc:
            pd.DataFrame(columns=expected_columns or []).to_csv(output_path, index=False)
            print(f"    ⚠ Query failed for {output_file}. Saved empty file. Reason: {exc}")
            return pd.DataFrame()

    def collect_network_logs(self):
        query = f"""
        WVDConnections
        {self._hostpool_filter_kql()}
        | extend UserName = tostring(column_ifexists("UserName", ""))
        | extend SessionHostName = tostring(column_ifexists("SessionHostName", ""))
        | extend CorrelationId = tostring(column_ifexists("CorrelationId", ""))
        | extend State = tostring(column_ifexists("State", ""))
        | extend ClientOS = tostring(column_ifexists("ClientOS", ""))
        | extend ClientType = tostring(column_ifexists("ClientType", ""))
        | extend ClientVersion = tostring(column_ifexists("ClientVersion", ""))
        | extend ResourceAlias = tostring(column_ifexists("ResourceAlias", ""))
        | extend RoundTripTimeMs = todouble(column_ifexists("EstRoundTripTimeInMs", real(null)))
        | extend UdpConnectionType = tostring(column_ifexists("UdpConnectionType", ""))
        | extend ClientSideIPAddress = tostring(column_ifexists("ClientSideIPAddress", ""))
        | extend GatewayRegion = tostring(column_ifexists("GatewayRegion", ""))
        | extend ConnectionType = tostring(column_ifexists("ConnectionType", ""))
        | extend SessionHostIPAddress = tostring(column_ifexists("SessionHostIPAddress", ""))
        | project TimeGenerated, UserName, SessionHostName, CorrelationId, State,
                  ClientOS, ClientType, ClientVersion, ResourceAlias,
                  RoundTripTimeMs, UdpConnectionType, ClientSideIPAddress,
                  GatewayRegion, ConnectionType, SessionHostIPAddress
        | order by TimeGenerated desc
        """
        return self._execute_query(
            query,
            "network_logs.csv",
            expected_columns=[
                "TimeGenerated",
                "UserName",
                "SessionHostName",
                "CorrelationId",
                "State",
                "ClientOS",
                "ClientType",
                "ClientVersion",
                "ResourceAlias",
                "RoundTripTimeMs",
                "UdpConnectionType",
                "ClientSideIPAddress",
                "GatewayRegion",
                "ConnectionType",
                "SessionHostIPAddress",
            ],
        )

    def collect_sessionhost_logs(self):
        query = f"""
        Perf
        {self._hostpool_filter_kql()}
        | where ObjectName in ("Processor", "Memory", "LogicalDisk", "PhysicalDisk")
        | where CounterName in ("% Processor Time", "Available MBytes", "% Free Space",
                                "Avg. Disk sec/Read", "Avg. Disk sec/Write",
                                "Current Disk Queue Length")
        | project TimeGenerated, Computer, ObjectName, CounterName, CounterValue, InstanceName
        | order by TimeGenerated desc
        """
        return self._execute_query(
            query,
            "sessionhost_logs.csv",
            expected_columns=[
                "TimeGenerated",
                "Computer",
                "ObjectName",
                "CounterName",
                "CounterValue",
                "InstanceName",
            ],
        )

    def collect_capacity_logs(self):
        query = f"""
        WVDAgentHealthStatus
        {self._hostpool_filter_kql()}
        | extend SessionHostName = tostring(column_ifexists("SessionHostName", ""))
        | extend Status = tostring(column_ifexists("Status", ""))
        | extend ActiveSessions = toint(column_ifexists("ActiveSessions", int(null)))
        | extend MaxSessionLimit = toint(column_ifexists("MaxSessionLimit", int(null)))
        | project TimeGenerated, SessionHostName, Status, ActiveSessions, MaxSessionLimit
        | summarize ActiveSessions=avg(ActiveSessions), MaxSessionLimit=max(MaxSessionLimit), Status=any(Status)
            by bin(TimeGenerated, 15m), SessionHostName
        | order by TimeGenerated desc
        """
        return self._execute_query(
            query,
            "capacity_logs.csv",
            expected_columns=[
                "TimeGenerated",
                "SessionHostName",
                "ActiveSessions",
                "MaxSessionLimit",
                "Status",
            ],
        )

    def collect_fslogix_logs(self):
        query = f"""
        let FslogixEventLogs =
            Event
            {self._hostpool_filter_kql()}
            | where Source in ("FSLogix-Apps", "FSLogix-CloudCache", "frxsvc", "frxdrv")
                or RenderedDescription has_cs "FSLogix"
            | extend EventID = toint(column_ifexists("EventID", int(null)))
            | extend EventLevelName = tostring(column_ifexists("EventLevelName", "Info"))
            | extend Computer = tostring(column_ifexists("Computer", ""))
            | extend Source = tostring(column_ifexists("Source", "Event"))
            | extend RenderedDescription = tostring(column_ifexists("RenderedDescription", ""))
            | project TimeGenerated, Computer, EventID, EventLevelName, RenderedDescription, Source;

        let FslogixCheckpointSignals =
            WVDCheckpoints
            {self._hostpool_filter_kql()}
            | extend ParametersText = tostring(column_ifexists("Parameters", ""))
            | extend CheckpointName = tostring(column_ifexists("Name", ""))
            | where ParametersText has "fslogix"
                or ParametersText has "profile"
                or ParametersText has "vhd"
                or ParametersText has "error"
                or CheckpointName has "error"
            | extend Computer = tostring(column_ifexists("Source", ""))
            | extend EventID = int(null)
            | extend EventLevelName = iif(ParametersText has "error" or CheckpointName has "error", "Error", "Info")
            | extend Source = "WVDCheckpoints"
            | extend RenderedDescription = strcat("Checkpoint=", CheckpointName, "; Parameters=", ParametersText)
            | project TimeGenerated, Computer, EventID, EventLevelName, RenderedDescription, Source;

        FslogixEventLogs
        | union FslogixCheckpointSignals
        | order by TimeGenerated desc
        """
        return self._execute_query(
            query,
            "fslogix_logs.csv",
            expected_columns=[
                "TimeGenerated",
                "Computer",
                "EventID",
                "EventLevelName",
                "RenderedDescription",
                "Source",
            ],
        )

    def collect_disk_logs(self):
        query = f"""
        Perf
        {self._hostpool_filter_kql()}
        | where ObjectName in ("LogicalDisk", "PhysicalDisk")
        | where CounterName in ("% Free Space", "Free Megabytes", "Avg. Disk sec/Read",
                                "Avg. Disk sec/Write", "Current Disk Queue Length",
                                "Disk Read Bytes/sec", "Disk Write Bytes/sec")
        | project TimeGenerated, Computer, InstanceName, CounterName, CounterValue
        | order by TimeGenerated desc
        """
        return self._execute_query(
            query,
            "disk_logs.csv",
            expected_columns=[
                "TimeGenerated",
                "Computer",
                "InstanceName",
                "CounterName",
                "CounterValue",
            ],
        )

    def collect_hygiene_logs(self):
        query = f"""
        let SessionDisconnectSignals =
            WVDConnections
            {self._hostpool_filter_kql()}
            | extend UserName = tostring(column_ifexists("UserName", ""))
            | extend SessionHostName = tostring(column_ifexists("SessionHostName", ""))
            | extend CorrelationId = tostring(column_ifexists("CorrelationId", ""))
            | extend State = tolower(tostring(column_ifexists("State", "")))
            | extend SessionStartTime = todatetime(column_ifexists("SessionStartTime", datetime(null)))
            | extend IsDisconnected = State in ("disconnected", "disconnectedbyclient", "clientdisconnected", "abruptlydisconnected")
            | extend DisconnectReason = iif(IsDisconnected, strcat("state:", State), "")
            | extend DisconnectedSessionDurationMinutes = tolong(iif(IsDisconnected and isnotnull(SessionStartTime), datetime_diff('minute', TimeGenerated, SessionStartTime), long(null)))
            | project TimeGenerated, UserName, SessionHostName, CorrelationId, State,
                      IsDisconnected, DisconnectReason, DisconnectedSessionDurationMinutes;

        let CheckpointDisconnectSignals =
            WVDCheckpoints
            {self._hostpool_filter_kql()}
            | extend CheckpointName = tolower(tostring(column_ifexists("Name", "")))
            | extend ParametersText = tolower(tostring(column_ifexists("Parameters", "")))
            | extend SourceText = tostring(column_ifexists("Source", ""))
            | extend UserName = tostring(column_ifexists("UserName", ""))
            | extend CorrelationId = tostring(column_ifexists("CorrelationId", ""))
            | where CheckpointName has "disconnect"
                or ParametersText has "disconnect"
                or ParametersText has "client disconnected"
                or ParametersText has "abrupt"
                or ParametersText has "terminated"
                or ParametersText has "broken"
                or ParametersText has "timeout"
            | extend SessionHostName = tostring(column_ifexists("SessionHostName", SourceText))
            | extend State = "checkpoint"
            | extend IsDisconnected = true
            | extend DisconnectReason = strcat("checkpoint:", CheckpointName)
            | extend DisconnectedSessionDurationMinutes = long(null)
            | project TimeGenerated, UserName, SessionHostName, CorrelationId, State,
                      IsDisconnected, DisconnectReason, DisconnectedSessionDurationMinutes;

        let Combined = union isfuzzy=true SessionDisconnectSignals, CheckpointDisconnectSignals;
        let HostCounts = Combined
            | where IsDisconnected
            | summarize DisconnectedSessionsPerHost = count() by SessionHostName;

        Combined
        | where IsDisconnected
        | join kind=leftouter HostCounts on SessionHostName
        | project TimeGenerated, UserName, SessionHostName, CorrelationId, State,
                  DisconnectReason, DisconnectedSessionDurationMinutes, DisconnectedSessionsPerHost
        | order by TimeGenerated desc
        """
        df = self._execute_query(
            query,
            "hygiene_logs.csv",
            expected_columns=[
                "TimeGenerated",
                "UserName",
                "SessionHostName",
                "CorrelationId",
                "State",
                "DisconnectReason",
                "DisconnectedSessionDurationMinutes",
                "DisconnectedSessionsPerHost",
            ],
        )
        self._save_hygiene_hourly_summary(df)
        return df

    def _save_hygiene_hourly_summary(self, hygiene_df: pd.DataFrame):
        """Save host/hour summary for hygiene disconnect metrics."""
        output_file = os.path.join(self.data_dir, "hygiene_summary_hourly.csv")

        if hygiene_df.empty:
            pd.DataFrame().to_csv(output_file, index=False)
            print("    ✓ Saved hygiene_summary_hourly.csv: 0 rows")
            return

        summary_df = hygiene_df.copy()
        summary_df["TimeGenerated"] = pd.to_datetime(summary_df["TimeGenerated"], errors="coerce")
        summary_df["SessionHostName"] = summary_df["SessionHostName"].fillna("unknown")
        summary_df["DisconnectedSessionDurationMinutes"] = pd.to_numeric(
            summary_df["DisconnectedSessionDurationMinutes"], errors="coerce"
        )
        summary_df = summary_df.dropna(subset=["TimeGenerated"])
        summary_df["Hour"] = summary_df["TimeGenerated"].dt.floor("h")

        grouped = (
            summary_df.groupby(["Hour", "SessionHostName"], dropna=False)
            .agg(
                DisconnectEventCount=("CorrelationId", "count"),
                AvgDisconnectedSessionDurationMinutes=("DisconnectedSessionDurationMinutes", "mean"),
                MaxDisconnectedSessionDurationMinutes=("DisconnectedSessionDurationMinutes", "max"),
            )
            .reset_index()
            .sort_values(["Hour", "SessionHostName"], ascending=[False, True])
        )

        grouped.to_csv(output_file, index=False)
        print(f"    ✓ Saved hygiene_summary_hourly.csv: {len(grouped)} rows")

    def collect_client_logs(self):
        query = f"""
        WVDCheckpoints
        {self._hostpool_filter_kql()}
        | project TimeGenerated, Name, Parameters, Source, UserName, CorrelationId
        | order by TimeGenerated desc
        """
        return self._execute_query(
            query,
            "client_logs.csv",
            expected_columns=[
                "TimeGenerated",
                "Name",
                "Parameters",
                "Source",
                "UserName",
                "CorrelationId",
            ],
        )

    def collect_all(self):
        print("Starting AVD data collection (last 30 days)...")
        print(f"Workspace ID: {self.workspace_id}\n")

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
