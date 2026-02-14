"""
Azure Virtual Desktop Host Pool Monitoring Module
Collects health metrics, session data, and diagnostic information
"""
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from azure.identity import ClientSecretCredential
from azure.mgmt.desktopvirtualization import DesktopVirtualizationMgmtClient
from azure.monitor.query import LogsQueryClient, LogsQueryStatus
import logging

logger = logging.getLogger(__name__)


@dataclass
class HostHealth:
    """Host health status and metrics"""
    host_name: str
    status: str
    last_heartbeat: Optional[datetime]
    sessions_active: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    errors: List[Dict]
    warnings: List[Dict]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.last_heartbeat:
            data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data


@dataclass
class HostPoolMetrics:
    """Aggregated host pool metrics"""
    pool_name: str
    total_hosts: int
    healthy_hosts: int
    unhealthy_hosts: int
    total_sessions: int
    available_capacity: int
    avg_cpu_usage: float
    avg_memory_usage: float
    critical_errors: int
    warnings: int
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary with datetime serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AVDMonitor:
    """Azure Virtual Desktop monitoring and data collection"""
    
    def __init__(self, subscription_id: str, tenant_id: str, 
                 client_id: str, client_secret: str, workspace_id: str):
        """Initialize Azure clients"""
        self.subscription_id = subscription_id
        self.workspace_id = workspace_id
        
        # Create credential
        self.credential = ClientSecretCredential(
            tenant_id=tenant_id,
            client_id=client_id,
            client_secret=client_secret
        )
        
        # Initialize clients
        self.avd_client = DesktopVirtualizationMgmtClient(
            credential=self.credential,
            subscription_id=subscription_id
        )
        
        self.logs_client = LogsQueryClient(self.credential)
        
    async def get_host_pool_info(self, resource_group: str, host_pool_name: str) -> Dict:
        """Get basic host pool information"""
        try:
            host_pool = self.avd_client.host_pools.get(
                resource_group_name=resource_group,
                host_pool_name=host_pool_name
            )
            
            return {
                'name': host_pool.name,
                'location': host_pool.location,
                'type': host_pool.host_pool_type,
                'load_balancer_type': host_pool.load_balancer_type,
                'max_session_limit': host_pool.max_session_limit,
                'friendly_name': host_pool.friendly_name
            }
        except Exception as e:
            logger.error(f"Error fetching host pool info: {e}")
            return {}
    
    async def get_session_hosts(self, resource_group: str, host_pool_name: str) -> List[Dict]:
        """Get all session hosts in the pool"""
        try:
            hosts = self.avd_client.session_hosts.list(
                resource_group_name=resource_group,
                host_pool_name=host_pool_name
            )
            
            host_list = []
            for host in hosts:
                host_list.append({
                    'name': host.name,
                    'status': host.status,
                    'sessions': host.sessions,
                    'last_heartbeat': host.last_heart_beat,
                    'update_state': host.update_state,
                    'os_version': host.os_version
                })
            
            return host_list
        except Exception as e:
            logger.error(f"Error fetching session hosts: {e}")
            return []
    
    async def query_host_metrics(self, host_name: str, hours: int = 1) -> Dict:
        """Query performance metrics from Log Analytics"""
        try:
            # KQL query for performance metrics
            query = f"""
            Perf
            | where Computer == "{host_name}"
            | where TimeGenerated > ago({hours}h)
            | where CounterName in ("% Processor Time", "% Committed Bytes In Use", "% Free Space")
            | summarize 
                AvgCPU = avg(iif(CounterName == "% Processor Time", CounterValue, 0)),
                AvgMemory = avg(iif(CounterName == "% Committed Bytes In Use", CounterValue, 0)),
                AvgDiskFree = avg(iif(CounterName == "% Free Space", CounterValue, 0))
            by Computer
            """
            
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=timedelta(hours=hours)
            )
            
            if response.status == LogsQueryStatus.SUCCESS:
                if response.tables and len(response.tables[0].rows) > 0:
                    row = response.tables[0].rows[0]
                    return {
                        'cpu_usage': float(row[1]) if row[1] else 0.0,
                        'memory_usage': float(row[2]) if row[2] else 0.0,
                        'disk_usage': 100 - float(row[3]) if row[3] else 0.0
                    }
            
            return {'cpu_usage': 0.0, 'memory_usage': 0.0, 'disk_usage': 0.0}
            
        except Exception as e:
            logger.error(f"Error querying metrics for {host_name}: {e}")
            return {'cpu_usage': 0.0, 'memory_usage': 0.0, 'disk_usage': 0.0}
    
    async def query_host_errors(self, host_name: str, hours: int = 24) -> List[Dict]:
        """Query errors and warnings from Log Analytics"""
        try:
            query = f"""
            Event
            | where Computer == "{host_name}"
            | where TimeGenerated > ago({hours}h)
            | where EventLevelName in ("Error", "Warning")
            | project TimeGenerated, EventLevelName, EventID, RenderedDescription
            | order by TimeGenerated desc
            | take 50
            """
            
            response = self.logs_client.query_workspace(
                workspace_id=self.workspace_id,
                query=query,
                timespan=timedelta(hours=hours)
            )
            
            errors = []
            if response.status == LogsQueryStatus.SUCCESS and response.tables:
                for row in response.tables[0].rows:
                    errors.append({
                        'timestamp': row[0],
                        'level': row[1],
                        'event_id': row[2],
                        'description': row[3]
                    })
            
            return errors
            
        except Exception as e:
            logger.error(f"Error querying events for {host_name}: {e}")
            return []
    
    async def get_host_health(self, resource_group: str, host_pool_name: str, 
                             host_name: str) -> HostHealth:
        """Get comprehensive health status for a single host"""
        
        # Get session host info
        hosts = await self.get_session_hosts(resource_group, host_pool_name)
        host_info = next((h for h in hosts if host_name in h['name']), None)
        
        if not host_info:
            return HostHealth(
                host_name=host_name,
                status="Unknown",
                last_heartbeat=None,
                sessions_active=0,
                cpu_usage=0.0,
                memory_usage=0.0,
                disk_usage=0.0,
                errors=[],
                warnings=[]
            )
        
        # Get metrics and errors in parallel
        metrics_task = self.query_host_metrics(host_name)
        errors_task = self.query_host_errors(host_name)
        
        metrics, all_events = await asyncio.gather(metrics_task, errors_task)
        
        # Separate errors and warnings
        errors = [e for e in all_events if e['level'] == 'Error']
        warnings = [e for e in all_events if e['level'] == 'Warning']
        
        return HostHealth(
            host_name=host_name,
            status=host_info['status'],
            last_heartbeat=host_info['last_heartbeat'],
            sessions_active=host_info['sessions'] or 0,
            cpu_usage=metrics['cpu_usage'],
            memory_usage=metrics['memory_usage'],
            disk_usage=metrics['disk_usage'],
            errors=errors[:10],  # Top 10 errors
            warnings=warnings[:10]  # Top 10 warnings
        )
    
    async def get_pool_health(self, resource_group: str, 
                             host_pool_name: str) -> tuple[HostPoolMetrics, List[HostHealth]]:
        """Get health status for entire host pool"""
        
        # Get all hosts
        hosts = await self.get_session_hosts(resource_group, host_pool_name)
        
        # Get health for each host in parallel
        health_tasks = []
        for host in hosts:
            host_name = host['name'].split('/')[-1]
            health_tasks.append(self.get_host_health(resource_group, host_pool_name, host_name))
        
        host_healths = await asyncio.gather(*health_tasks)
        
        # Calculate aggregate metrics
        total_hosts = len(host_healths)
        healthy_hosts = sum(1 for h in host_healths if h.status == 'Available')
        total_sessions = sum(h.sessions_active for h in host_healths)
        avg_cpu = sum(h.cpu_usage for h in host_healths) / total_hosts if total_hosts > 0 else 0
        avg_memory = sum(h.memory_usage for h in host_healths) / total_hosts if total_hosts > 0 else 0
        critical_errors = sum(len(h.errors) for h in host_healths)
        total_warnings = sum(len(h.warnings) for h in host_healths)
        
        pool_metrics = HostPoolMetrics(
            pool_name=host_pool_name,
            total_hosts=total_hosts,
            healthy_hosts=healthy_hosts,
            unhealthy_hosts=total_hosts - healthy_hosts,
            total_sessions=total_sessions,
            available_capacity=healthy_hosts * 10,  # Assuming 10 sessions per host
            avg_cpu_usage=avg_cpu,
            avg_memory_usage=avg_memory,
            critical_errors=critical_errors,
            warnings=total_warnings
        )
        
        return pool_metrics, host_healths
