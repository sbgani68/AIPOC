"""
Sample data and mock implementations for testing without Azure credentials
"""
from datetime import datetime, timedelta
from typing import List, Dict
import random


def generate_mock_host_health(host_name: str, status: str = "Available") -> Dict:
    """Generate mock host health data"""
    return {
        "host_name": host_name,
        "status": status,
        "last_heartbeat": (datetime.utcnow() - timedelta(minutes=random.randint(1, 5))).isoformat(),
        "sessions_active": random.randint(0, 10),
        "cpu_usage": random.uniform(20, 90),
        "memory_usage": random.uniform(30, 85),
        "disk_usage": random.uniform(40, 75),
        "errors": generate_mock_errors(),
        "warnings": generate_mock_warnings(),
        "timestamp": datetime.utcnow().isoformat()
    }


def generate_mock_errors() -> List[Dict]:
    """Generate mock error events"""
    common_errors = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "Error",
            "event_id": 1001,
            "description": "RD Gateway connection timeout - network connectivity issue detected"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "Error",
            "event_id": 1002,
            "description": "FSLogix Profile Container failed to attach -  storage performance degradation"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "Error",
            "event_id": 4625,
            "description": "User authentication failed - check AD connectivity"
        }
    ]
    
    num_errors = random.randint(0, 3)
    return random.sample(common_errors, num_errors)


def generate_mock_warnings() -> List[Dict]:
    """Generate mock warning events"""
    common_warnings = [
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "Warning",
            "event_id": 2001,
            "description": "High memory usage detected - consider adding more RAM"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "Warning",
            "event_id": 2002,
            "description": "Disk space running low on C: drive"
        },
        {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "Warning",
            "event_id": 2003,
            "description": "Windows Update pending restart"
        }
    ]
    
    num_warnings = random.randint(0, 5)
    return random.sample(common_warnings, num_warnings)


def generate_mock_pool_metrics(pool_name: str = "avd-pool-prod", num_hosts: int = 5) -> Dict:
    """Generate mock host pool metrics"""
    healthy_hosts = random.randint(int(num_hosts * 0.7), num_hosts)
    
    return {
        "pool_name": pool_name,
        "total_hosts": num_hosts,
        "healthy_hosts": healthy_hosts,
        "unhealthy_hosts": num_hosts - healthy_hosts,
        "total_sessions": random.randint(10, num_hosts * 8),
        "available_capacity": healthy_hosts * 10,
        "avg_cpu_usage": random.uniform(30, 75),
        "avg_memory_usage": random.uniform(40, 80),
        "critical_errors": random.randint(0, 5),
        "warnings": random.randint(0, 10),
        "timestamp": datetime.utcnow().isoformat()
    }


def generate_historical_data(days: int = 7, samples_per_day: int = 24) -> List[Dict]:
    """Generate historical metrics for testing predictive analytics"""
    historical = []
    base_time = datetime.utcnow() - timedelta(days=days)
    
    for day in range(days):
        for hour in range(samples_per_day):
            timestamp = base_time + timedelta(days=day, hours=hour)
            
            # Simulate business hours pattern
            is_business_hours = 8 <= timestamp.hour <= 17
            is_weekend = timestamp.weekday() >= 5
            
            # Adjust metrics based on time
            if is_business_hours and not is_weekend:
                cpu_base = 60
                memory_base = 65
                sessions_base = 40
            else:
                cpu_base = 30
                memory_base = 45
                sessions_base = 10
            
            # Add some noise
            historical.append({
                "pool_name": "avd-pool-prod",
                "total_hosts": 5,
                "healthy_hosts": random.randint(4, 5),
                "unhealthy_hosts": random.randint(0, 1),
                "total_sessions": int(sessions_base + random.uniform(-10, 20)),
                "available_capacity": 50,
                "avg_cpu_usage": cpu_base + random.uniform(-15, 15),
                "avg_memory_usage": memory_base + random.uniform(-10, 10),
                "critical_errors": random.randint(0, 3),
                "warnings": random.randint(0, 8),
                "timestamp": timestamp.isoformat()
            })
    
    return historical


# Sample AI analysis response for testing
SAMPLE_ANALYSIS = {
    "issue_summary": "Host experiencing high memory utilization with FSLogix profile attachment failures",
    "root_cause": "The session host is running out of available memory (85% utilization) causing FSLogix Profile Container attachments to fail. This is likely due to a memory leak in a running application or insufficient memory allocation for the expected workload. The recurring Event ID 1002 errors confirm storage performance issues during profile mounting.",
    "severity": "High",
    "impact": "Users are unable to access their personalized profiles, leading to data loss and poor user experience. Approximately 30% of login attempts are failing. New user sessions cannot be established until memory is freed.",
    "recommended_actions": [
        "Immediately restart the affected session host to clear memory",
        "Increase VM memory from current allocation to at least 16GB",
        "Investigate running processes using Process Explorer to identify memory leak source",
        "Review FSLogix storage backend performance metrics",
        "Configure FSLogix profile size limits to prevent oversized profiles"
    ],
    "prevention_tips": [
        "Implement automated memory monitoring with alerts at 75% threshold",
        "Configure FSLogix Cloud Cache for redundancy and performance",
        "Establish baseline memory requirements per user session",
        "Enable Application Insights for proactive issue detection",
        "Schedule regular host pool maintenance windows for updates and optimization"
    ],
    "confidence_score": 0.92
}


# Sample capacity prediction
SAMPLE_PREDICTION = {
    "predicted_sessions": 65,
    "predicted_cpu": 78.5,
    "predicted_memory": 82.0,
    "capacity_alert": True,
    "alert_reason": "High resource utilization predicted - recommend adding 2 hosts",
    "recommended_hosts": 7,
    "confidence": 0.85,
    "forecast_timestamp": (datetime.utcnow() + timedelta(hours=4)).isoformat()
}
