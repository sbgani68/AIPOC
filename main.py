"""
AVD AI Monitoring PoC - Main Entry Point
Demonstrates AI-powered monitoring and diagnostics for Azure Virtual Desktop
"""
import asyncio
import os
from dotenv import load_dotenv
from datetime import datetime
import logging

# Load environment variables
load_dotenv()

from src.azure_avd_monitor import AVDMonitor
from src.ai_analysis import AIAnalysisEngine
from src.predictive_analytics import PredictiveAnalytics
from config import settings

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_monitoring():
    """Demonstration of AVD monitoring capabilities"""
    
    print("\n" + "="*70)
    print("  AVD AI-Powered Monitoring & Diagnostics - Demo")
    print("="*70 + "\n")
    
    # Check configuration
    if not all([settings.azure_subscription_id, settings.azure_tenant_id]):
        print("⚠️  Azure credentials not configured")
        print("📝 Please create a .env file with your Azure credentials")
        print("   See .env.template for required variables\n")
        print("🚀 To start the API server instead, run: uvicorn api:app --reload\n")
        return
    
    print("✓ Configuration loaded")
    print(f"  - Subscription: {settings.azure_subscription_id[:8]}...")
    print(f"  - Resource Group: {settings.avd_resource_group}")
    print(f"  - Host Pool: {settings.avd_host_pool_name}")
    print(f"  - AI Provider: {settings.ai_provider}\n")
    
    try:
        # Initialize AVD Monitor
        print("🔄 Initializing Azure AVD Monitor...")
        monitor = AVDMonitor(
            subscription_id=settings.azure_subscription_id,
            tenant_id=settings.azure_tenant_id,
            client_id=settings.azure_client_id,
            client_secret=settings.azure_client_secret,
            workspace_id=settings.avd_workspace_id
        )
        print("✓ AVD Monitor initialized\n")
        
        # Get host pool information
        print("🔄 Fetching host pool information...")
        pool_info = await monitor.get_host_pool_info(
            settings.avd_resource_group,
            settings.avd_host_pool_name
        )
        
        print(f"\n📊 Host Pool: {pool_info.get('name', 'Unknown')}")
        print(f"   Location: {pool_info.get('location', 'N/A')}")
        print(f"   Type: {pool_info.get('type', 'N/A')}")
        print(f"   Max Sessions: {pool_info.get('max_session_limit', 'N/A')}\n")
        
        # Get pool health metrics
        print("🔄 Analyzing pool health...")
        pool_metrics, host_healths = await monitor.get_pool_health(
            settings.avd_resource_group,
            settings.avd_host_pool_name
        )
        
        print(f"\n📈 Pool Metrics:")
        print(f"   Total Hosts: {pool_metrics.total_hosts}")
        print(f"   Healthy: {pool_metrics.healthy_hosts} | Unhealthy: {pool_metrics.unhealthy_hosts}")
        print(f"   Active Sessions: {pool_metrics.total_sessions}")
        print(f"   Avg CPU: {pool_metrics.avg_cpu_usage:.1f}%")
        print(f"   Avg Memory: {pool_metrics.avg_memory_usage:.1f}%")
        print(f"   Critical Errors: {pool_metrics.critical_errors}")
        print(f"   Warnings: {pool_metrics.warnings}\n")
        
        # Display individual host health
        print(f"💻 Host Health Status:")
        for host in host_healths:
            status_emoji = "✓" if host.status == "Available" else "⚠️"
            print(f"\n   {status_emoji} {host.host_name}")
            print(f"      Status: {host.status}")
            print(f"      CPU: {host.cpu_usage:.1f}% | Memory: {host.memory_usage:.1f}% | Disk: {host.disk_usage:.1f}%")
            print(f"      Sessions: {host.sessions_active}")
            print(f"      Errors: {len(host.errors)} | Warnings: {len(host.warnings)}")
        
        # AI Analysis (if configured)
        if settings.anthropic_api_key or settings.openai_api_key:
            print("\n" + "="*70)
            print("🤖 AI-Powered Analysis")
            print("="*70 + "\n")
            
            ai_engine = AIAnalysisEngine(
                provider=settings.ai_provider,
                openai_key=settings.openai_api_key,
                anthropic_key=settings.anthropic_api_key,
                openai_model=settings.openai_model,
                anthropic_model=settings.anthropic_model
            )
            
            # Analyze first host with issues
            problem_host = next((h for h in host_healths if h.status != "Available" or len(h.errors) > 0), None)
            
            if problem_host:
                print(f"🔍 Analyzing {problem_host.host_name}...\n")
                analysis = await ai_engine.analyze_host_health(
                    problem_host.host_name,
                    problem_host.to_dict()
                )
                
                print(f"📋 Issue Summary:")
                print(f"   {analysis.issue_summary}\n")
                print(f"🔎 Root Cause:")
                print(f"   {analysis.root_cause}\n")
                print(f"⚡ Severity: {analysis.severity}")
                print(f"📊 Impact: {analysis.impact}\n")
                print(f"✅ Recommended Actions:")
                for i, action in enumerate(analysis.recommended_actions, 1):
                    print(f"   {i}. {action}")
                print(f"\n🛡️  Prevention Tips:")
                for i, tip in enumerate(analysis.prevention_tips, 1):
                    print(f"   {i}. {tip}")
                print(f"\n📊 Confidence Score: {analysis.confidence_score:.2%}\n")
            else:
                print("✓ All hosts are healthy - no issues to analyze\n")
        
        # Predictive Analytics (if enabled)
        if settings.enable_predictive_analytics:
            print("\n" + "="*70)
            print("🔮 Predictive Analytics")
            print("="*70 + "\n")
            
            predictor = PredictiveAnalytics()
            
            # Simulate some historical data for demo
            historical_data = [pool_metrics.to_dict()]
            
            print("📊 Note: Predictive models require historical data (minimum 10 samples)")
            print(f"   Current samples: {len(historical_data)}")
            print("   Run the API server and collect data over time for predictions\n")
        
        print("\n" + "="*70)
        print("✨ Demo Complete!")
        print("="*70)
        print("\n🚀 Next Steps:")
        print("   1. Start API server: uvicorn api:app --reload")
        print("   2. Access API docs: http://localhost:8000/docs")
        print("   3. View dashboard: http://localhost:8000/api/v1/dashboard")
        print("   4. Collect historical data for predictive analytics\n")
        
    except Exception as e:
        logger.error(f"Error during demo: {e}")
        print(f"\n❌ Error: {e}")
        print("   Check your Azure credentials and permissions\n")


def main():
    """Main entry point"""
    print("\n🚀 AVD AI Monitoring PoC")
    print("   Choose an option:")
    print("   1. Run demo (requires Azure credentials)")
    print("   2. Start API server")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(demo_monitoring())
    elif choice == "2":
        print("\n🌐 Starting API server on http://localhost:8000")
        print("📚 API documentation: http://localhost:8000/docs\n")
        import uvicorn
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    else:
        print("❌ Invalid choice\n")


if __name__ == "__main__":
    main()

