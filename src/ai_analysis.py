"""
AI-Powered Analysis Engine
Provides intelligent root cause analysis and recommendations using LLMs
"""
import json
import logging
from typing import List, Dict, Optional, Literal
from dataclasses import dataclass, asdict
from datetime import datetime
import openai
from anthropic import Anthropic

logger = logging.getLogger(__name__)


@dataclass
class RootCauseAnalysis:
    """AI-generated root cause analysis"""
    issue_summary: str
    root_cause: str
    severity: Literal["Critical", "High", "Medium", "Low"]
    impact: str
    recommended_actions: List[str]
    prevention_tips: List[str]
    confidence_score: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self):
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


class AIAnalysisEngine:
    """AI-powered analysis using OpenAI or Anthropic"""
    
    def __init__(self, provider: str = "anthropic", 
                 openai_key: str = None, anthropic_key: str = None,
                 openai_model: str = "gpt-4", 
                 anthropic_model: str = "claude-3-5-sonnet-20241022"):
        """
        Initialize AI analysis engine
        
        Args:
            provider: "openai" or "anthropic"
            openai_key: OpenAI API key
            anthropic_key: Anthropic API key
            openai_model: OpenAI model name
            anthropic_model: Anthropic model name
        """
        self.provider = provider.lower()
        self.openai_model = openai_model
        self.anthropic_model = anthropic_model
        
        if self.provider == "openai" and openai_key:
            openai.api_key = openai_key
        elif self.provider == "anthropic" and anthropic_key:
            self.anthropic_client = Anthropic(api_key=anthropic_key)
        else:
            logger.warning(f"No API key provided for {provider}")
    
    async def analyze_host_health(self, host_name: str, health_data: Dict) -> RootCauseAnalysis:
        """
        Analyze host health data and provide AI-powered diagnosis
        
        Args:
            host_name: Name of the host
            health_data: Dictionary containing health metrics and errors
        
        Returns:
            Root cause analysis with recommendations
        """
        
        # Build comprehensive context for AI
        prompt = self._build_analysis_prompt(host_name, health_data)
        
        # Get AI response
        if self.provider == "openai":
            analysis_text = await self._query_openai(prompt)
        else:
            analysis_text = await self._query_anthropic(prompt)
        
        # Parse AI response into structured format
        return self._parse_analysis_response(analysis_text)
    
    def _build_analysis_prompt(self, host_name: str, health_data: Dict) -> str:
        """Build detailed prompt for AI analysis"""
        
        prompt = f"""You are an expert Azure Virtual Desktop (AVD) system administrator and diagnostics specialist. 

Analyze the following AVD session host health data and provide a comprehensive diagnosis:

**Host:** {host_name}
**Status:** {health_data.get('status', 'Unknown')}
**Last Heartbeat:** {health_data.get('last_heartbeat', 'N/A')}

**Performance Metrics:**
- CPU Usage: {health_data.get('cpu_usage', 0):.1f}%
- Memory Usage: {health_data.get('memory_usage', 0):.1f}%
- Disk Usage: {health_data.get('disk_usage', 0):.1f}%
- Active Sessions: {health_data.get('sessions_active', 0)}

**Recent Errors ({len(health_data.get('errors', []))}):**
"""
        
        for i, error in enumerate(health_data.get('errors', [])[:5], 1):
            prompt += f"\n{i}. Event ID {error.get('event_id')}: {error.get('description', 'No description')[:200]}"
        
        prompt += f"\n\n**Recent Warnings ({len(health_data.get('warnings', []))}):**"
        for i, warning in enumerate(health_data.get('warnings', [])[:5], 1):
            prompt += f"\n{i}. Event ID {warning.get('event_id')}: {warning.get('description', 'No description')[:200]}"
        
        prompt += """

Provide your analysis in the following JSON format:
{
    "issue_summary": "Brief 1-2 sentence summary of the current state",
    "root_cause": "Detailed explanation of the root cause (3-5 sentences)",
    "severity": "Critical|High|Medium|Low",
    "impact": "Expected impact on users and business operations",
    "recommended_actions": ["Action 1", "Action 2", "Action 3"],
    "prevention_tips": ["Prevention tip 1", "Prevention tip 2"],
    "confidence_score": 0.95
}

Focus on:
1. Identifying patterns in errors and metrics
2. Correlating performance issues with specific errors
3. Providing actionable, specific recommendations
4. Explaining WHY issues are occurring, not just WHAT
5. Prioritizing actions by impact and urgency
"""
        
        return prompt
    
    async def _query_openai(self, prompt: str) -> str:
        """Query OpenAI GPT model"""
        try:
            response = openai.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert Azure Virtual Desktop administrator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1500
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return self._get_fallback_analysis()
    
    async def _query_anthropic(self, prompt: str) -> str:
        """Query Anthropic Claude model"""
        try:
            response = self.anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=1500,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return self._get_fallback_analysis()
    
    def _parse_analysis_response(self, response_text: str) -> RootCauseAnalysis:
        """Parse AI response into structured format"""
        try:
            # Try to extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                data = json.loads(json_str)
                
                return RootCauseAnalysis(
                    issue_summary=data.get('issue_summary', 'Analysis completed'),
                    root_cause=data.get('root_cause', 'See detailed analysis'),
                    severity=data.get('severity', 'Medium'),
                    impact=data.get('impact', 'Potential service degradation'),
                    recommended_actions=data.get('recommended_actions', []),
                    prevention_tips=data.get('prevention_tips', []),
                    confidence_score=float(data.get('confidence_score', 0.7))
                )
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            # Return fallback analysis
            return RootCauseAnalysis(
                issue_summary="Analysis parsing failed",
                root_cause="Unable to parse AI response. Manual review recommended.",
                severity="Medium",
                impact="Unknown - requires manual investigation",
                recommended_actions=["Review raw health data", "Check Azure Portal", "Contact support if issues persist"],
                prevention_tips=["Ensure proper monitoring is configured"],
                confidence_score=0.5
            )
    
    def _get_fallback_analysis(self) -> str:
        """Fallback analysis when AI is unavailable"""
        return json.dumps({
            "issue_summary": "AI analysis unavailable",
            "root_cause": "Unable to connect to AI service. Using fallback analysis.",
            "severity": "Low",
            "impact": "Limited diagnostic capability",
            "recommended_actions": [
                "Check API credentials",
                "Verify network connectivity",
                "Review logs manually"
            ],
            "prevention_tips": [
                "Ensure API keys are properly configured",
                "Monitor API quota limits"
            ],
            "confidence_score": 0.3
        })
    
    async def analyze_pool_trends(self, pool_metrics: Dict, 
                                  historical_data: List[Dict]) -> Dict:
        """
        Analyze pool-level trends and predict potential issues
        
        Args:
            pool_metrics: Current pool metrics
            historical_data: List of historical metric snapshots
        
        Returns:
            Trend analysis and predictions
        """
        
        prompt = f"""Analyze these Azure Virtual Desktop host pool trends:

**Current State:**
- Total Hosts: {pool_metrics.get('total_hosts')}
- Healthy: {pool_metrics.get('healthy_hosts')} | Unhealthy: {pool_metrics.get('unhealthy_hosts')}
- Active Sessions: {pool_metrics.get('total_sessions')}
- Avg CPU: {pool_metrics.get('avg_cpu_usage', 0):.1f}%
- Avg Memory: {pool_metrics.get('avg_memory_usage', 0):.1f}%
- Critical Errors: {pool_metrics.get('critical_errors')}

**Historical Trend (last {len(historical_data)} snapshots):**
"""
        
        for i, snapshot in enumerate(historical_data[-5:], 1):
            prompt += f"\n{i}. CPU: {snapshot.get('avg_cpu', 0):.1f}% | Memory: {snapshot.get('avg_memory', 0):.1f}% | Errors: {snapshot.get('errors', 0)}"
        
        prompt += """

Provide trend analysis in JSON format:
{
    "trend_direction": "improving|stable|degrading|critical",
    "key_findings": ["Finding 1", "Finding 2"],
    "risk_factors": ["Risk 1", "Risk 2"],
    "predictions": ["Prediction 1", "Prediction 2"],
    "recommended_actions": ["Action 1", "Action 2"]
}
"""
        
        if self.provider == "openai":
            response = await self._query_openai(prompt)
        else:
            response = await self._query_anthropic(prompt)
        
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0:
                return json.loads(response[json_start:json_end])
        except:
            pass
        
        return {
            "trend_direction": "stable",
            "key_findings": ["Insufficient data for trend analysis"],
            "risk_factors": [],
            "predictions": [],
            "recommended_actions": ["Continue monitoring"]
        }
