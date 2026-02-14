"""
Configuration management for AVD AI Monitoring PoC
"""
import os
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Azure Credentials
    azure_subscription_id: str = Field(default="", alias="AZURE_SUBSCRIPTION_ID")
    azure_tenant_id: str = Field(default="", alias="AZURE_TENANT_ID")
    azure_client_id: str = Field(default="", alias="AZURE_CLIENT_ID")
    azure_client_secret: str = Field(default="", alias="AZURE_CLIENT_SECRET")
    
    # Azure AVD Configuration
    avd_resource_group: str = Field(default="", alias="AVD_RESOURCE_GROUP")
    avd_host_pool_name: str = Field(default="", alias="AVD_HOST_POOL_NAME")
    avd_workspace_id: str = Field(default="", alias="AVD_WORKSPACE_ID")
    
    # AI Configuration
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    ai_provider: Literal["openai", "anthropic"] = Field(default="anthropic", alias="AI_PROVIDER")
    openai_model: str = Field(default="gpt-4", alias="OPENAI_MODEL")
    anthropic_model: str = Field(default="claude-3-5-sonnet-20241022", alias="ANTHROPIC_MODEL")
    
    # Application Settings
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    enable_predictive_analytics: bool = Field(default=True, alias="ENABLE_PREDICTIVE_ANALYTICS")
    cache_ttl_seconds: int = Field(default=300, alias="CACHE_TTL_SECONDS")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
