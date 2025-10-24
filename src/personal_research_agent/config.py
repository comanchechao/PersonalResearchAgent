"""
Configuration management for Personal Research Agent.
Follows 2025 best practices for secure configuration and multi-provider support.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
import os
from pathlib import Path


class LLMProvider(str, Enum):
    """Supported LLM providers for 2025."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # LM Studio, Ollama, etc.
    AZURE_OPENAI = "azure_openai"
    GOOGLE = "google"


class VectorDBProvider(str, Enum):
    """Supported vector database providers."""
    CHROMADB = "chromadb"
    FAISS = "faiss"
    REDIS = "redis"
    PINECONE = "pinecone"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: LLMProvider = LLMProvider.LOCAL
    model_name: str = "qwen2.5-coder-7b-instruct"
    api_key: Optional[str] = None
    api_base: Optional[str] = "http://localhost:1234/v1"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)
    timeout: int = Field(default=60, gt=0)
    
    # 2025 feature: Structured output support
    supports_structured_output: bool = True
    supports_function_calling: bool = True


class VectorDBConfig(BaseModel):
    """Configuration for vector databases."""
    provider: VectorDBProvider = VectorDBProvider.CHROMADB
    connection_string: Optional[str] = None
    collection_name: str = "research_knowledge"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = Field(default=1000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class SecurityConfig(BaseModel):
    """Security configuration following 2025 standards."""
    enable_auth: bool = True
    jwt_secret_key: str = Field(min_length=32)
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = Field(default=24, gt=0)
    
    # OAuth 2.0 / OpenID Connect support
    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    oauth_redirect_uri: Optional[str] = None
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, gt=0)
    rate_limit_window: int = Field(default=3600, gt=0)  # seconds


class MemoryConfig(BaseModel):
    """Configuration for agent memory and learning."""
    enable_memory: bool = True
    memory_provider: str = "redis"
    memory_connection: str = "redis://localhost:6379/0"
    
    # Personalization settings
    enable_personalization: bool = True
    user_preference_weight: float = Field(default=0.8, ge=0.0, le=1.0)
    learning_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    
    # Memory retention
    short_term_memory_hours: int = Field(default=24, gt=0)
    long_term_memory_days: int = Field(default=30, gt=0)


class ResearchConfig(BaseModel):
    """Configuration for research capabilities."""
    max_search_results: int = Field(default=10, gt=0, le=50)
    max_concurrent_searches: int = Field(default=3, gt=0, le=10)
    enable_web_search: bool = True
    enable_document_processing: bool = True
    
    # Search providers
    search_providers: List[str] = ["duckduckgo", "serper", "tavily"]
    
    # Document processing
    supported_formats: List[str] = ["pdf", "docx", "txt", "md", "html"]
    max_document_size_mb: int = Field(default=50, gt=0)


class ObservabilityConfig(BaseModel):
    """Configuration for monitoring and observability."""
    enable_telemetry: bool = True
    telemetry_endpoint: Optional[str] = None
    log_level: LogLevel = LogLevel.INFO
    enable_metrics: bool = True
    enable_tracing: bool = True


class Settings(BaseSettings):
    """Main application settings following 2025 best practices."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_nested_delimiter="__",
        extra="ignore"
    )
    
    # Application settings
    app_name: str = "Personal Research Agent"
    app_version: str = "0.1.0"
    debug: bool = False
    
    # Component configurations
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vector_db: VectorDBConfig = Field(default_factory=VectorDBConfig)
    security: SecurityConfig = Field(default_factory=lambda: SecurityConfig(
        jwt_secret_key=os.urandom(32).hex()
    ))
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    
    # Data directories
    data_dir: Path = Field(default_factory=lambda: Path.home() / ".personal_research_agent")
    cache_dir: Path = Field(default_factory=lambda: Path.home() / ".personal_research_agent" / "cache")
    logs_dir: Path = Field(default_factory=lambda: Path.home() / ".personal_research_agent" / "logs")
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
    
    @validator('llm')
    def validate_llm_config(cls, v):
        """Validate LLM configuration."""
        if v.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC] and not v.api_key:
            raise ValueError(f"API key required for {v.provider}")
        return v
    
    def get_llm_kwargs(self) -> Dict[str, Any]:
        """Get LLM initialization kwargs."""
        kwargs = {
            "model": self.llm.model_name,
            "temperature": self.llm.temperature,
            "max_tokens": self.llm.max_tokens,
            "timeout": self.llm.timeout,
        }
        
        if self.llm.api_key:
            kwargs["api_key"] = self.llm.api_key
        if self.llm.api_base:
            kwargs["base_url"] = self.llm.api_base
            
        return kwargs


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings


def update_settings(**kwargs) -> Settings:
    """Update settings with new values."""
    global settings
    settings = Settings(**{**settings.dict(), **kwargs})
    return settings
