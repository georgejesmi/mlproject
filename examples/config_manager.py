"""
Production-Grade Configuration Manager with Pydantic Validation
Demonstrates: Type safety, validation, environment-specific configs
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field, validator, HttpUrl
from pathlib import Path
import yaml
import os
from functools import lru_cache


class DataConfig(BaseModel):
    """Data configuration with validation"""
    raw_data_source: str
    train_test_split_ratio: float = Field(gt=0, lt=1)
    random_state: int = Field(ge=0)
    validation_split: float = Field(gt=0, lt=1)

    @validator('train_test_split_ratio', 'validation_split')
    def validate_ratio(cls, v):
        if not 0 < v < 1:
            raise ValueError('Ratio must be between 0 and 1')
        return v


class ModelConfig(BaseModel):
    """Model configuration with validation"""
    min_r2_threshold: float = Field(ge=0, le=1)
    max_training_time_seconds: int = Field(gt=0)
    cross_validation_folds: int = Field(ge=2, le=10)
    early_stopping_rounds: int = Field(gt=0)


class ModelRegistryConfig(BaseModel):
    """Model registry configuration"""
    enabled: bool = True
    backend: Literal["mlflow", "wandb", "local"] = "mlflow"
    tracking_uri: str
    experiment_name: str


class LoggingConfig(BaseModel):
    """Logging configuration"""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: Literal["json", "text"] = "json"
    log_to_file: bool = True
    log_to_console: bool = True
    rotation: Literal["daily", "weekly", "size"] = "daily"
    retention_days: int = Field(ge=1, le=365)
    include_correlation_id: bool = True


class MonitoringConfig(BaseModel):
    """Monitoring configuration"""
    enabled: bool = True
    metrics_backend: Literal["prometheus", "statsd", "cloudwatch"] = "prometheus"
    metrics_port: int = Field(ge=1024, le=65535)
    health_check_interval_seconds: int = Field(gt=0)
    data_drift_detection: bool = True
    model_performance_tracking: bool = True


class APIConfig(BaseModel):
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = Field(ge=1024, le=65535)
    workers: int = Field(ge=1, le=32)
    timeout_seconds: int = Field(gt=0)
    max_request_size_mb: int = Field(gt=0)
    rate_limit_per_minute: int = Field(gt=0)
    enable_cors: bool = True
    allowed_origins: List[str]


class SecurityConfig(BaseModel):
    """Security configuration"""
    enable_authentication: bool = True
    api_key_header: str = "X-API-Key"
    input_validation: bool = True
    sanitize_inputs: bool = True
    max_batch_size: int = Field(gt=0, le=10000)


class PerformanceConfig(BaseModel):
    """Performance configuration"""
    enable_caching: bool = True
    cache_ttl_seconds: int = Field(gt=0)
    batch_prediction_enabled: bool = True
    async_processing: bool = True
    model_lazy_loading: bool = True


class PathsConfig(BaseModel):
    """Paths configuration"""
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    models_dir: str = "models"
    data_dir: str = "data"
    cache_dir: str = "cache"


class FeatureEngineeringConfig(BaseModel):
    """Feature engineering configuration"""
    version: str
    numerical_features: List[str]
    categorical_features: List[str]
    scaling_method: Literal["standard", "minmax", "robust"] = "standard"
    encoding_method: Literal["onehot", "label", "target"] = "onehot"


class DataQualityConfig(BaseModel):
    """Data quality checks configuration"""
    check_missing_values: bool = True
    check_duplicates: bool = True
    check_outliers: bool = True
    check_schema: bool = True
    max_missing_percentage: float = Field(ge=0, le=100)
    outlier_method: Literal["iqr", "zscore"] = "iqr"


class AlertsConfig(BaseModel):
    """Alerts configuration"""
    enabled: bool = True
    email_notifications: bool = True
    slack_webhook_url: Optional[str] = None
    alert_on_model_degradation: bool = True
    alert_on_data_drift: bool = True
    alert_on_high_error_rate: bool = True
    error_rate_threshold: float = Field(ge=0, le=1)


class AppConfig(BaseModel):
    """Main application configuration"""
    environment: Literal["development", "staging", "production"]
    data: DataConfig
    model: ModelConfig
    model_registry: ModelRegistryConfig
    logging: LoggingConfig
    monitoring: MonitoringConfig
    api: APIConfig
    security: SecurityConfig
    performance: PerformanceConfig
    paths: PathsConfig
    feature_engineering: FeatureEngineeringConfig
    data_quality: DataQualityConfig
    alerts: AlertsConfig

    class Config:
        validate_assignment = True


class ConfigManager:
    """
    Configuration Manager with environment variable support
    
    Usage:
        config = ConfigManager.get_config()
        print(config.model.min_r2_threshold)
    """
    
    _instance: Optional[AppConfig] = None
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_config(cls, config_path: Optional[str] = None) -> AppConfig:
        """
        Load and validate configuration from YAML file
        
        Args:
            config_path: Path to config file. If None, uses CONFIG_PATH env var
            
        Returns:
            Validated AppConfig instance
        """
        if cls._instance is not None:
            return cls._instance
            
        # Determine config path
        if config_path is None:
            config_path = os.getenv("CONFIG_PATH", "config.yaml")
        
        # Load YAML
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Replace environment variables
        config_dict = cls._replace_env_vars(config_dict)
        
        # Validate and create config
        cls._instance = AppConfig(**config_dict)
        return cls._instance
    
    @staticmethod
    def _replace_env_vars(config_dict: dict) -> dict:
        """
        Replace ${ENV_VAR} placeholders with environment variables
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            Dictionary with environment variables replaced
        """
        def replace_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: replace_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [replace_value(item) for item in value]
            return value
        
        return {k: replace_value(v) for k, v in config_dict.items()}
    
    @classmethod
    def reload_config(cls, config_path: Optional[str] = None):
        """Force reload configuration"""
        cls._instance = None
        cls.get_config.cache_clear()
        return cls.get_config(config_path)


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = ConfigManager.get_config("examples/config.yaml")
    
    # Access configuration values with type safety
    print(f"Environment: {config.environment}")
    print(f"Min R2 Threshold: {config.model.min_r2_threshold}")
    print(f"API Port: {config.api.port}")
    print(f"Logging Level: {config.logging.level}")
    
    # Configuration is validated at load time
    # Invalid values will raise ValidationError

# Made with Bob
