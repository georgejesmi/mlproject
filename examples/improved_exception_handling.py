"""
Production-Grade Exception Handling with Structured Logging
Demonstrates: Custom exceptions, error tracking, correlation IDs
"""

import sys
import traceback
import logging
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import json
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for better classification"""
    DATA_VALIDATION = "data_validation"
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    MODEL_PREDICTION = "model_prediction"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class ErrorContext:
    """
    Context information for errors with correlation tracking
    """
    
    def __init__(
        self,
        correlation_id: Optional[str] = None,
        user_id: Optional[str] = None,
        request_id: Optional[str] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.user_id = user_id
        self.request_id = request_id
        self.timestamp = datetime.utcnow().isoformat()
        self.additional_context = additional_context or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary"""
        return {
            'correlation_id': self.correlation_id,
            'user_id': self.user_id,
            'request_id': self.request_id,
            'timestamp': self.timestamp,
            'additional_context': self.additional_context
        }


class BaseMLException(Exception):
    """
    Base exception class for ML pipeline with enhanced error tracking
    """
    
    def __init__(
        self,
        message: str,
        error_details: Optional[sys] = None,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        original_exception: Optional[Exception] = None
    ):
        """
        Initialize base exception
        
        Args:
            message: Error message
            error_details: sys module for traceback extraction
            category: Error category
            severity: Error severity
            context: Error context with correlation ID
            original_exception: Original exception if wrapping
        """
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.original_exception = original_exception
        
        # Extract traceback information
        if error_details:
            _, _, tb = error_details.exc_info()
            if tb:
                self.file_name = tb.tb_frame.f_code.co_filename
                self.line_number = tb.tb_lineno
                self.function_name = tb.tb_frame.f_code.co_name
                self.traceback_str = ''.join(traceback.format_tb(tb))
            else:
                self._set_default_traceback_info()
        else:
            self._set_default_traceback_info()
    
    def _set_default_traceback_info(self):
        """Set default traceback information"""
        self.file_name = "unknown"
        self.line_number = 0
        self.function_name = "unknown"
        self.traceback_str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for structured logging
        
        Returns:
            Dictionary representation of exception
        """
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'category': self.category.value,
            'severity': self.severity.value,
            'file_name': self.file_name,
            'line_number': self.line_number,
            'function_name': self.function_name,
            'context': self.context.to_dict(),
            'original_exception': str(self.original_exception) if self.original_exception else None,
            'traceback': self.traceback_str
        }
    
    def to_json(self) -> str:
        """Convert exception to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    def __str__(self) -> str:
        """String representation of exception"""
        return (
            f"[{self.severity.value.upper()}] {self.category.value}: {self.message}\n"
            f"Location: {self.file_name}:{self.line_number} in {self.function_name}\n"
            f"Correlation ID: {self.context.correlation_id}"
        )


class DataValidationException(BaseMLException):
    """Exception for data validation errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.DATA_VALIDATION
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class DataProcessingException(BaseMLException):
    """Exception for data processing errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.DATA_PROCESSING
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class ModelTrainingException(BaseMLException):
    """Exception for model training errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.MODEL_TRAINING
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.HIGH)
        super().__init__(message, **kwargs)


class ModelPredictionException(BaseMLException):
    """Exception for model prediction errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.MODEL_PREDICTION
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class ConfigurationException(BaseMLException):
    """Exception for configuration errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.CONFIGURATION
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.CRITICAL)
        super().__init__(message, **kwargs)


class ExternalServiceException(BaseMLException):
    """Exception for external service errors"""
    
    def __init__(self, message: str, **kwargs):
        kwargs['category'] = ErrorCategory.EXTERNAL_SERVICE
        kwargs['severity'] = kwargs.get('severity', ErrorSeverity.MEDIUM)
        super().__init__(message, **kwargs)


class StructuredLogger:
    """
    Structured logger with JSON formatting and correlation ID support
    """
    
    def __init__(self, name: str, log_level: str = "INFO"):
        """
        Initialize structured logger
        
        Args:
            name: Logger name
            log_level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level))
        
        # JSON formatter
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": %(message)s}'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _format_message(
        self,
        message: str,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Format message as JSON"""
        log_data = {
            'message': message,
            'correlation_id': correlation_id or str(uuid.uuid4())
        }
        log_data.update(kwargs)
        return json.dumps(log_data)
    
    def info(self, message: str, correlation_id: Optional[str] = None, **kwargs):
        """Log info message"""
        self.logger.info(self._format_message(message, correlation_id, **kwargs))
    
    def warning(self, message: str, correlation_id: Optional[str] = None, **kwargs):
        """Log warning message"""
        self.logger.warning(self._format_message(message, correlation_id, **kwargs))
    
    def error(self, message: str, correlation_id: Optional[str] = None, **kwargs):
        """Log error message"""
        self.logger.error(self._format_message(message, correlation_id, **kwargs))
    
    def critical(self, message: str, correlation_id: Optional[str] = None, **kwargs):
        """Log critical message"""
        self.logger.critical(self._format_message(message, correlation_id, **kwargs))
    
    def exception(self, exc: BaseMLException):
        """Log exception with full context"""
        self.logger.error(exc.to_json())


# Example usage
if __name__ == "__main__":
    # Initialize structured logger
    logger = StructuredLogger("ml_pipeline", "INFO")
    
    # Create error context
    context = ErrorContext(
        user_id="user_123",
        request_id="req_456",
        additional_context={'model_version': '1.0.0', 'dataset': 'student_performance'}
    )
    
    # Example 1: Data validation error
    try:
        # Simulate data validation error
        raise ValueError("Missing required column: 'math_score'")
    except Exception as e:
        exc = DataValidationException(
            message="Data validation failed",
            error_details=sys,
            context=context,
            original_exception=e
        )
        logger.exception(exc)
        print("\n" + str(exc))
    
    # Example 2: Model training error
    try:
        # Simulate model training error
        raise RuntimeError("Model convergence failed")
    except Exception as e:
        exc = ModelTrainingException(
            message="Model training failed after 100 iterations",
            error_details=sys,
            severity=ErrorSeverity.CRITICAL,
            context=context,
            original_exception=e
        )
        logger.exception(exc)
        print("\n" + str(exc))
    
    # Example 3: Structured logging without exception
    logger.info(
        "Model training started",
        correlation_id=context.correlation_id,
        model_type="RandomForest",
        n_estimators=100
    )

# Made with Bob
