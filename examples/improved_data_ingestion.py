"""
Production-Grade Data Ingestion with Error Handling and Validation
Demonstrates: Input validation, data quality checks, proper error handling
"""

import sys
import os
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import hashlib
import json

# Assuming these exist in your project
# from src.logger import logger
# from src.exception import CustomException

# For demonstration, using standard logging
logger = logging.getLogger(__name__)


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion with validation"""
    raw_data_source: str
    raw_data_path: str
    train_data_path: str
    test_data_path: str
    validation_data_path: Optional[str] = None
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
    def __post_init__(self):
        """Validate configuration after initialization"""
        if not 0 < self.test_size < 1:
            raise ValueError(f"test_size must be between 0 and 1, got {self.test_size}")
        if not 0 < self.validation_size < 1:
            raise ValueError(f"validation_size must be between 0 and 1, got {self.validation_size}")
        if self.test_size + self.validation_size >= 1:
            raise ValueError("test_size + validation_size must be less than 1")


class DataQualityChecker:
    """Data quality validation and checks"""
    
    @staticmethod
    def check_missing_values(df: pd.DataFrame, max_missing_pct: float = 5.0) -> dict:
        """
        Check for missing values in dataframe
        
        Args:
            df: Input dataframe
            max_missing_pct: Maximum allowed missing percentage
            
        Returns:
            Dictionary with missing value statistics
        """
        missing_stats = {}
        total_rows = len(df)
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            if missing_count > 0:
                missing_stats[col] = {
                    'count': int(missing_count),
                    'percentage': round(missing_pct, 2)
                }
                
                if missing_pct > max_missing_pct:
                    logger.warning(
                        f"Column '{col}' has {missing_pct:.2f}% missing values "
                        f"(threshold: {max_missing_pct}%)"
                    )
        
        return missing_stats
    
    @staticmethod
    def check_duplicates(df: pd.DataFrame) -> dict:
        """Check for duplicate rows"""
        duplicate_count = df.duplicated().sum()
        duplicate_pct = (duplicate_count / len(df)) * 100
        
        return {
            'count': int(duplicate_count),
            'percentage': round(duplicate_pct, 2)
        }
    
    @staticmethod
    def check_schema(df: pd.DataFrame, expected_columns: list) -> dict:
        """
        Validate dataframe schema
        
        Args:
            df: Input dataframe
            expected_columns: List of expected column names
            
        Returns:
            Dictionary with schema validation results
        """
        actual_columns = set(df.columns)
        expected_columns_set = set(expected_columns)
        
        missing_columns = expected_columns_set - actual_columns
        extra_columns = actual_columns - expected_columns_set
        
        return {
            'valid': len(missing_columns) == 0 and len(extra_columns) == 0,
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns)
        }
    
    @staticmethod
    def check_data_types(df: pd.DataFrame, expected_types: dict) -> dict:
        """Validate column data types"""
        type_issues = {}
        
        for col, expected_type in expected_types.items():
            if col in df.columns:
                actual_type = df[col].dtype
                if str(actual_type) != expected_type:
                    type_issues[col] = {
                        'expected': expected_type,
                        'actual': str(actual_type)
                    }
        
        return type_issues
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame, numerical_columns: list, method: str = 'iqr') -> dict:
        """
        Detect outliers in numerical columns
        
        Args:
            df: Input dataframe
            numerical_columns: List of numerical column names
            method: 'iqr' or 'zscore'
            
        Returns:
            Dictionary with outlier statistics
        """
        outlier_stats = {}
        
        for col in numerical_columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            else:  # zscore
                mean = df[col].mean()
                std = df[col].std()
                outliers = df[abs((df[col] - mean) / std) > 3]
            
            outlier_count = len(outliers)
            outlier_pct = (outlier_count / len(df)) * 100
            
            if outlier_count > 0:
                outlier_stats[col] = {
                    'count': int(outlier_count),
                    'percentage': round(outlier_pct, 2)
                }
        
        return outlier_stats


class DataIngestion:
    """
    Production-grade data ingestion with comprehensive validation
    """
    
    def __init__(self, config: DataIngestionConfig):
        """
        Initialize data ingestion
        
        Args:
            config: DataIngestionConfig instance
        """
        self.config = config
        self.quality_checker = DataQualityChecker()
        self.ingestion_metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__
        }
    
    def _validate_file_exists(self, file_path: str) -> None:
        """Validate that input file exists"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValueError(f"Data file is empty: {file_path}")
        
        logger.info(f"Validated file: {file_path} (size: {file_size} bytes)")
    
    def _compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute hash of dataframe for versioning"""
        data_str = df.to_json(orient='records')
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _save_metadata(self, output_dir: str) -> None:
        """Save ingestion metadata"""
        metadata_path = os.path.join(output_dir, 'ingestion_metadata.json')
        os.makedirs(output_dir, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.ingestion_metadata, f, indent=2)
        
        logger.info(f"Saved ingestion metadata to {metadata_path}")
    
    def perform_data_ingestion(self) -> Tuple[str, str, Optional[str]]:
        """
        Perform data ingestion with validation and quality checks
        
        Returns:
            Tuple of (train_path, test_path, validation_path)
        """
        logger.info("=" * 80)
        logger.info("Starting data ingestion pipeline")
        logger.info("=" * 80)
        
        try:
            # Step 1: Validate input file
            logger.info(f"Step 1: Validating input file: {self.config.raw_data_source}")
            self._validate_file_exists(self.config.raw_data_source)
            
            # Step 2: Load data
            logger.info("Step 2: Loading data from CSV")
            df = pd.read_csv(self.config.raw_data_source)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            # Step 3: Data quality checks
            logger.info("Step 3: Performing data quality checks")
            
            # Check missing values
            missing_stats = self.quality_checker.check_missing_values(df, max_missing_pct=5.0)
            if missing_stats:
                logger.warning(f"Missing values detected: {missing_stats}")
                self.ingestion_metadata['missing_values'] = missing_stats
            else:
                logger.info("✓ No missing values detected")
            
            # Check duplicates
            duplicate_stats = self.quality_checker.check_duplicates(df)
            if duplicate_stats['count'] > 0:
                logger.warning(f"Duplicate rows detected: {duplicate_stats}")
                self.ingestion_metadata['duplicates'] = duplicate_stats
            else:
                logger.info("✓ No duplicate rows detected")
            
            # Check schema (example - customize for your data)
            expected_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                              'lunch', 'test_preparation_course', 'math_score', 
                              'reading_score', 'writing_score']
            schema_validation = self.quality_checker.check_schema(df, expected_columns)
            if not schema_validation['valid']:
                logger.error(f"Schema validation failed: {schema_validation}")
                self.ingestion_metadata['schema_issues'] = schema_validation
            else:
                logger.info("✓ Schema validation passed")
            
            # Detect outliers
            numerical_cols = ['math_score', 'reading_score', 'writing_score']
            outlier_stats = self.quality_checker.detect_outliers(df, numerical_cols)
            if outlier_stats:
                logger.info(f"Outliers detected: {outlier_stats}")
                self.ingestion_metadata['outliers'] = outlier_stats
            
            # Step 4: Compute data hash for versioning
            data_hash = self._compute_data_hash(df)
            self.ingestion_metadata['data_hash'] = data_hash
            logger.info(f"Data hash: {data_hash}")
            
            # Step 5: Save raw data
            logger.info(f"Step 4: Saving raw data to {self.config.raw_data_path}")
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            df.to_csv(self.config.raw_data_path, index=False)
            
            # Step 6: Split data
            logger.info("Step 5: Splitting data into train/test/validation sets")
            
            # First split: train+val vs test
            train_val, test = train_test_split(
                df, 
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )
            
            # Second split: train vs validation
            val_size_adjusted = self.config.validation_size / (1 - self.config.test_size)
            train, validation = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=self.config.random_state
            )
            
            logger.info(f"Train set: {len(train)} rows ({len(train)/len(df)*100:.1f}%)")
            logger.info(f"Validation set: {len(validation)} rows ({len(validation)/len(df)*100:.1f}%)")
            logger.info(f"Test set: {len(test)} rows ({len(test)/len(df)*100:.1f}%)")
            
            # Step 7: Save splits
            logger.info("Step 6: Saving data splits")
            train.to_csv(self.config.train_data_path, index=False)
            test.to_csv(self.config.test_data_path, index=False)
            
            validation_path = None
            if self.config.validation_data_path:
                validation.to_csv(self.config.validation_data_path, index=False)
                validation_path = self.config.validation_data_path
            
            # Step 8: Save metadata
            self.ingestion_metadata['splits'] = {
                'train_rows': len(train),
                'validation_rows': len(validation),
                'test_rows': len(test),
                'total_rows': len(df)
            }
            
            output_dir = os.path.dirname(self.config.raw_data_path)
            self._save_metadata(output_dir)
            
            logger.info("=" * 80)
            logger.info("✓ Data ingestion completed successfully")
            logger.info("=" * 80)
            
            return self.config.train_data_path, self.config.test_data_path, validation_path
            
        except FileNotFoundError as e:
            logger.error(f"File not found error: {e}")
            raise
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty data error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during data ingestion: {e}", exc_info=True)
            raise


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create configuration
    config = DataIngestionConfig(
        raw_data_source="notebooks/data/stud.csv",
        raw_data_path="artifacts/data/raw_data.csv",
        train_data_path="artifacts/data/train_data.csv",
        test_data_path="artifacts/data/test_data.csv",
        validation_data_path="artifacts/data/validation_data.csv",
        test_size=0.2,
        validation_size=0.1,
        random_state=42
    )
    
    # Perform ingestion
    ingestion = DataIngestion(config)
    train_path, test_path, val_path = ingestion.perform_data_ingestion()
    
    print(f"\nData ingestion complete!")
    print(f"Train data: {train_path}")
    print(f"Test data: {test_path}")
    print(f"Validation data: {val_path}")

# Made with Bob
