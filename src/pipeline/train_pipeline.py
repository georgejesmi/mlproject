import sys

from src.logger import logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def __init__(self):
        logger.info("Training pipeline initialized")
    
    def start_training(self):
        try:
            # Step 1: Data Ingestion
            logger.info("Starting data ingestion...")
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.perform_data_ingestion()
            
            # Step 2: Data Transformation
            logger.info("Starting data transformation...")
            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.perform_data_transformation(
                train_data_path, test_data_path
            )
            
            # Step 3: Model Training
            logger.info("Starting model training...")
            model_trainer = ModelTrainer()
            model_trainer.perform_model_training(train_arr, test_arr)
            
            logger.info("Training pipeline completed successfully!")
            
        except Exception as e:
            logger.error("Training pipeline failed")
            raise CustomException(e, sys)
    