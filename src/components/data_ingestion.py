import sys
import os
import pandas as pd
from seaborn.external.docscrape import header
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.exception import CustomException
from dataclasses import dataclass
from src.config.paths import ARTIFACTS_DATA_PATH
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join(ARTIFACTS_DATA_PATH, "raw_data.csv")
    logger.info(f"Original data path : {raw_data_path}")
    train_data_path: str = os.path.join(ARTIFACTS_DATA_PATH, "train_data.csv")
    logger.info(f"Train data path :  {train_data_path}")
    test_data_path: str = os.path.join(ARTIFACTS_DATA_PATH, "test_data.csv")
    logger.info(f"Test data path : {test_data_path}")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def perform_data_ingestion(self):
        logger.info("Initiating data ingestion...")
        try:
            df = pd.read_csv("../../notebooks/data/stud.csv")
            logger.info("Successfully loaded data from CSV")

            # os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            # logger.info(f"Created Artifacts directory : {self.ingestion_config.raw_data_path}")

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logger.info(f"Saved raw data as CSV to Artifacts directory")

            logger.info("Splitting data into train and test sets...")
            train, test = train_test_split(df, test_size=0.2, random_state=1)

            train.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            logger.info(f"Saved train data as CSV to Artifacts directory")
            test.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info(f"Saved test data as CSV to Artifacts directory")
            # index = False, prevents the index of dataframe being written as a column in CSV File

            logger.info("Successfully split data into train and test sets.Data Ingestion completed !")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except FileNotFoundError as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.perform_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr = data_transformation.perform_data_transformation(train_data_path, test_data_path)

    model_trainer = ModelTrainer()
    model_trainer.perform_model_training(train_arr, test_arr)