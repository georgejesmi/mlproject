import sys
import os
from src.logger import logger
from src.exception import CustomException
from src.utils import save_object

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "preprocessor.pkl"))

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        try:
            num_cols = ['reading_score', 'writing_score']
            cat_cols = ['gender', 'race_ethnicity', 'lunch', 'parental_level_of_education', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )

            column_transformer_obj = ColumnTransformer(
                transformers=[
                    ('numerical_transformation', num_pipeline, num_cols),
                    ('categorical_transformation', cat_pipeline, cat_cols)
                ]
            )
            return column_transformer_obj
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

    def perform_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Loaded train and test data from artifacts for preprocessing")

            train_y = train_df['math_score']
            train_x = train_df.drop(['math_score'], axis=1)

            test_y = test_df['math_score']
            test_x = test_df.drop(['math_score'], axis=1)

            logger.info("Loading preprocessor pipeline object")
            preprocessor_pipeline_obj = self.get_data_transformer_object()

            logger.info("Initiating transformation on train data...")
            train_x_arr = preprocessor_pipeline_obj.fit_transform(train_x)
            train_arr = np.c_[train_x_arr, train_y]

            logger.info("Initiating transformation on test data...")
            test_x_arr = preprocessor_pipeline_obj.transform(test_x)
            test_arr = np.c_[test_x_arr, test_y]

            logger.info("Completed data preprocessing on train data and test data !")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_pipeline_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

