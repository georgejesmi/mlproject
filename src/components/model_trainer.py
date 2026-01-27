import os
import sys
from src.logger import logger
from src.exception import CustomException
from src.utils import evaluate_model_performance, save_object
from src.config.paths import ARTIFACTS_MODELS_PATH
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


@dataclass
class ModelTrainerConfig:
    model_path:str = os.path.join(ARTIFACTS_MODELS_PATH,"model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def perform_model_training(self, train_arr, test_arr):
        try:
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            logger.info(f"Train data shape : {x_train.shape}")
            logger.info(f"Test data shape : {x_test.shape}")

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "K Nearest Regressor": KNeighborsRegressor(),
                "Support Vector Regressor": SVR(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor()
            }
            model_scores = evaluate_model_performance(x_train, y_train, x_test, y_test, models)
            model_scores.sort_values(by=["r2_score"], ascending=False, inplace=True)
            logger.info(model_scores)
            logger.info(f"Best Model : {model_scores.loc[0, 'model_name']}, R2 Score: {model_scores.loc[0, 'r2_score']}")
            if model_scores.loc[0, 'r2_score'] < 0.05:
                raise CustomException("No good model found")
            best_model = model_scores.loc[0, 'model_name']
            save_object(self.model_trainer_config.model_path, models[best_model])
        except Exception as e:
            logger.error(e)
            raise CustomException(e, sys)

