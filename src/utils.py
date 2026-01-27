import sys
import dill
import os
import numpy as np
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as fp:
            dill.dump(obj, fp)
        logger.info(f"Saved object at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model_performance(x_train, y_train, x_test, y_test, models):
    models_r2_score = {}
    for model_name in models:
        model = models[model_name]
        model.fit(x_train, y_train)

        y_pred_test = model.predict(x_test)
        r2 = r2_score(y_test, y_pred_test)
        models_r2_score[model_name] = r2
    logger.info(f"Models R2 score: {models_r2_score}")
    models_r2_score_df = pd.DataFrame(models_r2_score.items(), columns=["model_name", "r2_score"])
    return models_r2_score_df
