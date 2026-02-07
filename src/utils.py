import sys
import dill
import os
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as fp:
            dill.dump(obj, fp)
        logger.info(f"Saved object at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model_performance(x_train, y_train, x_test, y_test, models, hyper_params):
    models_r2_score = {}
    trained_models = {}
    for model_name in models:
        model = models[model_name]
        logger.info(f"Evaluating model {model_name}")
        if model_name in hyper_params and model_name != 'CatBoost':
            gs = GridSearchCV(model, hyper_params[model_name], cv=5)
            gs.fit(x_train, y_train)
            best_estimator = gs.best_estimator_
            model = best_estimator
        model.fit(x_train, y_train)
        trained_models[model_name] = model

        y_pred_test = model.predict(x_test)
        r2 = r2_score(y_test, y_pred_test)
        models_r2_score[model_name] = r2
    logger.info(f"Models R2 score: {models_r2_score}")
    models_r2_score_df = pd.DataFrame(models_r2_score.items(), columns=["model_name", "r2_score"])
    return models_r2_score_df, trained_models
