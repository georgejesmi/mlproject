import sys
import dill
import os
from src.logger import logger
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as fp:
            dill.dump(obj, fp)
        logger.info(f"Saved object at {file_path}")
    except Exception as e:
        raise CustomException(e, sys)