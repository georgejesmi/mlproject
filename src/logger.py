import datetime
import logging
import os
import sys
from src.config.paths import LOGS_DIR

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('{asctime} - {filename} - {lineno} -  {levelname} - {message}', style='{', datefmt='%Y-%m-%d %H:%M')

LOG_FOLDER = f"{datetime.date.today()}"
log_dir = os.path.join(LOGS_DIR, LOG_FOLDER)
os.makedirs(log_dir, exist_ok=True)
LOG_FILE_PATH = os.path.join(log_dir, f"app-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")

# sends logs to the console
console_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(console_handler)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)

# sends logs to .log file
file_handler = logging.FileHandler(LOG_FILE_PATH)
logger.addHandler(file_handler)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.INFO)
