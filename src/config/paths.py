import os
from datetime import datetime

# unique id for logs file created on execution
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def getRunID():
    return RUN_ID

# project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", ".."))

# Original Data Path
DATA_PATH = os.path.join(PROJECT_ROOT, "notebooks", "data/stud.csv")

# projects artifacts and logs path
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# artifacts sub-folder
ARTIFACTS_DATA_PATH = os.path.join(ARTIFACTS_DIR, f"run_{getRunID()}", "data")
ARTIFACTS_MODELS_PATH = os.path.join(ARTIFACTS_DIR, f"run_{getRunID()}", "models")

for path in [ARTIFACTS_DIR, LOGS_DIR, ARTIFACTS_DATA_PATH, ARTIFACTS_MODELS_PATH]:
    os.makedirs(path, exist_ok=True)
