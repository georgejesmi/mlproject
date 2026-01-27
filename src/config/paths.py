import os
from datetime import datetime

# unique id for logs file created on execution
RUN_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# project root path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),"..", ".."))

# projects artifacts and logs path
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts", f"run_{RUN_ID}")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# artifacts sub-folder
ARTIFACTS_DATA_PATH = os.path.join(ARTIFACTS_DIR, "data")
ARTIFACTS_MODELS_PATH = os.path.join(ARTIFACTS_DIR, "models")

for path in [ARTIFACTS_DIR, LOGS_DIR, ARTIFACTS_DATA_PATH, ARTIFACTS_MODELS_PATH]:
    os.makedirs(path, exist_ok=True)
