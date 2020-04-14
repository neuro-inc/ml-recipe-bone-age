import os
from pathlib import Path

data_path_str = os.environ.get("DATA_ROOT_PATH_ENV")


ROOT_PATH = Path(__file__).absolute().parent.parent
DATA_PATH = Path(data_path_str or ROOT_PATH / 'data')
RESULT_DIR = ROOT_PATH / 'results'
MODELS_DIR = ROOT_PATH / 'models'
LOG_DIR = RESULT_DIR / 'logs'
