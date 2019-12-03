from pathlib import Path

ROOT_PATH = Path(__file__).absolute().parent.parent
DATA_PATH = ROOT_PATH / 'data'
RESULT_DIR = ROOT_PATH / 'results'
MODELS_DIR = ROOT_PATH / 'models'
LOG_DIR = RESULT_DIR / 'logs'