import os

PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(PATH, "data")

# Data paths
MODELS_PATH = os.path.join(DATA_PATH, "models")
PROD_MODEL_PATH = os.path.join(DATA_PATH, "prod_model")
PRICES_PATH = os.path.join(DATA_PATH, "prices")
SECRETS_PATH = os.path.join(PATH, "secrets.json")

# Default time zone
DEFAULT_TZ = "UTC"  # "Europe/Moscow"

# TEST/PROD
MODE = "PROD"
PROD_MODEL = 'model_31_2024_11_10'
