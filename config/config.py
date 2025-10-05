"""
Configuration module
"""

import os
import json
from dotenv import load_dotenv


load_dotenv()
host = os.getenv("HOST")
port = os.getenv("PORT")
database = os.getenv("DATABASE")
user = os.getenv("USER_DB")
password = os.getenv("PASSWORD")

db_params = {
    "host": host,
    "port": port,
    "database": database,
    "user": user,
    "password": password,
}

BACKTESTING_CONFIG = None
with open("parameter/in_sample.json", 'r', encoding="utf-8") as f:
    BACKTESTING_CONFIG = json.load(f)

OPTIMIZATION_CONFIG = None
with open("parameter/optimization.json", 'r', encoding="utf-8") as f:
    OPTIMIZATION_CONFIG = json.load(f)

BEST_CONFIG = None
# Create a default config if optimized_parameter.json doesn't exist
try:
    with open("parameter/optimized_parameter.json", 'r', encoding="utf-8") as f:
        BEST_CONFIG = json.load(f)
except FileNotFoundError:
    BEST_CONFIG = OPTIMIZATION_CONFIG  # Use optimization config as fallback
