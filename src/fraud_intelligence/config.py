#Description:Central configuration for reproducible data generation, modeling, and business-cost assumptions.
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"
DB_PATH = ROOT / "fraud_intelligence.db"

RANDOM_SEED = 42
N_TRANSACTIONS = 60000
FRAUD_BASE_RATE = 0.022

#Business costs used to tune the decision threshold.
MANUAL_REVIEW_COST = 4.50
FALSE_POSITIVE_CUSTOMER_COST = 18.00
MISSED_FRAUD_MULTIPLIER = 1.00

#Chronological data split proportions.
TRAIN_SHARE = 0.60
VALID_SHARE = 0.20
TEST_SHARE = 0.20
