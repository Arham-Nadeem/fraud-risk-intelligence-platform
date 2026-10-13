#Description:Persists processed transactions to SQLite and creates reusable analytical views.
from __future__ import annotations

import sqlite3
import pandas as pd


def load_to_sqlite(df: pd.DataFrame, db_path) -> None:
    #Persist one analytics-ready table and expose reusable SQL views for analyst workflows.
    with sqlite3.connect(db_path) as conn:
        df.to_sql("transactions", conn, if_exists="replace", index=False)
        conn.executescript(
            """
            DROP VIEW IF EXISTS vw_fraud_summary;
            DROP VIEW IF EXISTS vw_daily_risk;
            DROP VIEW IF EXISTS vw_merchant_risk;

            CREATE VIEW vw_fraud_summary AS
            SELECT
                COUNT(*) AS transactions,
                SUM(is_fraud) AS fraud_transactions,
                ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct,
                ROUND(SUM(amount), 2) AS transaction_value,
                ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value
            FROM transactions;

            CREATE VIEW vw_daily_risk AS
            SELECT
                DATE(timestamp) AS transaction_date,
                COUNT(*) AS transactions,
                SUM(is_fraud) AS fraud_transactions,
                ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct,
                ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value
            FROM transactions
            GROUP BY DATE(timestamp);

            CREATE VIEW vw_merchant_risk AS
            SELECT
                merchant_category,
                COUNT(*) AS transactions,
                SUM(is_fraud) AS fraud_transactions,
                ROUND(100.0 * AVG(is_fraud), 3) AS fraud_rate_pct,
                ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_value
            FROM transactions
            GROUP BY merchant_category;
            """
        )
