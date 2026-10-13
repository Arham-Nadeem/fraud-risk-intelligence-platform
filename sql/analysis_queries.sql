--Description:Reusable SQL queries for fraud operations, merchant risk, and high-risk transaction review.

--Executive fraud summary
SELECT * FROM vw_fraud_summary;

--Merchant categories with the highest observed fraud rates
SELECT *
FROM vw_merchant_risk
ORDER BY fraud_rate_pct DESC, fraud_value DESC;

--Daily fraud trend
SELECT *
FROM vw_daily_risk
ORDER BY transaction_date;

--High-value fraudulent transactions
SELECT
    transaction_id,
    timestamp,
    customer_id,
    amount,
    merchant_category,
    channel,
    country,
    new_device,
    amount_to_baseline,
    distance_from_home_km
FROM transactions
WHERE is_fraud = 1
ORDER BY amount DESC
LIMIT 50;
