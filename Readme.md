# Transformation Storage & Sample Queries

## Dataset & Tables
All cleaned and transformed data is stored in **BigQuery** under the dataset:

- **Dataset:** `dmml_assignment_churn_raw`

### Tables

#### `churn_clean`
Cleaned, type-safe dataset (one row per customer, includes `churn_label`, `ingest_date`).

| Column               | Type     | Description                       |
|----------------------|----------|-----------------------------------|
| customer_id          | STRING   | Unique customer identifier        |
| gender               | STRING   | Gender of customer                |
| senior_citizen       | INT64    | Senior citizen flag               |
| tenure               | INT64    | Months since signup               |
| internet_service     | STRING   | Internet service type             |
| contract             | STRING   | Contract type                     |
| payment_method       | STRING   | Payment method                    |
| monthly_charges      | FLOAT64  | Monthly charges                   |
| total_charges        | FLOAT64  | Total charges to date             |
| churn                | STRING   | "Yes"/"No"                        |
| churn_label          | INT64    | 1 = churn, 0 = no churn           |
| ingest_date          | DATE     | Load partition date               |

#### `churn_features`
Domain feature set for ML (categoricals left raw for sklearn encoding, plus engineered numeric/boolean features).

| Column                  | Type     | Description                                  |
|-------------------------|----------|----------------------------------------------|
| customer_id             | STRING   | Unique customer identifier                   |
| tenure                  | INT64    | Months since signup                          |
| monthly_charges         | FLOAT64  | Monthly charges                              |
| total_charges           | FLOAT64  | Total charges to date                        |
| is_senior               | BOOL     | Senior citizen flag                          |
| has_internet            | BOOL     | Internet service present                     |
| services_count          | INT64    | Number of add-on services                    |
| avg_charges_per_tenure  | FLOAT64  | Total charges / tenure                       |
| charges_per_service     | FLOAT64  | Monthly charges per active service           |
| internet_type           | STRING   | Internet type                                |
| contract_type           | STRING   | Contract category                            |
| payment_method          | STRING   | Payment method                               |
| contract_length_months  | INT64    | Contract length mapped to months             |
| tenure_bucket           | STRING   | Buckets of tenure ranges                     |
| is_high_spender         | BOOL     | Flag for charges above 80th percentile       |
| churn_label             | INT64    | 1 = churn, 0 = no churn                      |
| ingest_date             | DATE     | Load partition date                          |

> ⚠️ Encoding (one-hot, scaling) is **not** stored in DB. These are applied in the sklearn pipeline so the warehouse remains model-agnostic.

---

## Sample Queries

### 1. Counts & sanity checks
```sql
-- Row counts in clean vs features
SELECT
  (SELECT COUNT(*) FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_clean`)    AS clean_rows,
  (SELECT COUNT(*) FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`) AS feature_rows;

-- Churn distribution
SELECT churn_label, COUNT(*) AS n
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_clean`
GROUP BY churn_label;
```

### 2. Partitioned row counts
```sql
-- Last 7 days row count by ingest_date
SELECT ingest_date, COUNT(*) AS n
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
WHERE ingest_date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY) AND CURRENT_DATE()
GROUP BY ingest_date
ORDER BY ingest_date DESC;
```

### 3. Segment churn rates
```sql
-- Churn rate by contract type
SELECT contract_type, COUNT(*) AS n, AVG(churn_label) AS churn_rate
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
GROUP BY contract_type
ORDER BY churn_rate DESC;

-- Churn rate by internet type & payment method
SELECT internet_type, payment_method, COUNT(*) AS n, AVG(churn_label) AS churn_rate
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
GROUP BY internet_type, payment_method
ORDER BY churn_rate DESC
LIMIT 20;

```

### 4. Numeric correlations
```sql
-- Pearson correlation between numeric features and churn_label
WITH numeric_cols AS (
  SELECT
    monthly_charges, total_charges, tenure, services_count,
    avg_charges_per_tenure, charges_per_service, contract_length_months, churn_label
  FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
)
SELECT
  'monthly_charges'         AS feature, APPROX_CORR(monthly_charges, churn_label)        AS r UNION ALL
SELECT 'total_charges'      , APPROX_CORR(total_charges, churn_label)                    UNION ALL
SELECT 'tenure'             , APPROX_CORR(tenure, churn_label)                           UNION ALL
SELECT 'services_count'     , APPROX_CORR(services_count, churn_label)                   UNION ALL
SELECT 'avg_charges_per_tenure', APPROX_CORR(avg_charges_per_tenure, churn_label)        UNION ALL
SELECT 'charges_per_service', APPROX_CORR(charges_per_service, churn_label)              UNION ALL
SELECT 'contract_length_months', APPROX_CORR(contract_length_months, churn_label)
ORDER BY r DESC;

```

### 5. Cohort analysis

```sql
-- Churn rates by tenure cohorts
SELECT
  CASE
    WHEN tenure < 6  THEN '0-5m'
    WHEN tenure < 12 THEN '6-11m'
    WHEN tenure < 24 THEN '12-23m'
    WHEN tenure < 36 THEN '24-35m'
    ELSE '36m+'
  END AS tenure_cohort,
  COUNT(*) AS n,
  AVG(churn_label) AS churn_rate,
  AVG(monthly_charges) AS avg_monthly
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
GROUP BY tenure_cohort
ORDER BY churn_rate DESC;

```

### 6. Service bundle strength
```sql
-- Top high-churn segments by contract/payment
SELECT
  contract_type, payment_method,
  COUNT(*) AS n,
  AVG(churn_label) AS churn_rate
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
GROUP BY contract_type, payment_method
HAVING COUNT(*) >= 50
ORDER BY churn_rate DESC
LIMIT 15;

```

### 7. High-risk segments
```sql
-- Top high-churn segments by contract/payment
SELECT
  contract_type, payment_method,
  COUNT(*) AS n,
  AVG(churn_label) AS churn_rate
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
GROUP BY contract_type, payment_method
HAVING COUNT(*) >= 50
ORDER BY churn_rate DESC
LIMIT 15;

```

### 8. Customer-level retrieval (for API/demo)
```sql
-- Fetch one customer's features (for inference)
SELECT * EXCEPT(ingest_date)
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
WHERE customer_id = 'PUT_AN_ID_HERE'
LIMIT 1;

```

### 9. Partition hygiene
```sql
-- Analyze only the most recent partition
SELECT contract_type, COUNT(*) AS n, AVG(churn_label) AS churn_rate
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
WHERE ingest_date = (SELECT MAX(ingest_date) FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`)
GROUP BY contract_type
ORDER BY churn_rate DESC;

```

# Feature Store: Metadata & Retrieval

## 1. Feature Metadata Table
We maintain a metadata registry of all engineered features inside BigQuery for traceability.

```sql
CREATE OR REPLACE TABLE `dmml-assignment-469813.dmml_assignment_churn_raw.feature_metadata` (
  feature_name STRING,
  dtype STRING,
  description STRING,
  source STRING,
  version STRING,
  created_date DATE
);

## 2. Example Inserts

Each feature from the churn_features table is logged with type, description, and source.

```sql
INSERT INTO `dmml-assignment-469813.dmml_assignment_churn_raw.feature_metadata`
(feature_name, dtype, description, source, version, created_date)
VALUES
  ('tenure', 'INT64', 'Months since signup', 'churn_clean', 'v1', CURRENT_DATE()),
  ('monthly_charges', 'FLOAT64', 'Monthly charges', 'churn_clean', 'v1', CURRENT_DATE()),
  ('total_charges', 'FLOAT64', 'Total charges to date', 'churn_clean', 'v1', CURRENT_DATE()),
  ('is_senior', 'BOOL', 'Senior citizen flag', 'churn_features_sql', 'v1', CURRENT_DATE()),
  ('services_count', 'INT64', 'Number of active add-on services', 'churn_features_sql', 'v1', CURRENT_DATE()),
  ('avg_charges_per_tenure', 'FLOAT64', 'Total charges divided by tenure', 'churn_features_sql', 'v1', CURRENT_DATE()),
  ('internet_type', 'STRING', 'Internet service type', 'churn_clean', 'v1', CURRENT_DATE()),
  ('contract_type', 'STRING', 'Contract type', 'churn_clean', 'v1', CURRENT_DATE()),
  ('payment_method', 'STRING', 'Payment method', 'churn_clean', 'v1', CURRENT_DATE()),
  ('is_high_spender', 'BOOL', 'Above 80th percentile of monthly charges', 'churn_features_sql', 'v1', CURRENT_DATE());

```

## 3. Retrieval Examples
SQL Retrieval (single customer)
```sql
-- Retrieve latest features for one customer
SELECT * EXCEPT(ingest_date)
FROM `dmml-assignment-469813.dmml_assignment_churn_raw.churn_features`
WHERE customer_id = 'PUT_AN_ID_HERE'
ORDER BY ingest_date DESC
LIMIT 1;

```