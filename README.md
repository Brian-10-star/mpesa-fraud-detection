# M-Pesa Fraud Detection ML Platform

An end-to-end machine learning platform for real-time fraud detection on M-Pesa transactions. Built with a Kenyan fintech context using realistic M-Pesa transaction types, Kenyan names, KES amounts, and location data across 15 Kenyan cities.

![Dashboard](docs/dashboard.png)

---

## Architecture

The platform is built in 8 layers, each feeding the next:

```
Kafka Producer
     |
     v
mpesa_transactions (Kafka Topic, port 9093)
     |
     v
Consumer --> raw_transactions (PostgreSQL)
     |
     v
Feature Pipeline --> features table (20 features, 5 modules)
     |
     v
Label Generator --> fraud labels (rule-based heuristics)
     |
     v
Training Pipeline --> MLflow Experiment Tracking
     |
     v
Model Registry (mpesa-fraud-detector v2)
     |
     v
FastAPI (port 8000) --> prediction_log table
     |
     v
Drift Monitor --> drift_alerts table + HTML reports
     |
     v
Streamlit Dashboard (port 8501)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Streaming | Apache Kafka 7.4.0, confluent-kafka 2.6.1 |
| Database | PostgreSQL 18, SQLAlchemy 2.0.36, psycopg2 2.9.11 |
| Feature Engineering | Python 3.13, Pandas 2.2.3 |
| Machine Learning | scikit-learn 1.5.2, XGBoost 2.1.3, imbalanced-learn 0.12.4 |
| Experiment Tracking | MLflow 2.18.0 |
| API Serving | FastAPI 0.115.5, Uvicorn 0.32.1, Pydantic 2.10.3 |
| Drift Monitoring | scipy 1.17.1 (KS test + chi-squared test) |
| Dashboard | Streamlit 1.40.2 |
| Containerisation | Docker, Docker Compose |
| Testing | pytest 8.3.4 |
| CI | GitHub Actions |

---

## Project Structure

```
mpesa-fraud-detection/
|-- src/
|   |-- ingestion/
|   |   |-- schema.py           # MpesaTransaction dataclass
|   |   |-- producer.py         # Kafka transaction generator
|   |   `-- consumer.py         # Kafka to PostgreSQL writer
|   |-- features/
|   |   |-- feature_pipeline.py # Orchestrator
|   |   |-- temporal_features.py
|   |   |-- velocity_features.py
|   |   |-- amount_features.py
|   |   |-- behavioral_features.py
|   |   `-- feature_store.py
|   |-- training/
|   |   |-- train.py            # Main training script
|   |   |-- data_loader.py
|   |   |-- models.py
|   |   |-- evaluate.py
|   |   |-- register_model.py
|   |   `-- label_generator.py
|   |-- api/
|   |   |-- main.py             # FastAPI entry point
|   |   |-- schemas.py
|   |   |-- model_loader.py
|   |   |-- prediction_logger.py
|   |   |-- batch_scorer.py
|   |   `-- routes/
|   |       |-- predict.py
|   |       |-- health.py
|   |       |-- metrics.py
|   |       `-- model_info.py
|   |-- monitoring/
|   |   |-- monitor.py          # Orchestrator
|   |   |-- drift_detector.py
|   |   `-- alert_logger.py
|   `-- dashboard/
|       |-- app.py              # Streamlit entry point
|       `-- pages/
|           |-- live_feed.py
|           |-- model_metrics.py
|           `-- drift_report.py
|-- tests/
|   |-- test_features.py        # 13 tests
|   |-- test_model.py           # 6 tests
|   `-- test_api.py             # 8 tests
|-- data/
|   `-- reports/                # HTML drift reports (auto-generated)
|-- docker-compose.yml
|-- requirements.txt
`-- .env.example
```

---

## Database Schema (frauddb)

Five tables in PostgreSQL:

**raw_transactions** - every transaction received from Kafka
- 14 columns: transaction_id, type, sender/receiver phone and name, amount, balance before/after, location, device_fingerprint, timestamp

**features** - engineered feature row per transaction
- 20 features: temporal (6), velocity (4), amount (3), behavioral (4), plus fraud label columns

**prediction_log** - every prediction made by the API
- transaction_id, fraud_probability, is_fraud, model_version, predicted_at

**drift_alerts** - drift results logged over time
- feature_name, drift_score, alert_level, detected_at

**model_registry_meta** - local record of champion model version

---

## Feature Engineering

20 features engineered across 5 modules per transaction:

| Module | Features |
|---|---|
| Temporal | hour_of_day, day_of_week, is_night, is_weekend, is_month_start, is_month_end |
| Velocity | txn_count_last_10min, txn_count_last_1hr, txn_sum_last_10min, txn_sum_last_1hr |
| Amount | amount_zscore, amount_vs_sender_mean, is_large_amount |
| Behavioral | is_new_device, is_new_location, unique_receivers_last_1hr, type_frequency |
| Identity | amount, transaction_type (passed through to model) |

**Velocity** features query `raw_transactions` for each sender's recent history. **Amount** features compute a z-score by comparing the transaction amount against the sender's historical mean and standard deviation. **Behavioral** features detect account takeover signals: new device fingerprints, new locations, and high receiver diversity within one hour.

---

## Fraud Labeling

Labels are generated using rule-based heuristics (weak supervision). A transaction is labeled fraud if 2 or more of the following rules fire:

| Rule | Signal |
|---|---|
| Night (00:00-04:59) + amount > 3x sender mean | Suspicious timing and amount |
| 5+ transactions in last 10 minutes | Velocity attack |
| New device AND new location simultaneously | Account takeover |
| Amount z-score > 3.0 | Statistically extreme amount |
| Amount > KES 50,000 + new device | High-value unknown device |
| 4+ unique receivers in last 1 hour | Fund distribution pattern |

This is a legitimate industry technique called weak supervision, used to bootstrap ML labels without human-labeled ground truth data.

**Results on 339 transactions:**
- Fraud: 34 (10.0%)
- Legitimate: 305 (90.0%)

---

## ML Training

Three models trained on SMOTE-balanced data (22 fraud upsampled to 188 to match 188 legitimate):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

**Note on perfect scores:** The labels were generated by rules derived from the same features the model trains on. The model learns to replicate the rules exactly, producing perfect metrics. In a production system with human-labeled ground truth, scores would be lower and more informative. The value here is the complete pipeline architecture, not the metric values.

All experiments tracked in MLflow. Champion model registered as `mpesa-fraud-detector v2`.

![MLflow](docs/mlflow.png)

---

## API Endpoints

FastAPI server on port 8000. Interactive docs at `http://127.0.0.1:8000/docs`.

| Endpoint | Method | Description |
|---|---|---|
| `/predict` | POST | Score a transaction, returns fraud_probability and is_fraud |
| `/health` | GET | Model load status and API health |
| `/metrics` | GET | Aggregate prediction statistics from prediction_log |
| `/model-info` | GET | Current model name and version |
| `/` | GET | Service info and endpoint list |

**Sample request:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN-DEMO-001",
    "transaction_type": "Send Money",
    "sender_phone": "0712345678",
    "receiver_phone": "0798765432",
    "sender_name": "Brian Chira",
    "receiver_name": "Amina Wanjiku",
    "amount": 75000.00,
    "sender_balance_before": 80000.00,
    "sender_balance_after": 5000.00,
    "location": "Nairobi CBD",
    "device_fingerprint": "DEV-NEW-UNKNOWN-001",
    "timestamp": "2026-05-19T03:15:00"
  }'
```

**Sample response:**

```json
{
  "transaction_id": "TXN-DEMO-001",
  "fraud_probability": 1.0,
  "is_fraud": true,
  "model_version": "latest",
  "message": "FRAUD DETECTED"
}
```

![API Docs](docs/api.png)

---

## Drift Monitoring

The monitoring pipeline compares two datasets using statistical tests:

- **Reference data:** earliest 70% of labeled features (training window)
- **Current data:** most recent 200 predictions from prediction_log

**Tests used:**
- Kolmogorov-Smirnov (KS) test for numerical features
- Chi-squared test for boolean features

**Alert thresholds:**
- score >= 0.25: HIGH
- score >= 0.10: MEDIUM
- score < 0.10: OK

**Results from first monitoring run (339 transactions):**

| Feature | Drift Score | Alert Level |
|---|---|---|
| is_weekend | 1.0000 | HIGH |
| is_large_amount | 0.2496 | MEDIUM |
| hour_of_day | 0.2250 | MEDIUM |
| day_of_week | 0.1150 | MEDIUM |
| amount | 0.0835 | OK |
| txn_count_last_10min | 0.0042 | OK |
| amount_zscore | 0.0000 | OK |

HTML drift reports are saved to `data/reports/` on each run.

---

## Test Suite

27 tests across 3 files, 0 failures.

```
tests/test_features.py    13 tests   feature engineering correctness
tests/test_model.py        6 tests   model loading and prediction validity
tests/test_api.py          8 tests   API endpoints, validation, error handling
```

Run locally:

```bash
pytest tests/ -v
```

GitHub Actions CI runs `tests/test_features.py` automatically on every push to main using an isolated PostgreSQL test database.

---

## M-Pesa Transaction Types

The producer generates 7 transaction types weighted by real-world frequency:

| Type | Weight | Amount Range (KES) |
|---|---|---|
| Send Money | 30% | 50 - 70,000 |
| Buy Goods | 25% | 20 - 15,000 |
| Pay Bill | 15% | 100 - 50,000 |
| Withdraw | 10% | 100 - 70,000 |
| Pochi la Biashara | 8% | 10 - 5,000 |
| Airtime Purchase | 7% | 5 - 1,000 |
| Lipa na Mpesa | 5% | 50 - 30,000 |

Traffic volume adjusts by hour of day: LOW (00:00-05:59), HIGH (07:00-08:59 and 17:00-18:59), MEDIUM (all other hours).

---

## Setup and Running

### Prerequisites

- Python 3.13
- PostgreSQL 18
- Docker Desktop

### 1. Clone and install

```bash
git clone https://github.com/Brian-10-star/mpesa-fraud-detection.git
cd mpesa-fraud-detection
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your PostgreSQL credentials
```

### 3. Create the database

```bash
psql -U postgres -c "CREATE DATABASE frauddb;"
```

### 4. Start services

```bash
# Terminal 1: Kafka and Zookeeper
docker-compose up -d

# Terminal 2: MLflow tracking server
mlflow server --host 0.0.0.0 --port 5000

# Terminal 3: FastAPI
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 4: Streamlit dashboard
streamlit run src/dashboard/app.py
```

### 5. Generate data and train

```bash
# Generate transactions (run producer and consumer simultaneously)
python src/ingestion/producer.py
python src/ingestion/consumer.py

# Engineer features
python src/features/feature_pipeline.py

# Generate fraud labels
python src/training/label_generator.py

# Train models
python src/training/train.py

# Score all transactions
python src/api/batch_scorer.py

# Run drift monitoring
python src/monitoring/monitor.py
```

---

## Key Engineering Decisions

**confluent-kafka over kafka-python:** kafka-python 2.0.2 has a known incompatibility with Python 3.13 (invalid file descriptor error on the selector). confluent-kafka 2.6.1 is the actively maintained library used in production Kafka deployments.

**Weak supervision for labels:** No human-labeled fraud data exists for this dataset. Rule-based heuristics combining velocity, timing, and device signals are a standard industry technique for bootstrapping ML labels. Every rule is documented and every label includes the rules that fired.

**scipy over Evidently AI for drift detection:** Evidently 0.4.x through 0.5.x has a Pydantic v2 / Python 3.13 incompatibility. The scipy KS test and chi-squared test are the same statistical methods Evidently uses internally, implemented directly with full control over thresholds and scoring logic.

**PostgreSQL as feature store:** Feast and dedicated feature stores are deferred. PostgreSQL provides ACID guarantees, complex SQL queries across feature history, and zero additional infrastructure.

**localhost over WSL IP for PostgreSQL:** Python scripts run on Windows where PostgreSQL is installed. Using localhost eliminates the dynamic WSL IP address that changes on every system restart.

---

## Author

**Brian Mbugua Chira**
BSc Computer Science, Egerton University (Expected 2028)
Nairobi, Kenya

GitHub: [github.com/Brian-10-star](https://github.com/Brian-10-star)
LinkedIn: [linkedin.com/in/mbuguabrian](https://www.linkedin.com/in/mbuguabrian)