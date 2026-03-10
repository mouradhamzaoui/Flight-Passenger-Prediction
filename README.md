# ✈️ Delta Airlines — Flight Passenger Prediction Platform

> ML-Powered Load Factor Forecasting | Standard Airbus/Amadeus MLOps 2026

![CI](https://github.com/TON_USERNAME/flight-passenger-prediction/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/TON_USERNAME/flight-passenger-prediction/actions/workflows/cd.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![Carrier](https://img.shields.io/badge/Carrier-Delta%20DL-red)
![Model](https://img.shields.io/badge/Model-LightGBM-green)
![R2](https://img.shields.io/badge/R²-0.9991-brightgreen)
![MAE](https://img.shields.io/badge/MAE-0.363%25-brightgreen)

## 🎯 Problem Statement
Delta Air Lines transporte ~200M passagers/an.
Optimiser le Load Factor est l'enjeu #1 de rentabilité :
**+1% LF = ~$250M de revenus additionnels annuels.**

## 🏆 Model Performance
| Model | MAE | R² | MAPE |
|-------|-----|----|------|
| 🥇 LightGBM | 0.363% | 0.9991 | 0.55% |
| 🥈 XGBoost | 0.414% | 0.9989 | 0.62% |
| 🥉 GradientBoosting | 0.504% | 0.9983 | 0.77% |
| RandomForest | 2.774% | 0.9561 | 4.02% |

## 🚀 Stack
| Tech | Usage |
|------|-------|
| Python 3.11 + uv | Environment |
| LightGBM / XGBoost | ML Models |
| MLflow | Experiment Tracking |
| FastAPI | REST API |
| Streamlit | POC Dashboard |
| PostgreSQL | Metadata Store |
| Docker Compose | Orchestration |
| GitHub Actions | CI/CD |
| Evidently AI | ML Monitoring |

## 🛫 Services
| Service | URL |
|---------|-----|
| FastAPI Swagger | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

## ⚡ Quick Start
```bash
# Clone
git clone https://github.com/TON_USERNAME/flight-passenger-prediction
cd flight-passenger-prediction

# Setup
pip install uv
uv venv && .venv/Scripts/activate
uv add -r requirements.txt

# Generate data + train
python data/download_bts.py
python poc/feature_engineering.py
python poc/train_models.py

# Launch services
docker-compose up -d

# Run tests
pytest tests/ -v
```

## 📊 Architecture
```mermaid
flowchart TD
    A[BTS Data\nDelta DL Only] --> B[Feature Engineering\n63 Features]
    B --> C[ML Training\nLightGBM R²=0.9991]
    C --> D[MLflow Registry]
    D --> E[FastAPI\n/predict/flight\n/predict/route\n/predict/airport]
    E --> F[Streamlit Dashboard]
    E --> G[React Frontend]
    H[GitHub Actions\nCI/CD] --> I[Docker\nMulti-Services]
    J[Evidently AI\nMonitoring] --> E
```

## ✅ Tests
```
53/53 tests passed
```