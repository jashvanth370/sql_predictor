# ML-Based SQL Query Performance Prediction

Research project — predicting SQL query execution time using
multi-modal machine learning features.

## Research Topic
Machine Learning-Based SQL Query Performance Prediction
Using Multi-Modal Features (Query Plan, Semantic, and System Metrics)

## Project Structure
- `scripts/` — all pipeline scripts from data collection to prediction
- `data/`    — feature datasets (raw data excluded from repo)
- `models/`  — saved trained models (excluded from repo)
- `outputs/` — evaluation charts and figures

## Pipeline
| Phase | Script | Description |
|-------|--------|-------------|
| 2 | setup_tpch.py | Create TPC-H database |
| 2 | generate_data.py | Load benchmark data |
| 2 | collect_data.py | Collect query execution data |
| 3 | feature_engineering.py | Extract plan, semantic, system features |
| 4 | preprocess.py | Clean, scale, split dataset |
| 5 | train_model.py | Train Random Forest and XGBoost |
| 6 | evaluate.py | Evaluate and visualize results |
| 6 | predict.py | Predict execution time for new queries |

## Models Used
- Random Forest Regressor
- XGBoost Regressor

## Features
- Query plan features (operator types, depth, cost, rows)
- Semantic features (joins, conditions, subqueries, clauses)
- System metrics (CPU, memory, disk I/O)

## Requirements
```bash
pip install -r requirements.txt
```