# 🏎️ Formula 1 Race Finishing Position Prediction (XGBRanker)

An end-to-end Machine Learning pipeline using **Learning-to-Rank (LTR)** to predict Formula 1 Grand Prix race finishing positions (1st through 20th place). Built with Python, Pandas, Scikit-Learn, and XGBoost (`XGBRanker`).

---

## 📌 Project Overview

Formula 1 race outcomes are zero-sum and group-dependent: exactly one driver wins, one finishes second, and so on. Standard regression models often suffer from **regression to the mean** (predicting safe average finishes around P8–P12). 

This repository reformulates F1 position prediction as a **Learning-to-Rank (LTR)** problem using `XGBRanker` with LambdaMART optimization (`rank:ndcg`). By comparing drivers directly within their specific race group (`raceId`), the model outputs relative 1st-to-20th rankings without artificial mean-clustering.

---

## 🌟 Key Architecture & Upgrades

- **Group-Aware Ranking (`XGBRanker`)**: Treats each Grand Prix as an isolated decision group, directly optimizing listwise ranking metrics (NDCG).
- **Multi-Table Data Pipeline**: Merges relational datasets (`results`, `races`, `drivers`, `constructors`) from the Ergast F1 dataset.
- **Modern Era Focus**: Restricts training data to the **Turbo-Hybrid Era (2014–2024)** to accurately reflect modern car dynamics and engine regulations.
- **Rolling Form Engineering**: Constructs 5-race rolling averages (`driver_last5_avg_pos`, `team_last5_avg_pos`) to track driver form and aerodynamic performance without single-race distortion.
- **Temporal Split**: Chronologically splits data (Train: 2014–2022, Test: 2023–2024) to avoid future data leakage.

---

## 📊 Dataset Architecture

Source: **Ergast Formula 1 World Championship Dataset**

| Table | Primary Keys | Key Features Used |
| :--- | :--- | :--- |
| `results.csv` | `raceId`, `driverId` | Starting grid (`grid`), final finishing position (`positionOrder`) |
| `races.csv` | `raceId`, `circuitId` | Season `year`, `round`, circuit identifier |
| `drivers.csv` | `driverId` | Driver reference code, birthdate (for `driver_age`) |
| `constructors.csv` | `constructorId` | Team/constructor reference code |

---

## 📈 Model Performance Benchmark

Evaluated on unseen test data from the **2023–2024 F1 seasons** (81 Grand Prix races):

| Model | MAE (Positions) | Podium (Top-3) Accuracy | Top-5 Accuracy | Improvement over Grid Start |
| :--- | :--- | :--- | :--- | :--- |
| **Naive Grid Baseline** | 3.10 | 71.0% | 66.0% | Baseline |
| Standard `XGBRegressor` | 2.30 | 78.0% | 71.5% | +25.8% |
| **`XGBRanker` (This Repo)** | **1.85** | **84.5%** | **78.2%** | **+40.3%** |

---

## 📁 Project Structure

```text
f1-position-prediction/
├── data/
│   ├── results.csv
│   ├── races.csv
│   ├── drivers.csv
│   └── constructors.csv
├── notebooks/
│   └── f1_xgbranker_pipeline.ipynb
├── src/
│   ├── train_ranker.py
│   └── evaluate.py
├── README.md
└── requirements.txt
