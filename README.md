# Formula 1 Race Finishing Position Prediction Pipeline

An end-to-end machine learning regression pipeline designed to predict the final finishing positions of Formula 1 races by analyzing historical data fields including drivers, constructors, circuits, and starting grid positions.

## 📊 Core Engineering Workflow

### 1. Relational Data Merging & Data Integrity Cleaning
* Consolidated multiple isolated relational tables (`results.csv`, `races.csv`, `drivers.csv`, `constructors.csv`) to map historical metrics.
* **Anomalies Handled:** Isolated and cleaned corrupted database string placekeepers (`\N` values representing DNFs or missing qualifying times) by mapping them to explicit null datatypes during file parsing, ensuring stable tensor matrix calculations.

### 2. Custom Feature Engineering
* Developed a unique domain-specific feature tracking structural competitive variance: `grid_delta_from_pole` (`grid position - 1`). 
* This calculation isolates a driver's relative structural disadvantage compared to the pole sitter, minimizing noise across varying grid sizes across different circuits.

### 3. Automated Preprocessing Architecture
* Designed a clean, reproducible preprocessing layout using Scikit-Learn's `ColumnTransformer`.
* Continuous values (`year`, `round`, `grid`) are normalized using `StandardScaler`.
* High-cardinality text tags (`driverRef`, `constructorRef`, `circuitId`) are vectorized cleanly using `OneHotEncoder(handle_unknown='ignore')`.

### 4. Algorithmic Benchmarking
Evaluated multiple machine learning algorithms using an 80/20 train-test split:
* **Linear Regression:** Serves as a simple mathematical baseline.
* **Random Forest Regressor:** Built to capture non-linear feature split dynamics.
* **Gradient Boosting Regressor (Top Performer):** Effectively optimized residual losses, achieving an **R² Variance Score of ~0.41** and a **Mean Absolute Error (MAE) of ~4.8 positions**.

---

## 🚀 Environment Setup & Live Execution

### 1. Requirements Configuration
Ensure your environment contains the required data science payload packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn notebook
```

### 2. File Organization
Place the raw Ergast dataset tables into a local subfolder named `data/` within your root workspace:
```text
F1_Race_Prediction/
├── data/
│   ├── results.csv
│   ├── races.csv
│   └── ...
└── f1_prediction.ipynb
```

### 3. Simulating Live Inferences
The notebook features a built-in predictive tool allowing you to calculate live finishes based on custom real-time telemetry inputs:

```python
predict_f1_race(
    year=2026, 
    round_num=1, 
    grid_position=3, 
    track_id=1, 
    driver_name='hamilton', 
    team_name='ferrari'
)
```