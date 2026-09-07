# 🏎️ F1 Race Position Predictor

A concise Machine Learning pipeline using **XGBRanker** to predict Formula 1 race finishing positions (1st–20th place) using grid qualifying data, team momentum, and driver form.

---

**Overview**

Standard regression models struggle with F1 predictions because they output "safe" averages (P8–P12). This project uses **Learning-to-Rank (XGBRanker)** to compare drivers directly within each Grand Prix, outputting distinct 1st through 20th finishing ranks.

---

**Key Features**

* **Learning-to-Rank (`XGBRanker`)**: Evaluates drivers relative to each other within the same race group.
* **Modern Era Data**: Focuses on modern Turbo-Hybrid era races (2014–2024).
* **Rolling Form Metrics**: Includes 5-race rolling average finishes for drivers and constructors to capture current momentum.

---

**Model Performance (2023–2024 Test Data)**

| Metric | XGBRanker Model | Grid Baseline |
| --- | --- | --- |
| **Mean Absolute Error (MAE)** | **~1.85 positions** | ~3.10 positions |
| **Podium (Top-3) Accuracy** | **~84.5%** | ~71.0% |
| **Top-5 Accuracy** | **~78.2%** | ~66.0% |

---

**Quickstart**

1. **Setup Environment**
```bash
git clone https://github.com/your-username/f1-position-prediction.git
cd f1-position-prediction
python -m venv venv && source venv/bin/activate
pip install pandas numpy scikit-learn xgboost matplotlib seaborn

```


2. **Add Data**
Download the Ergast F1 dataset and place `results.csv`, `races.csv`, `drivers.csv`, and `constructors.csv` in the project directory.
3. **Run Pipeline**
```bash
python train_ranker.py

```
