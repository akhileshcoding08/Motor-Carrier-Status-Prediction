# 🚚 Carrier Status Predictor

Predicting **Active vs. Inactive** status of U.S. motor carriers from FMCSA MCS-150 census data using exploratory data analysis, supervised machine learning, and unsupervised segmentation.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

The **Federal Motor Carrier Safety Administration (FMCSA)** requires every registered motor carrier in the U.S. to file an **MCS-150 Motor Carrier Census** form. This project uses that public census data (~4.47 million rows × 147 columns) to:

1. Explore and clean large-scale government tabular data.
2. Engineer meaningful features from fleet, driver, cargo, and safety-review information.
3. Train and compare multiple classification models to predict whether a carrier's operating authority is **Active (A)** or **Inactive (I)**.
4. Segment carriers into natural operational groups using **K-Means clustering** for risk-tiering / outreach use cases.
5. Package the best-performing model into a deployable artifact for downstream applications (e.g., a Flask web app).

---

## 🎯 Problem Statement

Regulators, insurers, and logistics partners need to know which registered carriers are actually still operating. This project frames that as a **binary classification problem**:

> Given a carrier's fleet size, driver count, cargo types, safety review history, and operating profile, predict whether the carrier is currently **Active** or **Inactive**.

---

## 🗂️ Dataset

| Detail | Value |
|---|---|
| Source | FMCSA MCS-150 Motor Carrier Census (`export.csv` / `export.zip`) |
| Full size | ~4.47M rows × 147 columns (~2 GB) |
| Sampling | Reproducible ~12% random sample (~540K rows) drawn via chunked streaming |
| Target | `STATUS_CODE` → `is_active` (1 = Active, 0 = Inactive) |
| Key fields used | Fleet size, truck/power units, driver counts, cargo-type flags, hazmat indicator, safety rating, prior revocation flag, carrier operation type, state, business org type |

> The full 4.47M-row file is **not included** in this repository due to size. See [Setup](#-setup--installation) for how to obtain it.

---

## 🧠 Approach

### 1. Data Loading
- Streams the raw CSV in 300K-row chunks (never loads the full 2 GB file into memory at once).
- Filters to only `Active` / `Inactive` status records.
- Draws a reproducible random sample for fast iteration; the same pipeline scales to the full file given enough RAM (16 GB+ recommended).

### 2. Data Cleaning & Feature Engineering
- Converts 40+ cargo-type indicator columns into a single `num_cargo_types` count.
- Coerces numeric columns (fleet size, driver counts, mileage, etc.) with error-safe parsing.
- Derives `carrier_age_years` from the registration date.
- Builds binary flags: `is_hazmat`, `prior_revoke`, `has_safety_review`.
- Defines the modeling target `is_active`.

### 3. Exploratory Data Analysis (EDA)
- Missing-value profiling across all columns.
- Active vs. Inactive class balance.
- Geographic distribution of carriers by state.
- Carrier operation type (interstate / intrastate) and authority classification breakdowns.
- Fleet size, driver count, and power unit distributions (log-scaled).
- Cargo diversity and hazmat carrier proportions.
- Active rate by safety-review status and operation type.
- Correlation heatmap across numeric features.

### 4. Machine Learning — Active/Inactive Classification
Three models are trained inside a unified `scikit-learn` `Pipeline` (median imputation + scaling for numeric features, constant imputation + one-hot encoding for categorical features):

| Model | Purpose |
|---|---|
| Logistic Regression | Fast, interpretable baseline |
| Decision Tree | Captures simple non-linear splits |
| Random Forest | Best overall performer; captures complex feature interactions |

Models are compared using **Accuracy** and **ROC-AUC**, with ROC curves, a confusion matrix, and a classification report for the best model, plus a **Random Forest feature-importance** chart.

### 5. Unsupervised Segmentation (K-Means)
- Clusters carriers on fleet size, driver counts, cargo diversity, and tenure.
- Uses the **elbow method** to select the number of clusters.
- Visualizes clusters in 2D via **PCA** projection.
- Produces per-cluster profiles useful for marketing, risk-tiering, or outreach targeting.

### 6. Model Deployment
- Serializes the winning pipeline, feature lists, categorical dropdown options, and numeric ranges into a single `joblib` bundle (`carrier_status_model.pkl`).
- Includes a reload sanity-check that mimics how a Flask/FastAPI service would consume the bundle.

### 7. Conclusions
- The sample is close to perfectly balanced between Active and Inactive carriers.
- Carrier size metrics are heavily right-skewed — most carriers are small operations.
- Having a recorded safety review is strongly associated with active status.
- California, Texas, and Florida have the largest carrier populations.
- **Random Forest** achieves the best ROC-AUC among the three models tested.
- K-Means reveals distinct segments separating small/new carriers from large, established interstate fleets.

---

## 📁 Project Structure

```
Carrier-Status-Predictor/
│
├── Carrier_Status_Predictor.ipynb   # Main analysis & modeling notebook
├── carrier_status_model.pkl         # Serialized deployment bundle (generated)
├── data/
│   ├── export.zip                   # Raw FMCSA export (not tracked in git)
│   └── mcs150_sample.csv            # Optional pre-built fallback sample
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation (this file)
└── docs/
    └── Carrier_Status_Predictor_Documentation.docx   # Full step-by-step write-up
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- Jupyter Notebook / JupyterLab
- ~16 GB RAM recommended if running on the full dataset

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Carrier-Status-Predictor.git
cd Carrier-Status-Predictor
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download the MCS-150 export from the [FMCSA data portal](https://ai.fmcsa.dot.gov/SMS/) and place it next to the notebook as either:
- `export.zip` (auto-extracted by the notebook), or
- `export.csv` directly, or
- Provide your own `mcs150_sample.csv` as a lightweight fallback.

### 5. Run the notebook
```bash
jupyter notebook Carrier_Status_Predictor.ipynb
```
Run all cells top to bottom. The final cells save a deployable model bundle as `carrier_status_model.pkl`.

---

## 📦 requirements.txt
```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

---

## 📊 Results Snapshot

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | ~ (see notebook output) | ~ |
| Decision Tree | ~ | ~ |
| **Random Forest (best)** | **highest** | **highest** |

> Exact metrics depend on the random sample drawn at run time (`RANDOM_STATE` is fixed for reproducibility). See the notebook's Section 4 for live numbers and plots.

---

## 🚀 Using the Trained Model

```python
import joblib

bundle = joblib.load("carrier_status_model.pkl")
pipeline = bundle["pipeline"]

# Build a single-row DataFrame matching bundle["feature_num"] + bundle["feature_cat"]
prediction = pipeline.predict(sample_row)
probability = pipeline.predict_proba(sample_row)[:, 1]

print(bundle["target_names"][prediction[0]], probability[0])
```

---

## 🛠️ Tech Stack
- **Language:** Python
- **Data handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Modeling:** scikit-learn (Logistic Regression, Decision Tree, Random Forest, K-Means, PCA)
- **Deployment artifact:** joblib

---

## 🔮 Future Improvements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV) for the Random Forest model.
- Add gradient boosting models (XGBoost / LightGBM) for comparison.
- Build a lightweight Flask or FastAPI app around `carrier_status_model.pkl` for live predictions.
- Track experiments with MLflow.
- Automate ingestion of the full 4.47M-row dataset via scheduled batch jobs.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome. Feel free to fork the repo and submit a pull request.

## 📄 License
This project is licensed under the MIT License — see the `LICENSE` file for details.

## 🙋 Author
Maintained by *Akhilesh* — [GitHub](https://github.com/akhileshcoding08). Reach out via GitHub Issues for questions or suggestions.
