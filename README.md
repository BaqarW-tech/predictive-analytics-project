# 🤖 Nitaqat Breach Risk Predictor
### Predictive Analytics · Vision 2030 Portfolio Series · App 3 of 4

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://predictive-analytics-project-er6efuruhqkdxklr4rq46u.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange)
![Analytics](https://img.shields.io/badge/Analytics-Predictive%20ML-red)

---

## 🎯 Problem This Solves

Nitaqat penalties are imposed **after** a company breaches compliance — but the signals of an impending breach are visible **months earlier** in hiring trends, salary ratios, and workforce composition shifts.

This app trains three machine learning classifiers (Random Forest, Gradient Boosting, Logistic Regression) on company-level workforce features to predict which companies are likely to breach Nitaqat compliance in the next reporting cycle — enabling **pre-emptive intervention** rather than reactive enforcement.

---

## 🧠 Machine Learning Pipeline

### Feature Engineering (14 Predictors)

| Feature | Description |
|---|---|
| `saudization_pct` | Current Saudi worker % |
| `gap_to_target` | Distance from Nitaqat threshold |
| `recent_saudi_hire_pct` | % of last 2 years' hires that are Saudi |
| `saudization_trend` | Change in Saudization rate (recent vs older) |
| `salary_ratio` | Saudi avg salary ÷ overall avg salary |
| `edu_bachelor_pct` | % of Saudi staff with bachelor's degree or above |
| `female_pct` | Female workforce participation % |
| `contract_pct` | Temporary/contract employee ratio |
| `parttime_pct` | Part-time ratio |
| `ft_saudi_pct` | Saudi staff on full-time contracts |
| `q1_saudi_hire_ratio` | Q1 hiring concentration (audit-gaming signal) |
| `hiring_variance` | Std dev of monthly hires (stability signal) |
| `headcount` | Total company size |
| `senior_emp_pct` | Senior age group (45+) proportion |

### Models Compared

| Model | Strength |
|---|---|
| **Random Forest** | Handles feature interactions, robust to outliers |
| **Gradient Boosting** | Sequential error correction, often highest AUC |
| **Logistic Regression** | Interpretable baseline, good for regulatory explainability |

### Evaluation Metrics
- **ROC-AUC** — primary ranking metric
- **5-fold Stratified CV AUC** — generalisation estimate
- **Average Precision** — handles class imbalance correctly
- **Confusion Matrix** — visualise false positive / false negative trade-offs
- **Threshold sweep** — Precision / Recall / F1 vs decision threshold

---

## 📊 Dashboard Sections

| Section | Content |
|---|---|
| **KPI Row** | Best model AUC, selected model AUC, AP, % at-risk companies |
| **ROC Curves** | All 3 models overlaid for direct comparison |
| **PR Curves** | Precision-Recall for imbalanced class context |
| **Confusion Matrix** | True/False Positive/Negative breakdown |
| **Feature Importances** | Top 12 predictors from best tree model |
| **Threshold Sweep** | Interactive P/R/F1 vs threshold slider |
| **Probability Histogram** | Distribution separation of breach vs compliant |
| **Sector Risk Bar** | Avg predicted breach probability by sector |
| **Risk Scatter** | Saudization trend × breach prob, bubble = headcount |
| **Company Cards** | Every company ranked by predicted breach probability |
| **Model Scorecard** | Side-by-side model comparison table |
| **Policy Insight Box** | Auto-generated ML interpretation & recommendations |

---

## 🚀 Quick Start

```bash
git clone https://github.com/BaqarW-tech/saudi-workforce-predictive.git
cd saudi-workforce-predictive
cp ../saudi-workforce-descriptive/saudi_workforce_data.csv .
pip install -r requirements.txt
streamlit run app.py
```

### Google Colab
```python
!git clone https://github.com/BaqarW-tech/saudi-workforce-predictive
%cd saudi-workforce-predictive
!cp ../saudi-workforce-descriptive/saudi_workforce_data.csv .
!pip install -r requirements.txt -q

from pyngrok import ngrok
import subprocess, time
proc = subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"])
time.sleep(3)
print(ngrok.connect(8501).public_url)
```

---

## 📁 File Structure

```
saudi-workforce-predictive/
├── app.py                      # ML + Streamlit dashboard
├── saudi_workforce_data.csv    # Shared dataset (from App 1)
├── requirements.txt            # 5 dependencies
└── README.md
```

---

## 🔮 The Analytics Series

| App | Type | Question |
|---|---|---|
| App 1 ✅ | Descriptive  | *What* does the workforce look like? |
| App 2 ✅ | Diagnostic   | *Why* are companies failing? |
| **App 3 ✅** | **Predictive** | ***Will* a company breach next cycle? (ML)** |
| App 4     | Prescriptive | *What* hiring plan optimises compliance + cost? |

---

## 💡 Why This Stands Out on a CV

- **Three-model comparison** with proper cross-validation — not just "I ran a random forest"
- **Adjustable decision threshold** — shows understanding of precision/recall trade-offs
- **Policy framing**: positions the model as an HRSD early-warning tool, not an academic exercise
- **14 engineered features** — demonstrates feature engineering depth beyond raw columns
- **Interactive company-level predictions** — every company gets a score with progress bar

---

*Synthetic data calibrated to GASTAT/HRSD statistics. No real company or individual data used.*
