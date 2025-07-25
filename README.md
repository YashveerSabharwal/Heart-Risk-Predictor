﻿# Heart-Risk-Predictor
# ❤️ Heart Risk Predictor

> **Disclaimer 🩺** This tool is for educational purposes only and **must not** be used as a substitute for professional medical diagnosis.

A lightweight Streamlit web‑app that estimates the probability of heart disease from **11 standard clinical parameters**, supports batch CSV predictions, and benchmarks four classic ML models.

---

## Table of Contents

1. [Features](#features)
2. [Quick Demo](#quick-demo)
3. [Getting Started](#getting-started)
4. [Input Schema](#input-schema)
5. [Model Overview](#model-overview)
6. [Results & Evaluation](#results--evaluation)
7. [Project Structure](#project-structure)
8. [Roadmap](#roadmap)
9. [Contributing](#contributing)
10. [License](#license)

> ⭐ **Star** this repo if it helps you; it motivates continual improvement.

---

## Features<a name="features"></a>

| UI Tab                        | Purpose                                                                                                                                               |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tab 1 – Single Prediction** | Fill the 11‑field form → receive an immediate heart‑risk probability along with a risk category (Low / Moderate / High).                              |
| **Tab 2 – Bulk Prediction**   | Upload a CSV containing any number of patient records with the same 11 columns → download a copy augmented with predicted probabilities & categories. |
| **Tab 3 – Model Insights**    | Interactive dashboards: accuracy, precision‑recall, ROC curves, and feature importance for each model.                                                |

*Built with **Streamlit** for the UI and **scikit‑learn** for modeling.*

---

## Getting Started<a name="getting-started"></a>

### Prerequisites

```bash
# clone the repo
git clone https://github.com/<your‑username>/heart-risk-predictor.git
cd heart-risk-predictor

# optional: create a virtual environment
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# install python dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
streamlit run app.py
```

Then navigate to **[http://localhost:8501](http://localhost:8501)**.

---

## Input Schema<a name="input-schema"></a>

| Column            | Datatype / Range                                                  | UI Widget         | Notes                                   |
| ----------------- | ----------------------------------------------------------------- | ----------------- | --------------------------------------- |
| `age`             | int, 0 – 150 yrs                                                  | `st.number_input` | Patient age                             |
| `sex`             | {`male`, `female`}                                                | `st.selectbox`    | 1 = male, 0 = female encoded internally |
| `chest_pain_type` | {Typical Angina, Atypical Angina, Non‑Anginal Pain, Asymptomatic} | `st.selectbox`    | Categorical encoding                    |
| `resting_bp`      | int, 0 – 300 mm Hg                                                | `st.number_input` | Resting blood pressure                  |
| `cholesterol`     | int, 0 – ? mm/dl                                                  | `st.number_input` | Serum cholesterol                       |
| `fasting_bs`      | {`<=120 mg/dl`, `>120 mg/dl`}                                     | `st.selectbox`    | Encoded as 0/1                          |
| `resting_ecg`     | {Normal, ST‑T Abnormality, LV Hypertrophy}                        | `st.selectbox`    | ECG result                              |
| `max_hr`          | int, 60 – 202 bpm                                                 | `st.number_input` | Max heart rate achieved during test     |
| `exercise_angina` | {Yes, No}                                                         | `st.selectbox`    | Exercise‑induced angina                 |
| `oldpeak`         | float, 0.0 – 10.0                                                 | `st.number_input` | ST depression induced by exercise       |
| `st_slope`        | {Upsloping, Flat, Downsloping}                                    | `st.selectbox`    | Slope of peak exercise ST segment       |

> **Batch Mode:** CSV headers must exactly match the column names above (case‑insensitive).

---

## Model Overview<a name="model-overview"></a>

| Model                      | Library      | Key Hyper‑params  | Validation Accuracy |
| -------------------------- | ------------ | ----------------- | ------------------- |
| **Logistic Regression**    | scikit‑learn | C=1.0, penalty=l2 | 0.87                |
| **Decision Tree**          | scikit‑learn | max\_depth=8      | 0.84                |
| **Random Forest**          | scikit‑learn | n\_estimators=200 | **0.92**            |
| **Support Vector Machine** | scikit‑learn | kernel=rbf, C=10  | 0.89                |

Hyper‑parameters selected via 5‑fold cross‑validated Grid Search.

---

## Roadmap<a name="roadmap"></a>

* [ ] Integrate SHAP values for explainability
* [ ] Dockerfile & CI workflow
* [ ] Cloud deployment (Render / Fly.io)
* [ ] Add unit tests via `pytest`

---

## Contributing<a name="contributing"></a>

1. Fork this repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 🚀

---

