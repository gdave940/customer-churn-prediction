# Customer Churn Prediction

End-to-end machine learning solution for predicting telecom customer churn,
covering data preparation, EDA, feature engineering, model development,
evaluation, interpretation and a REST API for serving predictions.

**Dataset:** IBM Telco Customer Churn — 7,043 customers, 21 columns
**Target:** `Churn` (Yes / No), 26.5% positive class
**Model:** Decision Tree Classifier (`max_depth=7`, `min_samples_leaf=30`, `class_weight='balanced'`)

## Notebook contents

`notebook/churn_analysis_nagp.ipynb` runs top to bottom on a fresh kernel and covers
the eight stages below. Section numbers follow the assignment brief.

| Section | Contents |
|---|---|
| 1 | Data understanding — dtypes, missing values, duplicates, numerical/categorical identification, target distribution, cleaning |
| 2 | Exploratory data analysis — 7 visualisations, each with a business insight |
| 3 | Feature engineering — 4 features plus one tested and rejected |
| 4 | Model development — 4 Decision Tree configurations compared |
| 5 | Evaluation — accuracy, precision, recall, F1, confusion matrix, precision-vs-recall discussion |
| 6 | Interpretation — feature importance, tree visualisation, decision rules |
| 7 | Model saving — pipeline pickled with metadata |
| 8 | Additional work — GridSearchCV tuning, cross-validation, algorithm comparison |

Categorical encoding is handled inside the pipeline (`OneHotEncoder`) rather than
applied to the dataframe up front, so the identical transformation is guaranteed
at both training and serving time.

---

## Quick start

For the impatient. Full detail in [Setup](#setup).

```bash
git clone <repository-url> && cd customer_churn_project
python -m venv venv && source venv/bin/activate    # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt

jupyter notebook                                   # run the notebook end to end
uvicorn app:app --reload                           # then start the API
```

Open http://127.0.0.1:8000/docs and try `POST /predict` with the contents of
`sample_request.json`.

---

## Project structure

```
customer_churn_project/
├── data/
│   ├── TelcoCustomerChurn.csv
│   └── TelcoCustomerChurn_-_Data_Dictionary.csv
├── notebook/
│   └── churn_analysis_nagp.ipynb
├── model/
│   ├── churn_model.pkl
│   ├── model_metadata.json
│   └── decision_tree.png
├── app.py
├── requirements.txt
├── sample_request.json
├── sample_request_low_risk.json
├── .gitignore
└── README.md
```

---

## Setup

**Requires Python 3.11 or later.** scikit-learn 1.9 requires 3.11+ and FastAPI
requires 3.10+; the project was developed on Python 3.13.

### 1. Clone and enter the project

```bash
git clone <repository-url>
cd customer_churn_project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

```bash
# macOS / Linux
source venv/bin/activate
```

```powershell
# Windows PowerShell
venv\Scripts\Activate.ps1
```

If PowerShell refuses with "running scripts is disabled on this system", allow
it once for your user and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

The prompt should now start with `(venv)`.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Generate the model

A pickled model is tied to the scikit-learn version that produced it, so
regenerate it locally rather than relying on the committed copy:

```bash
jupyter notebook
```

Open `notebook/churn_analysis_nagp.ipynb` and run
**Kernel → Restart Kernel and Run All Cells**. It runs top to bottom on a fresh
kernel in about a minute, and writes `model/churn_model.pkl`,
`model/model_metadata.json` and `model/decision_tree.png`.

The notebook uses paths relative to its own folder (`../data/`, `../model/`),
so it must be executed with `notebook/` as the working directory. Opening it
through the Jupyter file browser does this automatically.

The committed model was trained with scikit-learn 1.9.0 on Python 3.13; the
exact version is recorded in `model/model_metadata.json`.

Confirm the artefacts exist before continuing — `churn_model.pkl` should be
roughly 17 KB:

```bash
ls -l model/            # Windows: Get-ChildItem model
```

### 5. Run the API

From the project root, with the virtual environment active:

```bash
uvicorn app:app --reload
```

Leave this terminal running. The API is a server, not a script that exits.

```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Interactive documentation: http://127.0.0.1:8000/docs

### 6. Verify it works

In a **second** terminal:

```bash
curl http://127.0.0.1:8000/health
```

Expect `{"status":"healthy","model_loaded":true}`. A 503 means the model file
is missing — go back to step 4.

Then send the sample request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

```powershell
# Windows PowerShell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" -InFile sample_request.json
```

Expect `{"prediction":"Yes","churn_probability":0.8697,"risk_level":"High"}`.

`sample_request_low_risk.json` should return `"No"` at probability 0.0605.

Stop the server with **Ctrl+C**.

### Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `pip install` resolver errors | Python below 3.11 — check with `python --version` |
| `/health` returns 503 | `model/churn_model.pkl` missing; re-run the notebook |
| `Unable to connect to the remote server` | The uvicorn terminal isn't running, or you stopped it |
| `Address already in use` | A stale server is running: `uvicorn app:app --reload --port 8001` |
| `FileNotFoundError` on the CSV in the notebook | Notebook run from the wrong directory; open it via the Jupyter file browser |

---

## Pipeline overview

```
Raw CSV
  → clean TotalCharges (11 blanks, all tenure=0, imputed as 0)
  → drop customerID
  → engineer 4 features
  → 70/30 stratified split (random_state=42)
  → ColumnTransformer: numerical passthrough + one-hot categorical
  → DecisionTreeClassifier
  → joblib pickle (preprocessing + model in one object)
```

Preprocessing lives inside a scikit-learn `Pipeline`, so the encoder is fitted
only on training data and the identical transformation is applied at serving
time. This prevents data leakage and removes any chance of the API and the
notebook disagreeing about preprocessing.

### Engineered features

| Feature | Definition | Rationale |
|---|---|---|
| `TenureBucket` | tenure binned at 12 / 24 / 48 months | churn falls 47.4% → 28.7% → 20.4% → 9.5% across bands |
| `NumAddOnServices` | count of the 6 optional services subscribed | proxy for how embedded the customer is in the product |
| `HasAutoPayment` | 1 for bank transfer / credit card | automatic payers churn at 16.0% vs 34.7% manual |
| `NewHighValue` | tenure ≤ 12 AND MonthlyCharges > 70 | 67.9% churn inside this group vs 20.7% outside |

All four are computed row-wise from a single customer's own values, using no
statistics pooled across the dataset. They are therefore safe to compute
before the train/test split and reproducible for a single unseen record.

---

## Results

Four Decision Tree configurations were compared on the held-out test set:

| Configuration | Train Acc | Test Acc | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Default (unpruned) | 0.998 | 0.731 | 0.494 | 0.478 | 0.486 |
| max_depth=5 | 0.794 | 0.782 | 0.696 | 0.319 | 0.438 |
| max_depth=5, balanced | 0.738 | 0.716 | 0.479 | 0.781 | 0.593 |
| **max_depth=7, balanced (final)** | 0.755 | 0.740 | 0.507 | **0.791** | **0.618** |

Final model on the test set (2,113 customers): accuracy 0.740, precision 0.507,
recall 0.791, F1 0.618, ROC-AUC 0.826.

Confusion matrix:

|  | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 1120 | 432 |
| **Actual Yes** | 117 | 444 |

The model catches 444 of 561 actual churners while flagging 876 customers in
total, reducing the retention team's contact list by 59% versus contacting
everyone.

### Hyperparameter tuning

A `GridSearchCV` over 48 combinations (`max_depth` 4-10, `min_samples_leaf`
10-50, gini vs entropy) with 5-fold stratified cross-validation selected
`entropy, max_depth=7, min_samples_leaf=50` at CV F1 0.6186.

On the held-out test set the tuned model scored F1 0.603 and recall 0.733,
both below the manually selected configuration (0.618 / 0.791). The search is
retained in the notebook because it confirms the manual choice against a
systematic sweep rather than replacing it — and because the gap illustrates
that a cross-validation optimum does not automatically transfer to unseen data.

### Comparison with other algorithms

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree (final) | 0.740 | 0.507 | 0.791 | 0.618 | 0.826 |
| Logistic Regression | 0.741 | 0.508 | 0.795 | 0.620 | 0.844 |
| Random Forest | 0.763 | 0.537 | 0.788 | 0.639 | 0.843 |

Random Forest is the strongest of the three at F1 0.639. It gains on the Decision
Tree through precision (0.537 vs 0.507) while holding recall essentially level
(0.788 vs 0.791) — it flags a shorter list without missing more churners.
Logistic Regression is close to the tree on F1 but has the best probability
ranking at ROC-AUC 0.844.

The Decision Tree is retained as the deliverable. The brief specifies one, the
margin is roughly 2.1 F1 points, and the tree's rules can be read directly by a
retention manager — not possible with a 300-tree ensemble or a table of
coefficients. For a first deployment where the team must trust the output enough
to act on it, that trade holds. Random Forest is the recommended next step if
predictive performance later outweighs explainability; the pipeline structure
makes swapping the estimator a one-line change.

### Class imbalance

The target is imbalanced at 73.5% / 26.5%. `class_weight='balanced'` reweights
the minority class inversely to its frequency during training, which lifted
recall from 0.319 to 0.781 at equivalent depth. No resampling was applied —
reweighting achieves the same effect without synthesising rows or discarding
data, and keeps the pipeline simple enough to serve directly.

### Why recall over precision

A false negative is a customer lost silently, costing their full remaining
lifetime value. A false positive is a retention offer sent to someone who was
staying, costing only the offer. Telecom churn is largely irreversible once the
customer ports out, so the asymmetry favours recall. The unpruned and
depth-5 models were rejected on this basis despite the latter having the
highest test accuracy.

`predict_proba` is exposed via the `churn_probability` field, so the threshold
can be tuned after deployment without retraining.

### Top drivers

Contract type accounts for 59% of the model's splits (`Contract_Two year` 0.359,
`Contract_One year` 0.233), followed by `tenure` (0.100),
`InternetService_Fiber optic` (0.094) and `TotalCharges` (0.051).

The tree uses 19 of the 36 encoded columns; the remaining 17 receive zero
importance.

Fiber optic customers churning at 41.9% is the most actionable finding — it is
the premium product and should not be the highest-risk segment, pointing to a
price-versus-value or service-quality issue worth escalating to the product team.

---

## API

### `POST /predict`

Accepts the 19 raw customer fields. Engineered features are derived
server-side, so callers do not need to know about them.

**Request** (`sample_request.json`):

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 94.40,
  "TotalCharges": 188.80
}
```

**Response** (200):

```json
{
  "prediction": "Yes",
  "churn_probability": 0.8697,
  "risk_level": "High"
}
```

A low-risk example is provided in `sample_request_low_risk.json` and returns
`"prediction": "No"` with probability 0.0605.

### Other endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Service info and model load status |
| GET | `/health` | 200 if the model is loaded, 503 otherwise |
| GET | `/docs` | Interactive Swagger UI |

### Error handling

| Condition | Status | Behaviour |
|---|---|---|
| Missing required field | 422 | Field name and reason returned |
| Value outside allowed set | 422 | Permitted values listed |
| Out-of-range number | 422 | Bound stated |
| Unrecognised extra field | 422 | Rejected (`extra="forbid"`) |
| Model file missing | 503 | Service reports unavailable rather than guessing |

Unseen categories are handled by `handle_unknown='ignore'` in the encoder, so
a novel category encodes as all-zeros rather than raising.

### Client examples

Beyond the Swagger UI, the endpoint can be called from anything that speaks
HTTP:

```bash
# curl
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

```powershell
# PowerShell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post `
  -ContentType "application/json" `
  -InFile sample_request.json
```

```python
# Python
import requests, json
payload = json.load(open("sample_request.json"))
print(requests.post("http://127.0.0.1:8000/predict", json=payload).json())
```

---

## Known limitations

- `build_features()` in `app.py` duplicates the notebook's feature engineering
  logic. Any change to one must be mirrored in the other; extracting it into a
  shared module would be the correct fix for production.
- The dataset is a static snapshot with no timestamps, so the model cannot
  capture trends such as recent bill increases or support-ticket volume.
- The engineered features contributed little to the final tree:
  `NumAddOnServices` scored 0.0030 and `TenureBucket`, `HasAutoPayment` and
  `NewHighValue` were not used at all. Each restates information the tree can
  already reach through the raw columns, so it prefers the unbinned original.
  They were strong as standalone predictors (`NewHighValue` alone separates
  67.9% churn from 20.7%) and would matter more in a linear model, which cannot
  construct interactions by itself. Retained and documented rather than removed.
- No monitoring for data drift; a production deployment would need periodic
  retraining and performance tracking.
- Model selection used a single held-out test set. Cross-validated scores
  differed from test scores by roughly 1.5 F1 points, so reported figures carry
  variance of that order and small differences between configurations should not
  be over-read.


  ---

  - `Github Link` : https://github.com/gdave940/customer-churn-prediction
  - `Recording` : https://nagarro-my.sharepoint.com/:v:/p/gaurav_dave/IQAonb98y2oITqKEcYu5Y_HtAW3t4usB5bwrFjDoEQ5BPgE
