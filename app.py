"""
Customer Churn Prediction API
POST /predict  -> churn prediction + probability for a single customer
"""

from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ConfigDict

MODEL_PATH = Path(__file__).parent / "model" / "churn_model.pkl"

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts whether a telecom customer is likely to churn.",
    version="1.0.0",
)

# ---------------------------------------------------------------- model load

model = None
load_error = None

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")
except Exception as exc:                      # noqa: BLE001
    load_error = str(exc)
    print(f"WARNING: could not load model -> {load_error}")


# ------------------------------------------------------------- request model

class Customer(BaseModel):
    """
    The 19 raw fields a real customer record contains.
    Engineered features are derived server-side so the caller never
    has to know about them.
    """

    model_config = ConfigDict(extra="forbid")

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0, le=500)
    TotalCharges: float = Field(ge=0, le=100000)


class PredictionResponse(BaseModel):
    prediction: str
    churn_probability: float
    risk_level: str


# --------------------------------------------------------- feature building

ADDON_COLUMNS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]


def build_features(payload: dict) -> pd.DataFrame:
    """
    Recreate the four engineered features exactly as the notebook does.
    Must stay in sync with the notebook's feature engineering section.
    """
    df = pd.DataFrame([payload])

    df["TenureBucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)

    # tenure above 72 falls outside the training range and becomes NaN
    df["TenureBucket"] = df["TenureBucket"].replace("nan", "49-72")

    df["NumAddOnServices"] = (df[ADDON_COLUMNS] == "Yes").sum(axis=1)
    df["HasAutoPayment"] = df["PaymentMethod"].str.contains("automatic").astype(int)
    df["NewHighValue"] = (
        (df["tenure"] <= 12) & (df["MonthlyCharges"] > 70)
    ).astype(int)

    return df


def risk_band(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


# -------------------------------------------------------------- error shape

@app.exception_handler(RequestValidationError)
async def validation_handler(request, exc: RequestValidationError):
    """Return a readable message instead of FastAPI's raw error list."""
    problems = [
        {"field": " -> ".join(str(p) for p in err["loc"][1:]), "problem": err["msg"]}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid input", "details": problems},
    )


# ----------------------------------------------------------------- endpoints

@app.get("/")
def root():
    return {
        "service": "Customer Churn Prediction API",
        "model_loaded": model is not None,
        "endpoints": {"predict": "POST /predict", "health": "GET /health", "docs": "GET /docs"},
    }


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model unavailable: {load_error}")
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: Customer):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded. Run the notebook to generate model/churn_model.pkl. ({load_error})",
        )

    try:
        features = build_features(customer.model_dump())
        probability = float(model.predict_proba(features)[0][1])
        label = "Yes" if probability >= 0.5 else "No"
    except Exception as exc:                  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    return PredictionResponse(
        prediction=label,
        churn_probability=round(probability, 4),
        risk_level=risk_band(probability),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
