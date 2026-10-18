"""Pydantic request/response models with clinical range validation."""
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PatientFeatures(BaseModel):
    """The 13 Cleveland features with plausible clinical ranges."""

    model_config = ConfigDict(json_schema_extra={"example": {
        "age": 63, "sex": 1, "cp": 4, "trestbps": 145, "chol": 233, "fbs": 1,
        "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3,
        "slope": 3, "ca": 0, "thal": 6}})

    age: float = Field(..., ge=18, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=60, le=250, description="Resting BP, mm Hg")
    chol: float = Field(..., ge=80, le=700, description="Serum cholesterol, mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result")
    thalach: float = Field(..., ge=50, le=250, description="Max heart rate")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak ST segment")
    ca: int = Field(..., ge=0, le=3, description="Major vessels coloured (0-3)")
    thal: int = Field(..., description="3 = normal, 6 = fixed, 7 = reversible")

    @field_validator("thal")
    @classmethod
    def check_thal(cls, v: int) -> int:
        """thal only takes the values 3, 6 or 7 in the dataset."""
        if v not in (3, 6, 7):
            raise ValueError("thal must be one of 3, 6, 7")
        return v


class PredictionResponse(BaseModel):
    """Single prediction result."""

    prediction: int
    label: str
    confidence: float
    probability_disease: float
    model_version: str


class BatchRequest(BaseModel):
    """Batch of patients (max 500)."""

    records: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    """Batch prediction result."""

    count: int
    predictions: list[PredictionResponse]
