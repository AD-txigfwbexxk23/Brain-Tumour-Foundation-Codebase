from pydantic import BaseModel
from functools import lru_cache
import os

from inference import BrainTumorPredictor


class PredictorInput(BaseModel):
    Age: float
    Tumor_Size: float
    Tumor_Growth_Rate: str
    Symptom_Severity: str
    Tumor_Location: str
    MRI_Findings: str
    Radiation_Exposure: str


@lru_cache(maxsize=1)
def get_predictor_singleton() -> BrainTumorPredictor:
    model_path = os.path.join(os.getcwd(), "multi_output_brain_tumor_model.pth")
    return BrainTumorPredictor(model_path=model_path)


