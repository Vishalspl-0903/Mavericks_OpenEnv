from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class VitalSigns(BaseModel):
    heart_rate: Optional[int] = None
    blood_pressure_systolic: Optional[int] = None
    blood_pressure_diastolic: Optional[int] = None
    respiratory_rate: Optional[int] = None
    oxygen_saturation: Optional[float] = None
    temperature: Optional[float] = None
    gcs: Optional[int] = None


class PatientPresentation(BaseModel):
    patient_id: str
    age: int
    sex: str
    chief_complaint: str
    symptoms: List[str] = Field(default_factory=list)
    vitals: VitalSigns
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    time_of_onset: str
    additional_info: Optional[str] = None


class TriageObservation(BaseModel):
    task_id: str
    step_number: int
    max_steps: int
    patient: PatientPresentation
    additional_info_revealed: bool
    clarification_history: List[str] = Field(default_factory=list)


class TriageAction(BaseModel):
    action_type: str
    esi_level: Optional[int] = None
    clarifying_question: Optional[str] = None
    reasoning: str
    recommended_actions: List[str] = Field(default_factory=list)
    confidence: float


class TriageReward(BaseModel):
    value: float
    esi_accuracy: float
    reasoning_quality: float
    action_appropriateness: float
    feedback: str
