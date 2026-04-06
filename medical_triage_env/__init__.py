"""Medical triage OpenEnv environment package."""

from .models import (
    PatientPresentation,
    TriageAction,
    TriageObservation,
    TriageReward,
    VitalSigns,
)

# Optional imports that require FastAPI
try:
    from .env import MedicalTriageEnv, app
    _ENV_AVAILABLE = True
except ImportError:
    _ENV_AVAILABLE = False
    MedicalTriageEnv = None
    app = None

__all__ = [
    "PatientPresentation",
    "TriageAction", 
    "TriageObservation",
    "TriageReward",
    "VitalSigns",
]

if _ENV_AVAILABLE:
    __all__.extend(["MedicalTriageEnv", "app"])

__version__ = "0.1.0"
