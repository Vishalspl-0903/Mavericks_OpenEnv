"""
Task configuration loader for medical triage environment.

Loads task scenarios from embedded YAML configuration, validates with Pydantic,
and provides cached access to task data.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .logs import get_logger
from .models import VitalSigns

logger = get_logger(__name__)

# YAML configuration embedded for portability
# In production, this would be loaded from config/tasks.yaml
TASKS_YAML_CONTENT = """
tasks:
  - id: classic-mi
    difficulty: easy
    esi_correct: 1
    chief_complaint: "crushing chest pain radiating to left arm"
    scenario: "Classic STEMI in a 58-year-old male with hypotension and diaphoresis."
    
    initial_vitals:
      heart_rate: 110
      blood_pressure_systolic: 90
      blood_pressure_diastolic: 60
      respiratory_rate: 22
      oxygen_saturation: 94.0
      temperature: 37.1
      gcs: 15
    
    patient_info:
      patient_id: "classic-mi-001"
      age: 58
      sex: "male"
      symptoms:
        - "severe chest pain"
        - "diaphoresis"
        - "nausea"
        - "shortness of breath"
        - "left arm pain"
      medical_history:
        - "hypertension"
        - "type 2 diabetes"
        - "smoker 20 pack-years"
      current_medications:
        - "metformin 500mg"
        - "amlodipine 5mg"
      time_of_onset: "30 minutes ago, sudden onset"
    
    hidden_info: []
    
    confounders:
      - "Patient mentions he had reflux last week (red herring)"
    
    vital_drift:
      per_step: {}
      starts_at_step: 10
    
    expected_clarify_steps: 0
    
    key_clarify_actions: []
    
    expected_actions:
      - "12-lead ECG immediately"
      - "IV access"
      - "oxygen therapy"
      - "aspirin 300mg"
      - "call cardiologist"
      - "prepare cath lab"
      - "troponin levels"
      - "continuous cardiac monitoring"
    
    key_reasoning_keywords:
      - "stemi"
      - "mi"
      - "myocardial"
      - "cardiac"
      - "ischemia"
      - "troponin"
      - "ecg"
      - "reperfusion"
      - "cardiogenic"
      - "shock"
    
    expected_severity: "critical"
    max_steps: 2
    why_difficulty: "Textbook STEMI with hypotension, tachycardia, chest pain, and diaphoresis; any clinically aware model should classify ESI 1 immediately."

  - id: meningitis-suspect
    difficulty: medium
    esi_correct: 1
    chief_complaint: "worst headache of my life, fever, stiff neck"
    scenario: "Suspected bacterial meningitis with non-blanching rash and fever."
    
    initial_vitals:
      heart_rate: 118
      blood_pressure_systolic: 105
      blood_pressure_diastolic: 70
      respiratory_rate: 20
      oxygen_saturation: 98.0
      temperature: 39.4
      gcs: 14
    
    patient_info:
      patient_id: "meningitis-suspect-001"
      age: 19
      sex: "female"
      symptoms:
        - "thunderclap headache"
        - "high fever"
        - "neck stiffness"
        - "photophobia"
        - "phonophobia"
        - "non-blanching petechial rash on legs"
        - "vomiting"
      medical_history:
        - "no significant past history"
      current_medications:
        - "combined oral contraceptive pill"
      time_of_onset: "8 hours ago, rapidly worsening"
    
    hidden_info: []
    
    confounders:
      - "On contraceptive pill (might suggest migraine with aura)"
      - "GCS 14 not fully alert but not critically low"
    
    vital_drift:
      per_step:
        heart_rate: 3
        gcs: -1
        temperature: 0.2
      starts_at_step: 2
    
    expected_clarify_steps: 1
    
    key_clarify_actions:
      - "check_rash"
      - "assess_meningismus"
    
    expected_actions:
      - "immediate isolation"
      - "blood cultures before antibiotics"
      - "IV ceftriaxone immediately"
      - "CT head"
      - "LP if CT clear"
      - "neurology consult"
      - "dexamethasone"
      - "meningococcal protocol"
    
    key_reasoning_keywords:
      - "meningitis"
      - "meningococcal"
      - "petechiae"
      - "non-blanching"
      - "kernig"
      - "brudzinski"
      - "septicemia"
      - "bacterial"
      - "antibiotics"
      - "lumbar puncture"
      - "lp"
    
    expected_severity: "critical"
    max_steps: 3
    why_difficulty: "The non-blanching rash is the critical cue, while the borderline GCS and oral contraceptive history can distract from a high-risk meningococcal presentation."

  - id: masked-sepsis
    difficulty: hard
    esi_correct: 2
    chief_complaint: "family says she has been confused and just not herself for 2 days"
    scenario: "Elderly patient with masked urosepsis, beta-blockade, CKD, and delirium."
    
    initial_vitals:
      heart_rate: 88
      blood_pressure_systolic: 118
      blood_pressure_diastolic: 72
      respiratory_rate: 18
      oxygen_saturation: 96.0
      temperature: 36.9
      gcs: 13
    
    patient_info:
      patient_id: "masked-sepsis-001"
      age: 82
      sex: "female"
      symptoms:
        - "acute confusion"
        - "general malaise"
        - "reduced oral intake"
        - "decreased mobility"
        - "mild abdominal discomfort"
      medical_history:
        - "atrial fibrillation"
        - "chronic kidney disease stage 3"
        - "type 2 diabetes"
        - "recurrent UTIs"
        - "osteoarthritis"
      current_medications:
        - "warfarin 3mg"
        - "metformin 500mg"
        - "bisoprolol 5mg"
        - "furosemide 40mg"
        - "paracetamol PRN"
      time_of_onset: "2 days gradual decline, acutely worse today"
    
    hidden_info:
      - trigger: "clarify"
        data:
          urinary_symptoms: "Urine is dark, cloudy and malodorous per family"
          fever_history: "Temperature was 38.9C at home last night per son"
          urine_output: "She has not passed urine since this morning"
          medication_adherence: "Last INR check was 3 weeks ago"
    
    confounders:
      - "Normal temperature and heart rate mask sepsis (beta-blocker effect)"
      - "Confusion could be simple delirium or dementia progression"
      - "Mild symptoms don't immediately suggest emergency"
      - "CKD and age are chronic conditions"
    
    vital_drift:
      per_step:
        heart_rate: 5
        respiratory_rate: 2
        oxygen_saturation: -1.0
        temperature: 0.3
        gcs: -1
      starts_at_step: 2
    
    expected_clarify_steps: 2
    
    key_clarify_actions:
      - "clarify"
      - "ask_history"
      - "detailed_assessment"
    
    expected_actions:
      - "sepsis 6 bundle"
      - "blood cultures x2"
      - "urine culture"
      - "serum lactate"
      - "cautious IV fluid challenge"
      - "hold metformin immediately"
      - "check INR urgently"
      - "renal function panel"
      - "urine dipstick"
      - "bladder scan"
      - "geriatrics consult"
    
    key_reasoning_keywords:
      - "sepsis"
      - "urosepsis"
      - "uti"
      - "beta blocker"
      - "bisoprolol"
      - "masked tachycardia"
      - "warfarin"
      - "inr"
      - "lactate"
      - "metformin"
      - "contraindicated"
      - "ckd"
      - "renal"
      - "elderly"
      - "atypical presentation"
      - "oliguria"
      - "confusion"
    
    expected_severity: "high"
    max_steps: 4
    why_difficulty: "Multiple masking factors act together: bisoprolol blunts tachycardia, CKD and age blunt fever, metformin may be unsafe in sepsis or AKI, warfarin requires urgent INR review, and the presentation can resemble simple delirium unless the agent asks for more detail."
"""


class HiddenInfoItem(BaseModel):
    """Hidden information revealed on clarification trigger."""
    trigger: str
    data: Dict[str, Any]
    
    model_config = ConfigDict(frozen=True)


class VitalDrift(BaseModel):
    """Configuration for how vitals change over time."""
    per_step: Dict[str, float] = Field(default_factory=dict)
    starts_at_step: int = 1
    
    model_config = ConfigDict(frozen=True)


class PatientInfo(BaseModel):
    """Patient demographic and clinical information."""
    patient_id: str
    age: int
    sex: str
    symptoms: List[str] = Field(default_factory=list)
    medical_history: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    time_of_onset: str
    
    model_config = ConfigDict(frozen=True)


class TaskConfig(BaseModel):
    """Complete task configuration validated with Pydantic."""
    id: str
    difficulty: str  # easy | medium | hard
    esi_correct: int  # 1-5
    chief_complaint: str
    scenario: str
    initial_vitals: Dict[str, Any]
    patient_info: PatientInfo
    hidden_info: List[HiddenInfoItem] = Field(default_factory=list)
    confounders: List[str] = Field(default_factory=list)
    vital_drift: VitalDrift
    expected_clarify_steps: int
    key_clarify_actions: List[str] = Field(default_factory=list)
    expected_actions: List[str] = Field(default_factory=list)
    key_reasoning_keywords: List[str] = Field(default_factory=list)
    expected_severity: str
    max_steps: int
    why_difficulty: str
    
    model_config = ConfigDict(frozen=True)


# Module-level cache: parsed once, reused
_TASKS_CACHE: Dict[str, TaskConfig] = {}
_TASK_LIST: List[str] = []


def _load_tasks_from_yaml() -> Dict[str, TaskConfig]:
    """Load and validate tasks from YAML content."""
    try:
        # Try loading from external file first
        config_path = Path(__file__).parent.parent / "config" / "tasks.yaml"
        if config_path.exists():
            logger.info("loading_tasks_from_file", path=str(config_path))
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_content = f.read()
        else:
            logger.info("loading_tasks_from_embedded_yaml")
            yaml_content = TASKS_YAML_CONTENT
        
        data = yaml.safe_load(yaml_content)
        
        if not isinstance(data, dict) or "tasks" not in data:
            raise ValueError("YAML must contain a 'tasks' key with a list of task configs")
        
        tasks: Dict[str, TaskConfig] = {}
        task_list: List[str] = []
        
        for task_data in data["tasks"]:
            try:
                task_config = TaskConfig.model_validate(task_data)
                tasks[task_config.id] = task_config
                task_list.append(task_config.id)
                logger.debug("task_loaded", task_id=task_config.id, difficulty=task_config.difficulty)
            except ValidationError as e:
                logger.error(
                    "task_validation_failed",
                    task_id=task_data.get("id", "unknown"),
                    error=str(e),
                )
                raise ValueError(f"Task validation failed for {task_data.get('id')}: {e}") from e
        
        logger.info("tasks_loaded_successfully", count=len(tasks), task_ids=task_list)
        
        return tasks, task_list
    
    except yaml.YAMLError as e:
        logger.error("yaml_parse_error", error=str(e))
        raise ValueError(f"Failed to parse YAML: {e}") from e
    except Exception as e:
        logger.error("task_loading_error", error=str(e), error_type=type(e).__name__)
        raise


# Load tasks at module import time (parse once, cache forever)
try:
    _TASKS_CACHE, _TASK_LIST = _load_tasks_from_yaml()
except Exception as e:
    logger.error("fatal_task_loading_error", error=str(e))
    raise


def get_task(task_id: str) -> TaskConfig:
    """
    Get a task configuration by ID.
    
    Args:
        task_id: Task identifier (e.g., "classic-mi")
        
    Returns:
        TaskConfig object with all task data
        
    Raises:
        KeyError: If task_id not found
    """
    if task_id not in _TASKS_CACHE:
        logger.warning("task_not_found", task_id=task_id, available_tasks=_TASK_LIST)
        raise KeyError(f"Unknown task_id: {task_id}. Available tasks: {', '.join(_TASK_LIST)}")
    
    return _TASKS_CACHE[task_id]


def get_task_list() -> List[str]:
    """Get list of all available task IDs."""
    return _TASK_LIST.copy()


def get_next_task(current_task_id: Optional[str]) -> TaskConfig:
    """
    Get the next task in sequence (cycles through task list).
    
    Args:
        current_task_id: Current task ID, or None to get first task
        
    Returns:
        TaskConfig for next task
    """
    if current_task_id is None:
        return _TASKS_CACHE[_TASK_LIST[0]]
    
    try:
        current_index = _TASK_LIST.index(current_task_id)
    except ValueError:
        logger.warning("current_task_not_found", task_id=current_task_id)
        return _TASKS_CACHE[_TASK_LIST[0]]
    
    next_index = (current_index + 1) % len(_TASK_LIST)
    return _TASKS_CACHE[_TASK_LIST[next_index]]


# Backward compatibility exports for existing code
TASK_LIST = _TASK_LIST
TASKS = {
    task_id: {
        "task_id": config.id,
        "difficulty": config.difficulty,
        "scenario": config.scenario,
        "patient": {
            "patient_id": config.patient_info.patient_id,
            "age": config.patient_info.age,
            "sex": config.patient_info.sex,
            "chief_complaint": config.chief_complaint,
            "symptoms": config.patient_info.symptoms,
            "vitals": config.initial_vitals,
            "medical_history": config.patient_info.medical_history,
            "current_medications": config.patient_info.current_medications,
            "time_of_onset": config.patient_info.time_of_onset,
            "additional_info": (
                "\n".join([
                    f"{k}: {v}" for item in config.hidden_info 
                    for k, v in item.data.items()
                ])
                if config.hidden_info else None
            ),
        },
        "correct_esi": config.esi_correct,
        "expected_actions": config.expected_actions,
        "key_reasoning_keywords": config.key_reasoning_keywords,
        "expected_severity": config.expected_severity,
        "max_steps": config.max_steps,
    }
    for task_id, config in _TASKS_CACHE.items()
}
