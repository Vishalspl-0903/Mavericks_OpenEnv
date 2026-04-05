from __future__ import annotations

from typing import Any, Dict, List, Optional

from .models import PatientPresentation, VitalSigns

TASK_LIST: List[str] = ["classic-mi", "meningitis-suspect", "masked-sepsis"]


def _patient_payload(**kwargs: Any) -> Dict[str, Any]:
    patient = PatientPresentation(**kwargs)
    return patient.model_dump(mode="python")


TASKS: Dict[str, Dict[str, Any]] = {
    "classic-mi": {
        "task_id": "classic-mi",
        "difficulty": "easy",
        "scenario": "Classic STEMI in a 58-year-old male with hypotension and diaphoresis.",
        "patient": _patient_payload(
            patient_id="classic-mi-001",
            age=58,
            sex="male",
            chief_complaint="crushing chest pain radiating to left arm",
            symptoms=[
                "severe chest pain",
                "diaphoresis",
                "nausea",
                "shortness of breath",
                "left arm pain",
            ],
            vitals=VitalSigns(
                heart_rate=110,
                blood_pressure_systolic=90,
                blood_pressure_diastolic=60,
                respiratory_rate=22,
                oxygen_saturation=94,
                temperature=37.1,
                gcs=15,
            ),
            medical_history=["hypertension", "type 2 diabetes", "smoker 20 pack-years"],
            current_medications=["metformin 500mg", "amlodipine 5mg"],
            time_of_onset="30 minutes ago, sudden onset",
            additional_info=None,
        ),
        "correct_esi": 1,
        "expected_actions": [
            "12-lead ECG immediately",
            "IV access",
            "oxygen therapy",
            "aspirin 300mg",
            "call cardiologist",
            "prepare cath lab",
            "troponin levels",
            "continuous cardiac monitoring",
        ],
        "key_reasoning_keywords": [
            "stemi",
            "mi",
            "myocardial",
            "cardiac",
            "ischemia",
            "troponin",
            "ecg",
            "reperfusion",
            "cardiogenic",
            "shock",
        ],
        "expected_severity": "critical",
        "max_steps": 2,
        "why_easy": "Textbook STEMI with hypotension, tachycardia, chest pain, and diaphoresis; any clinically aware model should classify ESI 1 immediately.",
    },
    "meningitis-suspect": {
        "task_id": "meningitis-suspect",
        "difficulty": "medium",
        "scenario": "Suspected bacterial meningitis with non-blanching rash and fever.",
        "patient": _patient_payload(
            patient_id="meningitis-suspect-001",
            age=19,
            sex="female",
            chief_complaint="worst headache of my life, fever, stiff neck",
            symptoms=[
                "thunderclap headache",
                "high fever",
                "neck stiffness",
                "photophobia",
                "phonophobia",
                "non-blanching petechial rash on legs",
                "vomiting",
            ],
            vitals=VitalSigns(
                heart_rate=118,
                blood_pressure_systolic=105,
                blood_pressure_diastolic=70,
                respiratory_rate=20,
                oxygen_saturation=98,
                temperature=39.4,
                gcs=14,
            ),
            medical_history=["no significant past history"],
            current_medications=["combined oral contraceptive pill"],
            time_of_onset="8 hours ago, rapidly worsening",
            additional_info=None,
        ),
        "correct_esi": 1,
        "expected_actions": [
            "immediate isolation",
            "blood cultures before antibiotics",
            "IV ceftriaxone immediately",
            "CT head",
            "LP if CT clear",
            "neurology consult",
            "dexamethasone",
            "meningococcal protocol",
        ],
        "key_reasoning_keywords": [
            "meningitis",
            "meningococcal",
            "petechiae",
            "non-blanching",
            "kernig",
            "brudzinski",
            "septicemia",
            "bacterial",
            "antibiotics",
            "lumbar puncture",
            "lp",
        ],
        "expected_severity": "critical",
        "max_steps": 3,
        "why_medium": "The non-blanching rash is the critical cue, while the borderline GCS and oral contraceptive history can distract from a high-risk meningococcal presentation.",
    },
    "masked-sepsis": {
        "task_id": "masked-sepsis",
        "difficulty": "hard",
        "scenario": "Elderly patient with masked urosepsis, beta-blockade, CKD, and delirium.",
        "patient": _patient_payload(
            patient_id="masked-sepsis-001",
            age=82,
            sex="female",
            chief_complaint="family says she has been confused and just not herself for 2 days",
            symptoms=[
                "acute confusion",
                "general malaise",
                "reduced oral intake",
                "decreased mobility",
                "mild abdominal discomfort",
            ],
            vitals=VitalSigns(
                heart_rate=88,
                blood_pressure_systolic=118,
                blood_pressure_diastolic=72,
                respiratory_rate=18,
                oxygen_saturation=96,
                temperature=36.9,
                gcs=13,
            ),
            medical_history=[
                "atrial fibrillation",
                "chronic kidney disease stage 3",
                "type 2 diabetes",
                "recurrent UTIs",
                "osteoarthritis",
            ],
            current_medications=[
                "warfarin 3mg",
                "metformin 500mg",
                "bisoprolol 5mg",
                "furosemide 40mg",
                "paracetamol PRN",
            ],
            time_of_onset="2 days gradual decline, acutely worse today",
            additional_info=(
                "Urine is dark, cloudy and malodorous per family. Temperature was 38.9C at home last night per son. "
                "She has not passed urine since this morning. Last INR check was 3 weeks ago."
            ),
        ),
        "correct_esi": 2,
        "expected_actions": [
            "sepsis 6 bundle",
            "blood cultures x2",
            "urine culture",
            "serum lactate",
            "cautious IV fluid challenge",
            "hold metformin immediately",
            "check INR urgently",
            "renal function panel",
            "urine dipstick",
            "bladder scan",
            "geriatrics consult",
        ],
        "key_reasoning_keywords": [
            "sepsis",
            "urosepsis",
            "uti",
            "beta blocker",
            "bisoprolol",
            "masked tachycardia",
            "warfarin",
            "inr",
            "lactate",
            "metformin",
            "contraindicated",
            "ckd",
            "renal",
            "elderly",
            "atypical presentation",
            "oliguria",
            "confusion",
        ],
        "expected_severity": "high",
        "max_steps": 4,
        "why_hard": (
            "Multiple masking factors act together: bisoprolol blunts tachycardia, CKD and age blunt fever, metformin may be unsafe in sepsis or AKI, warfarin requires urgent INR review, and the presentation can resemble simple delirium unless the agent asks for more detail."
        ),
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]


def get_next_task(current_task_id: Optional[str]) -> Dict[str, Any]:
    if current_task_id is None:
        return TASKS[TASK_LIST[0]]
    try:
        current_index = TASK_LIST.index(current_task_id)
    except ValueError:
        return TASKS[TASK_LIST[0]]
    next_index = (current_index + 1) % len(TASK_LIST)
    return TASKS[TASK_LIST[next_index]]
