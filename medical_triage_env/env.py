from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .graders import grade, compute_final_score
from .info_revealer import InfoRevealer
from .logs import get_logger
from .models import PatientPresentation, TriageAction, TriageObservation
from .tasks import load_all_tasks, TaskConfig

logger = get_logger(__name__)

app = FastAPI(title="medical-triage-env", version="0.1.0")


class MedicalTriageEnv:
    """Medical triage environment with progressive information disclosure."""
    
    def __init__(self, task_id: str) -> None:
        """
        Initialize environment for a specific task.
        
        Args:
            task_id: ID of the task to load (e.g., "classic-mi")
        """
        self.all_tasks, self.task_ids = load_all_tasks()
        if task_id not in self.all_tasks:
            raise ValueError(f"Task {task_id} not found. Available tasks: {list(self.all_tasks.keys())}")
        
        self.task_config = self.all_tasks[task_id]
        self.task_id = task_id
        self.session_id = str(uuid.uuid4())
        
        self.info_revealer = InfoRevealer(self.task_config)
        
        self.current_step = 0
        self.episode_rewards: List[float] = []
        self.action_history: List[TriageAction] = []
        self.done = False
        self.current_vitals: Dict = {}
        
        logger.info(
            "env_initialized",
            task_id=task_id,
            session_id=self.session_id
        )

    def build_observation(self) -> TriageObservation:
        """Build current observation with revealed information and drifted vitals."""
        drifted_vitals = self.info_revealer.apply_vital_drift(self.current_vitals, self.current_step)
        
        patient_payload = deepcopy(self.task_config.patient_info.model_dump())
        patient_payload["chief_complaint"] = self.task_config.chief_complaint
        patient_payload["vitals"] = drifted_vitals
        
        if self.task_id == "masked-sepsis" and not self.info_revealer.revealed_triggers:
            patient_payload["additional_info"] = None
        else:
            patient_payload["additional_info"] = patient_payload.get("additional_info")
        
        patient = PatientPresentation.model_validate(patient_payload)
        
        confounders = self.info_revealer.get_confounders()
        
        return TriageObservation(
            task_id=self.task_id,
            step_number=min(self.current_step + 1, self.task_config.max_steps),
            max_steps=self.task_config.max_steps,
            patient=patient,
            additional_info_revealed=bool(self.info_revealer.revealed_triggers),
            clarification_history=[f"Step {i+1}: {action.action_type}" for i, action in enumerate(self.action_history) if action.action_type == "clarify"],
        )

    def reset(self) -> TriageObservation:
        """Reset environment to initial state for the configured task."""
        self.current_step = 0
        self.episode_rewards = []
        self.action_history = []
        self.done = False
        
        self.current_vitals = self.info_revealer.get_initial_observation(self.current_step)
        
        logger.debug(
            "env_reset",
            task_id=self.task_id,
            session_id=self.session_id,
            max_steps=self.task_config.max_steps,
        )
        
        return self.build_observation()

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, dict]:
        """Execute one step in the environment."""
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() for a new task.")

        self.current_step += 1
        self.action_history.append(action)
        
        reward = 0.0
        raw_reward = 0.0
        grader_result = None
        
        logger.debug(
            "env_step_start",
            task_id=self.task_id,
            step=self.current_step,
            action_type=action.action_type,
        )

        if action.action_type == "clarify":
            clarify_type = action.clarifying_question or "general_clarify"
            revealed_info = self.info_revealer.process_clarify(clarify_type, self.current_step)
            
            self.current_vitals = self.info_revealer.apply_vital_drift(self.current_vitals, self.current_step)
            
            if "vitals" in revealed_info:
                self.current_vitals.update(revealed_info["vitals"])
            
            reward = 0.15 if revealed_info else 0.05
            
        elif action.action_type == "classify":
            if self.task_id == "masked-sepsis" and not self.info_revealer.revealed_triggers:
                logger.warning(
                    "early_classification_penalty",
                    task_id=self.task_id,
                    step=self.current_step
                )
                self.done = True
                final_reward = 0.15  # Max penalty score
                info = {
                    "raw_score": 0.15,
                    "step": self.current_step,
                    "grader_feedback": "Classified too early without gathering critical information",
                    "penalty_reason": "hard_case_early_classification"
                }
                next_obs = self.build_observation()
                self.episode_rewards.append(final_reward)
                return next_obs, final_reward, self.done, info
            
            task_dict = self.task_config.model_dump()
            task_dict["correct_esi"] = self.task_config.esi_correct
            grader_result = grade(action, task_dict)
            raw_reward = grader_result.value
            
            final_score, component_scores = compute_final_score(
                action=action,
                task=task_dict,
                action_history=self.action_history,
                esi_score=grader_result.esi_accuracy,
                steps_taken=self.current_step
            )
            
            reward = final_score
            self.done = True
            
        else:
            raise HTTPException(status_code=400, detail="action_type must be 'classify' or 'clarify'")

        if self.current_step >= self.task_config.max_steps:
            if not self.done:
                task_dict = self.task_config.model_dump()
                task_dict["correct_esi"] = self.task_config.esi_correct
                grader_result = grade(action, task_dict)
                raw_reward = grader_result.value
                reward = max(0.0, raw_reward - 0.10)
            self.done = True

        current_total = sum(self.episode_rewards)
        reward_headroom = max(0.0, 1.0 - current_total)
        final_reward = min(reward, reward_headroom)

        self.episode_rewards.append(final_reward)
        next_obs = self.build_observation()
        
        logger.debug(
            "env_step_end",
            task_id=self.task_id,
            step=self.current_step,
            reward=final_reward,
            done=self.done,
            cumulative_score=round(sum(self.episode_rewards), 2),
        )
        
        info = {
            "raw_score": raw_reward if action.action_type == "classify" else reward,
            "step": self.current_step,
            "grader_feedback": grader_result.feedback if grader_result else "",
        }
        
        if action.action_type == "classify" and 'component_scores' in locals():
            info.update(component_scores)
            
        return next_obs, final_reward, self.done, info

    def state(self) -> dict:
        """Get comprehensive environment state information."""
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "step": self.current_step,
            "done": self.done,
            "episode_rewards": list(self.episode_rewards),
            "cumulative_score": round(sum(self.episode_rewards), 2),
            "action_history_summary": [
                {
                    "step": i + 1,
                    "action_type": action.action_type,
                    "esi_level": action.esi_level,
                    "confidence": action.confidence
                } 
                for i, action in enumerate(self.action_history)
            ],
            "current_vitals": dict(self.current_vitals),
            "revealed_info_keys": self.info_revealer.get_revealed_info_keys(),
            "confounders": self.info_revealer.get_confounders(),
            "max_steps": self.task_config.max_steps,
        }



_active_environments: Dict[str, MedicalTriageEnv] = {}


@app.post("/reset")
def reset_endpoint(payload: Optional[dict] = Body(default=None)) -> Dict[str, Any]:
    """Reset environment for a specific task."""
    if not payload or "task_id" not in payload:
        raise HTTPException(status_code=400, detail="task_id is required in request body")
    
    task_id = payload["task_id"]
    
    try:
        env = MedicalTriageEnv(task_id)
        observation = env.reset()
        
        _active_environments[env.session_id] = env
        
        if len(_active_environments) > 10:
            oldest_sessions = list(_active_environments.keys())[:-10]
            for session_id in oldest_sessions:
                del _active_environments[session_id]
        
        logger.info(
            "environment_reset_success",
            task_id=task_id,
            session_id=env.session_id
        )
        
        return {
            "session_id": env.session_id,
            **observation.model_dump(),
        }
        
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("environment_reset_error", error=str(exc), task_id=task_id)
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {exc}") from exc


@app.post("/step")
def step_endpoint(payload: dict = Body(...)):
    """Execute a step in the environment."""
    if "session_id" not in payload:
        raise HTTPException(status_code=400, detail="session_id is required")
    if "action" not in payload:
        raise HTTPException(status_code=400, detail="action is required")
    
    session_id = payload["session_id"]
    
    if session_id not in _active_environments:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found. Call /reset first.")
    
    env = _active_environments[session_id]
    
    try:
        action = TriageAction.model_validate(payload["action"])
        observation, reward, done, info = env.step(action)
        
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as exc:
        logger.error("step_execution_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state_endpoint(session_id: str):
    """Get current environment state."""
    if session_id not in _active_environments:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    return _active_environments[session_id].state()


@app.get("/tasks")
def tasks_endpoint():
    """Get list of available tasks."""
    try:
        tasks, task_ids = load_all_tasks()
        return {
            "tasks": task_ids,
            "count": len(task_ids)
        }
    except Exception as exc:
        logger.error("tasks_list_error", error=str(exc))
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/health")
def health_endpoint() -> dict:
    return {"status": "ok"}


@app.get("/")
def root_endpoint() -> HTMLResponse:
        return HTMLResponse(
                content="""
<!doctype html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Medical Triage Environment</title>
</head>
<body>
    <h1>Medical Triage Environment is running</h1>
    <p>Available endpoints:</p>
    <ul>
        <li><code>/health</code></li>
        <li><code>/reset</code></li>
        <li><code>/step</code></li>
        <li><code>/state</code></li>
    </ul>
</body>
</html>
""".strip()
        )
