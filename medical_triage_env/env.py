from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .graders import grade
from .info_revealer import InfoRevealer
from .logs import get_logger
from .models import PatientPresentation, TriageAction, TriageObservation
from .tasks import load_all_tasks, TaskConfig

logger = get_logger(__name__)

app = FastAPI(title="medical-triage-env", version="0.1.0")


class MedicalTriageEnv:
    def __init__(self, task_id: str) -> None:
        self.all_tasks, self.task_ids = load_all_tasks()
        if task_id not in self.all_tasks:
            raise ValueError(
                f"Task {task_id} not found. "
                f"Available: {list(self.all_tasks.keys())}"
            )

        self.task_config: TaskConfig = self.all_tasks[task_id]
        self.task_id = task_id
        self.session_id = str(uuid.uuid4())

        self.info_revealer = InfoRevealer(self.task_config)

        self.current_step: int = 0
        self.episode_rewards: List[float] = []
        self.action_history: List[TriageAction] = []
        self.done: bool = False
        self.current_vitals: Dict = {}
        self._revealed_info: Dict[str, Any] = {}

        # Tracks whether hidden additional_info has been revealed to the agent.
        self._additional_info_revealed: bool = False

        logger.info("env_initialized", task_id=task_id, session_id=self.session_id)

    def build_observation(self) -> TriageObservation:
        drifted_vitals = self.info_revealer.apply_vital_drift(
            self.current_vitals, self.current_step
        )

        patient_payload = deepcopy(self.task_config.patient_info.model_dump())
        patient_payload["chief_complaint"] = self.task_config.chief_complaint
        patient_payload["vitals"] = drifted_vitals

        # Hide additional info until the first clarify reveal event.
        if not self._additional_info_revealed:
            patient_payload["additional_info"] = None
        else:
            if self._revealed_info:
                details = [f"{k.replace('_', ' ')}: {v}" for k, v in self._revealed_info.items()]
                patient_payload["additional_info"] = " | ".join(details)
            else:
                patient_payload["additional_info"] = "Additional history was elicited on clarification."

        patient = PatientPresentation.model_validate(patient_payload)

        return TriageObservation(
            task_id=self.task_id,
            step_number=min(self.current_step + 1, self.task_config.max_steps),
            max_steps=self.task_config.max_steps,
            patient=patient,
            additional_info_revealed=self._additional_info_revealed,
            clarification_history=[
                f"Step {i + 1}: {a.clarifying_question or 'clarify'}"
                for i, a in enumerate(self.action_history)
                if a.action_type == "clarify"
            ],
        )

    def reset(self) -> TriageObservation:
        self.current_step = 0
        self.episode_rewards = []
        self.action_history = []
        self.done = False
        self._additional_info_revealed = False
        self._revealed_info = {}

        self.current_vitals = self.info_revealer.get_initial_observation(self.current_step)

        logger.debug("env_reset", task_id=self.task_id, session_id=self.session_id)

        return self.build_observation()

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, dict]:
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() for a new task.")

        action = TriageAction(
            action_type=(action.action_type or "").strip().lower(),
            esi_level=action.esi_level,
            clarifying_question=action.clarifying_question,
            reasoning=action.reasoning,
            recommended_actions=action.recommended_actions or [],
            confidence=action.confidence,
        )

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
            # First clarify on tasks with hidden additional info yields reveal reward.
            if (
                not self._additional_info_revealed
                and bool(self.task_config.hidden_info)
            ):
                self._additional_info_revealed = True
                reward = 0.15
                logger.debug(
                    "additional_info_revealed",
                    task_id=self.task_id,
                    step=self.current_step,
                )
            else:
                reward = 0.05

            # Keep InfoRevealer behavior for trigger/vitals updates, but do not block rewarding.
            try:
                clarify_type = "clarify"
                revealed_extra = self.info_revealer.process_clarify(
                    clarify_type, self.current_step
                )
                if revealed_extra:
                    self._revealed_info.update(revealed_extra)
                if revealed_extra and "vitals" in revealed_extra:
                    self.current_vitals.update(revealed_extra["vitals"])
            except Exception as exc:
                logger.warning("info_revealer_error", error=str(exc))

        elif action.action_type == "classify":
            task_dict = self.task_config.model_dump()
            task_dict["correct_esi"] = self.task_config.esi_correct
            grader_result = grade(action, task_dict)
            raw_reward = grader_result.value

            correct_esi = self.task_config.esi_correct
            urgency_bonus = (
                0.10
                if action.esi_level == correct_esi and self.current_step <= 2
                else 0.0
            )
            step_penalty = 0.03 * (self.current_step - 1)

            reward = round(max(0.0, raw_reward + urgency_bonus - step_penalty), 2)
            self.done = True

        else:
            raise HTTPException(status_code=400, detail="action_type must be 'classify' or 'clarify'")

        if self.current_step >= self.task_config.max_steps:
            if not self.done:
                task_dict = self.task_config.model_dump()
                task_dict["correct_esi"] = self.task_config.esi_correct
                grader_result = grade(action, task_dict)
                raw_reward = grader_result.value
                reward = round(max(0.0, raw_reward - 0.10), 2)
            self.done = True

        reward = round(reward, 2)
        self.episode_rewards.append(reward)
        next_obs = self.build_observation()

        logger.debug(
            "env_step_end",
            task_id=self.task_id,
            step=self.current_step,
            reward=reward,
            done=self.done,
            cumulative=round(sum(self.episode_rewards), 2),
        )

        info = {
            "raw_score": raw_reward if action.action_type == "classify" else reward,
            "step": self.current_step,
            "grader_feedback": grader_result.feedback if grader_result else "",
            "additional_info_revealed": self._additional_info_revealed,
        }

        return next_obs, reward, self.done, info

    def state(self) -> dict:
        return {
            "session_id": self.session_id,
            "task_id": self.task_id,
            "step": self.current_step,
            "done": self.done,
            "episode_rewards": list(self.episode_rewards),
            "cumulative_score": round(sum(self.episode_rewards), 2),
            "additional_info_revealed": self._additional_info_revealed,
            "action_history_summary": [
                {
                    "step": i + 1,
                    "action_type": a.action_type,
                    "esi_level": a.esi_level,
                    "confidence": a.confidence,
                }
                for i, a in enumerate(self.action_history)
            ],
            "max_steps": self.task_config.max_steps,
        }



_active_environments: Dict[str, MedicalTriageEnv] = {}


@app.post("/reset")
def reset_endpoint(payload: Optional[dict] = Body(default=None)) -> Dict[str, Any]:
    if payload and "task_id" in payload:
        task_id = payload["task_id"]
    else:
        _, task_ids = load_all_tasks()
        if not task_ids:
            raise HTTPException(status_code=500, detail="No tasks available")
        task_id = task_ids[0]
    
    try:
        env = MedicalTriageEnv(task_id)
        observation = env.reset()

        _active_environments[env.session_id] = env

        # Keep session store bounded.
        if len(_active_environments) > 20:
            oldest_sessions = list(_active_environments.keys())[:-20]
            for sid in oldest_sessions:
                del _active_environments[sid]

        logger.info("environment_reset_success", task_id=task_id, session_id=env.session_id)

        return {
            "observation": observation.model_dump(),
            "info": {
                "session_id": env.session_id,
                "task_id": task_id,
            },
        }
        
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("environment_reset_error", error=str(exc), task_id=task_id)
        raise HTTPException(status_code=500, detail=f"Failed to reset environment: {exc}") from exc


@app.post("/step")
def step_endpoint(payload: dict = Body(...)) -> Dict[str, Any]:
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
            "observation": observation.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("step_execution_error", session_id=session_id, error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/state")
def state_endpoint(session_id: str):
    if session_id not in _active_environments:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return _active_environments[session_id].state()


@app.get("/tasks")
def tasks_endpoint():
    try:
        _, task_ids = load_all_tasks()
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
        return HTMLResponse(content="""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Medical Triage Environment</title>
</head>
<body>
    <h1>Medical Triage Environment</h1>
    <ul>
        <li><code>GET  /health</code></li>
        <li><code>POST /reset</code></li>
        <li><code>POST /step</code></li>
        <li><code>GET  /state?session_id=...</code></li>
        <li><code>GET  /tasks</code></li>
    </ul>
</body>
</html>""")


@app.get("/web")
def web_endpoint() -> HTMLResponse:
    return root_endpoint()
