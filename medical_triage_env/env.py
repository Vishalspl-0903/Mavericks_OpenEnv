from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException

from .graders import grade
from .models import PatientPresentation, TriageAction, TriageObservation
from .tasks import TASK_LIST, TASKS, get_next_task, get_task

app = FastAPI(title="medical-triage-env", version="0.1.0")


class MedicalTriageEnv:
    current_task: Optional[dict] = None
    current_step: int = 0
    episode_rewards: List[float] = []
    clarification_history: List[str] = []
    additional_info_revealed: bool = False
    done: bool = False

    def __init__(self) -> None:
        self.current_task = None
        self.current_step = 0
        self.episode_rewards = []
        self.clarification_history = []
        self.additional_info_revealed = False
        self.done = False
        self._task_order_index = -1

    def _select_task(self, task_id: Optional[str]) -> dict:
        if task_id:
            task = get_task(task_id)
            self._task_order_index = TASK_LIST.index(task_id)
            return task
        next_task = get_next_task(self.current_task["task_id"] if self.current_task else None)
        self._task_order_index = TASK_LIST.index(next_task["task_id"])
        return next_task

    def build_observation(self) -> TriageObservation:
        if self.current_task is None:
            raise RuntimeError("No active task. Call reset() first.")

        patient_payload = deepcopy(self.current_task["patient"])
        if self.current_task["task_id"] == "masked-sepsis" and not self.additional_info_revealed:
            patient_payload["additional_info"] = None
        elif not self.additional_info_revealed:
            patient_payload["additional_info"] = patient_payload.get("additional_info")
        patient = PatientPresentation.model_validate(patient_payload)

        return TriageObservation(
            task_id=self.current_task["task_id"],
            step_number=min(self.current_step + 1, int(self.current_task["max_steps"])),
            max_steps=int(self.current_task["max_steps"]),
            patient=patient,
            additional_info_revealed=self.additional_info_revealed,
            clarification_history=list(self.clarification_history),
        )

    def reset(self, task_id: Optional[str] = None) -> TriageObservation:
        self.current_task = self._select_task(task_id)
        self.current_step = 0
        self.episode_rewards = []
        self.clarification_history = []
        self.additional_info_revealed = False
        self.done = False
        return self.build_observation()

    def step(self, action: TriageAction) -> Tuple[TriageObservation, float, bool, dict]:
        if self.current_task is None:
            raise RuntimeError("No active task. Call reset() first.")
        if self.done:
            raise RuntimeError("Episode already finished. Call reset() for a new task.")

        self.current_step += 1
        task = self.current_task
        reward = 0.0
        raw_reward = 0.0
        grader_result = None
        final_reward = 0.0

        if action.action_type == "clarify":
            if task.get("patient", {}).get("additional_info") and not self.additional_info_revealed:
                self.additional_info_revealed = True
                question_text = action.clarifying_question or "Clarification requested"
                additional_info = task["patient"]["additional_info"]
                self.clarification_history.append(f"Q: {question_text} | A: {additional_info}")
                reward = 0.15
            else:
                reward = 0.05
            final_reward = reward
        elif action.action_type == "classify":
            grader_result = grade(action, task)
            raw_reward = grader_result.value
            urgency_bonus = 0.10 if (action.esi_level == task["correct_esi"] and self.current_step <= 2) else 0.0
            step_penalty = 0.03 * (self.current_step - 1)
            final_reward = round(max(0.0, raw_reward + urgency_bonus - step_penalty), 2)
            self.done = True
        else:
            raise HTTPException(status_code=400, detail="action_type must be 'classify' or 'clarify'")

        if self.current_step >= int(task["max_steps"]):
            if not self.done:
                grader_result = grade(action, task)
                raw_reward = grader_result.value
                final_reward = round(max(0.0, raw_reward - 0.10), 2)
            self.done = True

        current_total = round(sum(self.episode_rewards), 2)
        reward_headroom = max(0.0, 1.0 - current_total)
        final_reward = round(min(final_reward, reward_headroom), 2)

        self.episode_rewards.append(final_reward)
        next_obs = self.build_observation()
        info = {
            "raw_score": raw_reward if action.action_type == "classify" else reward,
            "step": self.current_step,
            "grader_feedback": grader_result.feedback if grader_result is not None and action.action_type == "classify" else "",
        }
        return next_obs, final_reward, self.done, info

    def state(self) -> dict:
        return {
            "task_id": self.current_task["task_id"] if self.current_task else None,
            "step": self.current_step,
            "done": self.done,
            "episode_rewards": list(self.episode_rewards),
            "cumulative_score": round(sum(self.episode_rewards), 2),
            "additional_info_revealed": self.additional_info_revealed,
            "clarifications_made": len(self.clarification_history),
        }


env = MedicalTriageEnv()


@app.post("/reset")
def reset_endpoint(payload: Optional[dict] = Body(default=None)) -> TriageObservation:
    task_id = payload.get("task_id") if payload else None
    try:
        return env.reset(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/step")
def step_endpoint(action: TriageAction):
    observation, reward, done, info = env.step(action)
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state")
def state_endpoint() -> dict:
    return env.state()


@app.get("/health")
def health_endpoint() -> dict:
    return {"status": "ok"}


@app.get("/")
def root_endpoint() -> dict:
    return {"name": "medical-triage-env", "version": "0.1.0"}
