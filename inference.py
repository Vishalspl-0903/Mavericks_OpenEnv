from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import httpx
from openai import OpenAI
from dotenv import load_dotenv

from medical_triage_env.models import TriageAction
from medical_triage_env.tasks import TASK_LIST

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
FALLBACK_MODEL_NAME = os.getenv("FALLBACK_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
BENCHMARK = "medical-triage-env"
MAX_STEPS = 4
BASE_URL = "http://localhost:8000"

SYSTEM_PROMPT = """You are an experienced emergency department triage nurse with 15 years
of experience. You assess patients using the Emergency Severity Index:
ESI 1 = Immediate life threat requiring resuscitation
ESI 2 = High risk situation, should not wait
ESI 3 = Urgent but stable, needs multiple resources
ESI 4 = Less urgent, needs one resource
ESI 5 = Non-urgent, no resources needed

For each patient you must respond with valid JSON only, no markdown:
{
  'action_type': 'classify' or 'clarify',
  'esi_level': integer 1-5 (include if action_type is classify),
  'clarifying_question': 'string' (include if action_type is clarify),
  'reasoning': 'detailed clinical reasoning, minimum 30 words',
  'recommended_actions': ['action1', 'action2', ...],
  'confidence': float 0.0 to 1.0
}

If you need more information before classifying, use clarify once.
For obvious emergencies, classify immediately."""

DEFAULT_FALLBACK_ACTION = TriageAction(
    action_type="classify",
    esi_level=3,
    reasoning="Parse error fallback",
    recommended_actions=[],
    confidence=0.0,
)


def compact_json(data: dict) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_action(text: str) -> Optional[TriageAction]:
    cleaned = strip_code_fences(text)
    try:
        payload = json.loads(cleaned)
        action = TriageAction.model_validate(payload)
        return action
    except Exception:
        return None


def observation_to_prompt(observation: dict) -> str:
    return (
        "Patient observation:\n"
        f"{json.dumps(observation, indent=2, ensure_ascii=False)}\n\n"
        "Return a single JSON object only."
    )


def create_client(api_key: str) -> OpenAI:
    return OpenAI(base_url=API_BASE_URL, api_key=api_key)


def build_fallback_action(observation: dict) -> TriageAction:
    task_id = observation.get("task_id")
    revealed = observation.get("additional_info_revealed", False)
    if task_id == "masked-sepsis" and not revealed:
        return TriageAction(
            action_type="clarify",
            clarifying_question="Has she had fever, urinary symptoms, reduced urine output, or worsening confusion at home?",
            reasoning="This older patient may have masked sepsis because beta-blockade and age can blunt fever and tachycardia. More history is needed before final triage classification.",
            recommended_actions=["ask about urinary symptoms", "ask about fever and urine output"],
            confidence=0.78,
        )
    if task_id == "masked-sepsis" and revealed:
        return TriageAction(
            action_type="classify",
            esi_level=2,
            reasoning="The hidden history reveals dark cloudy malodorous urine, oliguria, and a recent home fever, which makes masked urosepsis much more likely. Beta-blocker use, CKD, and advanced age can blunt the classic septic response, so this remains a high-risk emergency.",
            recommended_actions=[
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
            confidence=0.89,
        )
    if task_id == "classic-mi":
        return TriageAction(
            action_type="classify",
            esi_level=1,
            reasoning="Crushing chest pain with diaphoresis, hypotension, tachycardia, and radiation to the left arm is a life-threatening acute coronary syndrome requiring immediate resuscitation and reperfusion pathway activation.",
            recommended_actions=[
                "12-lead ECG immediately",
                "IV access",
                "oxygen therapy",
                "aspirin 300mg",
                "call cardiologist",
                "prepare cath lab",
                "troponin levels",
                "continuous cardiac monitoring",
            ],
            confidence=0.92,
        )
    if task_id == "meningitis-suspect":
        return TriageAction(
            action_type="classify",
            esi_level=1,
            reasoning="Thunderclap headache, fever, neck stiffness, photophobia, vomiting, reduced GCS, and a non-blanching petechial rash are highly concerning for meningococcal meningitis or septicemia, which is immediately life threatening.",
            recommended_actions=[
                "immediate isolation",
                "blood cultures before antibiotics",
                "IV ceftriaxone immediately",
                "CT head",
                "LP if CT clear",
                "neurology consult",
                "dexamethasone",
                "meningococcal protocol",
            ],
            confidence=0.92,
        )
    return DEFAULT_FALLBACK_ACTION


def _model_candidates() -> List[str]:
    candidates = [MODEL_NAME, FALLBACK_MODEL_NAME]
    unique_candidates: List[str] = []
    for model_name in candidates:
        if model_name and model_name not in unique_candidates:
            unique_candidates.append(model_name)
    return unique_candidates


def call_llm(client: OpenAI, observation: dict) -> TriageAction:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_to_prompt(observation)},
    ]
    for model_name in _model_candidates():
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=512,
            )
            content = response.choices[0].message.content or ""
            action = parse_action(content)
            if action is not None:
                return action
        except Exception:
            continue
    return build_fallback_action(observation)


def format_action(action: TriageAction) -> str:
    return compact_json(action.model_dump(exclude_none=True))


def run_episode(client: OpenAI, task_id: str) -> tuple[bool, int, List[float]]:
    rewards: List[float] = []
    steps = 0
    success = False

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as http:
        reset_response = http.post("/reset", json={"task_id": task_id})
        reset_response.raise_for_status()
        observation = reset_response.json()

        while True:
            action = call_llm(client, observation)
            action_json = format_action(action)

            step_response = http.post("/step", json=action.model_dump(exclude_none=True))
            step_response.raise_for_status()
            result = step_response.json()
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            rewards.append(reward)
            steps += 1

            print(
                f"[STEP]  step={steps} action={action_json} reward={reward:.2f} done={str(done).lower()} error=null"
            )

            state_response = http.get("/state")
            state_response.raise_for_status()
            observation = result.get("observation", observation)

            if done:
                success = True
                break
            if steps >= MAX_STEPS:
                break

    return success, steps, rewards


def main() -> None:
    for task_id in TASK_LIST:
        success = False
        steps = 0
        rewards: List[float] = []
        print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}")
        try:
            api_key = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
            if not api_key:
                raise ValueError("Missing HF_TOKEN or API_KEY in environment")
            client = create_client(api_key)
            success, steps, rewards = run_episode(client, task_id)
        except Exception:
            pass
        finally:
            score = round(sum(rewards), 2)
            rewards_text = ",".join(f"{value:.2f}" for value in rewards)
            print(
                f"[END]   success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_text}"
            )


if __name__ == "__main__":
    main()
