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

API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"
BASE_URL = os.getenv("BASE_URL") or "http://127.0.0.1:8000"
API_KEY = os.getenv("API_KEY") or "not-needed"
BENCHMARK = "medical-triage-env"
MAX_STEPS = 4

SYSTEM_PROMPT = """You are an experienced emergency department triage nurse with 15 years
of experience. You assess patients using the Emergency Severity Index:
ESI 1 = Immediate life-saving intervention or immediate life threat
ESI 2 = High risk situation, confused/lethargic/disoriented, or severe distress
ESI 3 = Urgent but stable, needs multiple resources
ESI 4 = Less urgent, needs one resource
ESI 5 = Non-urgent, no resources needed

Clinical safety policy:
- When in doubt, prioritize patient safety and choose higher acuity.
- Never undertriage high-risk presentations.
- Suspected sepsis with high-risk features must not be triaged below ESI 2.
- Hypotension, severe MI pattern, altered mental status with infection signs, or life-threatening presentation must be triaged as ESI 1.

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
        "Safety-first triage instruction: when uncertain, choose higher acuity to avoid undertriage. "
        "Use ESI 1 for immediate life threat/life-saving intervention and ESI 2 for high-risk, confused, or severe distress presentations. "
        "Return a single JSON object only."
    )


def call_llm(client: OpenAI, observation: dict) -> TriageAction:
    print(f"[DEBUG] Using API_BASE_URL={API_BASE_URL}", flush=True)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": observation_to_prompt(observation)},
    ]
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0,
        max_tokens=512,
    )
    content = response.choices[0].message.content or ""
    action = parse_action(content)
    if action is None:
        raise ValueError("LLM returned invalid JSON")
    return action


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
        session_id = observation.get("session_id")
        if not session_id:
            raise ValueError("/reset response missing session_id")

        while True:
            action = call_llm(client, observation)

            action_json = format_action(action)

            step_response = http.post(
                "/step",
                json={
                    "session_id": session_id,
                    "action": action.model_dump(exclude_none=True),
                },
            )
            step_response.raise_for_status()
            result = step_response.json()
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            rewards.append(reward)
            steps += 1

            print(f"[STEP] step={steps} action={action_json} reward={round(reward, 2)} done={str(done).lower()} error=null")

            state_response = http.get("/state", params={"session_id": session_id})
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
            client = OpenAI(
                base_url=API_BASE_URL,
                api_key=API_KEY,
            )
            success, steps, rewards = run_episode(client, task_id)
        except Exception as e:
            print(f"[ERROR] task={task_id} error={str(e)} error_type={type(e).__name__}")
        finally:
            score = round(sum(rewards), 2)
            rewards_str = ",".join(str(round(r, 2)) for r in rewards)
            print(f"[END] success={str(success).lower()} steps={steps} score={score} rewards={rewards_str}")


if __name__ == "__main__":
    main()
