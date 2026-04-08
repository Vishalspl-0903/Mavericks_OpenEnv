from __future__ import annotations

import json
import os
import re
from typing import List

import httpx
from openai import OpenAI
from dotenv import load_dotenv

from medical_triage_env.models import TriageAction
from medical_triage_env.tasks import TASK_LIST

load_dotenv()

BASE_URL = os.environ["BASE_URL"]
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
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

def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_action(text: str) -> TriageAction:
    cleaned = strip_code_fences(text)
    payload = json.loads(cleaned)
    return TriageAction.model_validate(payload)


def observation_to_prompt(observation: dict) -> str:
    return (
        "Patient observation:\n"
        f"{json.dumps(observation, indent=2)}\n\n"
        "Return JSON only."
    )


def call_llm(client: OpenAI, observation: dict) -> TriageAction:
    print(f"[DEBUG] PROXY CALL -> {API_BASE_URL}", flush=True)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": observation_to_prompt(observation)},
        ],
        temperature=0,
        max_tokens=512,
    )

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("Empty LLM response")

    return parse_action(content)


def run_episode(client: OpenAI, task_id: str) -> tuple[int, List[float]]:
    rewards: List[float] = []
    steps = 0

    with httpx.Client(base_url=BASE_URL, timeout=30.0) as http:
        reset = http.post("/reset", json={"task_id": task_id})
        reset.raise_for_status()

        data = reset.json()
        observation = data["observation"]
        session_id = data["info"]["session_id"]

        while True:
            action = call_llm(client, observation)

            step = http.post(
                "/step",
                json={
                    "session_id": session_id,
                    "action": action.model_dump(exclude_none=True),
                },
            )
            step.raise_for_status()

            result = step.json()

            reward = float(result["reward"])
            done = bool(result["done"])

            rewards.append(reward)
            steps += 1

            print(f"[STEP] {steps} reward={reward} done={done}")

            observation = result.get("observation", observation)

            if done or steps >= MAX_STEPS:
                break

    return steps, rewards


def main() -> None:
    client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY
    )

    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=5,
    )

    for task in TASK_LIST:
        print(f"[START] {task}")
        steps, rewards = run_episode(client, task)
        print(f"[END] steps={steps} score={sum(rewards)}")


if __name__ == "__main__":
    main()
