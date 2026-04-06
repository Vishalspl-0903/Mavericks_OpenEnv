from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .logs import get_logger
from .models import TriageAction, TriageReward

logger = get_logger(__name__)

STOPWORDS = {
    "a",
    "an",
    "and",
    "before",
    "check",
    "call",
    "consider",
    "consult",
    "continue",
    "immediate",
    "immediately",
    "initiate",
    "labs",
    "monitor",
    "move",
    "of",
    "prepare",
    "protocol",
    "request",
    "review",
    "start",
    "the",
    "to",
    "urgent",
    "urgently",
    "with",
    "within",
    "if",
    "for",
    "on",
    "by",
    "at",
    "from",
    "into",
    "plan",
    "therapy",
    "management",
    "bundle",
    "levels",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _first_significant_word(text: str) -> str:
    tokens = _tokenize(text)
    if not tokens:
        return ""
    for token in tokens:
        if token in STOPWORDS:
            continue
        if token.isdigit():
            continue
        return token
    return tokens[0]


def _count_keyword_matches(reasoning: str, keywords: Sequence[str]) -> List[str]:
    lowered = _normalize_text(reasoning)
    matches: List[str] = []
    for keyword in keywords:
        if _normalize_text(keyword) in lowered:
            matches.append(keyword)
    return matches


def _count_action_matches(recommended_actions: Sequence[str], expected_actions: Sequence[str]) -> Tuple[int, List[str]]:
    if not expected_actions:
        return 0, []
    normalized_recommendations = [_normalize_text(action) for action in recommended_actions]
    matched_expected: List[str] = []
    for expected_action in expected_actions:
        significant = _first_significant_word(expected_action)
        if not significant:
            continue
        if any(significant in recommendation for recommendation in normalized_recommendations):
            matched_expected.append(expected_action)
    return len(matched_expected), matched_expected


def build_feedback(action: TriageAction, task: Dict[str, Any], esi_score: float, reasoning_score: float, action_score: float) -> str:
    correct_esi = int(task["correct_esi"])
    matched_keywords = _count_keyword_matches(action.reasoning, task.get("key_reasoning_keywords", []))
    matched_count, matched_actions = _count_action_matches(action.recommended_actions, task.get("expected_actions", []))
    total_expected_actions = len(task.get("expected_actions", []))
    total_keywords = len(task.get("key_reasoning_keywords", []))
    reasoning_words = len(action.reasoning.split())

    parts: List[str] = []
    if action.action_type == "classify":
        if action.esi_level == correct_esi:
            parts.append(f"Correct ESI {correct_esi} classification.")
        else:
            parts.append(f"ESI {action.esi_level} selected; correct ESI is {correct_esi}.")
    else:
        parts.append("Clarification requested before classification.")

    if matched_keywords:
        parts.append(f"Matched reasoning keywords: {', '.join(matched_keywords[:5])}.")
    else:
        parts.append("No key reasoning keywords matched the rubric.")

    if reasoning_words < 20:
        parts.append("Reasoning was too short for full credit.")
    else:
        parts.append(f"Reasoning covered {len(matched_keywords)}/{total_keywords} key clinical concepts.")

    parts.append(f"Recommended actions covered {matched_count}/{total_expected_actions} expected actions.")

    if correct_esi <= 2 and action.action_type == "classify" and action.esi_level is not None and action.esi_level >= 4:
        parts.append("Dangerous undertriage penalty applied.")

    return " ".join(parts)


def grade(action: TriageAction, task: Dict[str, Any]) -> TriageReward:
    undertriage_penalty = False

    if action.action_type == "clarify":
        esi_score = 0.10
    else:
        correct_esi = int(task["correct_esi"])
        if action.esi_level is None:
            diff = 5
        else:
            diff = abs(int(action.esi_level) - correct_esi)
        if diff == 0:
            esi_score = 0.50
        elif diff == 1:
            esi_score = 0.25
        elif diff == 2:
            esi_score = 0.10
        else:
            esi_score = 0.00
        if correct_esi <= 2 and action.esi_level is not None and action.esi_level >= 4:
            undertriage_penalty = True

    matched_keywords = _count_keyword_matches(action.reasoning, task.get("key_reasoning_keywords", []))
    reasoning_score = min(0.30, 0.03 * len(matched_keywords))
    if len(action.reasoning.split()) < 20:
        reasoning_score = min(reasoning_score, 0.10)

    matched_count, _ = _count_action_matches(action.recommended_actions, task.get("expected_actions", []))
    expected_actions = task.get("expected_actions", [])
    action_score = 0.0
    if expected_actions:
        action_score = (matched_count / len(expected_actions)) * 0.20

    raw = esi_score + reasoning_score + action_score
    if undertriage_penalty:
        logger.warning(
            "undertriage_detected",
            correct_esi=task["correct_esi"],
            predicted_esi=action.esi_level,
            task_id=task.get("task_id", "unknown"),
            penalty_applied=True,
        )
        raw = raw * 0.25

    final = round(min(max(raw, 0.0), 1.0), 2)
    feedback = build_feedback(action, task, esi_score, reasoning_score, action_score)
    return TriageReward(
        value=final,
        esi_accuracy=esi_score,
        reasoning_quality=reasoning_score,
        action_appropriateness=action_score,
        feedback=feedback,
    )