from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence, Tuple

from .logs import get_logger
from .models import TriageAction, TriageReward
from .tasks import TaskConfig

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


def _keyword_matches(text: str, keywords: Sequence[str]) -> List[str]:
    """
    Flexible keyword matching that allows token-level matches for paraphrases.
    """
    lowered = _normalize_text(text)
    matches: List[str] = []
    for keyword in keywords:
        kw_lower = _normalize_text(keyword)
        if kw_lower in lowered:
            matches.append(keyword)
            continue
        tokens = [t for t in _tokenize(kw_lower) if t not in STOPWORDS and not t.isdigit()]
        if tokens and any(token in lowered for token in tokens):
            matches.append(keyword)
    return matches


def _action_matches(recommended: Sequence[str], expected: Sequence[str]) -> Tuple[int, List[str]]:
    """
    Match expected actions against recommended actions using token overlap.
    """
    if not expected:
        return 0, []

    rec_text = " ".join(_normalize_text(action) for action in recommended)
    matched: List[str] = []
    for expected_action in expected:
        tokens = [
            token
            for token in _tokenize(expected_action)
            if token not in STOPWORDS and not token.isdigit() and len(token) > 2
        ]
        if not tokens:
            continue
        if any(token in rec_text for token in tokens):
            matched.append(expected_action)

    return len(matched), matched


def _resolve_correct_esi(task: Dict[str, Any]) -> int | None:
    value = task.get("correct_esi", task.get("esi_correct"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def build_feedback(
    action: TriageAction,
    task: Dict[str, Any],
    esi_score: float,
    reasoning_score: float,
    action_score: float,
) -> str:
    correct_esi = _resolve_correct_esi(task)
    matched_keywords = _keyword_matches(action.reasoning, task.get("key_reasoning_keywords", []))
    matched_count, matched_actions = _action_matches(action.recommended_actions, task.get("expected_actions", []))
    total_expected = len(task.get("expected_actions", []))
    total_keywords = len(task.get("key_reasoning_keywords", []))
    reasoning_words = len(action.reasoning.split())

    parts: List[str] = []
    if action.action_type == "classify":
        if correct_esi is None:
            parts.append("Reference ESI unavailable.")
        elif action.esi_level == correct_esi:
            parts.append(f"Correct ESI {correct_esi} classification.")
        else:
            parts.append(f"ESI {action.esi_level} selected; correct ESI is {correct_esi}.")
    else:
        parts.append("Clarification requested before classification.")

    if matched_keywords:
        parts.append(
            f"Matched {len(matched_keywords)}/{total_keywords} reasoning keywords: "
            f"{', '.join(matched_keywords[:5])}."
        )
    else:
        parts.append("No key reasoning keywords matched.")

    if reasoning_words < 20:
        parts.append("Reasoning too short for full credit.")

    parts.append(f"Recommended actions covered {matched_count}/{total_expected} expected.")

    if correct_esi is not None and correct_esi <= 2 and action.action_type == "classify" and action.esi_level is not None and action.esi_level >= 4:
        parts.append("DANGEROUS undertriage penalty applied (0.25x multiplier).")

    return " ".join(parts)


def grade(action: TriageAction, task: Dict[str, Any]) -> TriageReward:
    undertriage_penalty = False
    correct_esi = _resolve_correct_esi(task)

    # ESI accuracy (0.0 - 0.50)
    if action.action_type == "clarify":
        esi_score = 0.02
    else:
        if action.esi_level is None or correct_esi is None:
            diff = 5
        else:
            diff = abs(int(action.esi_level) - correct_esi)
        if diff == 0:
            esi_score = 0.70
        elif diff == 1:
            if (
                correct_esi is not None
                and action.esi_level is not None
                and int(action.esi_level) > correct_esi
                and correct_esi <= 2
            ):
                esi_score = 0.08
            else:
                esi_score = 0.20
        elif diff == 2:
            esi_score = 0.00
        else:
            esi_score = -0.10
        if correct_esi is not None and correct_esi <= 2 and action.esi_level is not None and action.esi_level >= 4:
            undertriage_penalty = True

    # Reasoning quality (0.0 - 0.30)
    keywords = task.get("key_reasoning_keywords", [])
    matched_keywords = _keyword_matches(action.reasoning, keywords)
    total_keywords = len(keywords)
    if total_keywords > 0:
        keyword_ratio = len(matched_keywords) / total_keywords
        reasoning_score = round(min(0.15, keyword_ratio * 0.15), 4)
    else:
        reasoning_score = 0.10

    if len(action.reasoning.split()) < 20:
        reasoning_score = min(reasoning_score, 0.10)

    # Action appropriateness (0.0 - 0.20)
    expected_actions = task.get("expected_actions", [])
    matched_count, matched_actions = _action_matches(action.recommended_actions, expected_actions)
    if expected_actions:
        action_score = round((matched_count / len(expected_actions)) * 0.15, 4)
    else:
        action_score = 0.0

    raw = esi_score + reasoning_score + action_score
    if action.action_type == "classify" and diff == 0:
        raw += 0.05
    if undertriage_penalty:
        raw *= 0.25
        logger.warning(
            "undertriage_detected",
            correct_esi=correct_esi,
            predicted_esi=action.esi_level,
            task_id=task.get("task_id", "unknown"),
            penalty_applied=True,
        )

    final = float(min(max(raw, -1.0), 1.0))
    feedback = build_feedback(action, task, esi_score, reasoning_score, action_score)

    logger.debug(
        "grade_computed",
        task_id=task.get("task_id", "unknown"),
        esi_score=esi_score,
        reasoning_score=reasoning_score,
        action_score=action_score,
        matched_keywords=matched_keywords,
        matched_actions=matched_actions,
        final=final,
    )

    return TriageReward(
        value=final,
        esi_accuracy=esi_score,
        reasoning_quality=reasoning_score,
        action_appropriateness=action_score,
        feedback=feedback,
    )


class TemporalGrader:
    """Grades timing performance based on ESI urgency and steps taken."""
    
    @staticmethod
    def score_temporal(esi_correct: int, steps_taken: int, expected_steps: int) -> float:
        """
        Score temporal performance with urgency-based penalties and bonuses.
        
        Args:
            esi_correct: Correct ESI level (1-5)
            steps_taken: Number of steps actually taken
            expected_steps: Expected number of steps for this task
            
        Returns:
            Temporal score between 0.0 and 1.0
        """
        base_score = 1.0
        
        if esi_correct <= 2 and steps_taken > expected_steps + 1:
            extra_steps = steps_taken - (expected_steps + 1)
            penalty = 0.08 * extra_steps
            base_score -= penalty
            
            logger.debug(
                "temporal_penalty_applied",
                esi=esi_correct,
                steps_taken=steps_taken,
                expected_steps=expected_steps,
                extra_steps=extra_steps,
                penalty=penalty
            )
        
        elif esi_correct >= 4 and steps_taken < expected_steps - 1:
            saved_steps = (expected_steps - 1) - steps_taken
            bonus = 0.04 * saved_steps
            base_score += bonus
            
            logger.debug(
                "temporal_bonus_applied",
                esi=esi_correct,
                steps_taken=steps_taken,
                expected_steps=expected_steps,
                saved_steps=saved_steps,
                bonus=bonus
            )
        
        return max(0.0, min(1.0, base_score))


class ReasoningPathGrader:
    """Grades the reasoning path and clinical workflow quality."""
    
    @staticmethod
    def score_reasoning(action_history: List[TriageAction], task: TaskConfig) -> float:
        """
        Score reasoning path quality based on clinical workflow.
        
        Args:
            action_history: List of all actions taken during the episode
            task: TaskConfig with expected workflow information
            
        Returns:
            Reasoning path score between 0.0 and 1.0
        """
        score = 0.0
        
        asked_for_vitals = any(
            action.action_type == "clarify" and 
            any(keyword in action.clarifying_question.lower() if action.clarifying_question else ""
                for keyword in ["vital", "signs", "hr", "bp", "pulse", "temperature", "oxygen"])
            for action in action_history
        )
        
        if asked_for_vitals:
            score += 0.2
            logger.debug("reasoning_bonus_vitals_check", task_id=task.id)
        
        relevant_clarifications = 0
        for action in action_history:
            if action.action_type == "clarify" and action.clarifying_question:
                for key_clarify in task.key_clarify_actions:
                    if key_clarify.lower() in action.clarifying_question.lower():
                        relevant_clarifications += 1
                        break
        
        if relevant_clarifications >= 1:
            score += 0.3
            logger.debug(
                "reasoning_bonus_relevant_clarify", 
                task_id=task.id,
                count=relevant_clarifications
            )
        
        flagged_red_flags = False
        for action in action_history:
            if action.action_type == "classify" and action.reasoning:
                reasoning_lower = action.reasoning.lower()
                key_matches = sum(1 for keyword in task.key_reasoning_keywords 
                                if keyword.lower() in reasoning_lower)
                if key_matches >= 2:  # At least 2 key reasoning concepts mentioned
                    flagged_red_flags = True
                    break
        
        if flagged_red_flags:
            score += 0.3  
            logger.debug("reasoning_bonus_red_flags", task_id=task.id)
        
        irrelevant_clarifications = 0
        total_clarifications = sum(1 for action in action_history if action.action_type == "clarify")
        
        if total_clarifications > relevant_clarifications:
            irrelevant_clarifications = total_clarifications - relevant_clarifications
        
        if irrelevant_clarifications > 2:
            penalty = 0.2
            score -= penalty
            logger.debug(
                "reasoning_penalty_irrelevant", 
                task_id=task.id,
                irrelevant_count=irrelevant_clarifications,
                penalty=penalty
            )
        
        return max(0.0, min(1.0, score))


def compute_final_score(
    action: TriageAction,
    task: Dict[str, Any],
    action_history: List[TriageAction],
    esi_score: float,
    steps_taken: int
) -> Tuple[float, Dict[str, float]]:
    """
    Compute final score combining all grading components.
    
    Args:
        action: The final classification action
        task: Task configuration dictionary  
        action_history: Full history of actions taken
        esi_score: ESI accuracy score
        steps_taken: Number of steps taken
        
    Returns:
        Tuple of (final_score, component_scores_dict)
    """
    task_config = TaskConfig.model_validate(task)
    
    undertriage_penalty_factor = 1.0
    correct_esi = _resolve_correct_esi(task)
    if correct_esi is not None and correct_esi <= 2 and action.esi_level is not None and action.esi_level >= 4:
        undertriage_penalty_factor = 0.25
        logger.warning(
            "undertriage_penalty_applied",
            correct_esi=correct_esi,
            predicted_esi=action.esi_level,
            task_id=task.get("task_id", "unknown")
        )
    
    temporal_grader = TemporalGrader()
    reasoning_grader = ReasoningPathGrader()

    if action.action_type == "clarify":
        temporal_score = 0.2
        reasoning_score = 0.0
    else:
        temporal_score = temporal_grader.score_temporal(
            esi_correct=correct_esi if correct_esi is not None else 3,
            steps_taken=steps_taken,
            expected_steps=task.get("expected_clarify_steps", 2)
        )
        reasoning_score = reasoning_grader.score_reasoning(action_history, task_config)
    
    base_score = (
        0.80 * esi_score +
        0.10 * temporal_score +
        0.10 * reasoning_score
    )
    final_score = base_score * undertriage_penalty_factor
    
    component_scores = {
        "esi_score": round(esi_score, 4),
        "undertriage_penalty_factor": undertriage_penalty_factor,
        "temporal_score": round(temporal_score, 4),
        "reasoning_score": round(reasoning_score, 4),
        "base_score": round(max(0.0, min(1.0, base_score)), 4),
        "final_score": float(max(0.0, min(1.0, final_score))),
    }
    
    logger.debug(
        "final_score_computed",
        task_id=task.get("task_id", "unknown"),
        **component_scores
    )
    
    return component_scores["final_score"], component_scores
