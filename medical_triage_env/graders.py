from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple, TYPE_CHECKING

from .logs import get_logger
from .models import TriageAction, TriageReward

if TYPE_CHECKING:
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
        
        # High urgency cases (ESI 1-2): penalize for taking too many steps
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
        
        # Low urgency cases (ESI 4-5): bonus for efficient triage  
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
        
        # Clamp to valid range
        return max(0.0, min(1.0, base_score))


class ReasoningPathGrader:
    """Grades the reasoning path and clinical workflow quality."""
    
    @staticmethod
    def score_reasoning(action_history: List[TriageAction], task: "TaskConfig") -> float:
        """
        Score reasoning path quality based on clinical workflow.
        
        Args:
            action_history: List of all actions taken during the episode
            task: TaskConfig with expected workflow information
            
        Returns:
            Reasoning path score between 0.0 and 1.0
        """
        score = 0.0
        
        # Check if agent asked for vitals before classifying (+0.2)
        asked_for_vitals = any(
            action.action_type == "clarify" and 
            any(keyword in action.clarifying_question.lower() if action.clarifying_question else ""
                for keyword in ["vital", "signs", "hr", "bp", "pulse", "temperature", "oxygen"])
            for action in action_history
        )
        
        if asked_for_vitals:
            score += 0.2
            logger.debug("reasoning_bonus_vitals_check", task_id=task.id)
        
        # Check if agent asked at least 1 relevant clarifying question (+0.3)
        relevant_clarifications = 0
        for action in action_history:
            if action.action_type == "clarify" and action.clarifying_question:
                # Check if this clarification matches any key clarify actions for this task
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
        
        # Check if agent flagged correct red-flag symptoms in reasoning (+0.3)
        flagged_red_flags = False
        for action in action_history:
            if action.action_type == "classify" and action.reasoning:
                reasoning_lower = action.reasoning.lower()
                # Look for key clinical reasoning keywords that indicate red flags
                key_matches = sum(1 for keyword in task.key_reasoning_keywords 
                                if keyword.lower() in reasoning_lower)
                if key_matches >= 2:  # At least 2 key reasoning concepts mentioned
                    flagged_red_flags = True
                    break
        
        if flagged_red_flags:
            score += 0.3  
            logger.debug("reasoning_bonus_red_flags", task_id=task.id)
        
        # Penalty for too many irrelevant clarifications (-0.2 if >2 irrelevant)
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
        
        # Clamp to valid range
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
    # Import here to avoid circular dependency
    from .tasks import TaskConfig
    
    # Convert dict task to TaskConfig for new graders
    task_config = TaskConfig.model_validate(task)
    
    # Compute undertriage penalty factor (from existing logic)
    undertriage_penalty_factor = 1.0
    if task["correct_esi"] <= 2 and action.esi_level is not None and action.esi_level >= 4:
        undertriage_penalty_factor = 0.25
        logger.warning(
            "undertriage_penalty_applied",
            correct_esi=task["correct_esi"],
            predicted_esi=action.esi_level,
            task_id=task.get("task_id", "unknown")
        )
    
    # Compute temporal score
    temporal_grader = TemporalGrader()
    temporal_score = temporal_grader.score_temporal(
        esi_correct=task["correct_esi"],
        steps_taken=steps_taken,
        expected_steps=task.get("expected_clarify_steps", 2)
    )
    
    # Compute reasoning path score
    reasoning_grader = ReasoningPathGrader()
    reasoning_score = reasoning_grader.score_reasoning(action_history, task_config)
    
    # Combine all components with specified weights
    final_score = (
        0.40 * esi_score +
        0.25 * undertriage_penalty_factor * esi_score +  # Apply penalty as multiplicative factor
        0.20 * temporal_score +
        0.15 * reasoning_score
    )
    
    # Component scores for transparency
    component_scores = {
        "esi_score": esi_score,
        "undertriage_penalty_factor": undertriage_penalty_factor,
        "temporal_score": temporal_score,
        "reasoning_score": reasoning_score,
        "final_score": round(max(0.0, min(1.0, final_score)), 2)
    }
    
    logger.debug(
        "final_score_computed",
        task_id=task.get("task_id", "unknown"),
        **component_scores
    )
    
    return component_scores["final_score"], component_scores