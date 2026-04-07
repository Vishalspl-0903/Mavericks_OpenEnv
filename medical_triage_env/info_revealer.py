"""
Information revelation and vital sign drift management for medical triage environment.

The InfoRevealer class manages progressive information disclosure and vital sign changes
over time during a triage episode. It maintains state about what information has been
revealed and applies physiological vital sign drift patterns.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Set, Any

from .tasks import TaskConfig
from .logs import get_logger

logger = get_logger(__name__)


class InfoRevealer:
    """Manages progressive information disclosure and vital drift for triage sessions."""
    
    def __init__(self, task: TaskConfig) -> None:
        """
        Initialize InfoRevealer for a specific task.
        
        Args:
            task: The TaskConfig containing all task-specific information
        """
        self.task = task
        self.revealed_triggers: Set[str] = set()
        
        logger.debug(
            "info_revealer_initialized", 
            task_id=task.id,
            hidden_info_count=len(task.hidden_info),
            vital_drift_enabled=bool(task.vital_drift.per_step)
        )
    
    def get_initial_observation(self, step: int) -> Dict[str, Any]:
        """
        Get initial partial vital signs, omitting keys marked as hidden.
        
        Args:
            step: Current step number
            
        Returns:
            Dictionary with visible vital signs only
        """
        vitals = deepcopy(self.task.initial_vitals)
        
        # Remove keys that are marked as hidden in any hidden_info item
        hidden_keys = set()
        for hidden_item in self.task.hidden_info:
            # Check if this hidden item contains vital sign keys to hide initially
            data = hidden_item.data
            if "hidden_vitals" in data:
                hidden_keys.update(data["hidden_vitals"])
        
        # Remove hidden vital signs from initial observation
        for key in hidden_keys:
            vitals.pop(key, None)
        
        logger.debug(
            "initial_observation_created",
            task_id=self.task.id,
            step=step,
            visible_vitals=list(vitals.keys()),
            hidden_vitals=list(hidden_keys)
        )
        
        return vitals
    
    def process_clarify(self, action_type: str, step: int) -> Dict[str, Any]:
        """
        Process a clarification action and return newly revealed information.
        
        Each trigger can only unlock information once. Tracks revealed triggers internally.
        
        Args:
            action_type: The type of clarification action (e.g., "check_vitals", "ask_history")
            step: Current step number
            
        Returns:
            Dictionary with newly revealed information, or empty dict if nothing new
        """
        if action_type in self.revealed_triggers:
            logger.debug(
                "clarify_already_revealed",
                task_id=self.task.id,
                action_type=action_type,
                step=step
            )
            return {}
        
        revealed_info = {}
        
        # Find matching hidden info items for this trigger
        for hidden_item in self.task.hidden_info:
            if hidden_item.trigger == action_type:
                revealed_info.update(hidden_item.data)
                self.revealed_triggers.add(action_type)
                
                logger.debug(
                    "info_revealed",
                    task_id=self.task.id,
                    trigger=action_type,
                    step=step,
                    revealed_keys=list(hidden_item.data.keys())
                )
                break
        
        return revealed_info
    
    def apply_vital_drift(self, vitals: Dict[str, Any], step: int) -> Dict[str, Any]:
        """
        Apply per-step vital sign drift after the configured start step.
        
        Clamps values to physiologically valid ranges:
        - heart_rate: 20-300 bpm
        - oxygen_saturation (spo2): 50-100%  
        - blood_pressure_systolic (sbp): 40-300 mmHg
        - respiratory_rate: 4-60 bpm
        
        Args:
            vitals: Current vital signs dictionary
            step: Current step number
            
        Returns:
            Dictionary with drift-adjusted vital signs
        """
        if step < self.task.vital_drift.starts_at_step:
            return vitals
        
        if not self.task.vital_drift.per_step:
            return vitals
        
        drifted_vitals = deepcopy(vitals)
        steps_since_start = step - self.task.vital_drift.starts_at_step + 1
        
        # Apply drift for each configured vital sign
        for vital_key, drift_per_step in self.task.vital_drift.per_step.items():
            if vital_key in drifted_vitals and drifted_vitals[vital_key] is not None:
                current_value = float(drifted_vitals[vital_key])
                drifted_value = current_value + (drift_per_step * steps_since_start)
                
                # Apply physiological clamping
                if vital_key == "heart_rate":
                    drifted_value = max(20.0, min(300.0, drifted_value))
                elif vital_key == "oxygen_saturation":
                    drifted_value = max(50.0, min(100.0, drifted_value))  
                elif vital_key == "blood_pressure_systolic":
                    drifted_value = max(40.0, min(300.0, drifted_value))
                elif vital_key == "respiratory_rate":
                    drifted_value = max(4.0, min(60.0, drifted_value))
                
                # Convert back to appropriate type (int for most vitals, float for temperature/spo2)
                if vital_key in ["temperature", "oxygen_saturation"]:
                    drifted_vitals[vital_key] = round(drifted_value, 1)
                else:
                    drifted_vitals[vital_key] = int(round(drifted_value))
                
                logger.debug(
                    "vital_drift_applied",
                    task_id=self.task.id,
                    vital=vital_key,
                    original=current_value,
                    drifted=drifted_vitals[vital_key],
                    step=step,
                    steps_since_start=steps_since_start
                )
        
        return drifted_vitals
    
    def get_confounders(self) -> List[str]:
        """
        Get list of confounder strings to include in observation text.
        
        Returns:
            List of confounder description strings
        """
        return list(self.task.confounders)
    
    def get_revealed_info_keys(self) -> List[str]:
        """
        Get list of trigger keys that have been revealed so far.
        
        Returns:
            List of revealed trigger names
        """
        return list(self.revealed_triggers)