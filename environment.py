from typing import Dict, Any, Tuple, Optional
import random
from models import Observation, Action, Reward
from tasks import TASKS
from graders import grade_task
from rewards import calculate_reward


class CodeReviewEnv:
    """
    OpenEnv environment for AI code review agent training.

    Simulates realistic code review scenarios with bug detection,
    security audits, and performance improvements.
    """

    def __init__(self):
        self.current_task = None
        self.current_state = None
        self.step_count = 0
        self.max_steps = 50
        self.review_history = []

    def reset(self, task_id: Optional[str] = None) -> Observation:
        """Reset environment and return initial observation."""
        if task_id is None:
            task_id = random.choice(list(TASKS.keys()))

        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid options: {list(TASKS.keys())}")

        self.current_task = TASKS[task_id]
        self.step_count = 0
        self.review_history = []

        self.current_state = {
            "task_id": task_id,
            "code_snippet": self.current_task["code"],
            "expected_issues": self.current_task["issues"],
            "difficulty": self.current_task["difficulty"],
            "found_issues": [],
            "claimed_indices": set(),   # indices into expected_issues already matched
            "false_positives": 0,
            "step_count": 0
        }

        return Observation(
            code=self.current_state["code_snippet"],
            task_description=self.current_task["description"],
            review_history=self.review_history,
            step_count=self.step_count,
            remaining_steps=self.max_steps - self.step_count
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """Execute one environment step."""
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        self.step_count += 1
        self.current_state["step_count"] = self.step_count

        # Calculate reward using PRE-action state (found_issues before processing)
        reward_value = calculate_reward(
            self.current_state,
            action,
            self.current_task
        )

        # Process action (mutates found_issues / history)
        if action.action_type == "identify_issue":
            self._process_issue_identification(action)
        elif action.action_type == "suggest_fix":
            self._process_fix_suggestion(action)
        elif action.action_type == "approve":
            self._process_approval()
        elif action.action_type == "request_changes":
            self._process_request_changes()

        # Check episode termination
        done = (
            self.step_count >= self.max_steps
            or action.action_type == "approve"
            or action.action_type == "request_changes"
        )

        # Grade if done
        score = 0.0
        if done:
            score = grade_task(
                self.current_state["task_id"],
                self.current_state,
                self.review_history
            )

        # Create observation
        obs = Observation(
            code=self.current_state["code_snippet"],
            task_description=self.current_task["description"],
            review_history=list(self.review_history),
            step_count=self.step_count,
            remaining_steps=self.max_steps - self.step_count
        )

        # Create reward
        reward = Reward(
            value=reward_value,
            breakdown={
                "issue_detection": self._calculate_detection_score(),
                "false_positive_penalty": -0.3 * self.current_state["false_positives"],
                "step_efficiency": 1.0 - (self.step_count / self.max_steps)
            },
            final_score=score if done else 0.0
        )

        # Compute F1 for info dict (mirrors graders.py logic)
        tp = len(self.current_state["found_issues"])
        fp = self.current_state["false_positives"]
        expected_n = len(self.current_state["expected_issues"])
        recall_now = tp / expected_n if expected_n else 0.0
        precision_now = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_now = (2 * precision_now * recall_now / (precision_now + recall_now)
                  if (precision_now + recall_now) > 0 else 0.0)

        info = {
            "task_id": self.current_state["task_id"],
            "found_issues": tp,
            "expected_issues": expected_n,
            "false_positives": fp,
            "recall": round(recall_now, 4),
            "precision": round(precision_now, 4),
            "f1": round(f1_now, 4),
            "score": score
        }

        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        """Return current environment state (JSON-serializable)."""
        if self.current_state is None:
            return {}
        s = self.current_state.copy()
        # Convert set to list so FastAPI can JSON-serialize it
        s["claimed_indices"] = sorted(s.get("claimed_indices", set()))
        return s

    def _process_issue_identification(self, action: Action):
        issue = {
            "type": action.issue_type,
            "line": action.line_number,
            "description": action.description,
            "severity": action.severity
        }

        # Find unclaimed expected issues that match (±2 lines, same type)
        claimed = self.current_state["claimed_indices"]
        expected = self.current_state["expected_issues"]
        match_idx = next(
            (i for i, exp in enumerate(expected)
             if i not in claimed
             and exp["type"] == issue["type"]
             and abs(exp["line"] - issue["line"]) <= 2),
            None
        )

        if match_idx is not None:
            claimed.add(match_idx)
            self.current_state["found_issues"].append(issue)
        else:
            self.current_state["false_positives"] += 1

        is_valid = match_idx is not None

        self.review_history.append({
            "action": "identify_issue",
            "step": self.step_count,
            "issue": issue,
            "valid": is_valid
        })

    def _process_fix_suggestion(self, action: Action):
        self.review_history.append({
            "action": "suggest_fix",
            "step": self.step_count,
            "fix": action.suggested_fix
        })

    def _process_approval(self):
        self.review_history.append({
            "action": "approve",
            "step": self.step_count
        })

    def _process_request_changes(self):
        self.review_history.append({
            "action": "request_changes",
            "step": self.step_count,
            "found_count": len(self.current_state["found_issues"])
        })

    def _calculate_detection_score(self) -> float:
        if not self.current_state["expected_issues"]:
            return 1.0

        found = len(self.current_state["found_issues"])
        expected = len(self.current_state["expected_issues"])

        return min(1.0, found / expected)
