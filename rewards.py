from typing import Dict, Any


def calculate_reward(state: Dict[str, Any], action: Any, task: Dict[str, Any]) -> float:
    """Calculate reward for a single step with partial progress signals."""
    reward = 0.0

    if action.action_type == "identify_issue":
        # Check against UNCLAIMED expected issues (same claim logic as environment)
        claimed = state.get("claimed_indices", set())
        match_idx = next(
            (i for i, exp in enumerate(state["expected_issues"])
             if i not in claimed
             and exp["type"] == action.issue_type
             and abs(exp["line"] - action.line_number) <= 2),
            None
        )
        is_valid = match_idx is not None

        if is_valid:
            # Higher reward for critical issues
            severity_bonus = {
                "critical": 1.0,
                "high": 0.7,
                "medium": 0.4,
                "low": 0.2
            }.get(action.severity, 0.5)
            reward += 0.5 + severity_bonus
        else:
            # False positive or already-claimed issue
            reward -= 0.3

    elif action.action_type == "approve":
        # Check if all issues were found
        found = len(state["found_issues"])
        expected = len(state["expected_issues"])

        if found == expected and state["false_positives"] == 0:
            reward += 2.0  # Perfect approval
        elif found < expected:
            reward -= 1.0  # Premature approval

    elif action.action_type == "suggest_fix":
        reward += 0.1  # small reward for attempting fixes

    elif action.action_type == "request_changes":
        found = len(state["found_issues"])
        expected = len(state["expected_issues"])
        if found > 0:
            reward += 0.5 * (found / expected)

    # Step efficiency penalty
    reward -= 0.01 * state["step_count"]

    return reward
