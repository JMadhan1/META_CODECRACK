from typing import Dict, List, Any


def grade_task(task_id: str, state: Dict[str, Any], review_history: List[Dict]) -> float:
    """
    Grade agent performance on a task.
    Returns score between 0.0 and 1.0.

    Formula:  0.5 * recall  +  0.3 * precision  +  0.2 * severity_match

    Rationale:
      - Recall (0.5): finding real bugs is the primary objective.
        Missing a critical security issue is worse than a false alarm.
      - Precision (0.3): false positives waste developer time; penalise them
        but less than misses.
      - Severity match (0.2): correct classification helps triage; bonus signal.

    Line matching uses ±2 tolerance to handle LLM off-by-one errors.
    """
    expected = state["expected_issues"]
    found = state["found_issues"]
    false_positives = state.get("false_positives", 0)

    if not expected:
        return 0.999 if not found else 0.001

    def matches(found_issue: Dict, expected_issue: Dict) -> bool:
        type_match = found_issue["type"] == expected_issue["type"]
        line_close = abs(found_issue["line"] - expected_issue["line"]) <= 2
        return type_match and line_close

    # True positives with fuzzy matching (each expected slot claimed once)
    matched_expected: set = set()
    true_positives = 0
    severity_matches = 0

    for f in found:
        for i, e in enumerate(expected):
            if i not in matched_expected and matches(f, e):
                true_positives += 1
                matched_expected.add(i)
                if f.get("severity") == e.get("severity"):
                    severity_matches += 1
                break

    # Recall: fraction of real issues found
    recall = true_positives / len(expected)

    # Precision: fraction of reported issues that are real
    total_reported = true_positives + false_positives
    precision = true_positives / total_reported if total_reported > 0 else 0.0

    # Severity match: among all expected issues, what fraction got correct severity
    # (denominator is len(expected), not true_positives, to penalise missing issues)
    severity_score = severity_matches / len(expected)

    # Final weighted score
    final = (0.5 * recall) + (0.3 * precision) + (0.2 * severity_score)

    return max(0.001, min(0.999, final))
