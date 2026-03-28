from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal


class Observation(BaseModel):
    """Agent's view of the code review environment."""
    code: str = Field(description="Code snippet under review")
    task_description: str = Field(description="What the agent should review for")
    review_history: List[Dict] = Field(default_factory=list, description="Previous review actions")
    step_count: int = Field(description="Current step number")
    remaining_steps: int = Field(description="Steps left before timeout")


class Action(BaseModel):
    """Actions the agent can take during code review."""
    action_type: Literal["identify_issue", "suggest_fix", "approve", "request_changes"] = Field(
        description="Type of review action"
    )
    issue_type: Optional[Literal["bug", "security", "style", "logic", "performance"]] = Field(
        default=None,
        description="Category of issue (required for identify_issue)"
    )
    line_number: Optional[int] = Field(
        default=None,
        description="Line number of issue (required for identify_issue)"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description of the issue"
    )
    severity: Optional[Literal["critical", "high", "medium", "low"]] = Field(
        default=None,
        description="Issue severity"
    )
    suggested_fix: Optional[str] = Field(
        default=None,
        description="Code fix suggestion (required for suggest_fix)"
    )


class Reward(BaseModel):
    """Reward signal for the agent."""
    value: float = Field(description="Total reward for this step")
    breakdown: Dict[str, float] = Field(description="Reward components")
    final_score: float = Field(default=0.0, description="Task completion score (0.0-1.0)")
