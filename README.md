# Code Review Assistant — OpenEnv Environment

An AI agent training environment for automated code review. Agents learn to detect bugs, security vulnerabilities, and performance issues using a **hybrid rule-based + LLM approach** that is fast, accurate, and token-efficient.

---

## Overview

| | |
|---|---|
| **Domain** | Software Engineering |
| **Tasks** | 3 (easy / medium / hard) |
| **Reward range** | -2.0 to +5.0 |
| **Max steps/episode** | 50 |
| **Grading** | Deterministic (recall + precision + severity, ±2 line tolerance) |
| **Baseline strategy** | Pattern matching (Phase 1) + LLM fallback (Phase 2) |

---

## Baseline Performance

```
[easy  ] easy_sql_injection      : 1.000  (pattern match, 0 API calls)
[medium] medium_race_condition   : 0.950  (pattern match, 0-1 API calls)
[hard  ] hard_memory_leak        : 0.900  (pattern + LLM, ~3 API calls)

Average Score: 0.950
Execution time: < 15 seconds
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API key (Together AI recommended — $25 free credit)
cp .env.example .env
# Edit .env and set TOGETHER_API_KEY or GROQ_API_KEY

# Run baseline inference
python inference.py

# Validate environment
bash validate.sh

# Start local API server
python api.py
# → http://localhost:7860
```

---

## Environment API

### `reset(task_id=None) → Observation`
Resets the environment. If `task_id` is None, a random task is selected.

```python
from environment import CodeReviewEnv
env = CodeReviewEnv()
obs = env.reset(task_id="easy_sql_injection")
```

### `step(action) → (Observation, Reward, done, info)`
Executes one action in the environment.

```python
from models import Action
obs, reward, done, info = env.step(Action(
    action_type="identify_issue",
    issue_type="security",
    line_number=2,
    description="SQL injection via f-string interpolation",
    severity="critical"
))
```

### `state() → dict`
Returns raw environment state dictionary.

---

## Observation Space

```python
{
    "code": str,              # Code snippet under review (numbered)
    "task_description": str,  # Review objective
    "review_history": list,   # Prior actions this episode
    "step_count": int,        # Current step (1-indexed)
    "remaining_steps": int    # Steps left before timeout
}
```

## Action Space

```python
{
    "action_type": "identify_issue" | "suggest_fix" | "approve" | "request_changes",

    # Required for identify_issue:
    "issue_type": "bug" | "security" | "style" | "logic" | "performance",
    "line_number": int,
    "description": str,
    "severity": "critical" | "high" | "medium" | "low",

    # Required for suggest_fix:
    "suggested_fix": str
}
```

---

## Tasks

### Easy — SQL Injection Detection
- **Goal**: Find the SQL injection vulnerability
- **Issue count**: 1 (severity: critical)
- **Baseline**: Pattern match (instant)

### Medium — Race Condition Analysis
- **Goal**: Identify concurrency bugs in a BankAccount class
- **Issue count**: 2 (severity: high)
- **Baseline**: Pattern match (instant)

### Hard — Memory Leak & Iterator Bug
- **Goal**: Find memory leaks and a dictionary mutation bug
- **Issue count**: 3 (mixed severity)
- **Baseline**: Pattern match + LLM fallback

---

## Grading

Scores are in `[0.0, 1.0]` and computed as:

```
score = 0.4 × recall + 0.4 × precision + 0.2 × severity_match
```

- **Recall** — fraction of real issues found
- **Precision** — fraction of reported issues that are real
- **Severity match** — fraction of issues with correct severity classification
- **Line tolerance** — ±2 lines (handles LLM off-by-one errors)

False positives reduce precision directly.

---

## Hybrid Baseline Strategy

The baseline agent uses two phases:

**Phase 1 — Rule-based scan** (no API calls, ~0ms):
- Regex patterns catch SQL injections, race conditions, and common memory leaks instantly
- Zero tokens consumed for easy/medium tasks

**Phase 2 — LLM fallback** (only when patterns miss issues):
- Activates only for remaining unfound issues
- Prompt includes numbered code + already-found context
- Client-side dedup prevents re-reporting the same line

This minimizes API usage while maximizing accuracy.

---

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/reset?task_id=...` | Reset environment |
| POST | `/step` | Execute action |
| GET | `/state` | Get current state |
| GET | `/tasks` | List all tasks |
| GET | `/tasks/{task_id}` | Get task details |

---

## Docker Deployment

```bash
# Build
docker build -t code-review-env .

# Run locally
docker run -p 7860:7860 -e TOGETHER_API_KEY=your-key code-review-env
```

---

## Hugging Face Spaces Deployment

```bash
huggingface-cli login
huggingface-cli repo create code-review-env --type space --space_sdk docker
git init && git add . && git commit -m "Initial commit"
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/code-review-env
git push hf main
```

---

## License

MIT
