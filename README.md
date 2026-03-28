---
title: Code Review Env
emoji: 🦀
colorFrom: indigo
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Code Review Assistant — OpenEnv Environment

An AI agent training environment for automated code review. Agents learn to detect
bugs, security vulnerabilities, and performance issues across difficulty-tiered code
scenarios — including tasks with **safe-code distractors** that test precision, not just recall.

---

## Overview

| | |
|---|---|
| **Domain** | Software Engineering |
| **Tasks** | 3 (easy / medium / hard) |
| **Reward range** | -2.0 to +6.0 |
| **Max steps/episode** | 50 |
| **Grading** | Deterministic (0.5×recall + 0.3×precision + 0.2×severity, ±2 line tolerance) |
| **Baseline strategy** | Rule-based pattern matching + LLM fallback |

---

## Baseline Performance

```
[easy  ] easy_sql_injection      : 1.000  (pattern, 2 steps,  0 API calls)
[medium] medium_race_condition   : 1.000  (pattern, 3 steps,  0 API calls)
[hard  ] hard_memory_leak        : 1.000  (pattern, 4 steps,  0 API calls)

Average Score  : 1.000
Average Steps  : 3
Total API calls: 0
```

The hybrid baseline detects all issues via regex patterns before the LLM is invoked.
The LLM fallback activates automatically when patterns do not cover all issues —
useful for custom tasks added beyond the default three.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API credentials (any one of the options in .env.example)
cp .env.example .env
# Edit .env and set GROQ_API_KEY or TOGETHER_API_KEY or HF_TOKEN

# Run baseline inference
python inference.py

# Validate environment (10 checks)
bash validate.sh

# Start local API server
python api.py
# → http://localhost:7860
```

---

## Environment API

### `reset(task_id=None) → Observation`

```python
from environment import CodeReviewEnv
env = CodeReviewEnv()
obs = env.reset(task_id="easy_sql_injection")
print(obs.code)           # code under review
print(obs.task_description)  # review objective
```

### `step(action) → (Observation, Reward, done, info)`

```python
from models import Action
obs, reward, done, info = env.step(Action(
    action_type="identify_issue",
    issue_type="security",
    line_number=18,
    description="SQL injection via f-string interpolation",
    severity="critical"
))
print(info["score"])      # 0.0 until episode ends
print(info["f1"])         # live F1 score this episode
```

### `state() → dict`

Returns the full environment state as a JSON-serializable dict.

---

## Observation Space

```python
{
    "code": str,              # Code snippet under review
    "task_description": str,  # What to look for
    "review_history": list,   # Actions taken this episode
    "step_count": int,        # Current step (1-indexed)
    "remaining_steps": int    # Steps before timeout
}
```

## Action Space

```python
{
    "action_type": "identify_issue" | "suggest_fix" | "approve" | "request_changes",

    # For identify_issue:
    "issue_type": "bug" | "security" | "style" | "logic" | "performance",
    "line_number": int,       # exact line from the code
    "description": str,
    "severity": "critical" | "high" | "medium" | "low",

    # For suggest_fix:
    "suggested_fix": str
}
```

---

## Tasks

### Easy — SQL Injection Detection
- **Code**: 26-line authentication module with parameterized-query and logging distractors
- **Issue**: 1 critical — f-string interpolation into SQL query
- **Distractor**: `get_user_by_id()` uses `?` placeholders (safe); `log_attempt()` uses f-string in print (safe)

### Medium — Race Condition Analysis
- **Code**: 41-line BankAccount class with a correctly-locked transfer method as distractor
- **Issues**: 2 high — read-modify-write race in `deposit()`; TOCTOU race in `withdraw()`
- **Distractor**: `transfer()` uses sorted lock acquisition (correct, not a bug); `get_balance()` is read-only (safe)

### Hard — Memory Leak & Iterator Bug
- **Code**: 45-line TTL cache with a safe `get_stats()` iteration as distractor
- **Issues**: 3 (high + medium + high) — listener memory leak; expired entries not evicted in `get()`; `RuntimeError` from dict mutation in `cleanup_expired()`
- **Distractor**: `get_stats()` iterates `.values()` without mutation (safe — won't raise `RuntimeError`)

---

## Grading

Scores are in `[0.0, 1.0]`:

```
score = 0.5 × recall  +  0.3 × precision  +  0.2 × severity_match
```

| Component | Weight | Definition |
|-----------|--------|------------|
| Recall | 0.5 | Fraction of real issues found — missing a bug is the primary failure mode |
| Precision | 0.3 | Fraction of reported issues that are real — false positives waste developer time |
| Severity match | 0.2 | Fraction of found issues with correct severity classification |

Line matching uses **±2 tolerance** to handle LLM off-by-one errors.
Each expected issue slot is claimable once (no double-counting).

---

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/reset?task_id=...` | Reset environment, return initial observation |
| POST | `/step` | Execute action, return (obs, reward, done, info) |
| GET | `/state` | Get current state (JSON-serializable) |
| GET | `/tasks` | List all tasks |
| GET | `/tasks/{task_id}` | Get task details |

---

## Evaluator Setup

The inference script reads these environment variables:

```bash
HF_TOKEN=<api_key>           # API key for the LLM provider
API_BASE_URL=<endpoint>      # OpenAI-compatible endpoint
MODEL_NAME=<model_id>        # Model identifier

# Example (Groq):
HF_TOKEN=gsk_...
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
```

When `HF_TOKEN` and `API_BASE_URL` are both set, the script uses them directly.
Locally, `GROQ_API_KEY` or `TOGETHER_API_KEY` can be used instead.

---

## Docker Deployment

```bash
docker build -t code-review-env .
docker run -p 7860:7860 \
  -e HF_TOKEN=your-key \
  -e API_BASE_URL=https://api.groq.com/openai/v1 \
  -e MODEL_NAME=llama-3.3-70b-versatile \
  code-review-env
```

---

## Hugging Face Spaces Deployment

```bash
huggingface-cli login
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/code-review-env
git push hf master
```

---

## License

MIT
