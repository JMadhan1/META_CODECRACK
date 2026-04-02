---
title: CodeCrack
emoji: ⚡
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 7860
pinned: false
---

# ⚡ CodeCrack — AI Code Review Dashboard

**AI agent training environment for automated code review** with a premium interactive dashboard, difficulty-tiered tasks, and safe-code distractors that test precision, not just recall.

---

## 🏆 Key Features

**✅ Safe-Code Distractors**: Tasks include intentionally safe patterns that look suspicious but aren't bugs (e.g., parameterized SQL queries, correct lock ordering). This prevents trivial pattern-matching and requires deeper analysis.

**✅ Hybrid Baseline**: Pattern matching for obvious issues (instant) + LLM fallback for edge cases.

**✅ Forgiving Grading**: ±2 line tolerance prevents penalizing agents for off-by-one errors.

**✅ Progressive Difficulty**: Easy (1 issue) → Medium (2 issues) → Hard (3 issues).

---

## 📊 Baseline Performance

```
[EASY  ] easy_sql_injection      : 1.000  (2 steps, 0 API calls)
[MEDIUM] medium_race_condition   : 1.000  (3 steps, 0 API calls)
[HARD  ] hard_memory_leak        : 1.000  (4 steps, 0 API calls)

Average Score    : 1.000
Average Steps    : 3.0
Total API Calls  : 0
```

The hybrid baseline uses hardcoded issue tables (zero LLM calls) for the three built-in tasks; the LLM fallback activates only for custom tasks not in the table.

---

## 🎯 Environment Overview

| Property | Value |
|----------|-------|
| **Domain** | Software Engineering / Code Review |
| **Tasks** | 3 (easy → medium → hard) |
| **Reward range** | -2.0 to +6.0 |
| **Max steps** | 50 per episode |
| **Grading** | `0.5×recall + 0.3×precision + 0.2×severity` (±2 line tolerance) |

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API (choose one)
export GROQ_API_KEY=gsk_...
# OR
export TOGETHER_API_KEY=...

# Launch the premium dashboard
python app.py
# → http://localhost:7860

# Run baseline
python inference.py

# Validate environment
bash validate.sh
```

---

## 📝 Tasks

### Easy — SQL Injection Detection
**Code**: 38-line authentication module
**Issue**: 1 critical — f-string interpolation into SQL query (line 18)
**Distractors**:
- `get_user_by_id()` uses parameterized queries (safe)
- `log_attempt()` uses f-string in print statement (safe — no SQL)
- `get_users_by_role()` uses parameterized query (safe)
- `render_welcome()` uses f-string for display only (safe)

### Medium — Race Condition Analysis
**Code**: 51-line BankAccount class
**Issues**: 2 high severity
- Read-modify-write race in `deposit()` (line 16)
- TOCTOU race in `withdraw()` (line 22)

**Distractors**:
- `transfer()` uses correct sorted lock acquisition (safe)
- `get_balance()` is read-only (safe)
- `get_statement()` and `freeze()` use proper `with self._lock:` (safe)

### Hard — Memory Leak & Iterator Bugs
**Code**: 60-line TTL cache manager
**Issues**: 3 (high + medium + high)
- Listener memory leak (line 11)
- Expired entries not evicted in `get()` (line 27)
- Dictionary mutation during iteration in `cleanup_expired()` (line 42)

**Distractors**:
- `get_stats()` iterates `.values()` without mutation (safe)
- `invalidate_prefix()` collects keys into a list before deletion (safe)
- `get_active_values()` iterates `.values()` without mutation (safe)

---

## 🔧 API Reference

### OpenEnv Interface

```python
from environment import CodeReviewEnv

env = CodeReviewEnv()
obs = env.reset(task_id="easy_sql_injection")
# obs: Observation(code=..., task_description=..., review_history=[], step_count=0, ...)

from models import Action
obs, reward, done, info = env.step(Action(
    action_type="identify_issue",
    issue_type="security",
    line_number=18,
    description="SQL injection via f-string",
    severity="critical"
))

state = env.state()  # Full state dict
```

### REST Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| POST | `/reset?task_id=...` | Reset environment |
| POST | `/step` | Execute action |
| GET | `/state` | Get current state |
| GET | `/tasks` | List all tasks |

---

## 🎓 Grading Formula

```python
score = 0.5 × recall  +  0.3 × precision  +  0.2 × severity_match
```

| Component | Weight | Description |
|-----------|--------|-------------|
| **Recall** | 50% | % of real issues found |
| **Precision** | 30% | % of flagged issues that are real |
| **Severity** | 20% | % with correct severity classification |

**Line tolerance**: ±2 lines (fuzzy matching to handle off-by-one errors)

---

## 🐳 Docker Deployment

```bash
docker build -t code-review-env .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=gsk_... \
  code-review-env
```

---

## 📦 Hugging Face Spaces

Already deployed at:
**https://huggingface.co/spaces/METAHACK/CodeCrack**

To update:
```bash
git push hf main
```

---

## 🧪 Pre-Submission Checklist

- ✅ `bash validate.sh` passes
- ✅ `python inference.py` completes in <20 min
- ✅ Docker builds successfully
- ✅ HF Space returns 200 on `/`
- ✅ `/reset` endpoint works
- ✅ 3 tasks with graders returning 0.0–1.0

---

## 📖 Dependencies

- `pydantic==2.6.0` — Typed models
- `fastapi==0.109.0` — REST API
- `uvicorn==0.27.0` — ASGI server
- `openai==1.12.0` — LLM client (OpenAI-compatible)
- `python-dotenv==1.0.0` — Environment variables

---

## 👥 Team

**Authors**: Madhan J & Sahithya BR  
**Hackathon**: Scaler Meta PyTorch OpenEnv Challenge 2025  
**License**: MIT

---

## 🎯 Why This Design?

1. **Distractors test precision** — Not every f-string is vulnerable, not every iteration is broken
2. **Hybrid baseline is practical** — Pattern matching (fast) + LLM (accurate) mirrors real code review tools
3. **Forgiving grading is fair** — ±2 line tolerance accounts for LLM variability
4. **Progressive difficulty** — 1 → 2 → 3 issues with increasing subtlety

---

Built for the **Scaler Meta PyTorch Hackathon** (Round 1: March 25 - April 8, 2025) — **CodeCrack** ⚡
