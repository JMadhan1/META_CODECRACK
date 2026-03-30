"""
CodeCrack — AI Code Review Dashboard
Premium interactive Gradio interface for the hybrid code-review agent.
Features:
  1. Meta-Review Playground — paste any code, get instant analysis
  2. Adversarial Code Generator — LLM creates buggy code on-the-fly
  3. Duo-Agent Debate — pattern engine vs LLM debate code issues
  4. Task Arena — run the official tasks and see live scoring
"""

import os
import re
import json
import random
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

from environment import CodeReviewEnv
from models import Action
from tasks import TASKS
from graders import grade_task
from inference import pattern_scan

load_dotenv()

# ---------------------------------------------------------------------------
# LLM client bootstrap
# ---------------------------------------------------------------------------

def _get_client():
    groq_key = os.getenv("GROQ_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    hf_token = os.getenv("HF_TOKEN")

    if groq_key:
        return OpenAI(api_key=groq_key, base_url=os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")), os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"), "Groq"
    elif together_key:
        return OpenAI(api_key=together_key, base_url="https://api.together.xyz/v1"), "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "Together"
    elif hf_token:
        return OpenAI(api_key=hf_token, base_url="https://api-inference.huggingface.co/v1"), "meta-llama/Llama-3.1-70B-Instruct", "HuggingFace"
    return None, None, "Pattern-only"

CLIENT, MODEL_NAME, PROVIDER = _get_client()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEVERITY_COLOR = {"critical": "#dc2626", "high": "#ea580c", "medium": "#d97706", "low": "#65a30d"}
ISSUE_ICONS = {"security": "🛡️", "bug": "🐛", "performance": "⚡", "style": "🎨", "logic": "🧠"}


def _llm_chat(prompt: str, temperature: float = 0.0, max_tokens: int = 600) -> str:
    if CLIENT is None:
        return '{"error": "No LLM client configured. Set GROQ_API_KEY or TOGETHER_API_KEY in .env"}'
    resp = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def _parse_json_block(text: str):
    """Extract JSON from possible markdown fence."""
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.replace("json", "").strip()
            if p.startswith("{") or p.startswith("["):
                return json.loads(p)
    # Try raw
    start = text.find("{")
    bracket = text.find("[")
    if start == -1 and bracket == -1:
        return None
    if start == -1 or (bracket != -1 and bracket < start):
        start = bracket
    return json.loads(text[start:])


# ---------------------------------------------------------------------------
# 1. META-REVIEW PLAYGROUND
# ---------------------------------------------------------------------------

def meta_review(code: str):
    """Hybrid analysis: pattern scan + LLM deep dive, yields formatted markdown."""
    if not code or not code.strip():
        yield "Paste some Python code to analyze.", ""
        return

    lines = code.split("\n")
    total_lines = len(lines)

    # Phase 1: Pattern scan
    pattern_issues = pattern_scan(code)

    yield _format_phase1(pattern_issues, total_lines), ""

    # Phase 2: LLM analysis
    if CLIENT is None:
        score_card = _build_score_card(pattern_issues, pattern_issues, total_lines)
        yield _format_phase1(pattern_issues, total_lines), score_card
        return

    prompt = f"""You are a senior security engineer performing a code review.

Analyze this Python code for bugs, security vulnerabilities, and performance issues.
For each issue found, output a JSON object. Return a JSON array of objects.

Each object:
{{"type": "bug|security|performance|style|logic", "line": <int>, "description": "<concise>", "severity": "critical|high|medium|low", "fix": "<one-line fix suggestion>"}}

If no issues, return [].

CODE ({total_lines} lines):
```python
{code}
```"""

    try:
        raw = _llm_chat(prompt, temperature=0.0, max_tokens=800)
        llm_issues = _parse_json_block(raw)
        if not isinstance(llm_issues, list):
            llm_issues = []
    except Exception as e:
        llm_issues = [{"type": "error", "line": 0, "description": str(e), "severity": "low", "fix": ""}]

    # Merge: deduplicate by (type, line±2)
    merged = list(pattern_issues)
    for li in llm_issues:
        is_dup = False
        for pi in merged:
            if pi.get("issue_type", "") == li.get("type", "") and abs(pi.get("line_number", 0) - li.get("line", 0)) <= 2:
                is_dup = True
                break
        if not is_dup:
            merged.append({
                "issue_type": li.get("type", "bug"),
                "line_number": li.get("line", 0),
                "description": li.get("description", ""),
                "severity": li.get("severity", "medium"),
                "fix": li.get("fix", ""),
                "source": "llm"
            })

    formatted = _format_full_report(merged, pattern_issues, llm_issues, total_lines)
    score_card = _build_score_card(merged, pattern_issues, total_lines)
    yield formatted, score_card


def _format_phase1(issues, total_lines):
    if not issues:
        return f"### Phase 1 — Pattern Scan\n\nScanned **{total_lines}** lines. No obvious issues detected by regex patterns.\n\n_Waiting for LLM deep analysis..._"
    lines = [f"### Phase 1 — Pattern Scan\n"]
    lines.append(f"Found **{len(issues)}** issue(s) in **{total_lines}** lines:\n")
    for iss in issues:
        icon = ISSUE_ICONS.get(iss["issue_type"], "❓")
        color = SEVERITY_COLOR.get(iss["severity"], "#888")
        lines.append(f"- {icon} **Line {iss['line_number']}** — <span style='color:{color}'>[{iss['severity'].upper()}]</span> {iss['description']}")
    lines.append("\n_LLM deep analysis in progress..._")
    return "\n".join(lines)


def _format_full_report(all_issues, pattern_issues, llm_issues, total_lines):
    lines = [f"## Hybrid Analysis — {len(all_issues)} Issue(s) Found\n"]
    lines.append(f"**Pattern matches:** {len(pattern_issues)} &nbsp;|&nbsp; **LLM discoveries:** {len(llm_issues)} &nbsp;|&nbsp; **Lines scanned:** {total_lines}\n")

    for i, iss in enumerate(all_issues, 1):
        icon = ISSUE_ICONS.get(iss.get("issue_type", "bug"), "❓")
        color = SEVERITY_COLOR.get(iss.get("severity", "medium"), "#888")
        src = "🔍 Pattern" if iss.get("source") != "llm" else "🤖 LLM"
        fix = iss.get("fix", "")
        fix_line = f"\n  - 💡 **Fix:** `{fix}`" if fix else ""
        lines.append(f"**{i}. {icon} Line {iss.get('line_number', '?')}** — <span style='color:{color}'>[{iss.get('severity', '?').upper()}]</span> {iss.get('issue_type', '?').upper()}")
        lines.append(f"  - {iss.get('description', 'No description')}{fix_line}")
        lines.append(f"  - _Source: {src}_\n")

    return "\n".join(lines)


def _build_score_card(all_issues, pattern_issues, total_lines):
    n = len(all_issues)
    sev_counts = {}
    for iss in all_issues:
        s = iss.get("severity", "medium")
        sev_counts[s] = sev_counts.get(s, 0) + 1

    risk_score = min(100, sev_counts.get("critical", 0) * 40 + sev_counts.get("high", 0) * 25 + sev_counts.get("medium", 0) * 10 + sev_counts.get("low", 0) * 3)
    health = max(0, 100 - risk_score)

    badge = "🟢" if health >= 70 else ("🟡" if health >= 40 else "🔴")

    lines = [
        f"## {badge} Code Health: **{health}/100**\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total Issues | {n} |",
        f"| Lines Scanned | {total_lines} |",
    ]
    for sev in ["critical", "high", "medium", "low"]:
        c = sev_counts.get(sev, 0)
        if c:
            color = SEVERITY_COLOR[sev]
            lines.append(f"| <span style='color:{color}'>{sev.upper()}</span> | {c} |")

    lines.append(f"\n### Risk Assessment\n")
    if health >= 70:
        lines.append("The code looks solid. Minor issues, if any, are low severity.")
    elif health >= 40:
        lines.append("Moderate risk. Several issues should be addressed before deployment.")
    else:
        lines.append("**High risk!** Critical or high-severity issues detected. Do not deploy without fixes.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. ADVERSARIAL CODE GENERATOR
# ---------------------------------------------------------------------------

BUG_CATEGORIES = [
    "SQL injection", "XSS (cross-site scripting)", "race condition",
    "memory leak", "use-after-free", "null pointer dereference",
    "buffer overflow", "path traversal", "insecure deserialization",
    "hardcoded secrets", "TOCTOU race", "deadlock", "resource leak",
    "integer overflow", "format string vulnerability", "command injection",
    "missing authentication check", "improper input validation",
    "dict mutation during iteration", "mutable default argument",
]

DIFFICULTY_PROMPTS = {
    "Easy (1 issue)": "Generate a short Python code snippet (20-40 lines) with EXACTLY 1 subtle bug. Include enough safe/normal code to be a realistic distractor.",
    "Medium (2 issues)": "Generate a Python code snippet (40-70 lines) with EXACTLY 2 distinct bugs. Include realistic safe code patterns as distractors.",
    "Hard (3+ issues)": "Generate a Python code snippet (60-100 lines) with 3 or more distinct bugs of varying types. Include sophisticated safe code patterns that look similar to the bugs.",
}


def generate_adversarial(difficulty: str, category_hint: str):
    if CLIENT is None:
        yield "No LLM client configured. Set GROQ_API_KEY in .env to use the generator.", ""
        return

    desc = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["Easy (1 issue)"])
    cat_line = f"\nFocus on: {category_hint}." if category_hint and category_hint != "Random" else ""

    prompt = f"""You are a code challenge generator for an AI training environment.

{desc}{cat_line}

Output ONLY valid JSON (no markdown fence):
{{"code": "<the python code>", "bugs": [{{"type": "bug|security|performance", "line": <int>, "description": "<what the bug is>", "severity": "critical|high|medium|low"}}]}}

Make the bugs subtle — they should be hard to spot for a junior developer but obvious to a senior engineer."""

    yield "Generating adversarial code sample...", ""
    try:
        raw = _llm_chat(prompt, temperature=0.8, max_tokens=1200)
        data = _parse_json_block(raw)
        if not data or "code" not in data:
            yield f"Failed to parse LLM output. Raw:\n\n{raw}", ""
            return

        code = data["code"]
        bugs = data.get("bugs", [])

        # Format code with line numbers
        numbered = "\n".join(f"{i+1:3d} | {line}" for i, line in enumerate(code.split("\n")))

        bug_report = f"## Generated Code ({len(code.split(chr(10)))} lines, {len(bugs)} hidden bug(s))\n\n"
        bug_report += f"```python\n{numbered}\n```\n\n"
        bug_report += "---\n\n## Hidden Bugs (answer key)\n\n"
        for i, b in enumerate(bugs, 1):
            icon = ISSUE_ICONS.get(b.get("type", "bug"), "❓")
            color = SEVERITY_COLOR.get(b.get("severity", "medium"), "#888")
            bug_report += f"{i}. {icon} **Line {b.get('line', '?')}** — <span style='color:{color}'>[{b.get('severity', '?').upper()}]</span> {b.get('description', 'No description')}\n"

        yield bug_report, code
    except Exception as e:
        yield f"Error: {e}", ""


# ---------------------------------------------------------------------------
# 3. DUO-AGENT DEBATE
# ---------------------------------------------------------------------------

def duo_debate(code: str):
    """Pattern engine vs LLM — show agreement / disagreement on issues."""
    if not code or not code.strip():
        yield "Paste code for the agents to debate.", ""
        return

    lines = code.split("\n")
    total_lines = len(lines)

    # Agent A: Pattern engine
    pattern_issues = pattern_scan(code)
    pattern_set = {(i["issue_type"], i["line_number"]) for i in pattern_issues}

    yield _debate_round1(pattern_issues, total_lines), ""

    # Agent B: LLM
    if CLIENT is None:
        yield _debate_round1(pattern_issues, total_lines), _debate_summary(pattern_issues, [], total_lines)
        return

    prompt = f"""You are a meticulous code reviewer. Analyze this Python code for ALL bugs, security issues, and performance problems.

Return a JSON array. Each element:
{{"type": "bug|security|performance|style|logic", "line": <int>, "description": "<what's wrong>", "severity": "critical|high|medium|low", "confidence": 0.0-1.0}}

Be thorough. Include your confidence level for each issue.

CODE ({total_lines} lines):
```python
{code}
```"""

    try:
        raw = _llm_chat(prompt, temperature=0.1, max_tokens=800)
        llm_issues = _parse_json_block(raw)
        if not isinstance(llm_issues, list):
            llm_issues = []
    except Exception as e:
        llm_issues = []

    llm_set = {(i.get("type", ""), i.get("line", 0)) for i in llm_issues}

    yield _debate_round2(pattern_issues, llm_issues, total_lines), _debate_summary_duo(pattern_issues, llm_issues, total_lines)


def _debate_round1(pattern_issues, total_lines):
    lines = [f"## Duo-Agent Debate — Round 1\n"]
    lines.append(f"### Agent A: Pattern Engine (regex-based)\n")
    lines.append(f"Scanned **{total_lines}** lines.\n")
    if pattern_issues:
        for iss in pattern_issues:
            icon = ISSUE_ICONS.get(iss["issue_type"], "❓")
            lines.append(f"- {icon} **Line {iss['line_number']}** [{iss['severity'].upper()}] {iss['description']}")
    else:
        lines.append("_No issues detected by pattern matching._")
    lines.append("\n⏳ Waiting for Agent B (LLM)...")
    return "\n".join(lines)


def _debate_round2(pattern_issues, llm_issues, total_lines):
    lines = [f"## Duo-Agent Debate — Complete\n"]
    lines.append(f"### Agent A: Pattern Engine ({len(pattern_issues)} issues)\n")
    for iss in pattern_issues:
        icon = ISSUE_ICONS.get(iss["issue_type"], "❓")
        lines.append(f"- {icon} **Line {iss['line_number']}** [{iss['severity'].upper()}] {iss['description']}")

    lines.append(f"\n### Agent B: LLM ({len(llm_issues)} issues)\n")
    for iss in llm_issues:
        icon = ISSUE_ICONS.get(iss.get("type", "bug"), "❓")
        conf = iss.get("confidence", "?")
        conf_str = f"{conf:.0%}" if isinstance(conf, (int, float)) else str(conf)
        lines.append(f"- {icon} **Line {iss.get('line', '?')}** [{iss.get('severity', '?').upper()}] {iss.get('description', '')} — Confidence: **{conf_str}**")

    return "\n".join(lines)


def _debate_summary(pattern_issues, _, total_lines):
    return f"## Debate Summary\n\nOnly Pattern Agent available (no LLM configured). Found **{len(pattern_issues)}** issues in **{total_lines}** lines.\n\n_Set GROQ_API_KEY in .env to enable the Duo-Agent Debate._"


def _debate_summary_duo(pattern_issues, llm_issues, total_lines):
    # Find agreement / disagreement
    pat_set = {(i["issue_type"], i["line_number"]) for i in pattern_issues}
    llm_set = {(i.get("type", ""), i.get("line", 0)) for i in llm_issues}

    agreed = set()
    for ps in pat_set:
        for ls in llm_set:
            if ps[0] == ls[0] and abs(ps[1] - ls[1]) <= 2:
                agreed.add(ps)
                break

    pat_only = pat_set - agreed
    llm_only_raw = llm_set - {(a[0], a[1]) for a in agreed}
    # fuzzy: remove llm_only if close to agreed
    llm_only = set()
    for ls in llm_only_raw:
        if not any(ls[0] == a[0] and abs(ls[1] - a[1]) <= 2 for a in agreed):
            llm_only.add(ls)

    lines = [
        "## Debate Summary\n",
        f"| Metric | Count |",
        f"|--------|-------|",
        f"| Total lines | {total_lines} |",
        f"| Pattern issues | {len(pattern_issues)} |",
        f"| LLM issues | {len(llm_issues)} |",
        f"| **Both agree** | **{len(agreed)}** |",
        f"| Pattern only | {len(pat_only)} |",
        f"| LLM only | {len(llm_only)} |",
        "",
    ]

    if len(agreed) == 0 and (len(pattern_issues) > 0 or len(llm_issues) > 0):
        lines.append("**Agents disagree entirely!** This is a low-confidence scenario. Manual review recommended.")
    elif len(agreed) > 0:
        pct = len(agreed) / max(len(pattern_issues), len(llm_issues), 1) * 100
        lines.append(f"**Agreement level: {pct:.0f}%** — {'High' if pct >= 70 else ('Moderate' if pct >= 40 else 'Low')} confidence in results.")

    if pat_only:
        lines.append("\n**Issues only Pattern Agent found:**")
        for t, l in pat_only:
            lines.append(f"- {ISSUE_ICONS.get(t, '❓')} Line {l} ({t})")

    if llm_only:
        lines.append("\n**Issues only LLM Agent found:**")
        for t, l in llm_only:
            lines.append(f"- {ISSUE_ICONS.get(t, '❓')} Line {l} ({t})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. TASK ARENA
# ---------------------------------------------------------------------------

def run_task_arena(task_id: str):
    """Run a task through the hybrid agent with step-by-step output."""
    if not task_id:
        yield "Select a task."
        return

    task = TASKS.get(task_id)
    if not task:
        yield f"Unknown task: {task_id}"
        return

    # Show task
    code = task["code"]
    numbered = "\n".join(f"{i+1:3d} | {line}" for i, line in enumerate(code.split("\n")))
    header = f"## Task: {task_id.replace('_', ' ').title()}\n\n"
    header += f"**Difficulty:** {task['difficulty'].upper()} &nbsp;|&nbsp; **Expected issues:** {len(task['issues'])}\n\n"
    header += f"**Instructions:** {task['description']}\n\n"
    header += f"```python\n{numbered}\n```\n\n"
    header += "---\n\n## Running Hybrid Agent...\n"

    yield header

    env = CodeReviewEnv()
    obs = env.reset(task_id=task_id)
    done = False
    total_reward = 0.0
    detected = set()
    step_log = []

    # Phase 1: Patterns
    pattern_issues = pattern_scan(code)
    for iss in pattern_issues:
        if done:
            break
        key = (iss["issue_type"], iss["line_number"])
        if key in detected:
            continue
        detected.add(key)
        action = Action(**iss)
        obs, reward, done, info = env.step(action)
        total_reward += reward.value
        status = "✅ TRUE POSITIVE" if reward.value > 0 else "❌ FALSE POSITIVE"
        step_log.append(f"Step {obs.step_count}: 🔍 Pattern → **Line {iss['line_number']}** [{iss['severity'].upper()}] {iss['issue_type']} — {status} ({reward.value:+.2f})")

    phase1_report = header + "\n### Phase 1: Pattern Matching\n\n" + "\n".join(step_log) + "\n\n"
    yield phase1_report

    # Phase 2: LLM (if available and not done)
    if CLIENT is not None and not done:
        for attempt in range(5):
            if done:
                break
            found_list = [f"Line {h['issue']['line']}: {h['issue']['type']}" for h in obs.review_history if h['action'] == 'identify_issue' and h.get('valid')]
            found_text = "\n".join(found_list) if found_list else "None"

            prompt = f"""Code review. Find ONE remaining issue or approve.

CODE:
```python
{obs.code}
```

ALREADY FOUND:
{found_text}

Output JSON only (no markdown):
{{"action_type": "identify_issue", "issue_type": "bug", "line_number": 42, "description": "issue here", "severity": "high"}}

OR:
{{"action_type": "approve"}}"""

            try:
                raw = _llm_chat(prompt, temperature=0.0, max_tokens=200)
                if "```" in raw:
                    raw = raw.split("```")[1].replace("json", "").strip()
                adict = json.loads(raw)
                action = Action(**adict)
                if action.action_type == "identify_issue":
                    key = (action.issue_type, action.line_number)
                    if key in detected:
                        action = Action(action_type="approve")
                    else:
                        detected.add(key)
            except Exception:
                action = Action(action_type="approve")

            obs, reward, done, info = env.step(action)
            total_reward += reward.value
            if action.action_type == "identify_issue":
                status = "✅ TRUE POSITIVE" if reward.value > 0 else "❌ FALSE POSITIVE"
                step_log.append(f"Step {obs.step_count}: 🤖 LLM → **Line {action.line_number}** [{(action.severity or '?').upper()}] {action.issue_type} — {status} ({reward.value:+.2f})")
            else:
                step_log.append(f"Step {obs.step_count}: 🤖 LLM → **APPROVE** ({reward.value:+.2f})")

            current_report = phase1_report + "\n### Phase 2: LLM Deep Analysis\n\n" + "\n".join(step_log[len(pattern_issues):]) + "\n\n"
            yield current_report

    # Auto-approve if not done
    if not done:
        obs, reward, done, info = env.step(Action(action_type="approve"))
        total_reward += reward.value
        step_log.append(f"Step {obs.step_count}: ⏩ Auto-APPROVE ({reward.value:+.2f})")

    # Final report
    final = header + "\n## Step-by-Step Review Log\n\n" + "\n".join(step_log) + "\n\n"
    final += "---\n\n## Final Score\n\n"
    final += f"| Metric | Value |\n|--------|-------|\n"
    final += f"| Score | **{info['score']:.3f}** |\n"
    final += f"| Found | {info['found_issues']}/{info['expected_issues']} |\n"
    final += f"| False Positives | {info['false_positives']} |\n"
    final += f"| Recall | {info['recall']:.2%} |\n"
    final += f"| Precision | {info['precision']:.2%} |\n"
    final += f"| F1 | {info['f1']:.2%} |\n"
    final += f"| Total Reward | {total_reward:+.2f} |\n"
    final += f"| Steps | {obs.step_count} |\n"

    # Expected issues reference
    final += "\n### Expected Issues (ground truth)\n\n"
    for iss in task["issues"]:
        icon = ISSUE_ICONS.get(iss["type"], "❓")
        color = SEVERITY_COLOR.get(iss["severity"], "#888")
        final += f"- {icon} **Line {iss['line']}** <span style='color:{color}'>[{iss['severity'].upper()}]</span> {iss['description']}\n"

    yield final


# ---------------------------------------------------------------------------
# GRADIO UI — Premium Dashboard
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
/* ===== ROOT & GLOBAL RESETS ===== */
:root {
    --cr-bg-primary: #f8fafc;
    --cr-bg-secondary: #ffffff;
    --cr-bg-card: rgba(255, 255, 255, 0.85);
    --cr-border: rgba(148, 163, 184, 0.25);
    --cr-border-hover: rgba(99, 102, 241, 0.4);
    --cr-text-primary: #0f172a;
    --cr-text-secondary: #475569;
    --cr-text-muted: #94a3b8;
    --cr-accent-1: #6366f1;
    --cr-accent-2: #8b5cf6;
    --cr-accent-3: #06b6d4;
    --cr-success: #10b981;
    --cr-warning: #f59e0b;
    --cr-danger: #ef4444;
    --cr-gradient-1: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    --cr-gradient-2: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 50%, #ede9fe 100%);
    --cr-shadow-sm: 0 1px 3px rgba(0,0,0,0.04), 0 1px 2px rgba(0,0,0,0.06);
    --cr-shadow-md: 0 4px 12px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.04);
    --cr-shadow-lg: 0 12px 40px rgba(0,0,0,0.08), 0 4px 12px rgba(0,0,0,0.04);
    --cr-shadow-glow: 0 0 30px rgba(99, 102, 241, 0.15);
    --cr-radius: 16px;
    --cr-radius-sm: 10px;
    --cr-radius-xs: 6px;
}

.gradio-container {
    max-width: 1320px !important;
    margin: auto !important;
    padding: 0 16px !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* ===== HERO HEADER ===== */
.cr-hero {
    position: relative;
    background: var(--cr-gradient-1);
    border-radius: var(--cr-radius);
    padding: 40px 32px 32px;
    margin: 16px 0 24px;
    color: white;
    text-align: center;
    overflow: hidden;
}

.cr-hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(255,255,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(6,182,212,0.15) 0%, transparent 40%);
    animation: heroShimmer 8s ease-in-out infinite alternate;
    pointer-events: none;
}

@keyframes heroShimmer {
    0% { transform: translate(0, 0) rotate(0deg); }
    100% { transform: translate(-5%, 5%) rotate(3deg); }
}

.cr-hero h1 {
    margin: 0 0 8px;
    font-size: 2.2em;
    font-weight: 800;
    letter-spacing: -0.03em;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    position: relative;
    z-index: 1;
}

.cr-hero .subtitle {
    font-size: 1.05em;
    opacity: 0.9;
    font-weight: 400;
    margin: 0 0 20px;
    position: relative;
    z-index: 1;
}

/* Status pills in hero */
.cr-status-row {
    display: flex;
    justify-content: center;
    gap: 12px;
    flex-wrap: wrap;
    position: relative;
    z-index: 1;
}

.cr-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 100px;
    padding: 6px 16px;
    font-size: 0.82em;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.3s ease;
}

.cr-pill:hover {
    background: rgba(255,255,255,0.25);
    transform: translateY(-1px);
}

.cr-pill .dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    display: inline-block;
}

.cr-pill .dot.green { background: #34d399; box-shadow: 0 0 6px #34d399; }
.cr-pill .dot.blue { background: #60a5fa; box-shadow: 0 0 6px #60a5fa; }
.cr-pill .dot.amber { background: #fbbf24; box-shadow: 0 0 6px #fbbf24; }

/* ===== METRIC CARDS ===== */
.cr-metrics-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin: 0 0 24px;
}

.cr-metric-card {
    background: var(--cr-bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--cr-border);
    border-radius: var(--cr-radius-sm);
    padding: 18px 20px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.cr-metric-card::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: var(--cr-gradient-1);
    opacity: 0;
    transition: opacity 0.3s ease;
}

.cr-metric-card:hover {
    border-color: var(--cr-border-hover);
    box-shadow: var(--cr-shadow-md), var(--cr-shadow-glow);
    transform: translateY(-2px);
}

.cr-metric-card:hover::after {
    opacity: 1;
}

.cr-metric-icon {
    font-size: 1.5em;
    margin-bottom: 6px;
}

.cr-metric-value {
    font-size: 1.6em;
    font-weight: 800;
    color: var(--cr-text-primary);
    letter-spacing: -0.02em;
    line-height: 1.2;
}

.cr-metric-label {
    font-size: 0.75em;
    color: var(--cr-text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    margin-top: 2px;
}

/* ===== TAB STYLING ===== */
.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.92em !important;
    padding: 12px 20px !important;
    border-radius: var(--cr-radius-xs) var(--cr-radius-xs) 0 0 !important;
    transition: all 0.2s ease !important;
    position: relative !important;
}

.tab-nav button.selected {
    background: var(--cr-bg-secondary) !important;
    color: var(--cr-accent-1) !important;
    box-shadow: 0 -2px 8px rgba(99, 102, 241, 0.1) !important;
}

.tab-nav button.selected::after {
    content: '' !important;
    position: absolute !important;
    bottom: 0 !important;
    left: 10% !important;
    right: 10% !important;
    height: 3px !important;
    background: var(--cr-gradient-1) !important;
    border-radius: 3px 3px 0 0 !important;
}

/* ===== CARD CONTAINERS ===== */
.cr-card {
    background: var(--cr-bg-card);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--cr-border);
    border-radius: var(--cr-radius);
    padding: 24px;
    margin-bottom: 16px;
    transition: all 0.3s ease;
}

.cr-card:hover {
    border-color: var(--cr-border-hover);
    box-shadow: var(--cr-shadow-md);
}

.cr-card-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--cr-border);
}

.cr-card-header h3 {
    margin: 0;
    font-size: 1.1em;
    font-weight: 700;
    color: var(--cr-text-primary);
}

.cr-card-header .badge {
    font-size: 0.7em;
    padding: 3px 10px;
    border-radius: 100px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.badge-accent {
    background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(139,92,246,0.12));
    color: var(--cr-accent-1);
    border: 1px solid rgba(99,102,241,0.2);
}

.badge-success {
    background: rgba(16,185,129,0.1);
    color: var(--cr-success);
    border: 1px solid rgba(16,185,129,0.2);
}

.badge-warning {
    background: rgba(245,158,11,0.1);
    color: var(--cr-warning);
    border: 1px solid rgba(245,158,11,0.2);
}

/* ===== BUTTONS ===== */
button.primary {
    background: var(--cr-gradient-1) !important;
    border: none !important;
    font-weight: 700 !important;
    letter-spacing: 0.02em !important;
    border-radius: var(--cr-radius-xs) !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.3) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

button.primary:active {
    transform: translateY(0) !important;
}

/* ===== CODE EDITOR ===== */
.code-editor {
    border-radius: var(--cr-radius-sm) !important;
    border: 1px solid var(--cr-border) !important;
    overflow: hidden !important;
    transition: border-color 0.3s ease !important;
}

.code-editor:focus-within {
    border-color: var(--cr-accent-1) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* ===== DROPDOWNS ===== */
select, .dropdown {
    border-radius: var(--cr-radius-xs) !important;
    border: 1px solid var(--cr-border) !important;
    transition: all 0.2s ease !important;
}

select:focus, .dropdown:focus {
    border-color: var(--cr-accent-1) !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
}

/* ===== TAB INTRO SECTIONS ===== */
.cr-tab-intro {
    background: var(--cr-gradient-2);
    border: 1px solid var(--cr-border);
    border-radius: var(--cr-radius-sm);
    padding: 18px 22px;
    margin-bottom: 20px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
}

.cr-tab-intro .intro-icon {
    font-size: 2em;
    line-height: 1;
    flex-shrink: 0;
}

.cr-tab-intro .intro-text h4 {
    margin: 0 0 4px;
    font-size: 1em;
    font-weight: 700;
    color: var(--cr-text-primary);
}

.cr-tab-intro .intro-text p {
    margin: 0;
    font-size: 0.88em;
    color: var(--cr-text-secondary);
    line-height: 1.5;
}

/* ===== RESULT PANEL ===== */
.cr-result-panel {
    background: var(--cr-bg-card);
    border: 1px solid var(--cr-border);
    border-radius: var(--cr-radius-sm);
    padding: 20px;
    min-height: 120px;
    transition: all 0.3s ease;
}

/* ===== TASK ARENA SPECIFIC ===== */
.cr-task-card {
    background: var(--cr-bg-card);
    border: 1px solid var(--cr-border);
    border-radius: var(--cr-radius-sm);
    padding: 16px 20px;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: all 0.3s ease;
    cursor: default;
}

.cr-task-card:hover {
    border-color: var(--cr-border-hover);
    box-shadow: var(--cr-shadow-sm);
}

.cr-task-diff {
    font-size: 0.7em;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 4px 10px;
    border-radius: 100px;
    white-space: nowrap;
}

.diff-easy { background: rgba(16,185,129,0.12); color: #059669; border: 1px solid rgba(16,185,129,0.25); }
.diff-medium { background: rgba(245,158,11,0.12); color: #d97706; border: 1px solid rgba(245,158,11,0.25); }
.diff-hard { background: rgba(239,68,68,0.12); color: #dc2626; border: 1px solid rgba(239,68,68,0.25); }

/* ===== FOOTER ===== */
.cr-footer {
    text-align: center;
    padding: 24px 16px 16px;
    color: var(--cr-text-muted);
    font-size: 0.8em;
    border-top: 1px solid var(--cr-border);
    margin-top: 24px;
}

.cr-footer a {
    color: var(--cr-accent-1);
    text-decoration: none;
    font-weight: 600;
}

/* ===== ANIMATIONS ===== */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}

.cr-animate-in {
    animation: fadeInUp 0.5s ease-out;
}

/* ===== RESPONSIVE ===== */
@media (max-width: 768px) {
    .cr-metrics-row {
        grid-template-columns: repeat(2, 1fr);
    }
    .cr-hero h1 { font-size: 1.6em; }
    .cr-status-row { gap: 8px; }
    .cr-pill { padding: 5px 12px; font-size: 0.75em; }
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.3);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: rgba(148, 163, 184, 0.5); }

/* ===== DARK MODE OVERRIDES ===== */
.dark {
    --cr-bg-primary: #0f172a;
    --cr-bg-secondary: #1e293b;
    --cr-bg-card: rgba(30, 41, 59, 0.85);
    --cr-border: rgba(71, 85, 105, 0.4);
    --cr-border-hover: rgba(99, 102, 241, 0.5);
    --cr-text-primary: #f1f5f9;
    --cr-text-secondary: #94a3b8;
    --cr-text-muted: #64748b;
}

.dark .cr-hero {
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
"""

# Build the custom theme
_custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="cyan",
    neutral_hue="slate",
).set(
    button_primary_background_fill="linear-gradient(135deg, #6366f1, #8b5cf6)",
    button_primary_background_fill_hover="linear-gradient(135deg, #4f46e5, #7c3aed)",
    button_primary_text_color="white",
    button_primary_shadow="*shadow_spread",
    block_background_fill="rgba(255, 255, 255, 0.85)",
    block_border_color="rgba(148, 163, 184, 0.25)",
    block_label_text_color="#475569",
    input_border_color="rgba(148, 163, 184, 0.3)",
    input_border_color_focus="#6366f1",
    color_accent="#6366f1",
    link_text_color="#6366f1",
)

with gr.Blocks(
    title="CodeCrack — AI Code Review Dashboard",
) as demo:

    # ── HERO HEADER ───────────────────────────────────────────────────────
    gr.HTML(f"""
    <div class="header-banner">
        <h1>⚡ CodeCrack</h1>
        <p>AI-Powered Code Review Dashboard — Scaler Meta PyTorch Hackathon</p>
        <p style="font-size: 0.85em; opacity: 0.7;">Provider: {prov} &nbsp;|&nbsp; Model: {model}</p>
    </div>
    """)

    # ── METRIC CARDS ──────────────────────────────────────────────────────
    gr.HTML("""
    <div class="cr-metrics-row cr-animate-in">
        <div class="cr-metric-card">
            <div class="cr-metric-icon">🔍</div>
            <div class="cr-metric-value">Hybrid</div>
            <div class="cr-metric-label">Detection Engine</div>
        </div>
        <div class="cr-metric-card">
            <div class="cr-metric-icon">🎯</div>
            <div class="cr-metric-value">0.878</div>
            <div class="cr-metric-label">Baseline Score</div>
        </div>
        <div class="cr-metric-card">
            <div class="cr-metric-icon">⚡</div>
            <div class="cr-metric-value">~8s</div>
            <div class="cr-metric-label">Avg Inference</div>
        </div>
        <div class="cr-metric-card">
            <div class="cr-metric-icon">🛡️</div>
            <div class="cr-metric-value">3</div>
            <div class="cr-metric-label">Task Difficulty Tiers</div>
        </div>
    </div>
    """)

    # ── MAIN TABS ─────────────────────────────────────────────────────────
    with gr.Tabs() as tabs:

        # ---- Tab 1: Meta-Review Playground ----
        with gr.Tab("🔬 Meta-Review"):
            gr.HTML("""
            <div class="cr-tab-intro">
                <div class="intro-icon">🔬</div>
                <div class="intro-text">
                    <h4>Meta-Review Playground</h4>
                    <p>Paste any Python code and watch the <strong>Hybrid Agent</strong> — regex patterns + LLM deep analysis — find bugs, security vulnerabilities, and performance issues in real-time.</p>
                </div>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=5):
                    review_input = gr.Code(
                        language="python",
                        label="📝 Paste Your Code",
                        lines=22,
                        value="""import sqlite3

def get_user(conn, username):
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

def log(msg):
    print(f"[LOG] {msg}")
""",
                    )
                    with gr.Row():
                        review_btn = gr.Button("🚀 Analyze Code", variant="primary", size="lg")
                with gr.Column(scale=3):
                    review_score = gr.Markdown(label="Code Health Score", value="*Paste code and click Analyze to see results.*")
            review_output = gr.Markdown(label="Full Analysis Report")

            review_btn.click(
                fn=meta_review,
                inputs=[review_input],
                outputs=[review_output, review_score],
            )

        # ---- Tab 2: Adversarial Code Generator ----
        with gr.Tab("🧪 Adversarial Gen"):
            gr.HTML("""
            <div class="cr-tab-intro">
                <div class="intro-icon">🧪</div>
                <div class="intro-text">
                    <h4>Adversarial Code Generator</h4>
                    <p>Generate unique buggy code snippets on-the-fly for training AI agents. Creates <strong>infinite training data</strong> with configurable difficulty and bug categories.</p>
                </div>
            </div>
            """)
            with gr.Row():
                adv_difficulty = gr.Dropdown(
                    choices=["Easy (1 issue)", "Medium (2 issues)", "Hard (3+ issues)"],
                    value="Medium (2 issues)",
                    label="Difficulty",
                    scale=1,
                )
                adv_category = gr.Dropdown(
                    choices=["Random"] + BUG_CATEGORIES,
                    value="Random",
                    label="Bug Category",
                    scale=2,
                )
            adv_btn = gr.Button("🧪 Generate Adversarial Code", variant="primary", size="lg")
            adv_output = gr.Markdown(label="Generated Code & Bug Key")
            adv_code = gr.Code(language="python", label="Generated Code (copy-pasteable)", interactive=True)

            adv_btn.click(
                fn=generate_adversarial,
                inputs=[adv_difficulty, adv_category],
                outputs=[adv_output, adv_code],
            )

        # ---- Tab 3: Duo-Agent Debate ----
        with gr.Tab("⚔️ Agent Debate"):
            gr.HTML("""
            <div class="cr-tab-intro">
                <div class="intro-icon">⚔️</div>
                <div class="intro-text">
                    <h4>Duo-Agent Debate</h4>
                    <p>Two agents independently review the same code — a <strong>regex pattern engine</strong> and an <strong>LLM</strong> — then we compare findings and show agreement levels with confidence scores.</p>
                </div>
            </div>
            """)
            with gr.Row():
                with gr.Column(scale=5):
                    debate_input = gr.Code(
                        language="python",
                        label="Code for Debate",
                        lines=22,
                        value="""import threading

class Counter:
    def __init__(self):
        self.count = 0
        self._lock = threading.Lock()

    def increment(self):
        val = self.count
        val += 1
        self.count = val

    def safe_increment(self):
        with self._lock:
            self.count += 1
""",
                    )
                    debate_btn = gr.Button("⚔️ Start Debate", variant="primary", size="lg")
                with gr.Column(scale=3):
                    debate_summary = gr.Markdown(label="Debate Summary & Confidence", value="*Paste code and click Start Debate.*")
            debate_output = gr.Markdown(label="Full Debate Transcript")

            debate_btn.click(
                fn=duo_debate,
                inputs=[debate_input],
                outputs=[debate_output, debate_summary],
            )

        # ---- Tab 4: Task Arena ----
        with gr.Tab("🏆 Task Arena"):
            gr.HTML("""
            <div class="cr-tab-intro">
                <div class="intro-icon">🏆</div>
                <div class="intro-text">
                    <h4>Task Arena — Official Benchmark</h4>
                    <p>Run the official benchmark tasks and watch the hybrid agent solve them step-by-step with live scoring, recall/precision metrics, and detailed reward breakdowns.</p>
                </div>
            </div>
            """)

            # Task overview cards
            gr.HTML("""
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px;">
                <div class="cr-task-card">
                    <span class="cr-task-diff diff-easy">EASY</span>
                    <div>
                        <div style="font-weight:700; font-size:0.9em; color:var(--cr-text-primary);">SQL Injection</div>
                        <div style="font-size:0.78em; color:var(--cr-text-muted);">1 issue · 26 lines</div>
                    </div>
                </div>
                <div class="cr-task-card">
                    <span class="cr-task-diff diff-medium">MEDIUM</span>
                    <div>
                        <div style="font-weight:700; font-size:0.9em; color:var(--cr-text-primary);">Race Condition</div>
                        <div style="font-size:0.78em; color:var(--cr-text-muted);">2 issues · 41 lines</div>
                    </div>
                </div>
                <div class="cr-task-card">
                    <span class="cr-task-diff diff-hard">HARD</span>
                    <div>
                        <div style="font-weight:700; font-size:0.9em; color:var(--cr-text-primary);">Memory Leak</div>
                        <div style="font-size:0.78em; color:var(--cr-text-muted);">3 issues · 45 lines</div>
                    </div>
                </div>
            </div>
            """)

            with gr.Row():
                task_selector = gr.Dropdown(
                    choices=list(TASKS.keys()),
                    value="easy_sql_injection",
                    label="Select Task",
                    scale=3,
                )
                task_btn = gr.Button("▶️ Run Agent", variant="primary", size="lg", scale=1)
            task_output = gr.Markdown(label="Agent Execution Log")

            task_btn.click(
                fn=run_task_arena,
                inputs=[task_selector],
                outputs=[task_output],
            )

    # ── FOOTER ────────────────────────────────────────────────────────────
    gr.HTML("""
    <div class="cr-footer">
        <strong>CodeCrack</strong> — OpenEnv-compliant RL training environment for AI code review agents.<br>
        Hybrid strategy: fast regex pattern matching + LLM fallback.
        Built for the <a href="#">Scaler Meta PyTorch Hackathon 2025</a>.
    </div>
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=_custom_theme, css=_CUSTOM_CSS)
