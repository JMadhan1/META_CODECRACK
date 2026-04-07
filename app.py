"""
CodeCrack — AI Code Review Environment
"""

import os
import json
import gradio as gr
from fastapi import FastAPI, HTTPException
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

from environment import CodeReviewEnv
from models import Action
from tasks import TASKS
from inference import pattern_scan

load_dotenv()

# ---------------------------------------------------------------------------
# LLM client bootstrap
# ---------------------------------------------------------------------------

def _get_client():
    api_base     = os.getenv("API_BASE_URL")
    model_name   = os.getenv("MODEL_NAME")
    hf_token     = os.getenv("HF_TOKEN")
    groq_key     = os.getenv("GROQ_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")

    if api_base and model_name and hf_token:
        return OpenAI(api_key=hf_token, base_url=api_base), model_name, "HF/Custom"
    elif groq_key:
        return OpenAI(api_key=groq_key,
                      base_url=os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")),\
               os.getenv("MODEL_NAME", "llama-3.3-70b-versatile"), "Groq"
    elif together_key:
        return OpenAI(api_key=together_key, base_url="https://api.together.xyz/v1"),\
               "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "Together"
    elif hf_token:
        return OpenAI(api_key=hf_token,
                      base_url="https://api-inference.huggingface.co/v1"),\
               "meta-llama/Llama-3.1-70B-Instruct", "HuggingFace"
    return None, None, "Pattern-only"

CLIENT, MODEL_NAME, PROVIDER = _get_client()
prov  = PROVIDER
model = MODEL_NAME or "N/A"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEV_COLOR  = {"critical": "#ef4444", "high": "#f97316", "medium": "#eab308", "low": "#22c55e"}
ISSUE_ICON = {"security": "🛡️", "bug": "🐛", "performance": "⚡", "style": "🎨", "logic": "🧠"}

BUG_CATEGORIES = [
    "SQL injection", "XSS", "race condition", "memory leak", "use-after-free",
    "null pointer dereference", "buffer overflow", "path traversal",
    "insecure deserialization", "hardcoded secrets", "TOCTOU race", "deadlock",
    "resource leak", "integer overflow", "format string vulnerability",
    "command injection", "missing authentication check", "improper input validation",
    "dict mutation during iteration", "mutable default argument",
]

DIFFICULTY_PROMPTS = {
    "Easy (1 issue)":   "Generate a short Python snippet (20-40 lines) with EXACTLY 1 subtle bug.",
    "Medium (2 issues)":"Generate a Python snippet (40-70 lines) with EXACTLY 2 distinct bugs.",
    "Hard (3+ issues)": "Generate a Python snippet (60-100 lines) with 3+ distinct bugs.",
}

# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

def _llm(prompt: str, temperature: float = 0.0, max_tokens: int = 800) -> str:
    if CLIENT is None:
        return '{"error": "No LLM configured."}'
    r = CLIENT.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature, max_tokens=max_tokens,
    )
    return r.choices[0].message.content.strip()

def _parse_json(text: str):
    if "```" in text:
        for p in text.split("```"):
            p = p.replace("json", "").strip()
            if p.startswith(("{", "[")):
                return json.loads(p)
    s = text.find("{"); b = text.find("[")
    if s == -1 and b == -1: return None
    idx = s if b == -1 or (s != -1 and s < b) else b
    return json.loads(text[idx:])

# ---------------------------------------------------------------------------
# 1. Code Review
# ---------------------------------------------------------------------------

def meta_review(code: str):
    if not code or not code.strip():
        yield "Paste some Python code above.", ""; return

    lines = code.split("\n"); n = len(lines)
    pats  = pattern_scan(code)
    yield _fmt_phase1(pats, n), ""

    if CLIENT is None:
        yield _fmt_phase1(pats, n), _scorecard(pats, n); return

    prompt = (f"Senior security engineer code review.\n"
              f"Return a JSON array of issues. Each: "
              f'[{{"type":"bug|security|performance|logic","line":<int>,"description":"...","severity":"critical|high|medium|low","fix":"..."}}]\n'
              f"If clean, return [].\n\nCODE ({n} lines):\n```python\n{code}\n```")
    try:
        llm_issues = _parse_json(_llm(prompt)) or []
        if not isinstance(llm_issues, list): llm_issues = []
    except Exception as e:
        llm_issues = [{"type":"error","line":0,"description":str(e),"severity":"low","fix":""}]

    merged = list(pats)
    for li in llm_issues:
        if not any(pi.get("issue_type") == li.get("type") and
                   abs(pi.get("line_number",0) - li.get("line",0)) <= 2 for pi in merged):
            merged.append({"issue_type": li.get("type","bug"), "line_number": li.get("line",0),
                           "description": li.get("description",""), "severity": li.get("severity","medium"),
                           "fix": li.get("fix",""), "source": "llm"})

    yield _fmt_full(merged, pats, llm_issues, n), _scorecard(merged, n)


def _fmt_phase1(issues, n):
    if not issues:
        return f"### Pattern Scan\n\nScanned **{n}** lines — no pattern-level issues.\n\n*LLM deep scan running...*"
    out = [f"### Pattern Scan\n\n**{len(issues)}** issue(s) in **{n}** lines:\n"]
    for i in issues:
        c = SEV_COLOR.get(i["severity"], "#888")
        out.append(f"- {ISSUE_ICON.get(i['issue_type'],'❓')} **Line {i['line_number']}** "
                   f"— <span style='color:{c}'>[{i['severity'].upper()}]</span> {i['description']}")
    out.append("\n*LLM deep scan running...*")
    return "\n".join(out)


def _fmt_full(all_i, pats, llm_i, n):
    out = [f"### Analysis Complete — {len(all_i)} Issue(s)\n",
           f"**Pattern:** {len(pats)} · **LLM:** {len(llm_i)} · **Lines scanned:** {n}\n"]
    for idx, i in enumerate(all_i, 1):
        c   = SEV_COLOR.get(i.get("severity","medium"), "#888")
        src = "Pattern" if i.get("source") != "llm" else "LLM"
        fix = f"\n  - **Fix:** `{i['fix']}`" if i.get("fix") else ""
        out += [
            f"**{idx}. {ISSUE_ICON.get(i.get('issue_type','bug'),'❓')} Line {i.get('line_number','?')}**"
            f" — <span style='color:{c}'>[{i.get('severity','?').upper()}]</span>"
            f" `{i.get('issue_type','?').upper()}`",
            f"  - {i.get('description','')}{fix}",
            f"  - *Detected by: {src}*\n",
        ]
    return "\n".join(out)


def _scorecard(issues, n):
    sv = {}
    for i in issues: sv[i.get("severity","medium")] = sv.get(i.get("severity","medium"),0) + 1
    risk   = min(100, sv.get("critical",0)*40 + sv.get("high",0)*25 + sv.get("medium",0)*10 + sv.get("low",0)*3)
    health = max(0, 100 - risk)
    status = "PASS" if health >= 70 else ("WARN" if health >= 40 else "FAIL")
    rows   = [f"### Risk Summary\n\n**Code Health: {health}/100** — `{status}`\n",
              "| Metric | Value |", "|--------|-------|",
              f"| Total Issues | {len(issues)} |", f"| Lines Scanned | {n} |"]
    for s in ["critical","high","medium","low"]:
        if sv.get(s):
            rows.append(f"| <span style='color:{SEV_COLOR[s]}'>{s.upper()}</span> | {sv[s]} |")
    rows.append("")
    rows.append("No critical issues found." if health >= 70 else
                "Moderate risk — review before deploying." if health >= 40 else
                "**High risk.** Critical issues need immediate attention.")
    return "\n".join(rows)

# ---------------------------------------------------------------------------
# 2. Adversarial Generator
# ---------------------------------------------------------------------------

def generate_adversarial(difficulty: str, category_hint: str):
    if CLIENT is None:
        yield "No LLM configured. Set `GROQ_API_KEY` in `.env`.", ""; return

    cat  = f"\nFocus: {category_hint}." if category_hint and category_hint != "Random" else ""
    desc = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["Easy (1 issue)"])
    prompt = (f"You are a code challenge generator.\n{desc}{cat}\n"
              f'Output ONLY valid JSON: {{"code":"<python>","bugs":[{{"type":"bug|security|performance",'
              f'"line":<int>,"description":"...","severity":"critical|high|medium|low"}}]}}\n'
              f"Make bugs subtle — obvious to a senior engineer.")

    yield "Generating...", ""
    try:
        data = _parse_json(_llm(prompt, temperature=0.8, max_tokens=1500))
        if not data or "code" not in data:
            yield "Parse failed. Try again.", ""; return
        code, bugs = data["code"], data.get("bugs", [])
        numbered   = "\n".join(f"{i+1:3d} | {l}" for i,l in enumerate(code.split("\n")))
        report = (f"### Generated Code — {len(code.splitlines())} lines, {len(bugs)} hidden bug(s)\n\n"
                  f"```python\n{numbered}\n```\n\n---\n\n### Bug Key\n\n")
        for k, b in enumerate(bugs, 1):
            c = SEV_COLOR.get(b.get("severity","medium"), "#888")
            report += (f"{k}. {ISSUE_ICON.get(b.get('type','bug'),'❓')} **Line {b.get('line','?')}**"
                       f" — <span style='color:{c}'>[{b.get('severity','?').upper()}]</span>"
                       f" {b.get('description','')}\n")
        yield report, code
    except Exception as e:
        yield f"Error: {e}", ""

# ---------------------------------------------------------------------------
# 3. Agent Debate
# ---------------------------------------------------------------------------

def duo_debate(code: str):
    if not code or not code.strip():
        yield "Paste code to review.", ""; return

    n    = len(code.split("\n"))
    pats = pattern_scan(code)
    yield _debate1(pats, n), ""

    if CLIENT is None:
        yield _debate1(pats, n), _debate_sum_simple(pats, n); return

    prompt = (f"Analyze ALL bugs, security issues, and performance problems.\n"
              f'Return JSON array: [{{"type":"...","line":<int>,"description":"...","severity":"...","confidence":0.0-1.0}}]\n'
              f"\nCODE ({n} lines):\n```python\n{code}\n```")
    try:
        llm_i = _parse_json(_llm(prompt, temperature=0.1)) or []
        if not isinstance(llm_i, list): llm_i = []
    except Exception:
        llm_i = []

    yield _debate2(pats, llm_i), _debate_sum(pats, llm_i, n)


def _debate1(pats, n):
    out = [f"### Agent A — Pattern Engine\n\nScanned **{n}** lines.\n"]
    if pats:
        for i in pats:
            out.append(f"- {ISSUE_ICON.get(i['issue_type'],'❓')} **Line {i['line_number']}**"
                       f" `[{i['severity'].upper()}]` {i['description']}")
    else:
        out.append("*No issues detected.*")
    out.append("\n*Waiting for Agent B...*")
    return "\n".join(out)


def _debate2(pats, llm_i):
    out = ["### Agent A — Pattern Engine\n"]
    for i in pats:
        out.append(f"- {ISSUE_ICON.get(i['issue_type'],'❓')} **Line {i['line_number']}**"
                   f" `[{i['severity'].upper()}]` {i['description']}")
    if not pats: out.append("*No issues.*")
    out.append("\n### Agent B — LLM\n")
    for i in llm_i:
        cf = i.get("confidence","?")
        out.append(f"- {ISSUE_ICON.get(i.get('type','bug'),'❓')} **Line {i.get('line','?')}**"
                   f" `[{i.get('severity','?').upper()}]` {i.get('description','')} — conf"
                   f" {f'{cf:.0%}' if isinstance(cf,(int,float)) else cf}")
    if not llm_i: out.append("*No issues.*")
    return "\n".join(out)


def _debate_sum_simple(pats, n):
    return (f"### Summary\n\nPattern engine only (no LLM). "
            f"Found **{len(pats)}** issue(s) in **{n}** lines.\n\n"
            f"Set `GROQ_API_KEY` to enable dual-agent comparison.")


def _debate_sum(pats, llm_i, n):
    ps  = {(i["issue_type"], i["line_number"]) for i in pats}
    ls  = {(i.get("type",""), i.get("line",0)) for i in llm_i}
    agg = {p for p in ps if any(p[0]==l[0] and abs(p[1]-l[1])<=2 for l in ls)}
    po  = ps - agg
    lo  = {l for l in ls if not any(l[0]==a[0] and abs(l[1]-a[1])<=2 for a in agg)}
    pct = len(agg)/max(len(ps),len(ls),1)*100
    out = ["### Agreement Report\n",
           "| Metric | Count |", "|--------|-------|",
           f"| Lines | {n} |", f"| Pattern | {len(pats)} |",
           f"| LLM | {len(llm_i)} |", f"| **Agree** | **{len(agg)}** |",
           f"| Pattern only | {len(po)} |", f"| LLM only | {len(lo)} |", "",
           f"Agreement: **{pct:.0f}%** ({'High' if pct>=70 else 'Moderate' if pct>=40 else 'Low'})"]
    if po: out += ["\n**Pattern only:**"] + [f"- Line {l} `{t}`" for t,l in po]
    if lo: out += ["\n**LLM only:**"]    + [f"- Line {l} `{t}`" for t,l in lo]
    return "\n".join(out)

# ---------------------------------------------------------------------------
# 4. Benchmark
# ---------------------------------------------------------------------------

def run_task_arena(task_id: str):
    if not task_id: yield "Select a task."; return
    task = TASKS.get(task_id)
    if not task: yield f"Unknown task: {task_id}"; return

    code     = task["code"]
    numbered = "\n".join(f"{i+1:3d} | {l}" for i,l in enumerate(code.split("\n")))
    hdr      = (f"### {task_id.replace('_',' ').title()}\n\n"
                f"**Difficulty:** `{task['difficulty'].upper()}` · "
                f"**Expected:** {len(task['issues'])} issue(s)\n\n"
                f"{task['description']}\n\n```python\n{numbered}\n```\n\n---\n")
    yield hdr + "\n### Running hybrid agent...\n"

    env  = CodeReviewEnv()
    obs  = env.reset(task_id=task_id)
    done = False; total_r = 0.0; seen = set(); log = []

    for iss in pattern_scan(code):
        if done: break
        key = (iss["issue_type"], iss["line_number"])
        if key in seen: continue
        seen.add(key)
        obs, r, done, info = env.step(Action(**iss))
        total_r += r.value
        tag = "TP" if r.value > 0 else "FP"
        log.append(f"Step {obs.step_count}: `PATTERN` → Line {iss['line_number']}"
                   f" [{iss['severity'].upper()}] `{iss['issue_type']}` — **{tag}** ({r.value:+.2f})")

    yield hdr + "\n**Phase 1 — Pattern**\n\n" + "\n".join(log) + "\n\n"

    if CLIENT and not done:
        for _ in range(5):
            if done: break
            found = "\n".join(f"Line {h['issue']['line']}: {h['issue']['type']}"
                              for h in obs.review_history if h['action']=='identify_issue' and h.get('valid')) or "None"
            prompt = (f"Code review. Find ONE remaining issue or approve.\n"
                      f"CODE:\n```python\n{obs.code}\n```\nFOUND:\n{found}\n\n"
                      f'Output JSON only: {{"action_type":"identify_issue","issue_type":"bug","line_number":1,"description":"...","severity":"high"}}'
                      f'\nOR: {{"action_type":"approve"}}')
            try:
                raw = _llm(prompt, max_tokens=200)
                if "```" in raw: raw = raw.split("```")[1].replace("json","").strip()
                action = Action(**json.loads(raw))
                if action.action_type == "identify_issue":
                    key = (action.issue_type, action.line_number)
                    if key in seen: action = Action(action_type="approve")
                    else: seen.add(key)
            except Exception:
                action = Action(action_type="approve")

            obs, r, done, info = env.step(action)
            total_r += r.value
            if action.action_type == "identify_issue":
                tag = "TP" if r.value > 0 else "FP"
                log.append(f"Step {obs.step_count}: `LLM` → Line {action.line_number}"
                            f" [{(action.severity or '?').upper()}] `{action.issue_type}` — **{tag}** ({r.value:+.2f})")
            else:
                log.append(f"Step {obs.step_count}: `LLM` → APPROVE ({r.value:+.2f})")
            yield hdr + "\n**Phase 2 — LLM**\n\n" + "\n".join(log[-3:]) + "\n\n"

    if not done:
        obs, r, done, info = env.step(Action(action_type="approve"))
        total_r += r.value
        log.append(f"Step {obs.step_count}: `AUTO` → APPROVE ({r.value:+.2f})")

    gt = "".join(
        f"- {ISSUE_ICON.get(i['type'],'❓')} **Line {i['line']}** "
        f"<span style='color:{SEV_COLOR.get(i['severity'],'#888')}'>[{i['severity'].upper()}]</span> "
        f"{i['description']}\n"
        for i in task["issues"]
    )
    yield (hdr + "\n**Execution Log**\n\n" + "\n".join(log) +
           f"\n\n---\n\n### Results\n\n"
           f"| Metric | Value |\n|--------|-------|\n"
           f"| **Score** | **{info['score']:.3f}** |\n"
           f"| Found | {info['found_issues']}/{info['expected_issues']} |\n"
           f"| False Positives | {info['false_positives']} |\n"
           f"| Recall | {info['recall']:.2%} |\n"
           f"| Precision | {info['precision']:.2%} |\n"
           f"| F1 | {info['f1']:.2%} |\n"
           f"| Total Reward | {total_r:+.2f} |\n"
           f"| Steps | {obs.step_count} |\n"
           f"\n**Ground Truth**\n\n{gt}")


# ════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ════════════════════════════════════════════════════════════════════════════

_CSS = """
/* ── CodeCrack · Dark Cyber Theme ── */
:root {
    --cc-bg:    #080c14;
    --cc-surf:  #0d1117;
    --cc-card:  #0f1923;
    --cc-bdr:   #1a2535;
    --cc-bdr2:  #243044;
    --cc-cyan:  #00d4ff;
    --cc-purple:#a855f7;
    --cc-green: #2ed573;
    --cc-red:   #ff4757;
    --cc-yellow:#ffd32a;
    --cc-text:  #cdd6f4;
    --cc-dim:   #6272a4;
    --cc-muted: #2a3550;
    --color-accent:      #00d4ff;
    --color-accent-soft: rgba(0,212,255,.10);
}
@keyframes cc-hdr   { 0%,100%{background-position:0 50%} 50%{background-position:100% 50%} }
@keyframes cc-scan  { to{background-position:0 200%} }
@keyframes cc-pulse { 0%,100%{box-shadow:0 0 0 0 rgba(0,212,255,.6)} 60%{box-shadow:0 0 0 8px rgba(0,212,255,0)} }
@keyframes cc-up    { from{opacity:0;transform:translateY(7px)} to{opacity:1;transform:none} }

.gradio-container {
    max-width:100% !important; padding:0 !important; margin:0 !important;
    background:var(--cc-bg) !important; color:var(--cc-text) !important;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Inter',sans-serif !important;
}
.gradio-container > .main,
.gradio-container > .main > .wrap { padding:0 !important; gap:0 !important; }

/* Tabs */
.tab-nav {
    background:var(--cc-surf) !important;
    border-bottom:1px solid var(--cc-bdr) !important; padding:0 28px !important;
}
.tab-nav button {
    font-size:13px !important; font-weight:500 !important; color:var(--cc-dim) !important;
    padding:14px 20px !important; border:none !important;
    border-bottom:2px solid transparent !important; border-radius:0 !important;
    background:transparent !important; margin-bottom:-1px !important;
    transition:color .15s,border-color .15s,text-shadow .15s !important;
}
.tab-nav button:hover:not(.selected) { color:var(--cc-text) !important; }
.tab-nav button.selected {
    color:var(--cc-cyan) !important; font-weight:700 !important;
    border-bottom-color:var(--cc-cyan) !important;
    text-shadow:0 0 12px rgba(0,212,255,.5) !important;
}
.tabitem { background:var(--cc-bg) !important; padding:24px 28px !important; border:none !important; }

/* Blocks */
.gradio-container .block, .gradio-container fieldset {
    background:var(--cc-card) !important; border:1px solid var(--cc-bdr) !important;
    border-radius:10px !important; box-shadow:0 4px 24px rgba(0,0,0,.5) !important;
}

/* Button primary */
button.primary, .gradio-container button.primary, .wrap button.primary {
    background:linear-gradient(135deg,#004e6a,#002d55) !important;
    background-image:linear-gradient(135deg,#004e6a,#002d55) !important;
    border:1px solid rgba(0,212,255,.4) !important;
    color:#e6edf3 !important; font-weight:700 !important; font-size:14px !important;
    letter-spacing:.025em !important; border-radius:8px !important;
    box-shadow:0 0 16px rgba(0,212,255,.14),inset 0 1px 0 rgba(0,212,255,.07) !important;
    transition:all .2s cubic-bezier(.4,0,.2,1) !important;
}
button.primary:hover, .gradio-container button.primary:hover {
    background:linear-gradient(135deg,#006688,#003f77) !important;
    background-image:linear-gradient(135deg,#006688,#003f77) !important;
    box-shadow:0 0 28px rgba(0,212,255,.32),inset 0 1px 0 rgba(0,212,255,.14) !important;
    border-color:rgba(0,212,255,.65) !important; transform:translateY(-1px) !important;
}
button.primary:active { transform:translateY(0) !important; }

/* Inputs */
.gradio-container input, .gradio-container select, .gradio-container textarea {
    background:var(--cc-surf) !important; border:1px solid var(--cc-bdr) !important;
    border-radius:7px !important; color:var(--cc-text) !important;
    font-size:13.5px !important; transition:border-color .15s,box-shadow .15s !important;
}
.gradio-container input:focus, .gradio-container select:focus {
    border-color:var(--cc-cyan) !important; box-shadow:0 0 0 3px rgba(0,212,255,.1) !important;
}

/* Labels */
.block .label-wrap label, .block > label {
    font-size:11px !important; font-weight:700 !important;
    text-transform:uppercase !important; letter-spacing:.07em !important; color:var(--cc-dim) !important;
}

/* Markdown */
.prose, .markdown { font-size:13.5px !important; line-height:1.7 !important; color:var(--cc-text) !important; }
.prose pre, .markdown pre {
    background:#050810 !important; border:1px solid var(--cc-bdr) !important;
    border-radius:8px !important; font-size:12px !important;
}
.prose code, .markdown code {
    background:rgba(0,212,255,.07) !important; color:var(--cc-cyan) !important;
    border-radius:4px !important; padding:1px 5px !important; font-size:.88em !important;
}
.prose table th, .markdown table th {
    background:var(--cc-surf) !important; color:var(--cc-dim) !important; border-color:var(--cc-bdr) !important;
}
.prose table td, .markdown table td { border-color:var(--cc-bdr) !important; color:var(--cc-text) !important; }
.prose strong, .markdown strong, .gradio-container strong { color:var(--cc-cyan) !important; font-weight:700 !important; }
.prose a, .markdown a { color:var(--cc-cyan) !important; }

/* Dropdowns */
.gradio-container ul[role=listbox], .gradio-container .options {
    background:var(--cc-card) !important; border-color:var(--cc-bdr) !important;
}
.gradio-container li[role=option]:hover { background:rgba(0,212,255,.07) !important; }

/* Scrollbar */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--cc-bg); }
::-webkit-scrollbar-thumb { background:var(--cc-muted); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:var(--cc-dim); }
"""

# Dark cyber theme — CSS vars do the heavy lifting
_theme = gr.themes.Base(
    primary_hue="sky",
    secondary_hue="slate",
    neutral_hue="slate",
    text_size=gr.themes.sizes.text_sm,
    radius_size=gr.themes.sizes.radius_md,
).set(
    button_primary_background_fill="linear-gradient(135deg,#004e6a,#002d55)",
    button_primary_background_fill_hover="linear-gradient(135deg,#006688,#003f77)",
    button_primary_text_color="#e6edf3",
    button_primary_border_color="rgba(0,212,255,0.4)",
    button_primary_shadow="0 0 16px rgba(0,212,255,0.14)",
    button_primary_shadow_hover="0 0 28px rgba(0,212,255,0.32)",
    body_background_fill="#080c14",
    body_text_color="#cdd6f4",
    block_background_fill="#0f1923",
    block_border_color="#1a2535",
    block_border_width="1px",
    block_shadow="0 4px 24px rgba(0,0,0,0.5)",
    block_label_text_color="#6272a4",
    block_label_text_weight="700",
    block_label_text_size="11px",
    input_background_fill="#0d1117",
    input_border_color="#1a2535",
    input_border_color_focus="#00d4ff",
    input_shadow_focus="0 0 0 3px rgba(0,212,255,0.1)",
    color_accent="#00d4ff",
    color_accent_soft="rgba(0,212,255,0.1)",
    checkbox_background_color_selected="#00d4ff",
    slider_color="#00d4ff",
    link_text_color="#00d4ff",
)

# ── Pre-built HTML blocks ──────────────────────────────────────────────────

_HEADER = f"""
<div style="
    background:linear-gradient(-45deg,#080c14,#0a1020,#060d18,#0d1528);
    background-size:400% 400%; animation:cc-hdr 16s ease infinite;
    border-bottom:1px solid rgba(0,212,255,.12);
    padding:0 32px; height:66px;
    display:flex; align-items:center; justify-content:space-between;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    position:relative; overflow:hidden;
">
  <div style="position:absolute;inset:0;pointer-events:none;
    background:linear-gradient(180deg,transparent 0%,rgba(0,212,255,.025) 50%,transparent 100%);
    background-size:100% 200%; animation:cc-scan 6s linear infinite;"></div>
  <div style="position:absolute;inset:0;pointer-events:none;
    background-image:radial-gradient(circle,rgba(0,212,255,.055) 1px,transparent 1px);
    background-size:28px 28px;"></div>
  <div style="position:absolute;top:0;left:0;width:90px;height:2px;
    background:linear-gradient(90deg,#00d4ff,transparent);"></div>
  <div style="position:absolute;top:0;left:0;width:2px;height:44px;
    background:linear-gradient(180deg,#00d4ff,transparent);"></div>
  <div style="position:absolute;bottom:0;right:0;width:90px;height:1px;
    background:linear-gradient(270deg,rgba(168,85,247,.5),transparent);"></div>

  <div style="display:flex;align-items:center;gap:16px;position:relative;z-index:1;">
    <div style="
      width:40px;height:40px;border-radius:11px;
      background:linear-gradient(135deg,#003d55,#00223d);
      border:1px solid rgba(0,212,255,.35);
      display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;
      box-shadow:0 0 24px rgba(0,212,255,.25),0 0 60px rgba(0,212,255,.08);
    ">⚡</div>
    <div>
      <div style="line-height:1.1;font-size:17px;font-weight:800;letter-spacing:-.025em;">
        <span style="color:#e6edf3;">Code</span><span style="color:#00d4ff;text-shadow:0 0 14px rgba(0,212,255,.55);">Crack</span>
      </div>
      <div style="color:#2a3550;font-size:10px;font-weight:600;letter-spacing:.14em;margin-top:2px;">AI CODE REVIEW ENVIRONMENT</div>
    </div>
    <div style="width:1px;height:26px;background:rgba(255,255,255,.06);margin:0 2px;"></div>
    <div style="
      color:#00d4ff;font-size:10px;font-weight:700;letter-spacing:.1em;
      background:rgba(0,212,255,.07);border:1px solid rgba(0,212,255,.2);
      padding:4px 11px;border-radius:5px;box-shadow:0 0 10px rgba(0,212,255,.08);
    ">OpenEnv v1.0</div>
  </div>

  <div style="display:flex;align-items:center;gap:10px;position:relative;z-index:1;">
    <div style="
      display:flex;align-items:center;gap:8px;
      background:rgba(0,212,255,.05);border:1px solid rgba(0,212,255,.13);
      padding:7px 14px;border-radius:8px;
    ">
      <span style="width:7px;height:7px;border-radius:50%;background:#2ed573;display:inline-block;
        flex-shrink:0;box-shadow:0 0 8px #2ed573;animation:cc-pulse 2.5s ease-in-out infinite;"></span>
      <span style="font-size:12px;color:#6272a4;font-weight:500;">{prov}</span>
    </div>
    <div style="
      background:rgba(0,0,0,.4);border:1px solid rgba(255,255,255,.05);
      padding:7px 14px;border-radius:8px;
      font-family:'SF Mono','Fira Code',Consolas,monospace;font-size:11px;color:#2a3550;
    ">{model}</div>
  </div>
</div>
"""

_STATS = """
<div style="
    background:#0d1117;border-bottom:1px solid #1a2535;
    padding:0 32px;height:50px;
    display:flex;align-items:center;
    font-family:'SF Mono','Fira Code',Consolas,monospace;
    animation:cc-up .45s ease both;
">
  <div style="display:flex;align-items:center;gap:10px;padding-right:26px;margin-right:26px;border-right:1px solid #1a2535;">
    <span style="font-size:9px;color:#2a3550;text-transform:uppercase;letter-spacing:.14em;font-weight:700;">Baseline</span>
    <span style="font-size:19px;font-weight:900;color:#00d4ff;letter-spacing:-.04em;text-shadow:0 0 18px rgba(0,212,255,.5);">1.000</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;padding-right:26px;margin-right:26px;border-right:1px solid #1a2535;">
    <span style="font-size:9px;color:#2a3550;text-transform:uppercase;letter-spacing:.14em;font-weight:700;">Tasks</span>
    <span style="font-size:19px;font-weight:900;color:#cdd6f4;">3</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;padding-right:26px;margin-right:26px;border-right:1px solid #1a2535;">
    <span style="font-size:9px;color:#2a3550;text-transform:uppercase;letter-spacing:.14em;font-weight:700;">Strategy</span>
    <span style="font-size:14px;font-weight:700;color:#a855f7;text-shadow:0 0 10px rgba(168,85,247,.4);">Hybrid</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;padding-right:26px;margin-right:26px;border-right:1px solid #1a2535;">
    <span style="font-size:9px;color:#2a3550;text-transform:uppercase;letter-spacing:.14em;font-weight:700;">Tolerance</span>
    <span style="font-size:16px;font-weight:800;color:#cdd6f4;">±2</span>
    <span style="font-size:9px;color:#2a3550;">lines</span>
  </div>
  <div style="display:flex;align-items:center;gap:10px;">
    <span style="font-size:9px;color:#2a3550;text-transform:uppercase;letter-spacing:.14em;font-weight:700;">Grading</span>
    <span style="font-size:12px;font-weight:700;color:#00d4ff;">0.5R + 0.3P + 0.2S</span>
  </div>
</div>
"""

def _note(html: str) -> str:
    return f"""
<div style="
    background:rgba(0,212,255,.04);border:1px solid rgba(0,212,255,.12);
    border-left:3px solid #00d4ff;border-radius:8px;
    padding:13px 18px;margin-bottom:20px;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    font-size:13px;color:#6272a4;line-height:1.65;
    animation:cc-up .35s ease both;
">{html}</div>"""

_TASK_CARDS = """
<div style="
    display:grid;grid-template-columns:repeat(3,1fr);gap:16px;
    margin-bottom:24px;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    animation:cc-up .45s ease both;
">
  <div style="background:#0d1117;border:1px solid #1a2535;border-radius:12px;
    padding:20px;box-shadow:0 4px 24px rgba(0,0,0,.5);
    border-top:2px solid #2ed573;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:100px;height:100px;
      background:radial-gradient(circle at top right,rgba(46,213,115,.07),transparent);pointer-events:none;"></div>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
      <span style="background:rgba(46,213,115,.1);color:#2ed573;border:1px solid rgba(46,213,115,.3);
        font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.12em;
        padding:3px 9px;border-radius:4px;">Easy</span>
      <span style="font-size:21px;font-weight:900;color:#2ed573;font-family:'SF Mono',monospace;
        text-shadow:0 0 14px rgba(46,213,115,.5);">1.000</span>
    </div>
    <div style="font-size:14px;font-weight:700;color:#cdd6f4;margin-bottom:6px;">SQL Injection Detection</div>
    <div style="font-size:11px;color:#2a3550;font-family:'SF Mono',Consolas,monospace;">1 issue · 38 lines · security</div>
  </div>

  <div style="background:#0d1117;border:1px solid #1a2535;border-radius:12px;
    padding:20px;box-shadow:0 4px 24px rgba(0,0,0,.5);
    border-top:2px solid #ffd32a;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:100px;height:100px;
      background:radial-gradient(circle at top right,rgba(255,211,42,.07),transparent);pointer-events:none;"></div>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
      <span style="background:rgba(255,211,42,.1);color:#ffd32a;border:1px solid rgba(255,211,42,.3);
        font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.12em;
        padding:3px 9px;border-radius:4px;">Medium</span>
      <span style="font-size:21px;font-weight:900;color:#ffd32a;font-family:'SF Mono',monospace;
        text-shadow:0 0 14px rgba(255,211,42,.5);">1.000</span>
    </div>
    <div style="font-size:14px;font-weight:700;color:#cdd6f4;margin-bottom:6px;">Race Condition Analysis</div>
    <div style="font-size:11px;color:#2a3550;font-family:'SF Mono',Consolas,monospace;">2 issues · 51 lines · concurrency</div>
  </div>

  <div style="background:#0d1117;border:1px solid #1a2535;border-radius:12px;
    padding:20px;box-shadow:0 4px 24px rgba(0,0,0,.5);
    border-top:2px solid #ff4757;position:relative;overflow:hidden;">
    <div style="position:absolute;top:0;right:0;width:100px;height:100px;
      background:radial-gradient(circle at top right,rgba(255,71,87,.07),transparent);pointer-events:none;"></div>
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
      <span style="background:rgba(255,71,87,.1);color:#ff4757;border:1px solid rgba(255,71,87,.3);
        font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.12em;
        padding:3px 9px;border-radius:4px;">Hard</span>
      <span style="font-size:21px;font-weight:900;color:#ff4757;font-family:'SF Mono',monospace;
        text-shadow:0 0 14px rgba(255,71,87,.5);">1.000</span>
    </div>
    <div style="font-size:14px;font-weight:700;color:#cdd6f4;margin-bottom:6px;">Memory Leak &amp; Iterator Bug</div>
    <div style="font-size:11px;color:#2a3550;font-family:'SF Mono',Consolas,monospace;">3 issues · 60 lines · performance</div>
  </div>
</div>
"""

_FOOTER = """
<div style="
    background:#0d1117;border-top:1px solid #1a2535;
    padding:14px 32px;
    display:flex;justify-content:space-between;align-items:center;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    font-size:12px;color:#2a3550;
">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="width:24px;height:24px;border-radius:6px;
      background:linear-gradient(135deg,#003d55,#002240);
      border:1px solid rgba(0,212,255,.25);
      display:flex;align-items:center;justify-content:center;font-size:13px;
      box-shadow:0 0 10px rgba(0,212,255,.15);">⚡</div>
    <span style="font-weight:700;color:#6272a4;font-size:13px;">CodeCrack</span>
  </div>
  <span>OpenEnv-compliant RL environment for AI code review agents</span>
  <span>Scaler Meta PyTorch Hackathon 2025</span>
</div>
"""

_SQL_SAMPLE = """\
import sqlite3

def get_user(conn, username):
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE name = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()

def log(msg):
    print(f"[LOG] {msg}")
"""

_RACE_SAMPLE = """\
import threading

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
"""

# ── Gradio Blocks ──────────────────────────────────────────────────────────

with gr.Blocks(title="CodeCrack", css=_CSS, theme=_theme) as demo:

    gr.HTML(_HEADER)
    gr.HTML(_STATS)

    with gr.Tabs():

        # ── Code Review ────────────────────────────────────────────────────
        with gr.Tab("Code Review"):
            gr.HTML(_note(
                "<strong>Hybrid Analysis</strong> — Regex pattern matching runs instantly (zero latency, "
                "no API call), then the LLM performs a deep scan for edge cases. "
                "Results are merged and deduplicated with <strong>±2 line tolerance</strong>."
            ))
            with gr.Row():
                with gr.Column(scale=5):
                    review_input = gr.Code(
                        language="python", label="Source Code",
                        lines=22, value=_SQL_SAMPLE,
                    )
                    review_btn = gr.Button("Run Analysis →", variant="primary", size="lg")
                with gr.Column(scale=3):
                    review_score = gr.Markdown(
                        label="Risk Summary",
                        value="*Paste code and click **Run Analysis**.*",
                    )
            review_output = gr.Markdown(label="Analysis Report")
            review_btn.click(meta_review, [review_input], [review_output, review_score])

        # ── Generate ───────────────────────────────────────────────────────
        with gr.Tab("Generate"):
            gr.HTML(_note(
                "<strong>Adversarial Code Generator</strong> — The LLM generates novel buggy Python snippets "
                "with configurable difficulty and bug category, creating unlimited AI agent training data."
            ))
            with gr.Row():
                adv_diff = gr.Dropdown(
                    ["Easy (1 issue)", "Medium (2 issues)", "Hard (3+ issues)"],
                    value="Medium (2 issues)", label="Difficulty", scale=1,
                )
                adv_cat = gr.Dropdown(
                    ["Random"] + BUG_CATEGORIES, value="Random",
                    label="Bug Category", scale=2,
                )
            adv_btn    = gr.Button("Generate Sample →", variant="primary", size="lg")
            adv_output = gr.Markdown(label="Code and Bug Key")
            adv_code   = gr.Code(language="python", label="Generated Code", interactive=True)
            adv_btn.click(generate_adversarial, [adv_diff, adv_cat], [adv_output, adv_code])

        # ── Agent Debate ───────────────────────────────────────────────────
        with gr.Tab("Agent Debate"):
            gr.HTML(_note(
                "<strong>Dual-Agent Review</strong> — A deterministic regex engine and an LLM "
                "independently analyze the same code. Findings are compared with an agreement report "
                "and confidence score."
            ))
            with gr.Row():
                with gr.Column(scale=5):
                    debate_input = gr.Code(
                        language="python", label="Code Under Review",
                        lines=22, value=_RACE_SAMPLE,
                    )
                    debate_btn = gr.Button("Start Dual Review →", variant="primary", size="lg")
                with gr.Column(scale=3):
                    debate_sum = gr.Markdown(
                        label="Agreement Report",
                        value="*Paste code and click **Start Dual Review**.*",
                    )
            debate_out = gr.Markdown(label="Review Transcript")
            debate_btn.click(duo_debate, [debate_input], [debate_out, debate_sum])

        # ── Benchmark ──────────────────────────────────────────────────────
        with gr.Tab("Benchmark"):
            gr.HTML(_note(
                "<strong>Official Benchmark</strong> — Run the three graded tasks. "
                "The hybrid agent executes step-by-step, reporting recall, precision, F1, "
                "and cumulative reward in real time."
            ))
            gr.HTML(_TASK_CARDS)
            with gr.Row():
                task_sel = gr.Dropdown(
                    list(TASKS.keys()), value="easy_sql_injection",
                    label="Task", scale=3,
                )
                task_btn = gr.Button("Run Agent →", variant="primary", size="lg", scale=1)
            task_out = gr.Markdown(label="Execution Log")
            task_btn.click(run_task_arena, [task_sel], [task_out])

    gr.HTML(_FOOTER)


# ════════════════════════════════════════════════════════════════════════════
# FASTAPI — OpenEnv REST API
# ════════════════════════════════════════════════════════════════════════════

_api_env    = CodeReviewEnv()
fastapi_app = FastAPI(title="Code Review Environment API", version="1.0.0")


@fastapi_app.get("/api/health")
def api_health():
    return {"status": "ok", "version": "1.0.0"}


@fastapi_app.post("/reset")
def api_reset(task_id: Optional[str] = None):
    try:    return _api_env.reset(task_id=task_id).model_dump()
    except ValueError as e: raise HTTPException(400, str(e))


@fastapi_app.post("/step")
def api_step(action: Action):
    if _api_env.current_state is None:
        raise HTTPException(400, "Call /reset first.")
    obs, reward, done, info = _api_env.step(action)
    return {"observation": obs.model_dump(), "reward": reward.model_dump(), "done": done, "info": info}


@fastapi_app.get("/state")
def api_state():
    return _api_env.state()


@fastapi_app.get("/tasks")
def api_tasks():
    return {tid: {"difficulty": t["difficulty"], "description": t["description"],
                  "issue_count": len(t["issues"])} for tid, t in TASKS.items()}


@fastapi_app.get("/tasks/{task_id}")
def api_get_task(task_id: str):
    if task_id not in TASKS:
        raise HTTPException(404, f"Task '{task_id}' not found")
    t = TASKS[task_id]
    return {"task_id": task_id, "difficulty": t["difficulty"], "description": t["description"],
            "code": t["code"], "issue_count": len(t["issues"])}


app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
