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
/* ── Force indigo palette on Gradio CSS variables ── */
:root {
    --primary-50:  #eef2ff; --primary-100: #e0e7ff; --primary-200: #c7d2fe;
    --primary-300: #a5b4fc; --primary-400: #818cf8; --primary-500: #6366f1;
    --primary-600: #4f46e5; --primary-700: #4338ca; --primary-800: #3730a3;
    --primary-900: #312e81; --primary-950: #1e1b4b;
    --color-accent:      #4f46e5;
    --color-accent-soft: rgba(79,70,229,0.10);
    --button-primary-background-fill:       #4f46e5;
    --button-primary-background-fill-hover: #4338ca;
    --button-primary-text-color:            #ffffff;
    --button-primary-border-color:          #4338ca;
}

/* ── Keyframes ── */
@keyframes cc-hdr {
    0%   { background-position: 0% 50%;   }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%;   }
}
@keyframes cc-dot {
    0%,100% { box-shadow: 0 0 0 0   rgba(34,197,94,0.6); }
    60%     { box-shadow: 0 0 0 6px rgba(34,197,94,0);   }
}
@keyframes cc-fade-up {
    from { opacity:0; transform:translateY(6px); }
    to   { opacity:1; transform:translateY(0);   }
}

/* ── Root layout ── */
.gradio-container {
    max-width:100% !important;
    padding:0 !important;
    margin:0 !important;
    background:#f1f5f9 !important;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Inter',sans-serif !important;
}
.gradio-container > .main,
.gradio-container > .main > .wrap { padding:0 !important; gap:0 !important; }

/* ── Tab nav ── */
.tab-nav {
    background:#fff !important;
    border-bottom:1px solid #e2e8f0 !important;
    padding:0 28px !important;
}
.tab-nav button {
    font-size:13.5px !important; font-weight:500 !important;
    color:#64748b !important; padding:13px 18px !important;
    border:none !important; border-bottom:2px solid transparent !important;
    border-radius:0 !important; background:transparent !important;
    margin-bottom:-1px !important;
    transition:color .15s,border-color .15s !important;
    letter-spacing:.01em !important;
}
.tab-nav button:hover:not(.selected) { color:#1e293b !important; }
.tab-nav button.selected {
    color:#4f46e5 !important; font-weight:700 !important;
    border-bottom-color:#4f46e5 !important;
    background:transparent !important;
}

/* ── Tab content ── */
.tabitem { background:#f1f5f9 !important; padding:24px 28px !important; border:none !important; }

/* ── PRIMARY BUTTON (nuclear override) ── */
button.primary, .btn-primary,
.gradio-container button.primary,
.wrap button.primary {
    background:#4f46e5 !important;
    background-image:none !important;
    border:1px solid #4338ca !important;
    color:#fff !important;
    font-weight:700 !important; font-size:14px !important;
    letter-spacing:.015em !important;
    border-radius:8px !important;
    box-shadow:0 2px 10px rgba(79,70,229,.40) !important;
    transition:all .2s cubic-bezier(.4,0,.2,1) !important;
}
button.primary:hover, .gradio-container button.primary:hover {
    background:#4338ca !important;
    box-shadow:0 6px 22px rgba(79,70,229,.50) !important;
    transform:translateY(-2px) !important;
}
button.primary:active { background:#3730a3 !important; transform:translateY(0) !important; }

/* ── Inputs ── */
.gradio-container input,
.gradio-container select,
.gradio-container textarea {
    border-radius:7px !important; border-color:#e2e8f0 !important;
    font-size:13.5px !important; background:#fff !important;
    font-family:inherit !important;
    transition:border-color .15s,box-shadow .15s !important;
}
.gradio-container input:focus, .gradio-container select:focus {
    border-color:#4f46e5 !important;
    box-shadow:0 0 0 3px rgba(79,70,229,.12) !important;
}

/* ── Block labels ── */
.block .label-wrap label, .block > label {
    font-size:11px !important; font-weight:700 !important;
    text-transform:uppercase !important; letter-spacing:.07em !important;
    color:#64748b !important;
}

/* ── Markdown ── */
.prose, .markdown { font-size:13.5px !important; line-height:1.7 !important; color:#1e293b !important; }
.prose pre, .markdown pre {
    background:#0f172a !important; border-radius:8px !important;
    border:1px solid #1e293b !important; font-size:12px !important;
}
.prose code, .markdown code {
    background:rgba(99,102,241,.09) !important; color:#4f46e5 !important;
    border-radius:4px !important; padding:1px 5px !important; font-size:.88em !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#cbd5e1; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#94a3b8; }
"""

# Use Base theme — minimal, so our CSS wins
_theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    text_size=gr.themes.sizes.text_sm,
    radius_size=gr.themes.sizes.radius_md,
).set(
    # Buttons
    button_primary_background_fill="#4f46e5",
    button_primary_background_fill_hover="#4338ca",
    button_primary_text_color="#ffffff",
    button_primary_border_color="#4338ca",
    button_primary_shadow="0 2px 10px rgba(79,70,229,0.40)",
    button_primary_shadow_hover="0 6px 22px rgba(79,70,229,0.50)",
    # Layout
    body_background_fill="#f1f5f9",
    body_text_color="#1e293b",
    block_background_fill="#ffffff",
    block_border_color="#e2e8f0",
    block_border_width="1px",
    block_shadow="0 1px 4px rgba(0,0,0,0.06)",
    block_label_text_color="#64748b",
    block_label_text_weight="700",
    block_label_text_size="11px",
    # Inputs
    input_background_fill="#ffffff",
    input_border_color="#e2e8f0",
    input_border_color_focus="#4f46e5",
    input_shadow_focus="0 0 0 3px rgba(79,70,229,0.12)",
    # Accent
    color_accent="#4f46e5",
    color_accent_soft="rgba(79,70,229,0.10)",
    checkbox_background_color_selected="#4f46e5",
    slider_color="#4f46e5",
    link_text_color="#4f46e5",
)

# ── Pre-built HTML blocks (inline styles = reliable in any Gradio version) ──

_HEADER = f"""
<div style="
    background:linear-gradient(-45deg,#0f172a,#1a1040,#0f172a,#0c1535);
    background-size:400% 400%;
    animation:cc-hdr 14s ease infinite;
    border-bottom:1px solid rgba(99,102,241,.15);
    padding:0 32px; height:60px;
    display:flex; align-items:center; justify-content:space-between;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    position:relative; overflow:hidden;
">
  <!-- subtle grid overlay -->
  <div style="position:absolute;inset:0;
    background-image:linear-gradient(rgba(99,102,241,.04) 1px,transparent 1px),
                     linear-gradient(90deg,rgba(99,102,241,.04) 1px,transparent 1px);
    background-size:32px 32px; pointer-events:none;"></div>

  <!-- Left: brand -->
  <div style="display:flex;align-items:center;gap:14px;position:relative;z-index:1;">
    <div style="
      width:32px;height:32px;border-radius:9px;
      background:linear-gradient(135deg,#4f46e5,#818cf8);
      display:flex;align-items:center;justify-content:center;
      font-weight:900;font-size:17px;color:#fff;flex-shrink:0;
      box-shadow:0 0 0 2px rgba(129,140,248,.3),0 4px 14px rgba(79,70,229,.5);
      letter-spacing:-1px;
    ">C</div>
    <span style="color:#f1f5f9;font-size:16px;font-weight:800;letter-spacing:-.025em;">CodeCrack</span>
    <div style="width:1px;height:18px;background:rgba(255,255,255,.12);"></div>
    <span style="color:#64748b;font-size:12.5px;letter-spacing:.01em;">AI Code Review Environment</span>
    <span style="
      color:#a5b4fc;font-size:10.5px;font-weight:700;letter-spacing:.06em;
      background:rgba(99,102,241,.18);border:1px solid rgba(129,140,248,.3);
      padding:2px 10px;border-radius:5px;
    ">OpenEnv v1.0</span>
  </div>

  <!-- Right: status + model -->
  <div style="display:flex;align-items:center;gap:10px;position:relative;z-index:1;">
    <div style="
      display:flex;align-items:center;gap:7px;
      background:rgba(255,255,255,.06);border:1px solid rgba(255,255,255,.1);
      padding:6px 13px;border-radius:7px;backdrop-filter:blur(6px);
    ">
      <span style="
        width:7px;height:7px;border-radius:50%;background:#22c55e;
        display:inline-block;flex-shrink:0;
        animation:cc-dot 2s ease-in-out infinite;
      "></span>
      <span style="font-size:12px;color:#94a3b8;">{prov}</span>
    </div>
    <div style="
      background:rgba(0,0,0,.35);border:1px solid rgba(255,255,255,.1);
      padding:6px 13px;border-radius:7px;
      font-family:'SF Mono','Fira Code',Consolas,monospace;
      font-size:11px;color:#64748b;letter-spacing:.01em;
    ">{model}</div>
  </div>
</div>
"""

_STATS = """
<div style="
    background:#fff;border-bottom:1px solid #e2e8f0;
    padding:0 32px;height:50px;
    display:flex;align-items:center;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    animation:cc-fade-up .4s ease both;
">
  <div style="display:flex;align-items:baseline;gap:7px;padding-right:28px;margin-right:28px;border-right:1px solid #e2e8f0;">
    <span style="
      font-family:'SF Mono','Fira Code',Consolas,monospace;
      font-size:18px;font-weight:800;color:#4f46e5;letter-spacing:-.03em;
      text-shadow:0 0 20px rgba(99,102,241,.35);
    ">1.000</span>
    <span style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">Baseline</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:7px;padding-right:28px;margin-right:28px;border-right:1px solid #e2e8f0;">
    <span style="font-size:18px;font-weight:800;color:#0f172a;">3</span>
    <span style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">Tasks</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:7px;padding-right:28px;margin-right:28px;border-right:1px solid #e2e8f0;">
    <span style="font-size:16px;font-weight:800;color:#0f172a;">Hybrid</span>
    <span style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">Strategy</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:7px;padding-right:28px;margin-right:28px;border-right:1px solid #e2e8f0;">
    <span style="font-family:'SF Mono','Fira Code',Consolas,monospace;font-size:16px;font-weight:800;color:#0f172a;">±2</span>
    <span style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">Line Tolerance</span>
  </div>
  <div style="display:flex;align-items:baseline;gap:7px;">
    <span style="font-family:'SF Mono','Fira Code',Consolas,monospace;font-size:13px;font-weight:800;color:#4f46e5;">0.5R + 0.3P + 0.2S</span>
    <span style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;font-weight:700;">Grading Formula</span>
  </div>
</div>
"""

def _note(html: str) -> str:
    return f"""
<div style="
    background:linear-gradient(to right,rgba(79,70,229,.03),transparent);
    border:1px solid #e2e8f0;border-left:3px solid #4f46e5;
    border-radius:8px;padding:13px 18px;margin-bottom:20px;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    font-size:13px;color:#475569;line-height:1.65;
    box-shadow:0 1px 3px rgba(0,0,0,.04);
    animation:cc-fade-up .35s ease both;
">{html}</div>"""

_TASK_CARDS = """
<div style="
    display:grid;grid-template-columns:repeat(3,1fr);gap:14px;
    margin-bottom:22px;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    animation:cc-fade-up .4s ease both;
">
  <!-- EASY -->
  <div style="
    background:#fff;border:1px solid #e2e8f0;border-radius:10px;
    padding:18px;display:flex;align-items:flex-start;gap:13px;
    box-shadow:0 1px 4px rgba(0,0,0,.05);
    transition:box-shadow .2s,transform .2s;
    border-top:3px solid #22c55e;
  ">
    <span style="
      background:#f0fdf4;color:#166534;border:1px solid #a7f3d0;
      font-size:9.5px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;
      padding:3px 9px;border-radius:4px;white-space:nowrap;margin-top:2px;flex-shrink:0;
    ">Easy</span>
    <div>
      <div style="font-size:13.5px;font-weight:700;color:#0f172a;margin-bottom:5px;">SQL Injection Detection</div>
      <div style="font-size:11.5px;color:#94a3b8;font-family:'SF Mono','Fira Code',Consolas,monospace;">
        1 issue · 38 lines · score <strong style="color:#4f46e5;">1.000</strong>
      </div>
    </div>
  </div>
  <!-- MEDIUM -->
  <div style="
    background:#fff;border:1px solid #e2e8f0;border-radius:10px;
    padding:18px;display:flex;align-items:flex-start;gap:13px;
    box-shadow:0 1px 4px rgba(0,0,0,.05);
    border-top:3px solid #eab308;
  ">
    <span style="
      background:#fffbeb;color:#92400e;border:1px solid #fcd34d;
      font-size:9.5px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;
      padding:3px 9px;border-radius:4px;white-space:nowrap;margin-top:2px;flex-shrink:0;
    ">Medium</span>
    <div>
      <div style="font-size:13.5px;font-weight:700;color:#0f172a;margin-bottom:5px;">Race Condition Analysis</div>
      <div style="font-size:11.5px;color:#94a3b8;font-family:'SF Mono','Fira Code',Consolas,monospace;">
        2 issues · 51 lines · score <strong style="color:#4f46e5;">1.000</strong>
      </div>
    </div>
  </div>
  <!-- HARD -->
  <div style="
    background:#fff;border:1px solid #e2e8f0;border-radius:10px;
    padding:18px;display:flex;align-items:flex-start;gap:13px;
    box-shadow:0 1px 4px rgba(0,0,0,.05);
    border-top:3px solid #ef4444;
  ">
    <span style="
      background:#fef2f2;color:#991b1b;border:1px solid #fca5a5;
      font-size:9.5px;font-weight:800;text-transform:uppercase;letter-spacing:.08em;
      padding:3px 9px;border-radius:4px;white-space:nowrap;margin-top:2px;flex-shrink:0;
    ">Hard</span>
    <div>
      <div style="font-size:13.5px;font-weight:700;color:#0f172a;margin-bottom:5px;">Memory Leak &amp; Iterator Bug</div>
      <div style="font-size:11.5px;color:#94a3b8;font-family:'SF Mono','Fira Code',Consolas,monospace;">
        3 issues · 60 lines · score <strong style="color:#4f46e5;">1.000</strong>
      </div>
    </div>
  </div>
</div>
"""

_FOOTER = """
<div style="
    background:#fff;border-top:1px solid #e2e8f0;margin-top:8px;
    padding:14px 32px;display:flex;justify-content:space-between;align-items:center;
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
    font-size:12px;color:#94a3b8;
">
  <div style="display:flex;align-items:center;gap:8px;">
    <div style="width:20px;height:20px;border-radius:5px;
                background:linear-gradient(135deg,#4f46e5,#818cf8);
                display:flex;align-items:center;justify-content:center;
                font-weight:900;font-size:11px;color:#fff;">C</div>
    <span style="font-weight:700;color:#374151;font-size:13px;">CodeCrack</span>
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
