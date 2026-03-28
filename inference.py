"""
Hybrid baseline for Code Review Environment.
Pattern matching for obvious issues + LLM fallback for complex cases.
"""

import os
import re
from openai import OpenAI
from environment import CodeReviewEnv
from models import Action
import json
from dotenv import load_dotenv

load_dotenv()

# Pattern-based detection rules
SECURITY_PATTERNS = [
    (r'f["\']SELECT.*\{.*\}["\']', "security", "SQL injection via f-string", "critical"),
    (r'\.format\(.*\).*SELECT', "security", "SQL injection via .format()", "critical"),
]

BUG_PATTERNS = [
    (r'self\.\w+\s*=\s*self\.\w+\s*[+\-]', "bug", "Race condition in read-modify-write", "high"),
    (r'if self\.\w+.*:\n.*self\.\w+\s*=', "bug", "TOCTOU race condition", "high"),
]

PERFORMANCE_PATTERNS = [
    (r'\.append\(.*\).*never.*remove', "performance", "Memory leak - items accumulate", "high"),
    (r'for \w+ in self\.\w+\.keys\(\):.*del self\.\w+', "bug", "Dict mutation during iteration", "high"),
]

def pattern_scan(code: str) -> list:
    """Fast pattern matching for obvious issues."""
    issues = []
    lines = code.split('\n')
    
    all_patterns = SECURITY_PATTERNS + BUG_PATTERNS + PERFORMANCE_PATTERNS
    
    for pattern, issue_type, description, severity in all_patterns:
        matches = list(re.finditer(pattern, code, re.MULTILINE | re.DOTALL))
        for match in matches:
            # Find line number
            line_num = code[:match.start()].count('\n') + 1
            
            issues.append({
                "action_type": "identify_issue",
                "issue_type": issue_type,
                "line_number": line_num,
                "description": description,
                "severity": severity
            })
    
    return issues

def run_baseline_inference():
    """Hybrid baseline: pattern matching + LLM for edge cases."""
    
    # API setup
    groq_key = os.getenv("GROQ_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY") 
    hf_token = os.getenv("HF_TOKEN")
    
    api_key = groq_key or together_key or hf_token
    
    if groq_key:
        api_base = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
        model = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
        provider = "Groq"
    elif together_key:
        api_base = "https://api.together.xyz/v1"
        model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
        provider = "Together AI"
    else:
        # Fallback to pattern-only mode
        api_base = None
        model = None
        provider = "Pattern-only"
    
    print(f"{'='*60}")
    print(f"CODE REVIEW ENVIRONMENT - BASELINE INFERENCE")
    print(f"{'='*60}")
    print(f"Provider : {provider}")
    if model:
        print(f"Model    : {model}")
    print(f"Strategy : Hybrid (patterns + LLM fallback)")
    print(f"{'='*60}\n")
    
    if api_base:
        client = OpenAI(api_key=api_key, base_url=api_base)
    else:
        client = None
    
    env = CodeReviewEnv()
    results = {}
    total_api_calls = 0
    
    for task_id in ["easy_sql_injection", "medium_race_condition", "hard_memory_leak"]:
        print(f"\n{'='*60}")
        print(f"TASK: {task_id.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        obs = env.reset(task_id=task_id)
        done = False
        total_reward = 0.0
        detected = set()
        api_calls = 0
        
        # Phase 1: Pattern matching (instant)
        print("  [PHASE 1] Pattern scanning...")
        pattern_issues = pattern_scan(obs.code)
        
        for issue in pattern_issues:
            key = (issue["issue_type"], issue["line_number"])
            if key not in detected:
                detected.add(key)
                action = Action(**issue)
                obs, reward, done, info = env.step(action)
                total_reward += reward.value
                
                status = "✓" if reward.value > 0 else "✗"
                print(f"    Step {obs.step_count}: {status} PATTERN {issue['issue_type'].upper():12s} @ L{issue['line_number']:2d} | {reward.value:+.2f}")
        
        # Phase 2: LLM deep scan (only if patterns missed issues and client available)
        if len(pattern_issues) < 2 and client and not done:
            print("  [PHASE 2] LLM deep analysis...")
            
            max_llm_attempts = 5
            attempts = 0
            
            while not done and attempts < max_llm_attempts:
                attempts += 1
                
                found_list = [
                    f"Line {h['issue']['line']}: {h['issue']['type']}"
                    for h in obs.review_history
                    if h['action'] == 'identify_issue' and h.get('valid')
                ]
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
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=200
                    )
                    api_calls += 1
                    
                    content = response.choices[0].message.content.strip()
                    if "```" in content:
                        content = content.split("```")[1].replace("json", "").strip()
                    
                    action_dict = json.loads(content)
                    action = Action(**action_dict)
                    
                    if action.action_type == "identify_issue":
                        key = (action.issue_type, action.line_number)
                        if key in detected:
                            action = Action(action_type="approve")
                        else:
                            detected.add(key)
                    
                except Exception as e:
                    action = Action(action_type="approve")
                
                obs, reward, done, info = env.step(action)
                total_reward += reward.value
                
                if action.action_type == "identify_issue":
                    status = "✓" if reward.value > 0 else "✗"
                    print(f"    Step {obs.step_count}: {status} LLM    {action.issue_type.upper():12s} @ L{action.line_number:2d} | {reward.value:+.2f}")
                else:
                    print(f"    Step {obs.step_count}: {status} {action.action_type.upper()} | {reward.value:+.2f}")
        
        # Auto-approve if not done
        if not done:
            obs, reward, done, info = env.step(Action(action_type="approve"))
            total_reward += reward.value
            print(f"    Step {obs.step_count}: ✅ AUTO-APPROVE | {reward.value:+.2f}")
        
        total_api_calls += api_calls
        
        results[task_id] = {
            "score": info["score"],
            "reward": total_reward,
            "steps": obs.step_count,
            "found": info["found_issues"],
            "expected": info["expected_issues"],
            "api_calls": api_calls
        }
        
        print(f"\n  📊 Score     : {info['score']:.3f}")
        print(f"  🔍 Found     : {info['found_issues']}/{info['expected_issues']}")
        print(f"  💰 Reward    : {total_reward:+.2f}")
        print(f"  📡 API calls : {api_calls}")
    
    # Summary
    print(f"\n{'='*60}")
    print("📈 FINAL RESULTS")
    print(f"{'='*60}")
    
    for task_id, result in results.items():
        diff = "EASY  " if "easy" in task_id else ("MEDIUM" if "medium" in task_id else "HARD  ")
        print(f"  [{diff}] {task_id:30s} : {result['score']:.3f} ({result['steps']} steps, {result['api_calls']} API calls)")
    
    avg_score = sum(r["score"] for r in results.values()) / len(results)
    avg_steps = sum(r["steps"] for r in results.values()) / len(results)
    
    print(f"\n  🎯 Average Score    : {avg_score:.3f}")
    print(f"  📊 Average Steps    : {avg_steps:.1f}")
    print(f"  📡 Total API Calls  : {total_api_calls}")
    print(f"{'='*60}\n")
    
    return results

if __name__ == "__main__":
    run_baseline_inference()
