#!/bin/bash
set -e

echo "=== OpenEnv Pre-Submission Validation ==="
echo ""

echo "1. Testing environment import..."
python -c "from environment import CodeReviewEnv; print('   OK: Import successful')"

echo "2. Testing reset()..."
python -c "
from environment import CodeReviewEnv
env = CodeReviewEnv()
obs = env.reset()
assert obs.code, 'Observation code is empty'
assert obs.task_description, 'Task description is empty'
assert obs.remaining_steps == 50
print('   OK: reset() returns valid Observation')
"

echo "3. Testing step()..."
python -c "
from environment import CodeReviewEnv
from models import Action
env = CodeReviewEnv()
env.reset()
obs, reward, done, info = env.step(Action(action_type='approve'))
assert done, 'Episode should be done after approve'
assert 0.0 <= info['score'] <= 1.0, f'Score out of range: {info[\"score\"]}'
assert 'task_id' in info
print('   OK: step() returns (obs, reward, done, info)')
"

echo "4. Testing state()..."
python -c "
from environment import CodeReviewEnv
env = CodeReviewEnv()
env.reset()
s = env.state()
assert 'task_id' in s
assert 'found_issues' in s
assert 'expected_issues' in s
print('   OK: state() returns valid dict')
"

echo "5. Testing all 3 tasks..."
python -c "
from environment import CodeReviewEnv
env = CodeReviewEnv()
for t in ['easy_sql_injection', 'medium_race_condition', 'hard_memory_leak']:
    obs = env.reset(task_id=t)
    assert obs.code, f'No code for {t}'
print('   OK: All 3 tasks load correctly')
"

echo "6. Testing grader scores are in [0, 1]..."
python -c "
from graders import grade_task
from tasks import TASKS

for task_id, task in TASKS.items():
    # Empty state
    state_empty = {'expected_issues': task['issues'], 'found_issues': [], 'false_positives': 0}
    score_empty = grade_task(task_id, state_empty, [])
    assert 0.0 <= score_empty <= 1.0, f'Score out of range for {task_id}: {score_empty}'

    # Perfect state
    state_perfect = {'expected_issues': task['issues'], 'found_issues': task['issues'], 'false_positives': 0}
    score_perfect = grade_task(task_id, state_perfect, [])
    assert 0.0 <= score_perfect <= 1.0, f'Perfect score out of range: {score_perfect}'
    assert score_perfect >= score_empty, 'Perfect should beat empty'

print('   OK: Grader returns valid scores in [0, 1]')
"

echo "7. Testing Pydantic models..."
python -c "
from models import Observation, Action, Reward
a = Action(action_type='identify_issue', issue_type='security', line_number=2, description='SQL injection', severity='critical')
r = Reward(value=1.5, breakdown={'detection': 1.0}, final_score=0.85)
print('   OK: All models instantiate correctly')
"

echo "8. Checking required files..."
for f in Dockerfile inference.py openenv.yaml requirements.txt api.py environment.py models.py tasks.py graders.py rewards.py; do
    [ -f "$f" ] && echo "   OK: $f" || { echo "   MISSING: $f"; exit 1; }
done

echo ""
echo "=== All validations passed! ==="
echo "Ready to deploy to Hugging Face Spaces."
