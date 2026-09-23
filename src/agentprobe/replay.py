"""Audit historical traces with the current scorer without running any model."""
import argparse
import json
from pathlib import Path
from collections import defaultdict
from .trajectory import Trajectory
from .tasks_io import load_test
from .scorer import score_one, is_error
from .compare import summarize

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('runs/replay_audit.json'))
    args = parser.parse_args()
    tasks = {t.task_id:t for t in load_test()}
    groups = defaultdict(list)
    for line in args.input.read_text().splitlines():
        record = json.loads(line)
        traj = Trajectory(record['task_id'], final_answer=record.get('final_answer',''),
                          termination=record.get('termination','unknown'))
        for step in record['steps']:
            traj.add_step(step['tool'],step['args'],step['result'],
                          status=step.get('status'),kind=step.get('kind','tool'))
        score = score_one(traj,tasks[traj.task_id])
        tool_steps = [s for s in traj.steps if s.kind == 'tool']
        failed = sum(is_error(s) for s in tool_steps)
        score.update(tool_calls=len(tool_steps),failed_tool_calls=failed,
                     tool_error_rate=failed/len(tool_steps) if tool_steps else 0,
                     parse_errors=sum(s.kind == "parse" for s in traj.steps))
        groups[record['model']].append(score)
    result = {'status':'replay_only','scorer_version':'2.0',
              'warning':'Rescoring cannot restore actions discarded by the old parser. Fresh inference is required.',
              'models':[{'label':label,**summarize(rows)} for label,rows in groups.items()]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    for row in result['models']:
        print(row['label'],row['task_success_rate'])
    print(result['warning'])

if __name__ == '__main__':
    main()
