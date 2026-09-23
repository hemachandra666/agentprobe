"""Exercise the installed generator and preparation from outside the checkout.

Uses a scripted reference provider, not a live model. No GPU or Ollama needed.
"""
import json
import sys
from pathlib import Path
from agentprobe import engine, generate_teacher, prepare_training
from agentprobe.tasks_io import load_train


def main():
    work = Path.cwd()
    tasks = {task.task_id: task for task in load_train()}
    class ReferenceProvider(engine.Provider):
        def __init__(self, task):
            actions, value = [], task.numbers[0]
            for op, operand in zip(task.reference_tools, task.numbers[1:]):
                actions.append(engine.Action(op, {'a': value, 'b': operand}, None, op))
                value = value + operand if op == 'add' else value * operand
            actions.append(engine.Action(None, None, str(value), str(value)))
            self.actions = iter(actions)
        def act(self, question, history):
            return next(self.actions)
    def scripted(task_id, question, model=None):
        return engine.run(task_id, question, ReferenceProvider(tasks[task_id]))
    generate_teacher.agent.run = scripted
    sys.argv = ['generate_teacher', '--runs', '1', '--max-tasks', '2']
    generate_teacher.main()
    output = work / 'teacher_data.jsonl'
    assert output.is_file(), 'installed generator did not write in the working directory'
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 2
    prepare_training.ROOT = work
    sys.argv = ['prepare_training']
    prepare_training.main()
    assert (work / 'training_split.json').is_file()
    print('Installed generator and preparation passed with scripted reference actions.')


if __name__ == '__main__':
    main()
