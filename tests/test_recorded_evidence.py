"""Keep the published experiment and training snapshot reproducible on CPU."""
import json
import math
import shutil
import sys
from pathlib import Path

from agentprobe import prepare_training, replay

ROOT = Path(__file__).resolve().parents[1]


def test_published_evaluation_replays_without_changing_metrics(tmp_path, monkeypatch):
    evidence = ROOT / 'docs/results/2026-09-22'
    output = tmp_path / 'replay.json'
    monkeypatch.setattr(sys, 'argv', ['replay', '--input', str(evidence / 'comparison_runs.jsonl'),
                                     '--data-dir', str(ROOT / 'src/agentprobe/data'),
                                     '--output', str(output)])
    replay.main()
    saved = json.loads((evidence / 'comparison.json').read_text())
    actual = json.loads(output.read_text())
    assert actual['summary_checked']
    def equal(old, new):
        for key, value in old.items():
            if isinstance(value, dict):
                equal(value, new[key])
            elif isinstance(value, float):
                assert math.isclose(value, new[key], rel_tol=0, abs_tol=1e-12), key
            else:
                assert value == new[key], key
    assert len(saved['models']) == len(actual['models']) == 3
    for expected, reproduced in zip(saved['models'], actual['models']):
        equal(expected, reproduced)


def test_published_training_preparation_is_byte_identical(tmp_path, monkeypatch):
    shutil.copy(ROOT / 'teacher_data.jsonl', tmp_path / 'teacher_data.jsonl')
    monkeypatch.setattr(prepare_training, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['prepare_training', '--data-dir', str(ROOT / 'src/agentprobe/data')])
    prepare_training.main()
    for name in ['train_conversations.jsonl', 'validation_conversations.jsonl', 'training_split.json']:
        assert (tmp_path / name).read_bytes() == (ROOT / name).read_bytes(), name
