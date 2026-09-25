"""Local run identity. Missing remote revisions are explicit, never inferred."""
import hashlib
import json
import subprocess
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def adapter_identity(directory):
    root = Path(directory).resolve()
    names = ['adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json',
             'tokenizer_config.json', 'chat_template.jinja']
    hashes = {n: sha256(root / n) for n in names}
    config = json.loads((root / 'adapter_config.json').read_text())
    return {'directory': str(root), 'sha256': hashes,
            'declared_base_model': config.get('base_model_name_or_path'),
            'declared_base_revision': config.get('revision'),
            'resolved_base_revision': None,
            'revision_status': 'Not independently resolved by this capture.'}


def capture():
    def git(*args):
        try:
            p = subprocess.run(['git', *args], capture_output=True, text=True, timeout=10)
            return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {'error': str(exc)}
    packages = {}
    for name in ['torch', 'unsloth', 'transformers', 'peft', 'trl', 'ollama']:
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = None
    return {'schema_version': '1', 'git_commit': git('rev-parse', 'HEAD'),
            'git_status': git('status', '--short'), 'packages': packages,
            'source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
            'remote_model_identity': 'Ollama tags and Hugging Face revisions are not pinned by this runner.'}
