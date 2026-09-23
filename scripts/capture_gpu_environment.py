"""Record the current GPU environment without installing or changing packages."""
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

root = Path(__file__).resolve().parents[1]
packages = {}
for dist in distributions():
    name = dist.metadata.get("Name")
    if name and name.lower().replace("_", "-") != "agentprobe":
        packages[name] = dist.version
if "torch" not in {name.lower() for name in packages}:
    raise SystemExit("Run this script with .venv-gpu/bin/python; torch is missing.")
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
    check=True, capture_output=True, text=True,
)
metadata = {
    "captured_at_utc": datetime.now(timezone.utc).isoformat(),
    "note": "Current environment snapshot; not automatic proof of historical experiment identity.",
    "python": platform.python_version(),
    "platform": platform.platform(),
    "gpu_name_driver_memory": result.stdout.strip().splitlines(),
    "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
}
requirements = root / "requirements-gpu.txt"
requirements.write_text(
    "# Captured installed versions; not a validated portable GPU lockfile.\n"
    "# Project editable installation excluded; install this repository separately.\n"
    "# CUDA wheels need the indexes documented in README.md.\n"
    + "".join(f"{name}=={version}\n" for name, version in metadata["packages"].items())
)
output = root / "docs/results/2026-09-22/gpu_environment.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(metadata, indent=2) + "\n")
print(f"Recorded {len(packages)} packages in {requirements}")
print(f"Recorded hardware metadata in {output}")
