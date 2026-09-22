"""Display saved AgentProbe evaluation results."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AgentProbe",
    page_icon="🔍",
    layout="wide",
)

st.title("AgentProbe")
st.write("Teacher–student evaluation for arithmetic tool use.")

# Prefer the published evidence, then your completed experiment.
candidates = [
    Path("docs/results/2026-09-22/comparison.json"),
    Path("runs/40af95f76f6841daba30d05622990717/comparison.json"),
    Path("comparison.json"),
]
default_path = next(
    (path for path in candidates if path.is_file()),
    candidates[0],
)

with st.expander("Choose evaluation file"):
    selected_path = st.text_input(
        "Comparison JSON file",
        value=str(default_path),
        help="Enter a path relative to the repository directory.",
    )

path = Path(selected_path).expanduser()

try:
    data = json.loads(path.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    st.error(f"Could not load evaluation results: {exc}")
    st.info("Open 'Choose evaluation file' and enter a valid JSON path.")
    st.stop()

if not isinstance(data, dict):
    st.error("Expected a JSON object containing evaluation results.")
    st.stop()

if (
    data.get("scorer_version") != "2.0"
    or data.get("status") != "complete"
):
    st.warning(
        "Historical or incomplete experiment. "
        "These are not validated current task-success results."
    )
    with st.expander("Inspect saved data"):
        st.json(data)
    st.stop()

models = data.get("models", [])
if not models or not all(isinstance(model, dict) for model in models):
    st.error("This file contains no valid model summaries.")
    st.stop()

st.caption(
    f"Tasks: {data.get('num_test_tasks', 'Unknown')}  •  "
    f"Runs per task: {data.get('runs_per_task', 'Unknown')}  •  "
    f"Scorer: {data['scorer_version']}"
)


def percentage(value):
    return "N/A" if value is None else f"{value:.2%}"


def model_name(model):
    label = model.get("label", "Unknown model")
    names = {
        "teacher (qwen2.5:7b)": "Teacher · 7B",
        "untuned student (1.5B base)": "Untuned student · 1.5B",
        "tuned student (1.5B distilled)": "Tuned student · 1.5B",
    }
    return names.get(label, label)


# Headline results.
columns = st.columns(len(models))
for column, model in zip(columns, models):
    with column:
        st.metric(
            model_name(model),
            percentage(model.get("task_success_rate")),
        )
        st.caption("Task success")

st.subheader("Model comparison")

chart = pd.DataFrame(
    [
        {
            "Model": model_name(model),
            "Task success (%)": (
                model.get("task_success_rate", 0) * 100
            ),
            "Final answer correct (%)": (
                model.get("answer_correct_rate", 0) * 100
            ),
            "Tool result correct (%)": (
                model.get("tool_result_correct_rate", 0) * 100
            ),
        }
        for model in models
    ]
).set_index("Model")

st.bar_chart(chart)

summary = pd.DataFrame(
    [
        {
            "Model": model_name(model),
            "Task success": percentage(model.get("task_success_rate")),
            "Correct final answer": percentage(
                model.get("answer_correct_rate")
            ),
            "Correct tool result": percentage(
                model.get("tool_result_correct_rate")
            ),
            "Completion": percentage(model.get("completion_rate")),
            "Runs with parsing errors": model.get("parse_error_runs"),
            "Tool error rate": percentage(model.get("error_rate_micro")),
            "Runs with loops": percentage(model.get("loop_any_rate")),
            "Evaluated runs": model.get("total_runs"),
        }
        for model in models
    ]
)

st.dataframe(summary, hide_index=True, use_container_width=True)

st.caption(
    "Tool error rate measures failed tool executions. Parsing errors and "
    "incorrect answers are separate. Completion does not imply correctness."
)

st.subheader("Results by task family")

for model in models:
    with st.expander(model_name(model)):
        families = model.get("by_family", {})
        if not families:
            st.info("No family breakdown was saved.")
            continue

        family_table = pd.DataFrame(
            [
                {
                    "Task family": family,
                    "Runs": metrics.get("total_runs"),
                    "Task success": percentage(
                        metrics.get("task_success_rate")
                    ),
                    "Correct final answer": percentage(
                        metrics.get("answer_correct_rate")
                    ),
                    "Correct tool result": percentage(
                        metrics.get("tool_result_correct_rate")
                    ),
                    "Runs with parsing errors": metrics.get(
                        "parse_error_runs"
                    ),
                }
                for family, metrics in families.items()
            ]
        )

        st.dataframe(
            family_table,
            hide_index=True,
            use_container_width=True,
        )

st.info(
    "Scope: synthetic arithmetic tasks with two calculator tools. "
    "These results do not establish general agent reliability or "
    "measured speed and memory advantages."
)

with st.expander("Experiment details"):
    st.write("Experiment ID:", data.get("experiment_id", "Unknown"))
    st.write("Loaded file:", str(path))
    st.write("Task dataset hash:", data.get("task_sha256", "Unknown"))
    st.json(data.get("generation", {}))