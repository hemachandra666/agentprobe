"""AgentProbe dashboard: the honest three-way result.

Run with:
  uv run --no-sync streamlit run src/agentprobe/dashboard.py
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
COMPARISON_PATH = ROOT / "comparison.json"

st.set_page_config(page_title="AgentProbe", page_icon="🧭", layout="wide")

st.title("AgentProbe")
st.caption("Honest evaluation infrastructure for AI agents. It measures whether an agent "
           "actually behaves correctly, not just whether it looks like it did.")

st.subheader("The correction that is the whole point")
st.write(
    "A first version of this project reported that a distilled 1.5B student preserved "
    "**102%** of its 7B teacher's behavior. That number was an artifact of four measurement "
    "flaws: a leaky train/test split, a scorer that checked tool names but not answers, an "
    "unfair execution loop, and no untuned baseline. After fixing all four and re-measuring "
    "on genuinely unseen tasks, the honest result is very different."
)

st.divider()

st.header("The honest result")
if COMPARISON_PATH.exists():
    comp = json.loads(COMPARISON_PATH.read_text())
    rows = comp["models"]
    df = pd.DataFrame(rows)

    # pull success rates by label for the headline metrics
    def rate(substr):
        for r in rows:
            if substr in r["label"]:
                return r["task_success_rate"]
        return None

    teacher = rate("teacher")
    untuned = rate("untuned")
    tuned = rate("distilled")  # match the distinctive word; "tuned" also matches "untuned"

    c1, c2, c3 = st.columns(3)
    c1.metric("Teacher (7B)", f"{round((teacher or 0)*100)}%")
    c2.metric("Tuned student (1.5B)", f"{round((tuned or 0)*100)}%")
    c3.metric("Untuned student (1.5B)", f"{round((untuned or 0)*100)}%")

    st.write(
        f"Task success = the agent reached the **correct final answer** on unseen tasks, "
        f"through the same execution loop for every model. Fine-tuning moved the small model "
        f"from **{round((untuned or 0)*100)}%** to **{round((tuned or 0)*100)}%**: a real but "
        f"small gain, far below the teacher's **{round((teacher or 0)*100)}%**."
    )

    st.info(
        "The lesson: agent distillation is much harder than accuracy-style metrics suggest, "
        "and naive evaluations dramatically overstate it. The value here is the measurement "
        "that revealed the truth, not the distillation itself."
    )

    # honest bar chart of the true three-way success rates
    chart_df = pd.DataFrame({
        "task success": [teacher or 0, tuned or 0, untuned or 0],
    }, index=["teacher (7B)", "tuned student (1.5B)", "untuned student (1.5B)"])
    st.bar_chart(chart_df, horizontal=True, height=220)
    st.caption("Task success on unseen tasks. The tuned student barely clears the untuned "
               "baseline, and both sit far below the teacher.")

    st.subheader("Full metrics on unseen tasks")
    nice = df.rename(columns={
        "label": "model",
        "task_success_rate": "task success",
        "answer_correct_rate": "answer correct",
        "avg_step_efficiency": "step efficiency",
        "error_rate": "error rate",
        "loop_any_rate": "loop rate",
        "total_runs": "runs",
    }).set_index("model")
    st.dataframe(nice, use_container_width=True)

    st.caption(f"Evaluated on {comp.get('num_test_tasks', '?')} unseen tasks, "
               f"{comp.get('runs_per_task', '?')} run(s) each.")
else:
    st.warning("No comparison.json yet. Run: uv run --no-sync python -m agentprobe.compare --runs 1 --max-tasks 50")

st.divider()

st.header("Why naive metrics hide this")
st.write(
    "Two agents solve \"add 3 and 4, then multiply by 2.\" Both call add then multiply. "
    "One computes add(3,4)=7, multiply(7,2)=14. The other computes add(100,200)=300, "
    "multiply(300,2)=600. A tool-name check scores both a perfect 1.0. The answer is 14. "
    "AgentProbe checks the actual computed answer, so the second trace correctly fails. "
    "That single fix is most of why the honest numbers dropped so far."
)
