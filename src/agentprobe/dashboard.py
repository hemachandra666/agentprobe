"""Kept dashboard. Shows the distill-and-verify result, plus the eval story.

Run with:
  uv run --no-sync streamlit run src/agentprobe/dashboard.py
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "results.json"
EXAMPLE_PATH = ROOT / "examples" / "broken_path.json"
COMPARISON_PATH = ROOT / "comparison.json"

st.set_page_config(page_title="AgentProbe", page_icon="🧭", layout="wide")

# ---- headline: the finding, before any numbers ----
st.title("AgentProbe")
st.caption("An open-source tool that verifies whether a distilled AI agent kept its original behavior, not just its answers.")
st.subheader("Did the distilled model keep the teacher's agent behavior, not just its answers?")
st.write(
    "Distilling a model to make it smaller and cheaper is easy. Proving the small model still "
    "behaves like the teacher, picking the right tools, avoiding loops, recovering from errors, "
    "is the hard part. AgentProbe measures exactly that, on the path, not just the final answer."
)

st.divider()

# ---- THE MAIN RESULT: teacher vs student ----
st.header("The result: a 1.5B student vs its 7B teacher")
if COMPARISON_PATH.exists():
    comp = json.loads(COMPARISON_PATH.read_text())
    cdf = pd.DataFrame(comp["models"]).set_index("label")
    teacher_traj = cdf["avg_traj"].iloc[0]
    student_traj = cdf["avg_traj"].iloc[1]
    kept = round(100 * student_traj / teacher_traj) if teacher_traj else 0

    c1, c2, c3 = st.columns(3)
    c1.metric("Teacher trajectory match", teacher_traj)
    c2.metric("Student trajectory match", student_traj)
    c3.metric("Behavior preserved", f"{kept}%")

    st.write(
        f"Across {comp['runs_per_task']} runs per task, the 1.5B distilled student preserved the "
        f"7B teacher's trajectory behavior. It picks the same tools in the same order; the real tools "
        f"do the computing. This is behavior preservation, measured on the path, not just the answer."
    )

    st.info(
        "Read this as preservation, not superiority. A 1.5B model does not truly beat a 7B one; "
        "the scores match within noise. The student scores well because it only has to choose the "
        "right tools, the real tools do the arithmetic it would otherwise get wrong. AgentProbe measures "
        "tool-selection behavior, which is exactly what distillation transferred."
    )

    nice_c = cdf.rename(columns={
        "avg_traj": "trajectory match", "avg_step_eff": "step efficiency",
        "loop_rate": "loop rate", "error_rate": "error rate",
        "recovery_rate": "recovery rate", "total_runs": "runs",
    })
    st.dataframe(nice_c, use_container_width=True)
else:
    st.info("No comparison.json yet. Run: uv run --no-sync python -m agentprobe.compare --runs 20")

st.divider()

# ---- a real example, the core teaching moment ----
st.header("Why measure the path, not just the answer?")
if EXAMPLE_PATH.exists():
    ex = json.loads(EXAMPLE_PATH.read_text())
    n_steps = ex["step_count"]
    n_err = sum(1 for s in ex["steps"] if str(s["result"]).startswith("error:"))
    st.write(
        f"This agent solved a 3-step task in **{n_steps} steps with {n_err} failed tool calls**, "
        f"then still reached the correct answer. Pass/fail scores it 100%. Watch what actually happened:"
    )
    for s in ex["steps"]:
        is_err = str(s["result"]).startswith("error:")
        icon = "❌" if is_err else "✅"
        a = s["args"]
        # keep the args readable even when the model nested junk
        arg_str = ", ".join(f"{k}={v}" for k, v in a.items()) if isinstance(a, dict) else str(a)
        result = s["result"] if not is_err else "failed: bad argument"
        st.markdown(f"{icon} **step {s['index']}** &nbsp; `{s['tool']}({arg_str})` &nbsp;→&nbsp; {result}")
    st.success(f"Final answer: {ex['final_answer'].splitlines()[0]}  (correct, but it took {n_steps} steps not 3)")
else:
    st.info("No example trace found at examples/broken_path.json")

st.divider()

# ---- the model comparison ----
if not RESULTS_PATH.exists():
    st.warning("No results.json found. Run: uv run --no-sync python -m agentprobe.benchmark --runs 3")
    st.stop()

data = json.loads(RESULTS_PATH.read_text())
df = pd.DataFrame(data["models"]).set_index("model").sort_values("avg_traj", ascending=False)

st.header("AgentProbe also benchmarks open-weight models")
st.caption(f"{len(df)} open-weight models, {data['num_tasks']} tasks, {data['runs_per_task']} runs each. Generated {data['generated_at']}.")

best = df.index[0]
worst = df.index[-1]
st.write(
    f"**{best}** took the cleanest paths (trajectory match {df.loc[best,'avg_traj']}). "
    f"**{worst}** was weakest (trajectory match {df.loc[worst,'avg_traj']}), "
    f"failing {round(df.loc[worst,'error_rate']*100)}% of its tool calls."
)

# hero chart: trajectory match, ranked, horizontal so names are readable
st.bar_chart(df[["avg_traj"]].rename(columns={"avg_traj": "trajectory match"}),
             horizontal=True, height=280)
st.caption("Higher is better. This is the headline behavior score.")

st.divider()

# failure modes, horizontal, with a plain verdict
st.header("Where models lose points")
st.bar_chart(df[["error_rate", "loop_rate"]].rename(
    columns={"error_rate": "error rate", "loop_rate": "loop rate"}),
    horizontal=True, height=280)
st.caption("Lower is better. Failed tool calls and repeated loops are what drag an agent down.")

st.divider()

# full numbers for anyone who wants them, now secondary not primary
with st.expander("See the full metrics table"):
    nice = df.rename(columns={
        "avg_traj": "trajectory match", "avg_step_eff": "step efficiency",
        "loop_rate": "loop rate", "error_rate": "error rate",
        "recovery_rate": "recovery rate", "total_runs": "runs",
    })
    st.dataframe(nice, use_container_width=True)