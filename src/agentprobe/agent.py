"""Runs a tool-using agent loop and records its trajectory."""
from __future__ import annotations
from . import model as model_client, tools
from .trajectory import Trajectory
from .model import DEFAULT_MODEL

MAX_STEPS = 8  # safety cap so a confused agent cannot loop forever


def run(task_id: str, question: str, model: str = DEFAULT_MODEL) -> Trajectory:
    traj = Trajectory(task_id=task_id)
    messages = [
        {"role": "system", "content": "You are a calculator agent. Use the tools to compute, then give the final number."},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_STEPS):
        resp = model_client.chat(messages, model=model, tools=tools.SCHEMAS)
        msg = resp["message"]
        calls = msg.get("tool_calls") or []

        if not calls:
            traj.final_answer = msg.get("content", "").strip()
            break

        messages.append(msg)
        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            fn = tools.REGISTRY.get(name)
            try:
                if fn is None:
                    result = f"error: unknown tool {name}"
                elif not isinstance(args, dict):
                    result = f"error: bad arguments, expected an object, got {type(args).__name__}"
                else:
                    result = fn(**args)
            except Exception as e:
                result = f"error: {e}"
            traj.add_step(name, args, result)
            messages.append({"role": "tool", "content": str(result), "name": name})

    return traj


if __name__ == "__main__":
    t = run("smoke", "What is 6 times 7?")
    print("tools used:", t.tool_sequence())
    print("steps:", t.step_count)
    print("answer:", t.final_answer)