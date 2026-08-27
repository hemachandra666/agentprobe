"""Runs a tool-using agent loop and records its trajectory."""
from __future__ import annotations
from . import model, tools
from .trajectory import Trajectory

MAX_STEPS = 8  # safety cap so a confused agent cannot loop forever


def run(task_id: str, question: str) -> Trajectory:
    traj = Trajectory(task_id=task_id)
    messages = [
        {"role": "system", "content": "You are a calculator agent. Use the tools to compute, then give the final number."},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_STEPS):
        resp = model.chat(messages, tools=tools.SCHEMAS)
        msg = resp["message"]
        calls = msg.get("tool_calls") or []

        if not calls:
            traj.final_answer = msg.get("content", "").strip()
            break

        messages.append(msg)  # keep the model's tool-call turn in history
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