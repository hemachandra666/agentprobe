"""Teacher provider: an Ollama model with structured tool calling, in the shared loop."""
from __future__ import annotations
from . import model as model_client, tools, engine
from .model import DEFAULT_MODEL


class OllamaProvider(engine.Provider):
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def act(self, question: str, history: list) -> engine.Action:
        messages = [
            {"role": "system", "content": engine.SYSTEM},
            {"role": "user", "content": question},
        ]
        for role, content in history:
            if role == "assistant":
                messages.append({"role": "assistant", "content": content})
            else:
                messages.append({"role": "tool", "content": str(content), "name": "tool"})

        resp = model_client.chat(messages, model=self.model, tools=tools.SCHEMAS)
        msg = resp["message"]
        calls = msg.get("tool_calls") or []
        raw = str(msg)

        # Only treat as final when there are NO tool calls. If the model emitted
        # a tool call, execute it and continue, even if it also wrote text.
        if calls:
            call = calls[0]  # ONE action at a time (shared-loop rule)
            return engine.Action(
                tool=call["function"]["name"],
                args=call["function"]["arguments"],
                final_answer=None,
                raw=raw,
            )

        # Ollama returns a Message object; get content safely as a clean string.
        content = getattr(msg, "content", None)
        if content is None:
            content = msg.get("content", "") if hasattr(msg, "get") else ""
        content = content if isinstance(content, str) else ""
        return engine.Action(tool=None, args=None, final_answer=content, raw=raw)


def run(task_id: str, question: str, model: str = DEFAULT_MODEL):
    return engine.run(task_id, question, OllamaProvider(model))