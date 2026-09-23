"""Teacher provider: an Ollama model with structured tool calling, in the shared loop."""
from __future__ import annotations
import json
from . import model as model_client, tools, engine
from .model import DEFAULT_MODEL


def _clean_content(msg) -> str:
    content = getattr(msg, "content", None)
    if content is None:
        content = msg.get("content", "") if hasattr(msg, "get") else ""
    return content if isinstance(content, str) else ""


class OllamaProvider(engine.Provider):
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def _rebuild(self, question: str, history: list) -> list:
        """Turn engine history into a proper Ollama message list."""
        messages = [
            {"role": "system", "content": engine.SYSTEM},
            {"role": "user", "content": question},
        ]
        for index, entry in enumerate(history):
            kind = entry[0]
            if kind == "call":
                _, tool, args = entry
                # show the assistant making the tool call
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": tool, "arguments": args}}],
                })
            elif kind == "observation":
                if index and history[index - 1][0] == "call":
                    messages.append({"role": "tool", "content": str(entry[1]),
                                     "name": history[index - 1][1]})
                else:
                    messages.append({"role": "user", "content": str(entry[1])})
            elif kind == "assistant":
                messages.append({"role": "assistant", "content": str(entry[1])})
        return messages

    def act(self, question: str, history: list) -> engine.Action:
        messages = self._rebuild(question, history)
        resp = model_client.chat(messages, model=self.model, tools=tools.SCHEMAS)
        msg = resp["message"]
        calls = msg.get("tool_calls") or []
        raw = json.dumps(msg.model_dump() if hasattr(msg, "model_dump") else msg, default=str)

        if not isinstance(calls, list):
            return engine.Action(None, None, None, raw, error="tool_calls must be a list")
        if len(calls) > 1:
            return engine.Action(None, None, None, raw, error="multiple tool calls in one turn")

        if calls:
            call = calls[0]
            try:
                function = call["function"]
                name, arguments = function["name"], function["arguments"]
            except (KeyError, TypeError):
                return engine.Action(None, None, None, raw, error="malformed structured call")
            return engine.Action(
                tool=name,
                args=arguments,
                final_answer=None,
                raw=raw,
            )

        return engine.Action(tool=None, args=None, final_answer=_clean_content(msg), raw=raw)


def run(task_id: str, question: str, model: str = DEFAULT_MODEL):
    return engine.run(task_id, question, OllamaProvider(model))