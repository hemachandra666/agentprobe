"""Minimal model client. Wraps a local Ollama model behind one call."""
from __future__ import annotations
import ollama

DEFAULT_MODEL = "qwen2.5:7b"


def chat(messages, model: str = DEFAULT_MODEL, tools=None):
    """Send messages to the local model and return the raw response.

    messages: list of {"role": ..., "content": ...}
    tools:    optional list of tool schemas for tool-calling
    """
    return ollama.chat(model=model, messages=messages, tools=tools)


if __name__ == "__main__":
    resp = chat([{"role": "user", "content": "Reply with exactly one short sentence."}])
    print(resp["message"]["content"])