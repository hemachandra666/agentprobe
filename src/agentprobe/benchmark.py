"""Evaluate multiple Ollama models with the same current harness."""
from .compare import main
if __name__ == "__main__":
    main(default_models=["qwen2.5:7b", "llama3.1:8b", "mistral:7b"])
