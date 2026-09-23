"""Evaluate the teacher with the current scorer and dataset."""
from .compare import main
if __name__ == "__main__":
    main(default_models=["teacher"])
