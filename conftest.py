"""pytest bootstrap: ensure the repo root is importable.

Tests live in tests/ but import root-level modules (e.g. cluster_select). Under
pytest's default 'prepend' import mode only the test file's own directory is put
on sys.path, so we add the repo root (this file's directory) explicitly. With
this file present, `uv run pytest` works as well as `uv run python -m pytest`.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
