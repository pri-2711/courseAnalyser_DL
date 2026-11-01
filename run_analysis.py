"""Workspace entry point for the local Course Analyzer project."""
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).parent / "courseAnalyzer_dataPreparation"
if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT))
    runpy.run_path(str(PROJECT / "run_analysis.py"), run_name="__main__")
