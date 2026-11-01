"""Workspace entry point for the local Streamlit Course Analyzer."""
from pathlib import Path
import runpy
import sys

PROJECT = Path(__file__).parent / "courseAnalyzer_dataPreparation"
sys.path.insert(0, str(PROJECT))
runpy.run_path(str(PROJECT / "app.py"), run_name="__main__")
