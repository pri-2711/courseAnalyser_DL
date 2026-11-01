"""Build the submission ZIP from the project without modifying the project.

Files are copied to a staging folder, the staged copies are sanitised (absolute local paths and
test-account identifiers are rewritten), and the staging folder is zipped. Nothing in the project
folder is changed by this script.

Run from the project folder:  .venv\\Scripts\\python.exe scripts\\build_package.py
"""
from pathlib import Path
import json
import re
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_NAME = "CourseAnalyzer"
OUTPUT_ZIP = ROOT.parent / f"{ARCHIVE_NAME}_submission.zip"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "node_modules", "*.tmp*", "*.log")

COPY_TREES = ["backend", "src", "scripts", "tests", "models/all-MiniLM-L6-v2",
              "outputs/models", "outputs/training/reports", "outputs/training/graphs",
              "frontend/src", "frontend/dist"]
COPY_FILES = ["README.md", "requirements.txt", "requirements-dev.txt", "pytest.ini", ".gitignore",
              "run_analysis.py", "train_models.py", "models/reference_embeddings.npy",
              "frontend/package.json", "frontend/package-lock.json", "frontend/vite.config.js",
              "frontend/index.html", "outputs/verification/browser_verification.json"]
EMPTY_DIRS = ["data/input", "data/courses", "data/persisted"]
PATH_PREFIX = re.compile(r"D:\\\\university\\\\year4\\\\DL\\\\DL-project\\\\courseAnalyzer_dataPreparation\\\\")
ACCOUNT = re.compile(r"[\w.-]*browser-\d+@example\.test")


def sanitise(path):
    text = path.read_text(encoding="utf-8")
    cleaned = PATH_PREFIX.sub("", text)
    cleaned = ACCOUNT.sub("<test-account>", cleaned)
    if cleaned != text:
        path.write_text(cleaned, encoding="utf-8")
        json.loads(cleaned)  # the sanitised document must still be valid JSON
        return True
    return False


def main():
    stage = Path(tempfile.mkdtemp(prefix="course_analyzer_stage_")) / ARCHIVE_NAME
    stage.mkdir(parents=True)
    for tree in COPY_TREES:
        source = ROOT / tree
        if not source.exists():
            raise FileNotFoundError(f"Missing required folder: {tree}")
        shutil.copytree(source, stage / tree, ignore=IGNORE)
    for name in COPY_FILES:
        source = ROOT / name
        if not source.is_file():
            raise FileNotFoundError(f"Missing required file: {name}")
        (stage / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, stage / name)
    for folder in EMPTY_DIRS:
        (stage / folder).mkdir(parents=True, exist_ok=True)
        (stage / folder / ".gitkeep").write_text("", encoding="utf-8")
    sanitised = []
    for name in ("outputs/training/reports/training_report.json", "outputs/verification/browser_verification.json"):
        if sanitise(stage / name):
            sanitised.append(name)
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    files = 0
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage.parent).as_posix())
                files += 1
    shutil.rmtree(stage.parent, ignore_errors=True)
    print(f"sanitised: {sanitised}", flush=True)
    print(f"files: {files}", flush=True)
    print(f"zip: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / 1_048_576:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
