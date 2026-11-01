# Local Course Analyzer

The application lives in [`courseAnalyzer_dataPreparation`](courseAnalyzer_dataPreparation/README.md). It combines the preserved frozen SBERT analysis workflow with a local React/Vite frontend, FastAPI API, bcrypt account authentication, SQLite ownership records, and per-course review uploads.

## Launch the React + FastAPI app (PowerShell)

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

In terminal 1, from this workspace root:

```powershell
$taskSecret = [Environment]::GetEnvironmentVariable('COURSE_ANALYZER_SECRET', 'User')
if (-not $taskSecret) {
  $taskSecret = python -c "import secrets; print(secrets.token_urlsafe(48))"
  [Environment]::SetEnvironmentVariable('COURSE_ANALYZER_SECRET', $taskSecret, 'User')
}
$env:COURSE_ANALYZER_SECRET = $taskSecret
python -m uvicorn app.main:app --app-dir courseAnalyzer_dataPreparation/backend --host 127.0.0.1 --port 8000
```

In terminal 2:

```powershell
cd courseAnalyzer_dataPreparation/frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173>. Keep the API secret stable in your environment between launches. Do not commit it.

To use the existing command-line analyzer, run `python run_analysis.py` from this root. The previous Streamlit dashboard remains available with `streamlit run app.py`.

The recovered review data has no course ID, so it is never linked to the separate course catalog. Platform URL support currently extracts permitted public metadata; course reviews must be provided through local CSV/JSON uploads.
