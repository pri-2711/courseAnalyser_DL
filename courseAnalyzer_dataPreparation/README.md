# Course Analyzer

A local course intelligence application with a React/Vite interface, a FastAPI backend, SQLite account metadata, and the existing frozen `all-MiniLM-L6-v2` SBERT reference-embedding classifier. It supports per-user course workspaces, public course metadata when a page permits it, locally uploaded reviews, cached review embeddings, sentiment summaries, rating comparisons, and semantic review evidence retrieval.

The Streamlit dashboard has been removed; the React + FastAPI application is the only UI. The command-line pipelines (`run_analysis.py`, `train_models.py`) are retained. No Hadoop, Spark, MongoDB, Docker, paid service, or remote LLM is required.

## Fresh Windows setup from the ZIP

Requirements: Windows 10 or 11, Python 3.12 (python.org installer), and Node.js 20 or newer. Internet access is needed only for the one-time dependency installs and for course URL extraction. The trained models, the local SBERT encoder and the reference vectors are included, so nothing is downloaded or retrained at startup.

From the extracted project folder, in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
```

Start the backend in terminal 1. The first launch creates `data/app.sqlite3` automatically. Startup takes about a minute, while the packaged encoder and classifiers load:

```powershell
.\.venv\Scripts\Activate.ps1
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$taskSecret = [Environment]::GetEnvironmentVariable('COURSE_ANALYZER_SECRET', 'User')
if (-not $taskSecret) {
  $taskSecret = python -c "import secrets; print(secrets.token_urlsafe(48))"
  [Environment]::SetEnvironmentVariable('COURSE_ANALYZER_SECRET', $taskSecret, 'User')
}
$env:COURSE_ANALYZER_SECRET = $taskSecret
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Start the frontend in terminal 2:

```powershell
cd frontend
npm run preview -- --host 127.0.0.1 --port 5173 --strictPort
```

Open <http://127.0.0.1:5173>, create an account, and paste a course URL. Health check: <http://127.0.0.1:8000/health>. To run the test suite, install the development requirements first (`python -m pip install -r requirements-dev.txt`), then run `python -m pytest tests -q`.

The ZIP does not include the raw 280 MB review export (`data/input/reviews.json`), the local user database, or any user's saved analyses. None of these are needed to run the application. They are needed only to re-run the CLI training pipeline.

## Start the application (Windows PowerShell)

Install Python dependencies from the project directory:

```powershell
python -m pip install -r requirements.txt
```

In terminal 1, set a persistent secret in your user environment (choose a stable random value and keep it private), then start the API:

```powershell
$taskSecret = [Environment]::GetEnvironmentVariable('COURSE_ANALYZER_SECRET', 'User')
if (-not $taskSecret) {
  $taskSecret = python -c "import secrets; print(secrets.token_urlsafe(48))"
  [Environment]::SetEnvironmentVariable('COURSE_ANALYZER_SECRET', $taskSecret, 'User')
}
$env:COURSE_ANALYZER_SECRET = $taskSecret
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

The secret must remain the same between launches or existing signed sessions will no longer validate. Keep it in an environment variable; never commit it. Optional variables are `COURSE_ANALYZER_DB`, `COURSE_ANALYZER_DATA`, `COURSE_ANALYZER_ORIGINS`, and `COURSE_ANALYZER_COOKIE_SECURE` (set the last to `1` when serving through HTTPS).

In terminal 2:

```powershell
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173>. The API docs are at <http://127.0.0.1:8000/docs>.

## Reproducing the pipeline (offline, CPU is sufficient)

Run from the project directory after `python -m pip install -r requirements.txt`:

1. `python run_analysis.py` — validates `data/input/reviews.json` (568,453 Amazon Fine Food reviews), cleans text, generates SBERT (`all-MiniLM-L6-v2`) embeddings in resumable 5,000-row chunks, predicts reference-similarity sentiment, and writes the report, CSV summaries and PNG charts under `data/persisted/`. It checkpoints after every chunk and resumes automatically.
2. `python train_models.py` — trains the supervised classifiers on the cached embeddings (no re-embedding of completed rows), writes reports and graphs under `outputs/training/`, and packages the deployable model under `outputs/models/`.

**Labels are weak labels.** Ratings are mapped to sentiment: 1–2 negative, 3 neutral, 4–5 positive. They are not manually verified sentiment labels, so all metrics measure agreement with rating-derived labels.

### Model categories

| Model | Category |
|---|---|
| SBERT reference-similarity baseline | Transformer feature extractor (SBERT) with a non-trained similarity rule |
| Logistic Regression | Traditional Machine Learning |
| Linear SVM | Traditional Machine Learning |
| MLP (384→256→128→64→3, PyTorch) | Deep Learning (supervised classifier on SBERT embeddings) |

Training uses duplicate-safe, stratified group splits (normalized review text is the group key), so identical reviews never cross train/validation/test. Class weights are computed from training labels only. The MLP uses Adam (lr 0.001, weight decay 0.0001), CrossEntropyLoss, batch size 256, up to 30 epochs, ReduceLROnPlateau on validation loss (factor 0.5, patience 2, min lr 1e-5), and early stopping after 5 epochs without validation macro-F1 improvement. The best-validation-macro-F1 checkpoint is restored before test evaluation.

Results are in `outputs/training/reports/model_comparison.csv`, `training_report.json`, `evaluation_summary.md`, and the charts in `outputs/training/graphs/`.

### Portable package (`outputs/models/`)

- `sentiment_model.joblib` — Linear SVM (best traditional model by validation macro-F1)
- `mlp/mlp_classifier.pt` and `mlp/mlp_metadata.json` — MLP state dict, architecture, label mapping, feature standardisation parameters, and training configuration
- `sbert_model/` — complete local SentenceTransformer encoder
- `reference_embeddings.npy`, `model_metadata.json` — reference vectors and relative paths, default classifier, label mapping, preprocessing

The package uses relative paths only and does not read the Hugging Face cache. Recipients do not need to regenerate embeddings or retrain.

### Previous Streamlit dashboard

The Streamlit application (`app.py`) and its dependency were removed. The equivalent results are in the React Model Evaluation page and the Dashboard.

## How URL analysis works

The URL extractor validates HTTPS URLs and resolves host addresses before fetching. It rejects private, reserved, local, and metadata network destinations, does not automatically follow redirects, and checks `robots.txt` before reading a course page. Platform support for reviews, verified against live public pages:

| Platform | Course metadata | Public review text |
|---|---|---|
| Coursera | Yes (JSON-LD: title, description, provider, rating, ratings count) | Yes, for the first set of reviews embedded on the course page (about 20). Further pages are loaded from an API path disallowed by Coursera's robots.txt, so they are not requested. |
| edX | Yes | No. The public course page exposes no review text. |
| Udemy | No. robots.txt returns HTTP 403, so the page is not fetched. | No |
| Other educational sites (generic adapter, verified on freeCodeCamp and W3Schools) | Yes: JSON-LD Article/Course data, Open Graph and meta tags, heading outline. Fields the page does not state are listed as not published. | No public review text found, so none are counted. |

Extraction distinguishes these outcomes, shown in the dashboard panel:

| `extraction_status` | Meaning |
|---|---|
| `reviews_retrieved` | Metadata and at least one public review were retrieved |
| `metadata_only` | Resource metadata was retrieved. No public reviews are available, which is not an error |
| `extraction_failed` | The page could not be read or contained no usable metadata |
| `robots_disallowed` / `robots_unavailable` | robots.txt forbids the page, or permission could not be verified, so the page was not fetched |
| `access_restricted` | The page showed an access check (for example a bot challenge or "access denied" page). It is not bypassed, and no content is taken from it |

Failed statuses are shown as a failure in the panel with the reason. A `metadata_only` result is a success: the course information is shown above the review summary, and the absence of reviews is stated plainly.

Fetches are limited to one request per second per host, redirects are validated, private and reserved addresses are rejected, and `robots.txt` is checked before every page. Authentication, CAPTCHA and paywalls are never bypassed.

**Course URL workflow.** Submitting a URL runs extraction, saves the course metadata, and passes any retrieved reviews through the same embedding and classification path as uploads. Each review is de-duplicated by its review ID, so resubmitting a URL reuses the user's existing course record and adds only new reviews. The stored predictions carry the reference-classifier label and the MLP and linear-SVM labels.

Extraction never bypasses login, CAPTCHA, paywalls, or robots.txt, and it never invents reviews or ratings. Where review text is unavailable, the response says why, and a CSV/JSON/JSONL upload is the fallback. Resubmitting the same URL reuses the user's course record, and reviews are de-duplicated by review ID. Extracted reviews go through the same embedding and classification path as uploads, and the trained MLP and linear SVM labels are stored beside the reference-classifier labels. No model is retrained.

Supported metadata fields are populated when exposed by public structured data: course title, description, instructor/provider, platform, rating, review count, price, category, level, duration, language, and URL. Missing values stay missing. Extraction status, review count retrieved, and the upload requirement are returned to the interface.

## SBERT method and cache

The app retains the existing model (`sentence-transformers/all-MiniLM-L6-v2`) and reference phrases:

| Class | Reference phrases |
|---|---|
| Positive | excellent course; great learning; amazing content |
| Negative | bad course; poor explanation; waste of time |
| Neutral | average course; okay content |

Reviews and reference phrases are encoded by Sentence Transformers/PyTorch. Embeddings are L2 normalized, and the dot product is cosine similarity. For each class, the class score is the maximum similarity to its reference phrases; the highest class score wins. The confidence field is the winning score minus the runner-up score. A margin below `0.03` is marked uncertain, while the predicted label is preserved. This is semantic similarity classification, not a fine-tuned sentiment model; similarity and confidence are not accuracy or probability.

An uploaded file is fingerprinted. Re-uploading the same file returns its saved predictions. New review IDs are inferred once and appended to prior predictions; when no ID column (`review_id`, `_id`, or `id`) is provided, a stable hash of the row is used. **Rebuild analysis** explicitly reprocesses the supplied sample. Each account's course record is ownership-scoped in SQLite, and its review source, predictions, and embeddings are stored separately under `data/courses/<analysis-id>/`. The local model/tokenizer and reference vectors are saved under `models/` after successful analysis when not already present. Removing an analysis deletes only its own generated course directory and database record.

The API accepts review text fields recognized by the existing pipeline (`review_clean`, `review`, `review_text`, `text`, `comment`, or `content`), and optional `platform`, rating, review ID, and date fields. Empty review text is skipped. Uploads are limited to 20 MB and CSV, JSON, JSONL, or NDJSON. The initial user-supplied reviews are not assumed to belong to any recovered catalog item. Existing recovered review records currently have no course identifier and are not attached to courses.

## Metrics and evidence assistant

Course dashboards show review counts by positive/negative/neutral class, class share, sentiment value `(positive_count - negative_count) / total_reviews` (neutral rows remain in the denominator), average available review rating, and the rating-versus-predicted-label table/chart. These are descriptive metrics over the uploaded sample. No trend, reliability, course-quality, or ranking score is generated without supporting data and a defined method.

The assistant embeds the question with the same local SBERT model and retrieves the five nearest saved review vectors by cosine similarity. It returns the matching review excerpts with predicted class and similarity as evidence. It does not generate a free-form claim or use an external LLM.

The course dashboard accepts a separate manually labelled sample and displays accuracy, macro precision/recall/F1, per-class report, confusion matrix, and a small set of misclassified examples. The API only computes these metrics when you upload labels; labels must use a recognized review-text column and `label` or `sentiment`. The existing CLI evaluation flow in `run_analysis.py` supports the same metrics via `--evaluation`. No training metrics or fabricated evaluation results are reported.

## Local API and data layout

- `/auth/register`, `/auth/login`, `/auth/logout`, `/auth/me`
- `/recommendations` (content-based search over the signed-in user's own saved resources)
- `/model/status`, `/model/predict`, `/model/graphs/{name}` (packaged models; read-only, never retrains)
- `/courses/analyze`, `/courses`, `/courses/{id}`, `/courses/{id}/analytics`
- `/courses/{id}/reviews`, `/courses/{id}/rebuild`, and `/courses/{id}/evaluation` (local CSV/JSON upload)
- `/courses/compare`, `/courses/{id}/ask`, `/analyses/history`

Passwords are hashed using bcrypt. The signed session cookie is HttpOnly, SameSite=Lax, and has a seven-day expiration. Course queries require the account owner. Login attempts are rate-limited per source/email in-process. The database uses parameterized SQLite queries. Do not expose the development server to an untrusted network; the UI is intended for local use.

```text
backend/app/       FastAPI, auth, SQLite, URL adapters
frontend/src/      React responsive EduSphere-inspired UI
src/               Existing SBERT, scoring, loader and evaluation modules
data/courses/      Account-scoped uploaded reviews, predictions and vectors
data/persisted/    Existing versioned CLI output/cache (preserved)
models/            Local model and reference vectors (preserved)
tests/             CLI and API tests
```

## Course recommendations

The Recommendations page embeds the query with the local SBERT encoder and ranks the signed-in user's saved courses and resources by cosine similarity to their descriptions. Each resource's embedding is cached and recomputed only when its text changes. The final score is:

`score = 0.8 × relevance + 0.1 × (rating ÷ 5) + 0.1 × ((sentiment value + 1) ÷ 2)`

A signal a resource does not publish (no rating, no analysed reviews) contributes 0. Resources below a relevance of 0.2 are not shown. Each result states its own similarity, rating, sentiment and score. Resources without any title or description are listed as unmatched. A catalogue with no resources says so, and you can widen it by adding URLs.

## Per-model sentiment statistics

The course dashboard shows the supervised MLP by default. Tabs switch to the linear SVM and the SBERT reference classifier. Each tab is labelled with the model it shows, and predictions from different models are never combined. The validation-selected model in the training report is unchanged and is still reported by `/model/status`. Counts for a classifier include only reviews that have that classifier's prediction.

## Course dashboard layout

For an analysed course, the page shows course information (title, provider, description, instructor, duration, price, rating and URL, where published) above the review summary cards. The rating chart plots observed star ratings (x-axis) against review counts (y-axis), coloured by sentiment. Its explainer states that it compares ratings with sentiment, and it shows an insufficient-data message when too few reviews carry a star rating. Missing ratings are left out of the chart and are never inferred. The Model Evaluation page's comparison table keeps its first two columns and header in place while scrolling, and it reflows on narrow screens. Historical records without a prediction are reported as such and never counted as neutral. `scripts/backfill_supervised.py` fills blank MLP and SVM cells from the stored embeddings and saved classifiers, without retraining. It refuses to write when the embedding rows don't line up with the predictions.

## Browser verification

`scripts/browser_verify.py` drives the running application in headless Chromium through registration, login, extraction (Coursera, freeCodeCamp, W3Schools, Udemy), duplicate prevention, the MLP default and model tabs, the course layout order, recommendations, CSV upload, the model page (including the sticky comparison-table columns), and logout. It writes `outputs/verification/browser_verification.json`. Start the backend and the built frontend first (see below).

## Running the tests

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest tests -q
```

The live Coursera test needs internet access. Tests marked `network` can be skipped with `-m "not network"`.

## Current limits

- The recovered review corpus has no course relationship field, and its review content is not established as course feedback. It is not joined to the recovered course catalog or loaded by this account-based app automatically.
- Review text is extracted only from Coursera course pages, and only the approximately 20 reviews embedded in each page. Coursera's further review pages are loaded through an API path that its robots.txt disallows, so they are not requested. edX pages expose no public review text, and Udemy's robots.txt returns HTTP 403, so Udemy pages are not fetched.
- Other websites are supported at the metadata level only, through the generic adapter. It was verified on freeCodeCamp and W3Schools pages; other sites may expose less metadata. Adding a site-specific review source requires a new adapter and verification against live pages.
- Recommendations use the local SBERT encoder against the courses and resources saved by the signed-in user. Relevance is semantic similarity, not a learned preference model. Results are only as good as the saved descriptions.
- The supervised classifiers were trained on rating-derived weak labels. Their accuracy against true sentiment has not been measured.
- The API upload route processes a bounded file in memory. The legacy CLI remains the route for larger newline-delimited datasets and independent manual evaluation.
- Local cookies use HTTP only for loopback development. Enable HTTPS and `COURSE_ANALYZER_COOKIE_SECURE=1` before any non-loopback deployment.
- No fine-tuning takes place. Human-labelled evaluation is needed before drawing conclusions about classifier quality.
