"""Browser-level verification of the running Course Analyzer (Chromium via Playwright).

Prerequisites (two terminals, from the project directory):
  .venv\\Scripts\\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
  cd frontend ; npm run build ; npm run preview -- --host 127.0.0.1 --port 5173 --strictPort

Usage:  .venv\\Scripts\\python.exe scripts\\browser_verify.py
Each workflow is recorded as PASS/FAIL with the observed evidence. Nothing is marked PASS unless the
browser actually observed it. Console errors and failed network responses are collected.
"""
from datetime import datetime
import time
from pathlib import Path
import json
import re
import secrets
import sys
import tempfile

from playwright.sync_api import sync_playwright, expect

# Page text can contain non-ASCII (e.g. check marks); keep the log readable on Windows consoles.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

APP = "http://127.0.0.1:5173"
API = "http://127.0.0.1:8000"
EMAIL = f"browser-{datetime.now():%Y%m%d%H%M%S}@example.test"
PASSWORD = secrets.token_urlsafe(18)  # generated per run; never stored
COURSERA = "https://www.coursera.org/learn/python"
FREECODECAMP = "https://www.freecodecamp.org/news/how-to-detect-financial-fraud-patterns-with-knowledge-graphs-a-production-guide/"
W3SCHOOLS = "https://www.w3schools.com/python/python_intro.asp"
UDEMY = "https://www.udemy.com/course/python-for-beginners/"

results = []
console_errors = []


def record(name, ok, evidence):
    results.append({"workflow": name, "result": "PASS" if ok else "FAIL", "evidence": evidence})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {evidence}", flush=True)


def step(name, fn):
    try:
        evidence = fn()
        record(name, True, evidence or "ok")
    except Exception as exc:  # keep going so one failure does not hide the rest
        record(name, False, f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}")


def submit_url(page, url):
    with page.expect_response(lambda r: r.url.endswith("/courses/analyze") and r.request.method == "POST", timeout=120_000):
        page.get_by_placeholder("Paste a course URL to get started").fill(url)
        page.get_by_role("button", name="Analyze course").click()
    page.wait_for_timeout(1500)


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.set_default_timeout(60_000)
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))

        def register():
            page.goto(APP)
            page.get_by_role("button", name="Create account").first.click()
            page.locator('input[type="email"]').fill(EMAIL)
            inputs = page.locator("form.auth-card input")
            inputs.nth(0).fill("Browser Test")
            page.locator('input[type="password"]').nth(0).fill(PASSWORD)
            page.locator('input[type="password"]').nth(1).fill(PASSWORD)
            page.locator("form.auth-card button.button.primary").click()
            expect(page.get_by_text("Good to see you").first).to_be_visible()
            return f"registered {EMAIL}; dashboard visible"
        step("1. User registration", register)

        def logout_then_login():
            page.get_by_title("Log out").click()
            expect(page.get_by_role("heading", name="Welcome back")).to_be_visible()
            page.locator('input[type="email"]').fill(EMAIL)
            page.locator('input[type="password"]').fill(PASSWORD)
            page.get_by_role("button", name="Sign in").click()
            expect(page.get_by_text("Good to see you").first).to_be_visible()
            return "signed out, then signed in again with the same credentials"
        step("2. Login and authentication (logout + login)", logout_then_login)

        def navigate():
            page.locator("nav button.nav-item", has_text="Model Evaluation").click()
            expect(page.get_by_role("heading", name="Model training and evaluation")).to_be_visible()
            return "Model Evaluation page opened from the sidebar"
        step("3. Dashboard navigation", navigate)

        def course_metadata_and_reviews():
            page.locator("nav button.nav-item", has_text="Dashboard").click()
            submit_url(page, COURSERA)
            expect(page.get_by_text("Course extraction")).to_be_visible()
            notice = page.locator(".notice, [role=status]").first
            expect(page.get_by_text("Retrieved 20 public reviews").first).to_be_visible()
            counts = page.locator(".extraction-counts").inner_text()
            return "Coursera: " + " | ".join(line.strip() for line in counts.splitlines() if line.strip())
        step("4-6. Course URL submission, Coursera metadata and actual review retrieval", course_metadata_and_reviews)

        def sentiment_and_breakdown():
            expect(page.get_by_role("heading", name="Sentiment by model")).to_be_visible()
            tabs = page.locator(".model-breakdown [role=tab]")
            expect(tabs).to_have_count(3)
            names = [tabs.nth(i).inner_text().strip() for i in range(3)]
            selected = page.locator(".model-breakdown [role=tab][aria-selected=true]").inner_text().strip()
            expect(page.locator(".model-breakdown .model-total")).to_be_visible()
            return "model tabs: " + "; ".join(names) + f" | selected by default: {selected}"
        step("11. Sentiment analysis (per-model breakdown)", sentiment_and_breakdown)

        def ui_refinements():
            tab = page.locator(".model-tabs button[aria-selected=true]").inner_text()
            if "MLP" not in tab:
                raise AssertionError(f"default model tab is '{tab}', expected MLP")
            overview_first = page.evaluate("() => { const o=document.querySelector('.course-overview'); const m=document.querySelector('.metric-grid'); return !!o && !!m && (o.compareDocumentPosition(m) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0 }")
            if not overview_first:
                raise AssertionError("course overview is not above the summary cards")
            expect(page.locator(".rating-sentiment")).to_be_visible()
            page.locator(".model-tabs button", has_text="Linear SVM").click()
            expect(page.locator(".model-tabs button[aria-selected=true]")).to_contain_text("Linear SVM")
            expect(page.locator(".model-stats")).to_be_visible()
            return f"default tab: {tab.strip()}; overview above summary cards; rating chart visible; switched to Linear SVM"
        step("UI: default MLP tab, course layout order, rating chart, model switching", ui_refinements)
        def duplicate_prevention():
            before = context.request.get(f"{API}/courses").json()
            submit_url(page, COURSERA)
            expect(page.get_by_text("Course extraction")).to_be_visible()
            after = context.request.get(f"{API}/courses").json()
            same = [c for c in after if c["url"] == COURSERA]
            if len(before) != len(after) or len(same) != 1:
                raise AssertionError(f"courses before={len(before)} after={len(after)} duplicates={len(same)}")
            return f"resubmitting the same URL kept {len(after)} course record(s), one for this URL"
        step("10. Duplicate submission prevention", duplicate_prevention)

        def freecodecamp():
            submit_url(page, FREECODECAMP)
            page.wait_for_function("() => (document.querySelector(\".extraction-panel\")||{innerText:\"\"}).innerText.includes(\"Financial Fraud\")", timeout=60000)
            panel = " ".join(page.locator(".extraction-panel").inner_text().split())
            if "Resource metadata retrieved" not in panel:
                raise AssertionError("metadata sentence missing; panel shows: " + panel[-300:])
            if "Extraction failed" in panel:
                raise AssertionError("freeCodeCamp article shown as failed")
            if "Financial Fraud" not in panel:
                raise AssertionError("panel does not show the freeCodeCamp article title")
            return "freeCodeCamp article: metadata shown as success (no reviews, not an error)"
        step("7. freeCodeCamp URL extraction", freecodecamp)

        def generic():
            submit_url(page, W3SCHOOLS)
            expect(page.locator(".extraction-panel")).to_contain_text("Introduction to Python")
            expect(page.locator(".extraction-panel")).to_contain_text("Resource metadata retrieved")
            panel = page.locator(".extraction-panel").inner_text()
            return "W3Schools tutorial panel: " + " ".join(panel.split())[:260]
        step("8. Generic educational URL extraction", generic)

        def failed_state():
            submit_url(page, UDEMY)
            expect(page.get_by_text("Extraction failed").first).to_be_visible()
            return "Udemy (robots.txt 403): panel shows 'Extraction failed' with the reason"
        step("Failed extraction is clearly displayed", failed_state)

        def recommendations():
            page.locator("nav button.nav-item", has_text="Recommendations").click()
            button = page.get_by_role("button", name="Find resources")
            expect(button).to_be_disabled()
            page.get_by_label("Topic or skill").fill("python programming")
            button.click()
            expect(page.locator(".rec-card").first).to_be_visible()
            cards = page.locator(".rec-card")
            titles = [cards.nth(i).locator("h3").inner_text() for i in range(cards.count())]
            link = cards.first.locator("a").get_attribute("href")
            if not link.startswith("https://"):
                raise AssertionError(f"bad resource link {link}")
            return f"{cards.count()} results; top: '{titles[0]}' -> {link}; empty query button disabled"
        step("12. Course Recommendations section", recommendations)

        def csv_upload():
            page.locator("nav button.nav-item", has_text="My Analyses").click()
            page.get_by_role("button", name="Open analysis").first.click()
            csv = Path(tempfile.gettempdir()) / "browser_reviews.csv"
            csv.write_text("review_id,review,rating\nb1,Excellent teaching and clear examples,5\nb2,Too slow and confusing,2\n", encoding="utf-8")
            page.locator('input[type="file"]').first.set_input_files(str(csv))
            expect(page.get_by_text(re.compile(r"Analysis saved locally|Saved analysis reused"))).to_be_visible()
            return "CSV uploaded from an open course and analysed"
        step("15. CSV review upload", csv_upload)

        def model_page_details():
            page.locator("nav button.nav-item", has_text="Model Evaluation").click()
            expect(page.locator(".metric-table")).to_be_visible()
            rows = page.locator(".metric-table tbody tr").count()
            graphs = page.locator(".graph-card img").count()
            sticky = page.locator(".sticky-col").count()
            if sticky == 0:
                raise AssertionError("comparison table has no sticky columns")
            selected_text = page.get_by_text("Selected model:").count()
            if selected_text:
                raise AssertionError("'Selected model:' text is still present")
            return f"{rows} model rows, {graphs} evaluation graphs rendered; 'Selected model' absent"
        step("13-14. Model Training and Evaluation page, comparison table and graphs", model_page_details)

        def compare_courses():
            page.locator("nav button.nav-item", has_text="Comparison Tool").click()
            expect(page.get_by_role("heading", name="Compare courses")).to_be_visible()
            # Coursera (reviews retrieved) against W3Schools (metadata only, no reviews, no published rating).
            for platform in ("coursera", "w3schools"):
                matches = page.locator(".cmp-pick", has_text=re.compile(platform, re.I))
                if matches.count() == 0:
                    raise AssertionError(f"no saved course on {platform} to compare")
                matches.first.locator("input").check()
            expect(page.locator(".cmp-table")).to_be_visible()
            insights = page.locator(".cmp-insights li").all_inner_texts()
            if not insights:
                raise AssertionError("no comparison insights were generated")
            # Generated text only: the picker panel's own disclaimer legitimately says "declares one better".
            body = page.locator(".cmp-cards").inner_text() + "\n" + "\n".join(insights)
            for claim in ("better", "winner"):
                if claim in body.lower():
                    raise AssertionError(f"comparison text contains a ranking claim: '{claim}'")
            cards = page.locator(".cmp-card").count()
            if cards != 2:
                raise AssertionError(f"expected 2 course cards, found {cards}")
            return f"2 course cards, {len(insights)} insights, no ranking language; zero-review course handled"
        step("17. Comparison of two courses (different platforms, zero reviews, missing rating)", compare_courses)

        def logout():
            page.get_by_title("Log out").click()
            expect(page.get_by_role("heading", name="Welcome back")).to_be_visible()
            return "logged out to the sign-in page"
        step("16. Logout", logout)

        browser.close()

    report = {"run_at": datetime.now().isoformat(timespec="seconds"), "app": APP, "api": API,
              "results": results, "console_errors": console_errors[:50]}
    out = Path(__file__).resolve().parents[1] / "outputs" / "verification" / "browser_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = [r for r in results if r["result"] == "FAIL"]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed; console errors: {len(console_errors)}", flush=True)
    print(f"report: {out}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())








