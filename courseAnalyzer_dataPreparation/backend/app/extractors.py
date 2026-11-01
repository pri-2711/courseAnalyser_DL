"""Safe public course/resource extraction and public review extraction.

Extraction reads only what a public page itself exposes: JSON-LD / Schema.org, Open Graph and
standard meta tags, visible heading structure, and review records embedded in the page data.
It never bypasses login, CAPTCHA, paywalls or robots.txt, never requests disallowed paths, and
never fabricates ratings, reviews, authors, durations or other fields.

Adapters are small classes. A platform adapter only overrides `reviews_from_page` (review
source) or `metadata_from_page` when a site needs special handling; everything else goes through
the shared pipeline in `BaseCourseExtractor.extract`.

Extraction status vocabulary (`extraction_status`):
  reviews_retrieved   metadata and at least one public review were retrieved
  metadata_only      resource metadata was retrieved; no public reviews exist or are exposed
  extraction_failed  the page could not be read or contained no usable metadata
  access_restricted   the site returned an access-check page; nothing was extracted
  robots_disallowed / robots_unavailable  robots.txt forbids, or could not confirm, fetching the page
"""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import ipaddress
import json
import re
import socket
import threading
import time
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx


class UnsafeURL(ValueError):
    pass


USER_AGENT = "CourseAnalyzerLocal/1.0 (public metadata reader; single user; no crawling)"
MIN_SECONDS_BETWEEN_REQUESTS = 1.0
MAX_HTML_BYTES = 2_000_000
METADATA_FIELDS = ("title", "description", "provider", "instructor", "rating", "review_count",
                   "price", "duration", "category", "level", "language")
PLATFORMS = {
    "udemy": ("udemy.com",),
    "coursera": ("coursera.org",),
    "edx": ("edx.org", "open.edx.org"),
}
STRUCTURED_TYPES = {"course", "learningresource", "article", "blogposting", "newsarticle", "techarticle",
                    "educationalorganization", "creativework", "tutorial", "howto"}
_last_request = {}
_request_lock = threading.Lock()


def validate_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise UnsafeURL("Use a public HTTPS course URL without embedded credentials.")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".test")):
        raise UnsafeURL("Local and private hosts are not allowed.")
    try:
        addresses = {ipaddress.ip_address(host)} if _is_ip(host) else {
            ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        }
    except (OSError, ValueError):
        raise UnsafeURL("The course host could not be validated.")
    if not addresses or any(not ip.is_global for ip in addresses):
        raise UnsafeURL("Private, reserved, and metadata network addresses are not allowed.")
    return url.strip()


def _is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def detect_platform(url: str) -> str:
    host = urlparse(validate_url(url)).hostname.lower().removeprefix("www.")
    for platform, domains in PLATFORMS.items():
        if any(host == domain or host.endswith("." + domain) for domain in domains):
            return platform
    return "generic"


def _throttle(host):
    """Keep at least MIN_SECONDS_BETWEEN_REQUESTS between requests to the same host."""
    with _request_lock:
        wait = MIN_SECONDS_BETWEEN_REQUESTS - (time.monotonic() - _last_request.get(host, 0.0))
        _last_request[host] = time.monotonic() + max(wait, 0.0)
    if wait > 0:
        time.sleep(wait)


def _clean(value):
    if value is None:
        return None
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text or None


def _name_of(value):
    if isinstance(value, list):
        value = next((item for item in value if item), None)
    if isinstance(value, dict):
        return _clean(value.get("name"))
    return _clean(value)


def _is_access_check_page(html):
    head = html[:20000].lower()
    title = re.search(r"<title[^>]*>(.*?)</title>", head, re.S)
    return bool(title) and any(marker in title.group(1) for marker in CHALLENGE_MARKERS)


def _types(item):
    raw = item.get("@type", [])
    raw = [raw] if isinstance(raw, str) else raw
    return {str(kind).lower().rsplit("/", 1)[-1].rsplit("#", 1)[-1] for kind in raw if isinstance(kind, str)}


CHALLENGE_MARKERS = ("client challenge", "just a moment", "attention required", "access denied", "verify you are human")


def _person(value):
    """An author counts as an instructor only when it is a person, not the site that publishes the page."""
    if isinstance(value, list):
        value = next((item for item in value if item), None)
    if isinstance(value, dict):
        kinds = _types(value)
        return _clean(value.get("name")) if not kinds or "person" in kinds else None
    return None


def _duration(value):
    """Turn an ISO 8601 duration such as PT24H or PT1H30M into readable text. Other values pass through."""
    text = _clean(value)
    match = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?", text or "")
    if not text or not match or not any(match.groups()):
        return text
    days, hours, minutes = (int(part) if part else 0 for part in match.groups())
    parts = [f"{n} {unit}{'s' if n != 1 else ''}" for n, unit in ((days, "day"), (hours, "hour"), (minutes, "minute")) if n]
    return " ".join(parts)


def parse_page_metadata(html, url, platform):
    """Extract standard metadata and a heading outline from one public HTML page.

    Returns only fields that the page itself states; every other field is None and is listed in
    `fields_unavailable`.
    """
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html[:MAX_HTML_BYTES], "html.parser")
    found = {}
    resource_type = None
    for node in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(node.string or node.get_text())
        except ValueError:
            continue
        for item in (data if isinstance(data, list) else [data]):
            if not isinstance(item, dict):
                continue
            kinds = _types(item)
            if not kinds & STRUCTURED_TYPES or found.get("title"):
                continue
            rating = item.get("aggregateRating") if isinstance(item.get("aggregateRating"), dict) else {}
            offers = item.get("offers") if isinstance(item.get("offers"), dict) else {}
            found.update({
                "title": _clean(item.get("name") or item.get("headline")),
                "description": _clean(item.get("description")),
                "provider": _name_of(item.get("provider")) or _name_of(item.get("publisher")),
                "instructor": _name_of(item.get("instructor")) or _person(item.get("author")),
                "rating": rating.get("ratingValue"),
                "review_count": rating.get("ratingCount") or rating.get("reviewCount"),
                "price": offers.get("price"),
                "duration": _duration(item.get("timeRequired")),
                "category": _clean(item.get("category")),
                "level": _clean(item.get("educationalLevel")),
                "language": _clean(item.get("inLanguage")),
            })
            resource_type = "course" if kinds & {"course"} else "learning resource" if kinds & {"learningresource", "tutorial", "howto"} else "article" if kinds & {"article", "blogposting", "newsarticle", "techarticle"} else "resource"
    meta = {}
    for tag in soup.find_all("meta"):
        key = tag.get("property") or tag.get("name")
        if key and tag.get("content") is not None:
            meta[key.lower()] = tag.get("content")
    og_title = _clean(meta.get("og:title"))
    fallback = {
        "title": og_title or _clean(soup.title.get_text(" ", strip=True) if soup.title else None),
        "description": _clean(meta.get("og:description") or meta.get("description")),
        "provider": _clean(meta.get("og:site_name")),
        "instructor": _clean(meta.get("author") or meta.get("article:author")),
    }
    for key, value in fallback.items():
        if not found.get(key) and value:
            found[key] = value
    # Coerce numeric rating fields only when they parse; never invent a value.
    for key in ("rating", "review_count", "price"):
        try:
            found[key] = float(found[key]) if found.get(key) not in (None, "") else None
        except (TypeError, ValueError):
            found[key] = None
    if found.get("review_count") is not None:
        found["review_count"] = int(found["review_count"])
    outline = [_clean(h.get_text(" ", strip=True)) for h in soup.find_all(["h2", "h3"])]
    outline = [text for text in outline if text][:25]
    result = {key: found.get(key) for key in METADATA_FIELDS}
    result["resource_type"] = resource_type or ("course" if platform in PLATFORMS else "resource")
    result["learning_outline"] = outline
    result["fields_unavailable"] = [key for key in ("title", "description", "provider", "instructor", "rating", "review_count", "price", "duration") if result[key] in (None, "")]
    result["metadata_extracted"] = bool(result["title"] or result["description"])
    result["platform"] = platform
    result["url"] = url
    return result


COURSERA_REVIEW = re.compile(r'\{\s*"__typename"\s*:\s*"DescriptionPage_Review"[^{}]*\}')


def parse_coursera_reviews(html, source_url):
    """Read the reviews Coursera embeds in a public course page.

    Each embedded record carries the review ID, display name, comment text, rating and timestamp.
    Records without text are skipped; a missing rating is kept as null rather than invented.
    The page embeds only its first set of reviews. Further pages are loaded from an API path that
    Coursera's robots.txt disallows, so they are not requested.
    """
    extracted_at = datetime.now(timezone.utc).isoformat()
    reviews, seen, skipped = [], set(), Counter()
    for match in COURSERA_REVIEW.finditer(html):
        try:
            item = json.loads(match.group(0))
        except ValueError:
            skipped["unreadable_record"] += 1
            continue
        review_id = str(item.get("id") or "").strip()
        if not review_id:
            skipped["missing_review_id"] += 1
            continue
        if review_id in seen:
            skipped["duplicate_on_page"] += 1
            continue
        seen.add(review_id)
        text = str(item.get("comment") or "").strip()
        if not text:
            skipped["empty_text"] += 1
            continue
        try:
            rating = float(item["rating"]) if item.get("rating") is not None else None
        except (TypeError, ValueError):
            rating = None
        reviews.append({"review_id": f"coursera:{review_id}", "review_text": text, "rating": rating,
                        "review_date": item.get("timestamp"), "reviewer": item.get("authorName"),
                        "platform": "coursera", "source_url": source_url, "extracted_at": extracted_at})
    if reviews:
        status = (f"Retrieved {len(reviews)} public reviews embedded on the course page. Coursera loads further "
                  "reviews through an API path that its robots.txt disallows, so only this set is available.")
    else:
        status = "No public review text was embedded on this course page."
    return {"reviews": reviews, "skipped_reviews": dict(skipped), "review_status": status}


@dataclass
class BaseCourseExtractor:
    platform: str

    def extract(self, url: str) -> dict:
        safe_url = validate_url(url)
        if detect_platform(safe_url) != self.platform and self.platform != "generic":
            raise ValueError("URL does not match the selected platform extractor.")
        current = safe_url
        with httpx.Client(timeout=httpx.Timeout(8.0), follow_redirects=False,
                          headers={"User-Agent": USER_AGENT}, trust_env=False) as client:
            for _ in range(4):
                validate_url(current)
                parsed = urlparse(current)
                robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
                _throttle(parsed.netloc)
                robots_response = client.get(robots_url)
                if robots_response.status_code == 200:
                    rules = RobotFileParser(robots_url)
                    rules.parse(robots_response.text.splitlines() + [""])
                    if not rules.can_fetch(USER_AGENT.split("/")[0], current):
                        return self._blocked("robots_disallowed", url, "The site's robots.txt disallows fetching this page.")
                elif robots_response.status_code not in (404, 410):
                    return self._blocked("robots_unavailable", url, f"robots.txt returned HTTP {robots_response.status_code}, so permission could not be verified. No page was fetched.")
                _throttle(parsed.netloc)
                response = client.get(current)
                if response.status_code in (301, 302, 303, 307, 308):
                    location = response.headers.get("location")
                    if not location:
                        break
                    current = urljoin(current, location)
                    continue
                response.raise_for_status()
                return self._result(safe_url, response.text)
        return self._blocked("extraction_failed", url, "The course page could not be read.")

    def _blocked(self, status, url, message):
        return {"title": None, "url": url, "platform": self.platform, "extraction_status": status,
                "metadata_extracted": False, "reviews": [], "reviews_retrieved": 0, "skipped_reviews": {},
                "review_status": message, "fields_unavailable": list(METADATA_FIELDS)}

    def _result(self, url, html):
        if _is_access_check_page(html):
            return self._blocked("access_restricted", url, "The site returned an access-check page instead of course content, so nothing was extracted.")
        result = self.metadata_from_page(html, url)
        result.update(self.reviews_from_page(html, url))
        result["reviews_retrieved"] = len(result.get("reviews", []))
        if result["reviews"]:
            result["extraction_status"] = "reviews_retrieved"
        elif result.get("metadata_extracted"):
            result["extraction_status"] = "metadata_only"
        else:
            result["extraction_status"] = "extraction_failed"
            result["review_status"] = "No resource metadata could be read from this page. Upload review data instead."
        return result

    def metadata_from_page(self, html, url):
        return parse_page_metadata(html, url, self.platform)

    def reviews_from_page(self, html, url):
        """Platforms without a supported public review source report that explicitly."""
        return {"reviews": [], "skipped_reviews": {},
                "review_status": "No public review text is available for this site. Metadata and learning content were extracted without reviews."}


class UdemyExtractor(BaseCourseExtractor):
    def __init__(self): super().__init__("udemy")


class CourseraExtractor(BaseCourseExtractor):
    def __init__(self): super().__init__("coursera")

    def reviews_from_page(self, html, url):
        return parse_coursera_reviews(html, url)


class EdXExtractor(BaseCourseExtractor):
    def __init__(self): super().__init__("edx")


class GenericCourseExtractor(BaseCourseExtractor):
    def __init__(self): super().__init__("generic")


EXTRACTORS = {"udemy": UdemyExtractor, "coursera": CourseraExtractor, "edx": EdXExtractor,
              "generic": GenericCourseExtractor}
