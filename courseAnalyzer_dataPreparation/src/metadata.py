"""Independent analysis of course catalog metadata; no review joins are performed."""
import pandas as pd
from .data_loader import _field


def analyze_courses(frame):
    if frame.empty:
        return {"course_count": 0, "platform_distribution": {}, "average_rating": None}, frame
    data = frame.copy()
    cols = {
        "title": _field(data.columns, ("course_title", "title", "course_name", "name")),
        "platform": _field(data.columns, ("platform",)),
        "rating": _field(data.columns, ("rating", "avg_rating", "average_rating", "course_rating")),
        "price": _field(data.columns, ("price", "course_price", "cost")),
        "subscribers": _field(data.columns, ("num_subscribers", "subscribers", "enrollment", "students", "course_students_enrolled")),
        "reviews": _field(data.columns, ("num_reviews", "review_count", "reviews", "number_of_reviews", "course_review_count")),
    }
    if cols["platform"]:
        data["platform"] = data[cols["platform"]].fillna("unknown").astype(str).str.strip().str.lower()
        data.loc[data.platform.isin(("", "nan", "none", "null")), "platform"] = "unknown"
    else:
        data["platform"] = "unknown"
    for field in ("rating", "price", "subscribers", "reviews"):
        col = cols[field]
        if col:
            values = data[col].astype(str).str.strip()
            if field == "price":
                free = values.str.lower().isin(("free", "free course"))
                values = values.str.replace(r"[^0-9.\-]", "", regex=True)
                data[field] = pd.to_numeric(values, errors="coerce")
                data.loc[free, field] = 0
            else:
                values = values.str.replace(",", "", regex=False)
                data[field] = pd.to_numeric(values, errors="coerce")
        else:
            data[field] = pd.NA
    summary = {"course_count": int(len(data)),
               "platform_distribution": data.platform.value_counts(dropna=False).to_dict(),
               "average_rating": float(data.rating.mean()) if data.rating.notna().any() else None,
               "average_price": float(data.price.mean()) if data.price.notna().any() else None,
               "average_subscribers": float(data.subscribers.mean()) if data.subscribers.notna().any() else None,
               "average_course_review_count": float(data.reviews.mean()) if data.reviews.notna().any() else None,
               "price_quantiles": data.price.quantile([0, .25, .5, .75, 1]).to_dict() if data.price.notna().any() else None,
               "subscriber_quantiles": data.subscribers.quantile([0, .25, .5, .75, 1]).to_dict() if data.subscribers.notna().any() else None,
               "course_review_count_quantiles": data.reviews.quantile([0, .25, .5, .75, 1]).to_dict() if data.reviews.notna().any() else None,
               "platform_average_rating": data.groupby("platform").rating.mean().dropna().to_dict(),
               "available_fields": [name for name, col in cols.items() if col]}
    if cols["title"]:
        data["course_title"] = data[cols["title"]].astype(str)
    return summary, data
