"""Save compact plots for aggregate sentiment results."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LABELS = ("positive", "negative", "neutral")
COLORS = {"positive": "#2e8b57", "negative": "#c74440", "neutral": "#7b8794"}


def create_plots(summary, platform, ratings, output_dir, trends=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(LABELS, [summary["sentiment_counts"][x] for x in LABELS], color=[COLORS[x] for x in LABELS])
    ax.set(title="Overall review sentiment", ylabel="Review count")
    fig.tight_layout(); fig.savefig(output_dir / "sentiment_distribution.png", dpi=150); plt.close(fig)

    if platform:
        frame = pd.DataFrame(platform).set_index("platform")[[f"{x}_count" for x in LABELS]]
        frame.columns = list(LABELS)
        ax = frame.plot(kind="bar", stacked=True, figsize=(9, 4), color=[COLORS[x] for x in LABELS])
        ax.set(title="Sentiment by platform", xlabel="Platform", ylabel="Review count")
        ax.figure.tight_layout(); ax.figure.savefig(output_dir / "platform_sentiment.png", dpi=150); plt.close(ax.figure)

    if ratings:
        frame = pd.DataFrame(ratings).set_index("rating")[[f"{x}_count" for x in LABELS]]
        frame.columns = list(LABELS)
        ax = frame.plot(kind="bar", stacked=True, figsize=(8, 4), color=[COLORS[x] for x in LABELS])
        ax.set(title="Predicted sentiment by rating", xlabel="Observed rating", ylabel="Review count")
        ax.figure.tight_layout(); ax.figure.savefig(output_dir / "rating_vs_sentiment.png", dpi=150); plt.close(ax.figure)
    if trends:
        frame = pd.DataFrame(trends)
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, group in frame.groupby("platform", sort=False):
            group = group.sort_values("date")
            ax.plot(group.date, group.sentiment_value, marker=".", label=name)
        ax.axhline(0, color="#7b8794", linewidth=1, linestyle="--")
        ax.set(title="Sentiment value over time", xlabel="Review date", ylabel="(positive − negative) / reviews")
        if frame.platform.nunique() <= 12:
            ax.legend(title="Platform", bbox_to_anchor=(1.02, 1), loc="upper left")
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(output_dir / "sentiment_trends.png", dpi=150); plt.close(fig)


def create_course_plots(courses, output_dir, suffix=""):
    if "course_title" not in courses:
        return
    for field, title, filename in (("rating", "Highest rated courses", f"top_rated_courses{suffix}.png"),
                                   ("subscribers", "Most subscribed courses", f"top_subscribed_courses{suffix}.png")):
        if field not in courses or not courses[field].notna().any():
            continue
        top = courses.dropna(subset=[field]).nlargest(10, field).sort_values(field)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(top.course_title.astype(str).str.slice(0, 48), top[field], color="#4776a8")
        ax.set(title=title, xlabel=field.replace("_", " ").title())
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=150); plt.close(fig)
    for field, title, filename in (("price", "Course price distribution", f"course_price_distribution{suffix}.png"),
                                   ("subscribers", "Course subscriber distribution", f"course_subscriber_distribution{suffix}.png"),
                                   ("reviews", "Catalog review-count distribution", f"course_review_count_distribution{suffix}.png")):
        if field not in courses or not courses[field].notna().any():
            continue
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(courses[field].dropna(), bins=30, color="#4776a8", edgecolor="white")
        ax.set(title=title, xlabel=field.replace("_", " ").title(), ylabel="Course count")
        fig.tight_layout(); fig.savefig(output_dir / filename, dpi=150); plt.close(fig)
    if "platform" in courses and "rating" in courses and courses.rating.notna().any():
        means = courses.groupby("platform").rating.mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(means.index.astype(str), means.values, color="#4776a8")
        ax.set(title="Average course rating by platform", ylabel="Average catalog rating")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout(); fig.savefig(output_dir / f"course_average_rating_by_platform{suffix}.png", dpi=150); plt.close(fig)
