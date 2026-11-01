"""SQLite persistence with explicit per-user ownership."""
import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.environ.get("COURSE_ANALYZER_DB", Path(__file__).resolve().parents[2] / "data" / "app.sqlite3"))


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    return db


def initialize():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY, email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL, display_name TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions(
            id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            expires_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analyses(
            id TEXT PRIMARY KEY, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            url TEXT NOT NULL, platform TEXT NOT NULL, title TEXT NOT NULL,
            metadata_json TEXT NOT NULL, summary_json TEXT NOT NULL,
            fingerprint TEXT NOT NULL, reviews_path TEXT NOT NULL, embeddings_path TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS analyses_owner ON analyses(user_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS evaluations(
            analysis_id TEXT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
            user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            metrics_json TEXT NOT NULL, PRIMARY KEY(analysis_id,user_id)
        );
        """)
