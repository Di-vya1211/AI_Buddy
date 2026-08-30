import json
import os
import sqlite3
from collections import Counter
from datetime import date, datetime, timedelta

import streamlit as st

DB_PATH = os.path.join(os.path.dirname(__file__), "study_buddy.db")


@st.cache_resource
def get_connection():
    """Open (or create) the SQLite database and ensure schema is up to date.

    Uses ``@st.cache_resource`` so the same connection object is reused across
    every Streamlit re-run without opening a new file handle each time.

    Returns
    -------
    sqlite3.Connection
        A shared connection with ``check_same_thread=False`` (safe here because
        Streamlit's threading model is single-user per process).
    """
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            score INTEGER NOT NULL,
            total INTEGER NOT NULL,
            weak_topics TEXT
        )"""
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL
        )"""
    )
    conn.commit()
    return conn


def log_activity(conn, action):
    """Record a timestamped activity event (used for streak calculation).

    Parameters
    ----------
    conn : sqlite3.Connection
    action : str
        Short label for the feature used, e.g. ``"explain"``, ``"quiz"``.
    """
    conn.execute(
        "INSERT INTO activity (timestamp, action) VALUES (?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action),
    )
    conn.commit()


def log_quiz_attempt(conn, score, total, weak_topics):
    """Persist a completed quiz attempt and fire an activity event.

    Parameters
    ----------
    conn : sqlite3.Connection
    score : int
        Number of correct answers.
    total : int
        Total number of questions.
    weak_topics : list[str]
        Topic labels for every question the student got wrong.
    """
    conn.execute(
        "INSERT INTO quiz_attempts (timestamp, score, total, weak_topics) "
        "VALUES (?, ?, ?, ?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), score, total, json.dumps(weak_topics)),
    )
    conn.commit()
    log_activity(conn, "quiz")


def get_recent_weak_topics(conn, attempts=5, limit=5):
    """Return the most-frequently-missed topics across the last *attempts* quizzes.

    Parameters
    ----------
    conn : sqlite3.Connection
    attempts : int
        How many recent quiz attempts to look back through.
    limit : int
        Maximum number of topics to return.

    Returns
    -------
    list[str]
        Topics ranked by how often they appeared as wrong answers, most
        common first.  Empty list if no quiz history exists.
    """
    cur = conn.execute(
        "SELECT weak_topics FROM quiz_attempts ORDER BY id DESC LIMIT ?",
        (attempts,),
    )
    counter = Counter()
    for (raw,) in cur.fetchall():
        if raw:
            counter.update(json.loads(raw))
    return [topic for topic, _ in counter.most_common(limit)]


def get_last_score_percentage(conn):
    """Return the most recent quiz score as an integer percentage (0-100).

    Returns
    -------
    int or None
        Percentage, or None if no attempts have been recorded yet.
    """
    cur = conn.execute(
        "SELECT score, total FROM quiz_attempts ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    if not row or not row[1]:  # row[1] is total; guard against zero
        return None
    score, total = row
    return round(score / total * 100)


def get_score_history(conn):
    """Fetch every quiz attempt in chronological order.

    Returns
    -------
    list[tuple[str, int, int]]
        Rows of (timestamp, score, total).
    """
    cur = conn.execute(
        "SELECT timestamp, score, total FROM quiz_attempts ORDER BY id"
    )
    return cur.fetchall()


def get_quiz_stats(conn):
    """Return aggregate quiz statistics.

    Returns
    -------
    tuple[int, int | None]
        ``(attempt_count, average_percentage)`` where *average_percentage* is
        an integer (0-100) or None if no attempts exist.
        The SQLite ``AVG`` expression uses ``NULLIF(total, 0)`` to avoid
        dividing by zero if a row with ``total=0`` was ever inserted.
    """
    cur = conn.execute(
        "SELECT COUNT(*), AVG(score * 100.0 / NULLIF(total, 0)) FROM quiz_attempts"
    )
    count, avg = cur.fetchone()
    return count or 0, (round(avg) if avg is not None else None)


def get_streak(conn):
    """Calculate the student's current consecutive-day activity streak.

    A streak is the longest unbroken run of calendar days ending today (or
    yesterday, to forgive opening the app after midnight).

    Returns
    -------
    int
        Number of consecutive active days.  0 if the student has not been
        active today or yesterday.
    """
    cur = conn.execute("SELECT DISTINCT date(timestamp) AS d FROM activity")
    dates = {date.fromisoformat(row[0]) for row in cur.fetchall()}
    if not dates:
        return 0

    cursor_date = date.today()
    if cursor_date not in dates:
        # Give a one-day grace period (e.g. opened app just after midnight)
        cursor_date -= timedelta(days=1)
        if cursor_date not in dates:
            return 0

    streak = 0
    while cursor_date in dates:
        streak += 1
        cursor_date -= timedelta(days=1)
    return streak


def get_active_days(conn):
    """Return all distinct days on which any activity was recorded.

    Returns
    -------
    list[str]
        ISO-format date strings (``"YYYY-MM-DD"``), newest first.
    """
    cur = conn.execute("SELECT DISTINCT date(timestamp) AS d FROM activity ORDER BY d DESC")
    return [row[0] for row in cur.fetchall()]
