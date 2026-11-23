"""Sisi's memory: a small SQLite database (stays on your computer, never committed).

Two tables:
  messages  the conversation itself (you + Sisi), so a chat can continue later
  notes     durable facts Sisi chose to remember ("his name is Amine", "3 months talking")
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, session TEXT, role TEXT, text TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, note TEXT UNIQUE, created_at TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Memory:
    def __init__(self, path: str | Path = "data/agent/sisi.db"):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)

    # conversation ---------------------------------------------------------------
    def add_message(self, session: str, role: str, text: str) -> None:
        self.db.execute("INSERT INTO messages (session, role, text, created_at) VALUES (?,?,?,?)",
                        (session, role, text, _now()))
        self.db.commit()

    def history(self, session: str, limit: int = 40) -> list[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT role, text FROM messages WHERE session = ? ORDER BY id DESC LIMIT ?",
            (session, limit)).fetchall()
        return rows[::-1]

    # long-term notes ------------------------------------------------------------
    def remember(self, note: str) -> bool:
        note = note.strip()
        if not note:
            return False
        cur = self.db.execute("INSERT OR IGNORE INTO notes (note, created_at) VALUES (?, ?)",
                              (note, _now()))
        self.db.commit()
        return cur.rowcount == 1

    def notes(self) -> list[str]:
        return [r[0] for r in self.db.execute("SELECT note FROM notes ORDER BY id")]

    def forget_everything(self) -> None:
        self.db.executescript("DELETE FROM messages; DELETE FROM notes;")
        self.db.commit()
