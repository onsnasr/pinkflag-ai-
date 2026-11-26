"""Sisi's memory: one small SQLite database (stays on your computer, never committed).

Tables:
  messages   the conversation (you + Sisi), one session per guy, so chats continue later
  notes      durable facts Sisi chose to remember, scoped per guy ("talking for 2 months")
  guys       the roster: name, status, mini profile (works/studies, where you met, age, detail)
  analyses   every message of his that the model analysed: label + confidence + date
             -> powers the receipts, the weekly timeline and the vibe score
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY, session TEXT, role TEXT, text TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, note TEXT, created_at TEXT, scope TEXT DEFAULT 'default');
CREATE TABLE IF NOT EXISTS guys (
    id INTEGER PRIMARY KEY, name TEXT NOT NULL, status TEXT DEFAULT 'talking stage',
    works_or_studies TEXT DEFAULT '', met_at TEXT DEFAULT '', age TEXT DEFAULT '',
    detail TEXT DEFAULT '', emoji TEXT DEFAULT '💌', created_at TEXT);
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY, guy_id INTEGER, text TEXT, label TEXT, confidence REAL,
    created_at TEXT);
"""
PROFILE_FIELDS = ("works_or_studies", "met_at", "age", "detail")
GUY_FIELDS = ("name", "status", "emoji") + PROFILE_FIELDS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Memory:
    def __init__(self, path: str | Path = "data/agent/sisi.db"):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the web server may call from different threads
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._migrate()

    def _migrate(self) -> None:
        """Older databases had notes without a scope column."""
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(notes)")}
        if "scope" not in cols:
            self.db.execute("ALTER TABLE notes ADD COLUMN scope TEXT DEFAULT 'default'")
            self.db.commit()

        if "image" not in {r["name"] for r in self.db.execute("PRAGMA table_info(messages)")}:
            self.db.execute("ALTER TABLE messages ADD COLUMN image BLOB")
            self.db.commit()

    # conversation ---------------------------------------------------------------
    def add_message(self, session: str, role: str, text: str, image: bytes | None = None) -> None:
        self.db.execute(
            "INSERT INTO messages (session, role, text, created_at, image) VALUES (?,?,?,?,?)",
            (session, role, text, _now(), image))
        self.db.commit()

    def chat_history(self, session: str, limit: int = 200) -> list[dict]:
        rows = self.db.execute(
            "SELECT role, text, CASE WHEN image IS NOT NULL THEN id END AS image_id "
            "FROM messages WHERE session = ? ORDER BY id DESC LIMIT ?", (session, limit))
        return [dict(row) for row in reversed(rows.fetchall())]

    def message_image(self, session: str, message_id: int) -> bytes | None:
        row = self.db.execute("SELECT image FROM messages WHERE session = ? AND id = ?",
                              (session, message_id)).fetchone()
        return row["image"] if row else None

    def history(self, session: str, limit: int = 40) -> list[tuple[str, str]]:
        rows = self.db.execute(
            "SELECT role, text FROM messages WHERE session = ? ORDER BY id DESC LIMIT ?",
            (session, limit)).fetchall()
        return [(r["role"], r["text"]) for r in rows[::-1]]

    # long-term notes ------------------------------------------------------------
    def remember(self, note: str, scope: str = "default") -> bool:
        note = note.strip()
        if not note:
            return False
        exists = self.db.execute("SELECT 1 FROM notes WHERE note = ? AND scope = ?",
                                 (note, scope)).fetchone()
        if exists:
            return False
        self.db.execute("INSERT INTO notes (note, created_at, scope) VALUES (?, ?, ?)",
                        (note, _now(), scope))
        self.db.commit()
        return True

    def notes(self, scope: str = "default") -> list[str]:
        return [r["note"] for r in self.db.execute(
            "SELECT note FROM notes WHERE scope = ? ORDER BY id", (scope,))]

    # the roster -----------------------------------------------------------------
    def add_guy(self, name: str, status: str = "talking stage", emoji: str = "💌") -> int:
        cur = self.db.execute("INSERT INTO guys (name, status, emoji, created_at) VALUES (?,?,?,?)",
                              (name.strip(), status.strip(), emoji, _now()))
        self.db.commit()
        return cur.lastrowid

    def guy(self, guy_id: int) -> dict | None:
        row = self.db.execute("SELECT * FROM guys WHERE id = ?", (guy_id,)).fetchone()
        return dict(row) if row else None

    def guys(self) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM guys ORDER BY id")]

    def update_guy(self, guy_id: int, **fields) -> None:
        fields = {k: str(v).strip() for k, v in fields.items() if k in GUY_FIELDS and v is not None}
        if fields:
            sets = ", ".join(f"{k} = ?" for k in fields)
            self.db.execute(f"UPDATE guys SET {sets} WHERE id = ?", (*fields.values(), guy_id))
            self.db.commit()

    def delete_guy(self, guy_id: int) -> None:
        scope = f"guy-{guy_id}"
        self.db.execute("DELETE FROM analyses WHERE guy_id = ?", (guy_id,))
        self.db.execute("DELETE FROM notes WHERE scope = ?", (scope,))
        self.db.execute("DELETE FROM messages WHERE session = ?", (scope,))
        self.db.execute("DELETE FROM guys WHERE id = ?", (guy_id,))
        self.db.commit()

    # analysed messages ----------------------------------------------------------
    def add_analyses(self, guy_id: int, results: list[dict]) -> None:
        self.db.executemany(
            "INSERT INTO analyses (guy_id, text, label, confidence, created_at) VALUES (?,?,?,?,?)",
            [(guy_id, r["message"], r["label"], r["confidence"], _now()) for r in results])
        self.db.commit()

    def analyses(self, guy_id: int) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT text, label, confidence, created_at FROM analyses WHERE guy_id = ? ORDER BY id",
            (guy_id,))]

    def forget_everything(self) -> None:
        self.db.executescript("DELETE FROM messages; DELETE FROM notes; DELETE FROM guys; "
                              "DELETE FROM analyses;")
        self.db.commit()
