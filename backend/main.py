"""Hattama — сервер.

Всё работает локально: распознавание речи, разбор поручений, сборка протокола.
Ни один запрос не уходит в интернет.
"""

import json
import os
import shutil
import sqlite3
import tempfile
import threading
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import sys

sys.path.insert(0, str(Path(__file__).parent))

import transcribe as asr
import extract as tasks_extractor
import dates as dates_mod
import export as docx_export

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
DB_PATH = DATA / "hattama.db"

app = FastAPI(title="Hattama", description="Протокол совещаний с контролем поручений")


def db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meetings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT, meeting_date TEXT, status TEXT,
                audio_path TEXT, transcript TEXT, summary TEXT,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER, text TEXT, assignee TEXT, author TEXT,
                deadline_raw TEXT, deadline TEXT, deadline_rule TEXT,
                quote TEXT, status TEXT
            );
            """
        )


init_db()


def process_meeting(meeting_id: int, audio_path: str):
    """Полный разбор: звук -> реплики -> поручения -> даты -> база."""
    try:
        segments = asr.process(audio_path)
        text = asr.to_text(segments)
        with db() as conn:
            conn.execute(
                "UPDATE meetings SET transcript=?, status=? WHERE id=?",
                (json.dumps(segments, ensure_ascii=False), "разбираю поручения", meeting_id),
            )

        found = tasks_extractor.extract_tasks(text)
        found = dates_mod.enrich(found, date.today())

        with db() as conn:
            for t in found:
                conn.execute(
                    """INSERT INTO tasks
                       (meeting_id, text, assignee, author, deadline_raw, deadline,
                        deadline_rule, quote, status)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (meeting_id, t["text"], t["assignee"], t.get("author", ""),
                     t["deadline_raw"], t.get("deadline"), t.get("deadline_rule", ""),
                     t["quote"], t["status"]),
                )
            conn.execute("UPDATE meetings SET status=? WHERE id=?", ("готово", meeting_id))
    except Exception as e:
        with db() as conn:
            conn.execute(
                "UPDATE meetings SET status=? WHERE id=?", (f"ошибка: {e}", meeting_id)
            )


@app.post("/api/meetings")
async def upload_meeting(file: UploadFile = File(...), title: str = "Совещание"):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    dest = DATA / f"upload_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with db() as conn:
        cur = conn.execute(
            "INSERT INTO meetings (title, meeting_date, status, audio_path, created_at) VALUES (?,?,?,?,?)",
            (title, date.today().isoformat(), "распознаю речь", str(dest),
             datetime.now().isoformat()),
        )
        meeting_id = cur.lastrowid

    threading.Thread(target=process_meeting, args=(meeting_id, str(dest)), daemon=True).start()
    return {"id": meeting_id, "status": "распознаю речь"}


@app.get("/api/meetings")
def list_meetings():
    with db() as conn:
        rows = conn.execute(
            "SELECT id, title, meeting_date, status FROM meetings ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/meetings/{meeting_id}")
def get_meeting(meeting_id: int):
    with db() as conn:
        m = conn.execute("SELECT * FROM meetings WHERE id=?", (meeting_id,)).fetchone()
        if not m:
            raise HTTPException(404, "совещание не найдено")
        rows = conn.execute("SELECT * FROM tasks WHERE meeting_id=?", (meeting_id,)).fetchall()

    today = date.today().isoformat()
    tasks = []
    for r in rows:
        t = dict(r)
        if t["deadline"] and t["deadline"] < today and t["status"] == "в работе":
            t["status"] = "просрочено"
        tasks.append(t)

    return {
        "id": m["id"],
        "title": m["title"],
        "meeting_date": m["meeting_date"],
        "status": m["status"],
        "transcript": json.loads(m["transcript"]) if m["transcript"] else [],
        "tasks": tasks,
        "summary": m["summary"] or "",
    }


class TaskPatch(BaseModel):
    text: str | None = None
    assignee: str | None = None
    deadline: str | None = None
    status: str | None = None


@app.patch("/api/tasks/{task_id}")
def patch_task(task_id: int, patch: TaskPatch):
    fields = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not fields:
        return {"ok": True}
    sets = ", ".join(f"{k}=?" for k in fields)
    with db() as conn:
        conn.execute(f"UPDATE tasks SET {sets} WHERE id=?", (*fields.values(), task_id))
    return {"ok": True}


@app.get("/api/meetings/{meeting_id}/export")
def export_meeting(meeting_id: int):
    data = get_meeting(meeting_id)
    speakers = sorted({s.get("speaker", "") for s in data["transcript"] if s.get("speaker")})
    out = tempfile.mktemp(suffix=".docx")
    docx_export.build_protocol(
        out,
        title=data["title"],
        meeting_date=date.fromisoformat(data["meeting_date"]),
        participants=speakers,
        transcript=data["transcript"],
        tasks=data["tasks"],
        summary=data["summary"],
    )
    return FileResponse(
        out,
        filename=f"Протокол_{data['meeting_date']}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


WEB = ROOT / "frontend"
if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    page = WEB / "index.html"
    if page.exists():
        return page.read_text(encoding="utf-8")
    return "<h1>Hattama</h1><p>Интерфейс не найден</p>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
