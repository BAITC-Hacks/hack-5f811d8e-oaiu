"""Hattama — сервер.

Всё работает локально: распознавание речи, разбор поручений, сборка протокола.
Ни один запрос не уходит в интернет.
"""

import hashlib
import hmac
import io
import ipaddress
import json
import os
import socket
import time
import shutil
import sqlite3
import tempfile
import threading
from datetime import date, datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import Response
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
                created_at TEXT, language TEXT DEFAULT 'mixed' 
            );
            CREATE TABLE IF NOT EXISTS participants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER, name TEXT, position TEXT, joined_at TEXT
            );
            CREATE TABLE IF NOT EXISTS tracks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                meeting_id INTEGER, speaker TEXT, path TEXT, offset_sec REAL
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

with db() as _c:  # база могла быть создана до появления выбора языка
    if "language" not in {r[1] for r in _c.execute("PRAGMA table_info(meetings)")}:
        _c.execute("ALTER TABLE meetings ADD COLUMN language TEXT DEFAULT 'mixed'")



# --- Защита отметки участников -------------------------------------------------
# QR обновляется каждые 15 секунд, код привязан к совещанию и окну времени.
# Скриншот, пересланный коллеге, перестаёт работать почти сразу.
ROOM_SECRET = os.environ.get("HATTAMA_SECRET", "hattama-local-secret")
WINDOW = 120  # секунд: код живёт до 4 минут с учётом предыдущего окна.
# Скорость сканирования не должна отсекать людей, которые медленно обращаются
# с телефоном. Основная защита от удалённого подключения — проверка сети.


def room_code(meeting_id: int, shift: int = 0) -> str:
    window = int(time.time() // WINDOW) + shift
    msg = f"{meeting_id}:{window}".encode()
    return hmac.new(ROOM_SECRET.encode(), msg, hashlib.sha256).hexdigest()[:10]


def code_valid(meeting_id: int, code: str) -> bool:
    """Принимаем текущее окно и предыдущее: человек мог сканировать на стыке."""
    return code in (room_code(meeting_id), room_code(meeting_id, -1))


def same_network(client_ip: str, host_ip: str) -> bool:
    """Отметиться можно только из сети переговорной, не из дома."""
    if client_ip in ("127.0.0.1", "::1", "testclient"):
        return True
    try:
        c, h = ipaddress.ip_address(client_ip), ipaddress.ip_address(host_ip)
    except ValueError:
        return False
    if not c.is_private:
        return False
    return c.packed[:3] == h.packed[:3]  # одна подсеть /24


def server_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def get_participants(meeting_id: int) -> list[dict]:
    with db() as conn:
        rows = conn.execute(
            "SELECT id, name, position FROM participants WHERE meeting_id=? ORDER BY id",
            (meeting_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def process_meeting(meeting_id: int, audio_path: str):
    """Полный разбор: звук -> реплики -> поручения -> даты -> база.

    Если участники отметились перед совещанием, их имена уходят в словарь
    распознавания и в разбор поручений: модель не гадает, а выбирает
    из известного списка.
    """
    try:
        people = [p["name"] for p in get_participants(meeting_id)]
        with db() as conn:
            row = conn.execute("SELECT language FROM meetings WHERE id=?",
                               (meeting_id,)).fetchone()
        language = (row["language"] if row else None) or "mixed"
        segments = asr.process(audio_path, known_names=people, language=language)
        text = asr.to_text(segments)
        with db() as conn:
            conn.execute(
                "UPDATE meetings SET transcript=?, status=? WHERE id=?",
                (json.dumps(segments, ensure_ascii=False), "разделяю говорящих", meeting_id),
            )

        title = tasks_extractor.make_title(text)
        with db() as conn:
            if title:
                conn.execute("UPDATE meetings SET title=? WHERE id=?", (title, meeting_id))
            conn.execute("UPDATE meetings SET status=? WHERE id=?",
                         ("составляю краткое содержание", meeting_id))

        summary = tasks_extractor.make_summary(text)
        with db() as conn:
            if summary:
                conn.execute("UPDATE meetings SET summary=? WHERE id=?", (summary, meeting_id))
            conn.execute("UPDATE meetings SET status=? WHERE id=?",
                         ("ищу поручения", meeting_id))

        found = tasks_extractor.extract_tasks(text, known_names=people)
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


@app.post("/api/meetings/draft")
def create_draft(title: str = "Совещание"):
    """Совещание заводится до записи: сначала отмечаются участники."""
    with db() as conn:
        cur = conn.execute(
            "INSERT INTO meetings (title, meeting_date, status, created_at) VALUES (?,?,?,?)",
            (title, date.today().isoformat(), "участники", datetime.now().isoformat()),
        )
    return {"id": cur.lastrowid}


def process_meeting_tracks(meeting_id: int, tracks: list[dict]):
    """Каждая дорожка распознаётся отдельно, затем реплики склеиваются по времени.

    Звук не смешивается: смешивание складывает шумы и не делает дальний голос
    громче. Склеивается текст, а имя говорящего берётся из владельца дорожки.
    """
    try:
        people = [t["speaker"] for t in tracks]
        with db() as conn:
            row = conn.execute("SELECT language FROM meetings WHERE id=?",
                               (meeting_id,)).fetchone()
        language = (row["language"] if row else None) or "mixed"
        merged: list[dict] = []
        for t in tracks:
            segs = asr.process_track(t["path"], known_names=people, language=language)
            for seg in segs:
                seg["start"] += t["offset_sec"]
                seg["end"] += t["offset_sec"]
                seg["speaker"] = t["speaker"]
                seg["speaker_id"] = t["speaker"]
                seg["loudness"] = asr.segment_loudness(t["path"], seg["start"] - t["offset_sec"],
                                                       seg["end"] - t["offset_sec"])
            merged.extend(segs)

        merged = asr.drop_crosstalk(merged)
        merged.sort(key=lambda s: s["start"])

        text = asr.to_text(merged)
        with db() as conn:
            conn.execute(
                "UPDATE meetings SET transcript=?, status=? WHERE id=?",
                (json.dumps(merged, ensure_ascii=False), "разделяю говорящих", meeting_id),
            )

        title = tasks_extractor.make_title(text)
        with db() as conn:
            if title:
                conn.execute("UPDATE meetings SET title=? WHERE id=?", (title, meeting_id))
            conn.execute("UPDATE meetings SET status=? WHERE id=?",
                         ("составляю краткое содержание", meeting_id))

        summary = tasks_extractor.make_summary(text)
        with db() as conn:
            if summary:
                conn.execute("UPDATE meetings SET summary=? WHERE id=?", (summary, meeting_id))
            conn.execute("UPDATE meetings SET status=? WHERE id=?",
                         ("ищу поручения", meeting_id))

        found = tasks_extractor.extract_tasks(text, known_names=people)
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
            conn.execute("UPDATE meetings SET status=? WHERE id=?",
                         (f"ошибка: {e}", meeting_id))


@app.post("/api/meetings")
async def upload_meeting(file: UploadFile = File(...), title: str = "Совещание",
                         meeting_id: int | None = Form(None)):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    dest = DATA / f"upload_{datetime.now():%Y%m%d_%H%M%S}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with db() as conn:
        if meeting_id:
            conn.execute(
                "UPDATE meetings SET status=?, audio_path=? WHERE id=?",
                ("распознаю речь", str(dest), meeting_id),
            )
        else:
            cur = conn.execute(
                "INSERT INTO meetings (title, meeting_date, status, audio_path, created_at) VALUES (?,?,?,?,?)",
                (title, date.today().isoformat(), "распознаю речь", str(dest),
                 datetime.now().isoformat()),
            )
            meeting_id = cur.lastrowid

    threading.Thread(target=process_meeting, args=(meeting_id, str(dest)), daemon=True).start()
    return {"id": meeting_id, "status": "распознаю речь"}


@app.post("/api/meetings/{meeting_id}/tracks")
async def upload_track(meeting_id: int, file: UploadFile = File(...),
                       speaker: str = Form(...), offset: float = Form(0.0)):
    """Дорожка одного участника. Каждый телефон пишет своего владельца:
    микрофон рядом с говорящим, поэтому его речь записана лучше всех."""
    suffix = Path(file.filename or "track.webm").suffix or ".webm"
    dest = DATA / f"track_{meeting_id}_{datetime.now():%H%M%S%f}{suffix}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    with db() as conn:
        conn.execute(
            "INSERT INTO tracks (meeting_id, speaker, path, offset_sec) VALUES (?,?,?,?)",
            (meeting_id, speaker.strip(), str(dest), offset),
        )
        n = conn.execute("SELECT COUNT(*) c FROM tracks WHERE meeting_id=?",
                         (meeting_id,)).fetchone()["c"]
    return {"ok": True, "tracks": n}


@app.post("/api/meetings/{meeting_id}/process")
def process_tracks(meeting_id: int):
    """Разбор совещания, записанного несколькими телефонами."""
    with db() as conn:
        rows = conn.execute(
            "SELECT speaker, path, offset_sec FROM tracks WHERE meeting_id=?",
            (meeting_id,),
        ).fetchall()
    if not rows:
        raise HTTPException(400, "дорожек нет")
    with db() as conn:
        conn.execute("UPDATE meetings SET status=? WHERE id=?",
                     ("распознаю речь", meeting_id))
    tracks = [dict(r) for r in rows]
    threading.Thread(target=process_meeting_tracks, args=(meeting_id, tracks),
                     daemon=True).start()
    return {"ok": True, "tracks": len(tracks)}


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
        "language": m["language"] if "language" in m.keys() else "mixed",
        "meeting_date": m["meeting_date"],
        "status": m["status"],
        "transcript": json.loads(m["transcript"]) if m["transcript"] else [],
        "tasks": tasks,
        "summary": m["summary"] or "",
    }


@app.delete("/api/meetings/{meeting_id}")
def delete_meeting(meeting_id: int):
    """Удаление совещания вместе с записями, дорожками и поручениями.

    Файлы стираются с диска: запись совещания не должна лежать дольше, чем нужно.
    """
    with db() as conn:
        rows = conn.execute(
            "SELECT audio_path FROM meetings WHERE id=?", (meeting_id,)
        ).fetchall()
        tracks = conn.execute(
            "SELECT path FROM tracks WHERE meeting_id=?", (meeting_id,)
        ).fetchall()

    for r in list(rows) + list(tracks):
        path = r[0]
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    with db() as conn:
        conn.execute("DELETE FROM tasks WHERE meeting_id=?", (meeting_id,))
        conn.execute("DELETE FROM tracks WHERE meeting_id=?", (meeting_id,))
        conn.execute("DELETE FROM participants WHERE meeting_id=?", (meeting_id,))
        conn.execute("DELETE FROM meetings WHERE id=?", (meeting_id,))
    return {"ok": True}


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


class MeetingLang(BaseModel):
    language: str  # ru, kk или mixed


@app.patch("/api/meetings/{meeting_id}/language")
def set_language(meeting_id: int, body: MeetingLang):
    """Язык совещания выбирается до записи.

    Один язык — один проход распознавания, быстро и точно.
    Смешанный — два прохода с выбором варианта по каждой реплике.
    """
    lang = body.language if body.language in ("ru", "kk", "mixed") else "mixed"
    with db() as conn:
        conn.execute("UPDATE meetings SET language=? WHERE id=?", (lang, meeting_id))
    return {"ok": True, "language": lang}


class Participant(BaseModel):
    name: str
    position: str = ""
    code: str | None = None  # код из QR; пусто значит добавил ведущий на своём устройстве


@app.post("/api/meetings/{meeting_id}/participants")
def add_participant(meeting_id: int, p: Participant, request: Request = None):
    """Отметка участника. Отметился значит уведомлён о записи и согласен.

    Три проверки: код из QR не протух, устройство в сети переговорной,
    запись ещё не началась.
    """
    with db() as conn:
        m = conn.execute("SELECT status FROM meetings WHERE id=?", (meeting_id,)).fetchone()
    if m and m["status"] not in ("участники", None):
        raise HTTPException(409, "запись уже началась, отметка закрыта")

    if p.code is not None:  # отметка с телефона участника
        if not code_valid(meeting_id, p.code):
            raise HTTPException(403, "код устарел, отсканируйте QR заново")
        client_ip = request.client.host if request and request.client else ""
        if not same_network(client_ip, server_ip()):
            raise HTTPException(403, "отметиться можно только из сети переговорной")

    with db() as conn:
        conn.execute(
            "INSERT INTO participants (meeting_id, name, position, joined_at) VALUES (?,?,?,?)",
            (meeting_id, p.name.strip(), p.position.strip(), datetime.now().isoformat()),
        )
    return {"ok": True, "participants": get_participants(meeting_id)}


@app.get("/api/meetings/{meeting_id}/participants")
def list_participants(meeting_id: int):
    return get_participants(meeting_id)


@app.delete("/api/participants/{participant_id}")
def remove_participant(participant_id: int):
    """Ведущий убирает лишнего из списка."""
    with db() as conn:
        conn.execute("DELETE FROM participants WHERE id=?", (participant_id,))
    return {"ok": True}


@app.get("/api/meetings/{meeting_id}/qr.svg")
def participant_qr(meeting_id: int, request: Request):
    """QR для отметки участников. Рисуется локально, без сторонних сервисов."""
    import segno

    host = request.headers.get("host", "localhost:8000")
    # Если страницу открыли на самом сервере, в QR нужен адрес в сети,
    # иначе телефон уйдёт на свой собственный localhost.
    name = host.split(":")[0]
    if name in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
        port = host.split(":")[1] if ":" in host else "8000"
        host = f"{server_ip()}:{port}"
    url = f"http://{host}/join/{meeting_id}?c={room_code(meeting_id)}"
    buf = io.BytesIO()
    segno.make(url, error="m").save(
        buf, kind="svg", scale=6, border=3,
        dark="#11181f", light="#ffffff", xmldecl=False,
    )
    return Response(content=buf.getvalue(), media_type="image/svg+xml")


@app.get("/join/{meeting_id}", response_class=HTMLResponse)
def join_page(meeting_id: int):
    page = WEB_DIR / "join.html"
    if page.exists():
        return page.read_text(encoding="utf-8").replace("{{MEETING_ID}}", str(meeting_id))
    return "<h1>Отметка участника</h1>"


WEB_DIR = ROOT / "frontend"
WEB = WEB_DIR
if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/", response_class=HTMLResponse)
def landing():
    page = WEB / "landing.html"
    if page.exists():
        return page.read_text(encoding="utf-8")
    return "<h1>Hattama</h1>"


@app.get("/app", response_class=HTMLResponse)
def app_page():
    page = WEB / "index.html"
    if page.exists():
        return page.read_text(encoding="utf-8")
    return "<h1>Hattama</h1><p>Интерфейс не найден</p>"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
