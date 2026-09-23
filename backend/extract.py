"""Извлечение поручений из транскрипта совещания.

Работает полностью локально через Ollama. Наружу ничего не уходит.
На вход — реплики совещания, на выход — список поручений:
кто, что, к какому сроку, и цитата-источник.
"""

import json
import re
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3:4b"

PROMPT = """Ты — секретарь совещания. Твоя задача — выписать из стенограммы ВСЕ поручения.

Поручение — это когда один человек поручает другому что-то сделать.
Признаки: "подготовьте", "организуйте", "жду от вас", "сделайте", "дайте смету",
"соберите совещание", "направьте", а также подведение итогов в конце встречи.

Для каждого поручения укажи:
- text: что именно нужно сделать, кратко и по-деловому
- assignee: кому поручено (имя из стенограммы; если названа должность или отдел — пиши их)
- author: кто поручил
- deadline_raw: срок ровно теми словами, которые прозвучали ("до конца недели", "за две недели",
  "на этой неделе", "к 15 октября"). Если срок не назван — пустая строка
- quote: точная цитата из стенограммы, откуда взято поручение

ВАЖНО:
- Не придумывай ничего, чего нет в тексте.
- Если одно и то же поручение прозвучало дважды (в обсуждении и в итогах) — выпиши один раз.
- Отвечай ТОЛЬКО массивом JSON, без пояснений и без markdown.

Стенограмма:
---
{transcript}
---

Ответ — массив JSON:"""


def _call_ollama(prompt: str, timeout: int = 300) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0.1, "num_ctx": 16384},
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_json_array(raw: str):
    """Модель иногда оборачивает ответ в markdown или добавляет рассуждения."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"В ответе модели нет массива JSON: {raw[:200]}")
    return json.loads(raw[start : end + 1])


def extract_tasks(transcript: str, known_names: list[str] | None = None) -> list[dict]:
    prompt = PROMPT.format(transcript=transcript)
    if known_names:
        people = ", ".join(known_names)
        prompt = prompt.replace(
            "Стенограмма:",
            f"В совещании участвуют: {people}.\n"
            "В поле assignee указывай имя ТОЛЬКО из этого списка. "
            "Если поручение адресовано кому-то вне списка, пиши как в тексте.\n\n"
            "Стенограмма:",
        )
    raw = _call_ollama(prompt)
    tasks = _parse_json_array(raw)
    out = []
    for i, t in enumerate(tasks, 1):
        out.append(
            {
                "id": i,
                "text": (t.get("text") or "").strip(),
                "assignee": (t.get("assignee") or "").strip(),
                "author": (t.get("author") or "").strip(),
                "deadline_raw": (t.get("deadline_raw") or "").strip(),
                "quote": (t.get("quote") or "").strip(),
                "status": "в работе",
            }
        )
    return [t for t in out if t["text"]]


if __name__ == "__main__":
    import sys
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else "samples/meeting1_transcript.txt"
    text = open(path, encoding="utf-8").read()

    t0 = time.time()
    tasks = extract_tasks(text)
    print(f"Найдено поручений: {len(tasks)} за {time.time() - t0:.1f} сек\n")
    for t in tasks:
        print(f"[{t['id']}] {t['text']}")
        print(f"    кому: {t['assignee']}  |  срок: {t['deadline_raw'] or 'не назван'}")
        print(f"    цитата: {t['quote'][:90]}...")
        print()
