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


def _call_ollama(prompt: str, timeout: int = 300, max_tokens: int | None = None,
                 fmt: str | None = None) -> str:
    options = {"temperature": 0.1, "num_ctx": 16384}
    if max_tokens:
        options["num_predict"] = max_tokens
    resp = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            **({"format": fmt} if fmt else {}),
            "options": options,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def _parse_json_array(raw: str):
    """Модель иногда оборачивает ответ в markdown или добавляет рассуждения."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S).strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start = raw.find("[")
    if start == -1:
        raise ValueError(f"В ответе модели нет массива JSON: {raw[:200]}")
    # модель иногда дописывает пояснения после массива: берём только массив
    try:
        data, _ = json.JSONDecoder().raw_decode(raw[start:])
        return data
    except json.JSONDecodeError:
        end = raw.rfind("]")
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


TITLE_PROMPT = """Придумай короткое название для совещания по его стенограмме.
Ответ строго в формате JSON: {{"title": "название"}}

Требования:
- от двух до пяти слов
- по сути обсуждения, а не общими словами
- без кавычек, без точки в конце
- примеры хороших: Поставки сырья и претензия, Инвестпрограмма и подрядчики,
  Охрана труда и переаттестация
- плохие примеры: Совещание, Рабочая встреча, Обсуждение вопросов

Стенограмма:
---
{transcript}
---

Ответ JSON:"""


def make_title(transcript: str) -> str:
    """Название совещания по содержанию. Локально, как и всё остальное.

    Ответ короткий, поэтому ограничиваем длину и отключаем размышления модели:
    иначе на заголовок уходит больше времени, чем на разбор поручений.
    """
    try:
        raw = _call_ollama(
            "/no_think\n" + TITLE_PROMPT.format(transcript=transcript[:1500]),
            timeout=90, max_tokens=60, fmt="json",
        )
        title = json.loads(raw).get("title", "").strip().strip('"«». ')
        words = title.split()
        if 1 < len(words) <= 8:
            return title[0].upper() + title[1:]
    except Exception:
        pass
    return ""
