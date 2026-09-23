"""Перевод устных сроков в конкретные даты.

Делается кодом, а не моделью: правило должно быть объяснимым и повторяемым.
Рядом с датой всегда сохраняется формулировка, которая прозвучала на совещании,
и правило, по которому дата получена.
"""

import re
from datetime import date, timedelta

MONTHS = {
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "ма": 5, "июн": 6,
    "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
}
WEEKDAYS = {
    "понедельник": 0, "вторник": 1, "сред": 2, "четверг": 3,
    "пятниц": 4, "суббот": 5, "воскресень": 6,
}


def resolve(deadline_raw: str, meeting_date: date) -> dict:
    """Возвращает {date, rule} — дату и человеческое объяснение, как она получена."""
    if not deadline_raw:
        return {"date": None, "rule": "срок на совещании не назван"}

    s = deadline_raw.lower().strip()

    # «до 15 октября», «к 26 сентября»
    m = re.search(r"(\d{1,2})\s*([а-яё]{3,})", s)
    if m:
        day, mon_word = int(m.group(1)), m.group(2)
        for prefix, num in MONTHS.items():
            if mon_word.startswith(prefix):
                year = meeting_date.year
                d = date(year, num, day)
                if d < meeting_date:
                    d = date(year + 1, num, day)
                return {"date": d, "rule": f"названа конкретная дата: {day} {mon_word}"}

    # «до конца недели», «на этой неделе», «текущая неделя»
    if "конца недели" in s or "этой недел" in s or "текущ" in s:
        d = meeting_date + timedelta(days=(4 - meeting_date.weekday()) % 7)
        return {"date": d, "rule": "конец текущей рабочей недели — ближайшая пятница"}

    # «на следующей неделе»
    if "следующ" in s and "недел" in s:
        d = meeting_date + timedelta(days=(4 - meeting_date.weekday()) % 7 + 7)
        return {"date": d, "rule": "пятница следующей недели"}

    # «за две недели», «через 3 недели», «1 неделя»
    m = re.search(r"(\d+|одн|две|три)\s*недел", s)
    if m:
        word = m.group(1)
        n = {"одн": 1, "две": 2, "три": 3}.get(word, None)
        n = n if n is not None else int(word)
        d = meeting_date + timedelta(weeks=n)
        return {"date": d, "rule": f"{n} нед. от даты совещания ({meeting_date:%d.%m.%Y})"}

    # «за 10 дней», «через 3 дня»
    m = re.search(r"(\d+)\s*дн", s)
    if m:
        n = int(m.group(1))
        return {
            "date": meeting_date + timedelta(days=n),
            "rule": f"{n} дн. от даты совещания ({meeting_date:%d.%m.%Y})",
        }

    # «до пятницы», «к среде»
    for name, idx in WEEKDAYS.items():
        if name in s:
            delta = (idx - meeting_date.weekday()) % 7
            delta = delta or 7
            return {
                "date": meeting_date + timedelta(days=delta),
                "rule": f"ближайший день недели после совещания",
            }

    # «до конца месяца»
    if "конца месяца" in s:
        nxt = (meeting_date.replace(day=28) + timedelta(days=4)).replace(day=1)
        return {"date": nxt - timedelta(days=1), "rule": "последний день текущего месяца"}

    return {"date": None, "rule": f"формулировку «{deadline_raw}» не удалось перевести в дату"}


def enrich(tasks: list[dict], meeting_date: date) -> list[dict]:
    today = date.today()
    for t in tasks:
        r = resolve(t.get("deadline_raw", ""), meeting_date)
        t["deadline"] = r["date"].isoformat() if r["date"] else None
        t["deadline_rule"] = r["rule"]
        if r["date"] and r["date"] < today and t.get("status") == "в работе":
            t["status"] = "просрочено"
    return tasks


if __name__ == "__main__":
    md = date(2026, 9, 23)
    for raw in ["до конца недели", "за две недели", "на этой неделе", "1 неделя",
                "к 15 октября", "до пятницы", "следующая неделя", ""]:
        r = resolve(raw, md)
        d = f"{r['date']:%d.%m.%Y}" if r["date"] else "—"
        print(f"{raw or '(не назван)':<20} → {d:<12} {r['rule']}")
