"""Распознавание речи и разделение говорящих. Всё локально.

Распознавание — whisper.cpp с моделью large-v3-turbo на видеоядре Mac.
Разделение говорящих — по паузам и смене реплик, с уточнением через имена
из контекста разговора.
"""

import json
import os
import re
import subprocess
import tempfile

MODEL_TURBO = os.path.expanduser("~/Desktop/models/ggml-large-v3-turbo.bin")
MODEL_FULL = os.path.expanduser("~/Desktop/models/ggml-large-v3.bin")
# Качество важнее скорости: если полная модель скачана целиком — работаем на ней.
def _pick_model() -> str:
    if os.path.exists(MODEL_FULL) and os.path.getsize(MODEL_FULL) > 2_900_000_000:
        return MODEL_FULL
    return MODEL_TURBO


MODEL_PATH = _pick_model()
WHISPER = "/opt/homebrew/bin/whisper-cli"

# Подсказка модели помогает с именами, но у неё есть обратная сторона:
# длинный русский текст в подсказке склоняет модель к русскому языку, и казахская
# речь начинает распознаваться хуже. Поэтому в подсказке только имена собственные,
# они не привязаны к языку. Отключить целиком: HATTAMA_PROMPT=off
INITIAL_PROMPT = (
    "Данияр Серикович, Ботагоз Нурлановна, Жандос Талгатович, "
    "Ерболат Мухтарович, Салтанат Ерболовна, Ерлан."
)
USE_PROMPT = os.environ.get("HATTAMA_PROMPT", "on").lower() != "off"
# Обработка звука вытягивает дальних говорящих, но на смешанной речи иногда мешает.
# Отключить: HATTAMA_AUDIO_CLEAN=off
USE_CLEAN = os.environ.get("HATTAMA_AUDIO_CLEAN", "on").lower() != "off"


# Обработка звука перед распознаванием. В переговорной часть людей сидит далеко
# от микрофона, их голоса тише и тонут в гуле техники. Фильтры вытягивают речь:
#   highpass/lowpass — срезают гул кондиционера и шипение
#   afftdn           — подавление постоянного шума
#   speechnorm       — выравнивание громкости между ближними и дальними голосами
# Агрессивное шумоподавление съедает тихие слова, поэтому его тут нет:
# только срез гула техники и мягкое выравнивание громкости, чтобы дальние
# голоса звучали сопоставимо с ближними.
AUDIO_FILTERS = "highpass=f=70,speechnorm=e=6.25:r=0.00001:l=1"


def to_wav(src: str, clean: bool = True) -> str:
    """Любое аудио -> моно 16 кГц с обработкой речи, как требует модель."""
    out = tempfile.mktemp(suffix=".wav")
    cmd = ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le"]
    if clean:
        cmd += ["-af", AUDIO_FILTERS]
    cmd += [out, "-loglevel", "error"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # если фильтры недоступны в этой сборке ffmpeg, работаем без них
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
             out, "-loglevel", "error"],
            check=True,
        )
    return out


def build_prompt(known_names: list[str] | None = None) -> str:
    """Словарь для модели: только имена участников, без фраз на одном языке.

    Так модель реже ошибается в именах и при этом не склоняется к русскому,
    когда на совещании звучит казахская или смешанная речь.
    """
    if not USE_PROMPT:
        return ""
    if known_names:
        return ", ".join(known_names) + "."
    return INITIAL_PROMPT


def transcribe(audio_path: str, language: str = "auto",
               known_names: list[str] | None = None) -> list[dict]:
    """Возвращает реплики: [{start, end, text}] в секундах."""
    model = _pick_model()
    wav = to_wav(audio_path, clean=USE_CLEAN)
    out_prefix = tempfile.mktemp()
    cmd = [
        WHISPER, "-m", model, "-f", wav,
        "-l", language,             # auto = определять язык самостоятельно
        "-oj", "-of", out_prefix,   # вывод в JSON
        "-t", "8",                  # потоки процессора
        "-bs", "5",                 # поиск лучшего варианта вместо первого попавшегося
        "-bo", "5",
        "-et", "2.6",               # порог, после которого пробуется другой вариант
        *(["--prompt", build_prompt(known_names)] if build_prompt(known_names) else []),
        "--max-len", "0",
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    with open(out_prefix + ".json", encoding="utf-8") as f:
        data = json.load(f)

    segments = []
    for seg in data.get("transcription", []):
        offsets = seg.get("offsets", {})
        text = seg.get("text", "").strip()
        if not text:
            continue
        segments.append(
            {
                "start": offsets.get("from", 0) / 1000,
                "end": offsets.get("to", 0) / 1000,
                "text": text,
            }
        )

    for p in (wav, out_prefix + ".json"):
        try:
            os.remove(p)
        except OSError:
            pass
    return segments


def group_speakers(segments: list[dict], pause_threshold: float = 1.2) -> list[dict]:
    """Грубое разделение говорящих: длинная пауза = смена говорящего.

    Работает без моделей и не требует загрузки весов. Имена подставляются
    отдельным шагом по контексту разговора, а менеджер может поправить вручную.
    """
    speaker = 1
    for i, seg in enumerate(segments):
        if i > 0:
            gap = seg["start"] - segments[i - 1]["end"]
            if gap > pause_threshold:
                speaker += 1
        seg["speaker_id"] = f"Говорящий {speaker}"
        seg["speaker"] = seg["speaker_id"]
    return segments


NAME_PATTERN = re.compile(
    r"\b([А-ЯЁ][а-яё]+(?:бек|бай|жан|ат|ов|ев|ин|нов)?)\s+([А-ЯЁ][а-яё]+(?:вич|евич|ович|овна|евна|ызы|улы))\b"
)


def guess_names(segments: list[dict]) -> dict:
    """Ищет обращения вида «Ботагоз Нурлановна, вам слово» и привязывает имя
    к следующему говорящему."""
    mapping = {}
    for i, seg in enumerate(segments):
        m = NAME_PATTERN.search(seg["text"])
        if not m:
            continue
        name = f"{m.group(1)} {m.group(2)}"
        for nxt in segments[i + 1 : i + 3]:
            if nxt["speaker_id"] != seg["speaker_id"]:
                mapping.setdefault(nxt["speaker_id"], name)
                break
    return mapping


def apply_names(segments: list[dict], known_names: list[str] | None = None) -> list[dict]:
    mapping = guess_names(segments)
    if known_names:
        # Имена отметившихся: если в реплике прозвучало обращение к участнику
        # из списка, привязываем следующего говорящего к нему.
        for i, seg in enumerate(segments):
            for name in known_names:
                first = name.split()[0]
                if first and first.lower() in seg["text"].lower():
                    for nxt in segments[i + 1 : i + 3]:
                        if nxt["speaker_id"] != seg["speaker_id"]:
                            mapping.setdefault(nxt["speaker_id"], name)
                            break
    for seg in segments:
        seg["speaker"] = mapping.get(seg["speaker_id"], seg["speaker_id"])
    return segments


def process(audio_path: str, known_names: list[str] | None = None) -> list[dict]:
    if USE_MIXED:
        segments = transcribe_mixed(audio_path, known_names=known_names)
    else:
        segments = transcribe(audio_path, known_names=known_names)
    segments = group_speakers(segments)
    segments = apply_names(segments, known_names)
    return segments


def to_text(segments: list[dict]) -> str:
    return "\n".join(f"{s['speaker']}: {s['text']}" for s in segments)


if __name__ == "__main__":
    import sys
    import time

    path = sys.argv[1]
    t0 = time.time()
    segs = process(path)
    print(f"Реплик: {len(segs)} за {time.time() - t0:.1f} сек\n")
    for s in segs[:25]:
        print(f"[{s['start']:6.1f}] {s['speaker']}: {s['text']}")


def segment_loudness(audio_path: str, start: float, end: float) -> float:
    """Средняя громкость куска записи в децибелах.

    Нужна, чтобы при записи с нескольких телефонов понять, чей это голос:
    в дорожке владельца он звучит громко, в чужих дорожках тихо и издалека.
    """
    if end <= start:
        return -99.0
    try:
        out = subprocess.run(
            # volumedetect печатает результат на уровне info, поэтому вывод не глушим
            ["ffmpeg", "-hide_banner", "-nostats", "-ss", str(max(0.0, start)),
             "-t", str(end - start),
             "-i", audio_path, "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, check=True,
        )
        for line in out.stderr.splitlines():
            if "mean_volume" in line:
                return float(line.split(":")[1].strip().split()[0])
    except Exception:
        pass
    return -99.0


def drop_crosstalk(segments: list[dict], overlap: float = 0.6) -> list[dict]:
    """Одну реплику слышат все микрофоны, поэтому она попадает в несколько дорожек.

    Из пересекающихся во времени вариантов оставляем тот, где голос громче:
    кто громче, тот и говорит. Остальные версии отбрасываем, чтобы
    в стенограмме не задваивались реплики и чужие слова не приписывались молчавшим.
    """
    ordered = sorted(segments, key=lambda s: (-s.get("loudness", -99.0), s["start"]))
    kept: list[dict] = []
    for seg in ordered:
        clash = False
        for k in kept:
            if k["speaker"] == seg["speaker"]:
                continue
            inter = min(seg["end"], k["end"]) - max(seg["start"], k["start"])
            shorter = min(seg["end"] - seg["start"], k["end"] - k["start"]) or 1
            if inter / shorter > overlap:
                clash = True
                break
        if not clash:
            kept.append(seg)
    kept.sort(key=lambda s: s["start"])
    return kept


# --- Смешанная речь ------------------------------------------------------------
# Whisper определяет язык один раз на всю запись. На совещании, где русский
# и казахский чередуются, это даёт перекос: казахские фразы «переводятся»
# в похожие русские слова. Поэтому запись прогоняется дважды, а затем по каждой
# реплике выбирается тот вариант, который действительно на своём языке.

KZ_LETTERS = set("әғқңөұүһі")
# Частые казахские слова без особых букв: по ним тоже видно язык реплики
KZ_WORDS = {
    "бойынша", "керек", "қажет", "және", "үшін", "болады", "деп", "бар", "жоқ",
    "мен", "сіз", "біз", "осы", "бұл", "сол", "енді", "жақсы", "рахмет", "ия",
    "ме", "ма", "ба", "бе", "па", "пе", "та", "те", "да", "де",
    "аптада", "айда", "жылы", "күні", "уақыт", "жиналыс", "есеп", "тапсырма",
}
USE_MIXED = os.environ.get("HATTAMA_MIXED", "on").lower() != "off"


def kazakh_score(text: str) -> float:
    """Доля казахских слов в реплике.

    Считаем по словам, а не по буквам: когда казахский проход слышит русскую речь,
    он иногда вставляет одну казахскую букву в слово, и оценка по буквам
    ошибочно объявляет всю фразу казахской.

    Казахским считаем слово, в котором есть буква из казахского алфавита
    либо которое входит в список частых казахских слов.
    """
    words = re.findall(r"[^\W\d_]+", text.lower(), flags=re.UNICODE)
    if not words:
        return 0.0
    kz = 0
    for w in words:
        if set(w) & KZ_LETTERS or w in KZ_WORDS:
            kz += 1
    return kz / len(words)


def _overlap(a: dict, b: dict) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    shorter = min(a["end"] - a["start"], b["end"] - b["start"]) or 1
    return inter / shorter


def merge_languages(ru: list[dict], kk: list[dict], threshold: float = 0.25) -> list[dict]:
    """Склейка двух проходов.

    Русский проход берём за основу: деловая часть совещания обычно на русском.
    Реплику заменяем казахским вариантом только если он уверенно казахский:
    четверть слов и больше. Тогда единичные казахские слова в русской фразе
    не переключают всю реплику.
    """
    out = []
    for seg in ru:
        best = seg
        for other in kk:
            if _overlap(seg, other) < 0.5:
                continue
            score_kk, score_ru = kazakh_score(other["text"]), kazakh_score(seg["text"])
            if score_kk >= threshold and score_kk > score_ru:
                best = {**seg, "text": other["text"], "lang": "kk"}
            break
        out.append(best)

    # казахские реплики, которых русский проход не услышал вовсе
    for other in kk:
        if kazakh_score(other["text"]) < threshold:
            continue
        if any(_overlap(other, s) > 0.5 for s in out):
            continue
        out.append({**other, "lang": "kk"})

    out.sort(key=lambda s: s["start"])
    return out


def transcribe_mixed(audio_path: str, known_names: list[str] | None = None) -> list[dict]:
    """Два прохода: русский и казахский, затем выбор варианта по каждой реплике."""
    ru = transcribe(audio_path, language="ru", known_names=known_names)
    kk = transcribe(audio_path, language="kk", known_names=known_names)
    if not kk:
        return ru
    if not ru:
        return kk
    return merge_languages(ru, kk)
