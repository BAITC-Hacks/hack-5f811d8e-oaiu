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

# Подсказка модели: имена, отчества и деловые обороты, которых она иначе не знает.
# Резко снижает число ошибок в именах — а имя нам нужно для привязки поручения.
INITIAL_PROMPT = (
    "Оперативное совещание. Участники: Данияр Серикович, Ботагоз Нурлановна, "
    "Жандос Талгатович, Ерболат Мухтарович, Салтанат Ерболовна, Ерлан. "
    "Обсуждаются поручения, сроки и показатели: до конца недели, за две недели, "
    "на этой неделе, к пятнадцатому октября, претензия поставщику, смета, "
    "подрядчики, документация, переаттестация. Сәлеметсіздер ме, жақсы, рахмет."
)


def to_wav(src: str) -> str:
    """Любое аудио -> моно 16 кГц, как требует модель."""
    out = tempfile.mktemp(suffix=".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
         out, "-loglevel", "error"],
        check=True,
    )
    return out


def transcribe(audio_path: str, language: str = "auto") -> list[dict]:
    """Возвращает реплики: [{start, end, text}] в секундах."""
    model = _pick_model()
    wav = to_wav(audio_path)
    out_prefix = tempfile.mktemp()
    cmd = [
        WHISPER, "-m", model, "-f", wav,
        "-l", language,             # auto = определять язык самостоятельно
        "-oj", "-of", out_prefix,   # вывод в JSON
        "-t", "8",                  # потоки процессора
        "-bs", "5",                 # поиск лучшего варианта вместо первого попавшегося
        "-bo", "5",
        "-et", "2.6",               # порог, после которого пробуется другой вариант
        "--prompt", INITIAL_PROMPT, # словарь имён и терминов
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


def apply_names(segments: list[dict]) -> list[dict]:
    mapping = guess_names(segments)
    for seg in segments:
        seg["speaker"] = mapping.get(seg["speaker_id"], seg["speaker_id"])
    return segments


def process(audio_path: str) -> list[dict]:
    segments = transcribe(audio_path)
    segments = group_speakers(segments)
    segments = apply_names(segments)
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
