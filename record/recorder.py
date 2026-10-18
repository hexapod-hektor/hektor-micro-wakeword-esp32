"""Записувач: зберігає звук з вух робота між «start» і «stop» у WAV + запис у data/manifest.json.

Стоїть першим після transport.input(), пропускає всі кадри далі без змін. Керується з record.py (ця ж тека)
(HTTP /rec/start, /rec/stop): Mac грає кліп у динамік, робот чує, ми зберігаємо те, що прийшло
через INMP441 → Opus → WebRTC — саме те, що бачить детектор wake-слова.

Файли: data/robot/<name>.wav (перезаписи кліпів) або data/background/<name>.wav (фон:
серво, динамік, кімната). Формат як у решти набору: 16 кГц, 16 біт, моно; у перезаписах тиша
по краях обрізана, у фоні — ні.
"""
import datetime as dt
import json
import time
import wave
from pathlib import Path

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

WAKE = Path(__file__).resolve().parents[1] / "data"
MANIFEST = WAKE / "manifest.json"


def trim(x: np.ndarray, rate: int, pad_ms=100, drop_db=25.0, gap_ms=700) -> np.ndarray:
    """Тиша по краях геть. Кадри 10 мс → RMS у дБ → медіана по 5 кадрах (поодинокі сплески шуму вух,
    −33…−40 дБ, зникають) → «мова» = кадри гучніші за пік − drop_db. Беремо шматок мови навколо піку
    і доклеюємо сусідні шматки, якщо пауза між ними < gap_ms (після коми в «Привіт, Гектор» буває 0.6 с). pad_ms запасу з боків.
    Шум вух робота ≈ −52 дБ зі сплесками, мова −5…−20 (перевірено 05.10.2026)."""
    f = rate // 100
    n = len(x) // f
    if n < 5:
        return x
    rms = np.sqrt((x[: n * f].astype(np.float32).reshape(n, f) ** 2).mean(axis=1))
    db = 20 * np.log10(rms + 1e-9)
    db = np.array([np.median(db[max(0, i - 2): i + 3]) for i in range(n)])
    loud = db > db.max() - drop_db
    # суцільні шматки мови
    segs, start = [], None
    for i, l in enumerate(list(loud) + [False]):
        if l and start is None:
            start = i
        elif not l and start is not None:
            segs.append((start, i)); start = None
    if not segs:
        return x
    peak = int(np.argmax(db))
    k = next(j for j, (a, b) in enumerate(segs) if a <= peak < b)
    a, b = segs[k]
    gap = gap_ms // 10
    for s0, e0 in reversed(segs[:k]):      # вліво
        if a - e0 <= gap: a = s0
        else: break
    for s0, e0 in segs[k + 1:]:            # вправо
        if s0 - b <= gap: b = e0
        else: break
    pad = rate * pad_ms // 1000
    return x[max(0, a * f - pad): min(len(x), b * f + pad)]


class Recorder(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._buf = b""
        self._armed = False
        self._rate = 16000
        self._job = None          # (name, set, meta) поточного запису
        self._started_at = 0.0
        self.frames_seen = 0      # для /rec/status: чи звук узагалі йде

    # ---- керування (з HTTP) ----
    def start(self, name: str, set_: str, meta: dict):
        if self._armed:
            logger.warning(f"rec: {self._job[0]} ще пишеться — закриваю і починаю {name}")
            self.stop()
        self._buf = b""
        self._job = (name, set_, meta)
        self._armed = True
        self._started_at = time.monotonic()

    def stop(self) -> dict | None:
        if not self._armed:
            return None
        self._armed = False
        name, set_, meta = self._job
        x = np.frombuffer(self._buf, dtype=np.int16)
        if set_ != "background":
            x = trim(x, self._rate)
        d = WAKE / ("background" if set_ == "background" else "robot")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(self._rate)
            w.writeframes(x.tobytes())
        entry = {
            "set": set_, "source": "robot", "mic": "spine-ears",
            "sample_rate": self._rate, "duration_s": round(len(x) / self._rate, 2),
            "created": dt.date.today().isoformat(),
            **meta,
        }
        rel = str(path.relative_to(WAKE))
        peak_db = 20 * np.log10(np.abs(x).max() / 32768 + 1e-9) if len(x) else -99.0
        entry["peak_db"] = round(float(peak_db), 1)
        if peak_db > -1.0:   # упирається в межу: динамік тихіше або далі, інакше кліпінг
            logger.warning(f"rec: {name}: пік {peak_db:.1f} дБ — занадто гучно, збав гучність Mac")
        m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}}
        m["files"][rel] = entry
        MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
        logger.info(f"rec: {rel} {entry['duration_s']} с")
        return {"file": rel, "duration_s": entry["duration_s"]}

    def status(self) -> dict:
        return {"armed": self._armed, "job": self._job[0] if self._job else None,
                "frames": self.frames_seen, "rate": self._rate,
                "buffered_s": round(len(self._buf) / 2 / self._rate, 2)}

    # ---- пайплайн ----
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self.frames_seen += 1
            self._rate = frame.sample_rate
            if self._armed:
                self._buf += frame.audio
        await self.push_frame(frame, direction)
