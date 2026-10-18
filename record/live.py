"""Живий тест wake-моделей через вуха робота: вердикт у терміналі і смужки в браузері.

    cd record && uv run live.py models/hektor_v4.onnx -t webrtc --esp32 --host 192.168.0.25
    → http://127.0.0.1:8092  (основний бот і record.py зупинити — порт 7860 один)

Моделі — шляхи до .onnx (openWakeWord) і models/<назва>/manifest.json (microWakeWord, поруч model.tflite)
серед аргументів; без них бере всі models/*.onnx плюс «hey jarvis» як контроль і всі models/*/manifest.json.
Поріг — `--threshold 0.2` (типово 0.5, як у воротах бота); у micro-моделей свій поріг з manifest.json.

Той самий транспорт, що й record.py, але замість записувача — Scorer: кожні 80 мс рахує score моделей
на звуку з вух. Кожен сплеск звуку (рівень вище -30 дБ, далі 0.7 с тиші) — одна репліка: у термінал
іде рядок з піком score кожної моделі і ✓/✗ проти порогу, в кінці — підсумок «спрацювало N з M».
Сторінка через SSE (/events, 10 разів на секунду): смужки score, рівень, журнал вище порогу.
Нічого не пише на диск і не йде в STT — лише дивимось, що бачать ворота в реальному часі.
"""
import asyncio
import datetime as dt
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np
from aiohttp import web
from dotenv import load_dotenv
from loguru import logger
from openwakeword.model import Model
from pymicro_wakeword import MicroWakeWord, MicroWakeWordFeatures
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.runner.types import RunnerArguments
from pipecat.workers.runner import WorkerRunner

from recorder import MANIFEST, WAKE, trim
from transport import make_transport

load_dotenv(".env.local", override=True)
PORT = 8092
HERE = Path(__file__).resolve().parent
_scorer = None
_http_started = False
_log = []            # останні події автозапису для сторінки
WAKE_PAT = re.compile(r"гект|хект|ект[оа]р|хактор|ректор")   # як Whisper чує «Гектор» на одиночному слові
SPEECH_DB, SILENCE_MS, MIN_MS, MAX_MS, PRE_MS = -25.0, 600, 250, 3000, 250   # гудіння серво дає піки −34…−40 дБ
HALLUC = re.compile(r"дякую|перегляд|субтитр|продовження|підпис")   # так Whisper «чує» тишу й шум
_whisper = None


def hear(path: Path):
    """(текст, ймовірність «тут нема мови»). Whisper на шумі вигадує «Дякую за перегляд!» — тому no_speech_prob."""
    import mlx_whisper
    r = mlx_whisper.transcribe(str(path), path_or_hf_repo="mlx-community/whisper-large-v3-turbo", language="uk")
    nsp = max((seg.get("no_speech_prob", 0.0) for seg in r.get("segments", [])), default=1.0)
    return r["text"].strip(), float(nsp)


def save_clip(x: np.ndarray, set_: str, key: str, text: str, meta: dict, whisper_text: str) -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    name = f"live-{key}-{meta.get('speaker', 'martyn')}-{meta.get('distance_m', 2)}m-{meta.get('position', 'front')}-{stamp}"
    out = WAKE / "robot" / f"{name}.wav"
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.tobytes())
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}}
    m["files"][f"robot/{name}.wav"] = {
        "set": set_, "phrase": key, "text": text, "source": "robot", "mic": "spine-ears", "via": "human",
        "speaker": meta.get("speaker", "martyn"), "distance_m": float(meta.get("distance_m", 2)), "position": meta.get("position", "front"),
        "run": meta.get("run", "live-page"), "emotion": "neutral", "speed": 1.0, "sample_rate": 16000,
        "duration_s": round(len(x) / 16000, 2), "peak_db": round(float(20 * np.log10(np.abs(x).max() / 32768 + 1e-9)), 1),
        "whisper_text": whisper_text, "scores": dict(meta.get("scores", {})), "created": dt.date.today().isoformat(),
    }
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    return f"{name}.wav"


SPEECH_DB = -25.0     # гучніше — вважаємо, що хтось говорить (гудіння серво сягає −34)
GAP_S = 0.7           # стільки тиші закриває репліку


def _cli():
    """Витягує з argv свої аргументи (шляхи .onnx і --threshold), решту лишає runner-у Pipecat."""
    models, threshold, rest = [], 0.5, []
    args = iter(sys.argv[1:])
    for a in args:
        if a.endswith(".onnx") or a.endswith("manifest.json"):
            models.append(a)
        elif a.endswith(".tflite"):
            models.append(str(Path(a).with_name("manifest.json")))
        elif a == "--threshold":
            threshold = float(next(args))
        else:
            rest.append(a)
    sys.argv[1:] = rest
    return models, threshold


MODELS, THRESHOLD = _cli()


class Scorer(FrameProcessor):
    def __init__(self, models: list[str], threshold: float):
        super().__init__()
        if not models:
            models = sorted(str(p) for p in (HERE / "models").glob("*.onnx")) + ["hey_jarvis"] \
                + sorted(str(p) for p in (HERE / "models").glob("*/manifest.json"))
        oww = [p for p in models if not p.endswith("manifest.json")]
        mww = [p for p in models if p.endswith("manifest.json")]
        self._model = Model(wakeword_models=oww, inference_framework="onnx")
        # microWakeWord: один фронтенд (40 мел-бінів кожні 10 мс) на всі моделі, кожна зі своїм порогом
        self._micro = {Path(p).parent.name: MicroWakeWord.from_config(p) for p in mww}
        self._feat = MicroWakeWordFeatures()
        self.names = [Path(p).stem for p in oww] + list(self._micro)
        self.threshold = threshold
        self.thr = {n: threshold for n in self.names}
        self.thr.update({n: m.probability_cutoff for n, m in self._micro.items()})
        self._buf = b""
        self.ring = b""                      # останні ~4 с звуку для кнопок «це був Гектор» на сторінці
        self.scores = {n: 0.0 for n in self.names}
        self.level_db = -99.0
        self.frames = 0
        self.connected = False
        self.auto = True                     # автозапис: сегментувати мову, Whisper, зберігати
        self.force = ""                      # "" = Whisper вирішує; "hektor"/"hektore"/"other" = мітити все так
        self.meta = {"speaker": "martyn", "distance_m": 2, "position": "front"}
        self._utt = None                     # bytes поточної фрази або None
        self._utt_peaks = {}                 # пік кожної моделі в межах поточної фрази — це і йде в маніфест
        self._quiet_ms = 0
        self._pre = b""
        self.pending = 0
        # репліка: від першого гучного кадру до GAP_S тиші; пам'ятаємо пік кожної моделі
        self._in_speech = False
        self._last_loud = 0.0
        self._peaks = {n: 0.0 for n in self.names}
        self.utterances = 0
        self.hits = {n: 0 for n in self.names}
        self.verdicts = []                   # останні репліки: хто почув, хто ні (для сторінки)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InputAudioRawFrame):
            self.frames += 1
            self._buf += frame.audio
            self.ring = (self.ring + frame.audio)[-16000 * 2 * 4:]
            while len(self._buf) >= 2560:
                chunk, self._buf = self._buf[:2560], self._buf[2560:]
                x = np.frombuffer(chunk, dtype=np.int16)
                self.level_db = float(20 * np.log10(np.abs(x).max() / 32768 + 1e-9))
                pred = self._model.predict(x)
                for n in self.scores:
                    if n in pred:
                        self.scores[n] = float(pred[n])
                best = {n: 0.0 for n in self._micro}
                for f in self._feat.process_streaming(chunk):
                    for n, m in self._micro.items():
                        best[n] = max(best[n], m.process_streaming_prob(f) or 0.0)
                self.scores.update(best)
                if self.auto:
                    self._segment(chunk)
                self._track(time.monotonic())
        await self.push_frame(frame, direction)

    def _track(self, now: float):
        if self.level_db > SPEECH_DB:
            self._in_speech = True
            self._last_loud = now
        if self._in_speech:
            for n in self.names:
                self._peaks[n] = max(self._peaks[n], self.scores[n])
            if now - self._last_loud >= GAP_S:
                self._finish()

    def _finish(self):
        self.utterances += 1
        parts = []
        for n in self.names:
            ok = self._peaks[n] > self.thr[n]
            self.hits[n] += ok
            parts.append(f"{n} {self._peaks[n]:.2f} {'✓' if ok else '✗'}")
        logger.info(f"wake #{self.utterances}: " + " | ".join(parts))
        self.verdicts = ([{"n": self.utterances, "t": time.strftime("%H:%M:%S"),
                           "peaks": {n: round(self._peaks[n], 2) for n in self.names},
                           "ok": {n: bool(self._peaks[n] > self.thr[n]) for n in self.names}}] + self.verdicts)[:20]
        self._in_speech = False
        self._peaks = {n: 0.0 for n in self.names}

    def summary(self):
        if not self.utterances:
            return "wake: реплік не було"
        return "wake: підсумок — " + ", ".join(f"{n} {self.hits[n]} з {self.utterances}" for n in self.names) \
            + " (пороги " + ", ".join(f"{n} {t:.2f}" for n, t in self.thr.items()) + ")"


    def _segment(self, chunk: bytes):
        """Фраза = звук гучніший за SPEECH_DB, кінець — SILENCE_MS тиші. Порція 80 мс."""
        loud = self.level_db > SPEECH_DB
        if self._utt is None:
            self._pre = (self._pre + chunk)[-16000 * 2 * PRE_MS // 1000:]
            if loud:
                self._utt = self._pre + chunk
                self._quiet_ms = 0
                self._utt_peaks = dict(self.scores)
            return
        self._utt += chunk
        for n in self.names:
            self._utt_peaks[n] = max(self._utt_peaks[n], self.scores[n])
        self._quiet_ms = 0 if loud else self._quiet_ms + 80
        ms = len(self._utt) / 32
        if self._quiet_ms >= SILENCE_MS or ms >= MAX_MS:
            utt, self._utt, self._pre = self._utt, None, b""
            if ms - self._quiet_ms >= MIN_MS:
                self.pending += 1
                asyncio.get_event_loop().create_task(self._save(utt, dict(self.meta, scores={n: round(v, 3) for n, v in self._utt_peaks.items()})))

    async def _save(self, utt: bytes, meta: dict):
        try:
            x = trim(np.frombuffer(utt, dtype=np.int16), 16000)
            tmp = Path("/tmp/live-utt.wav")
            with wave.open(str(tmp), "wb") as w:
                w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.tobytes())
            text, nsp = await asyncio.to_thread(hear, tmp)
            tl = text.lower()
            if self.force:                   # сеанс «я кажу лише X»: Whisper тільки для тексту в маніфесті
                key = self.force
                set_ = "negative" if key == "other" else "positive"
                label = {"hektor": "Гектор", "hektore": "Гекторе"}.get(key, text[:40])
                f = save_clip(x, set_, key, label, meta, text[:120])
                _log.insert(0, f"{dt.datetime.now():%H:%M:%S}  «{text[:40]}» → {set_} {key} (вручну)  {len(x)/16000:.1f} с  {f[-20:]}")
                return
            if not re.search(r"[а-яіїєґa-z]", tl) or nsp > 0.5 or HALLUC.search(tl):
                _log.insert(0, f"{dt.datetime.now():%H:%M:%S}  шум, пропущено ({len(x)/16000:.1f} с, «{text[:20]}», no_speech {nsp:.2f})")
                return
            if WAKE_PAT.search(tl):
                key = "hektore" if re.search(r"(торе|торі)\b", tl) else "hektor"
                set_, label = "positive", ("Гекторе" if key == "hektore" else "Гектор")
            else:
                key, set_, label = "other", "negative", text[:40]
            f = save_clip(x, set_, key, label, meta, text[:120])
            _log.insert(0, f"{dt.datetime.now():%H:%M:%S}  «{text[:40]}» → {set_} {key}  {len(x)/16000:.1f} с  {f[-20:]}")
        except Exception as e:
            _log.insert(0, f"{dt.datetime.now():%H:%M:%S}  помилка: {str(e)[:80]}")
        finally:
            self.pending -= 1
            del _log[40:]


async def _events(request):
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
    await resp.prepare(request)
    try:
        while True:
            s = _scorer
            data = {"connected": bool(s and s.connected), "frames": s.frames if s else 0, "level_db": s.level_db if s else -99,
                    "scores": s.scores if s else {}, "t": time.time(), "thr": s.thr if s else {},
                    "verdicts": s.verdicts if s else [], "hits": s.hits if s else {}, "utterances": s.utterances if s else 0, "auto": bool(s and s.auto), "pending": s.pending if s else 0, "log": _log[:15]}
            await resp.write(f"data: {json.dumps(data)}\n\n".encode())
            await asyncio.sleep(0.1)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


async def _mark(request):
    """Зберегти останні N с з вух як кліп набору: {"set": "positive"|"negative", "phrase": "hektor", "text": "Гектор",
    "seconds": 2.5, "speaker": "martyn", "distance_m": 2, "position": "front"}. Запис у manifest.json одразу."""
    import datetime as dt
    import wave
    d = await request.json()
    s = _scorer
    if not s or not s.connected:
        return web.json_response({"ok": False, "error": "плата не підключена"}, status=503)
    n = int(float(d.get("seconds", 2.5)) * 16000 * 2)
    x = np.frombuffer(s.ring[-n:], dtype=np.int16)
    x = trim(x, 16000)
    if len(x) < 16000 * 0.25:
        return web.json_response({"ok": False, "error": "у буфері тиша"})
    set_ = d.get("set", "positive"); key = d.get("phrase", "hektor")
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"live-{key}-{d.get('speaker', 'martyn')}-{d.get('distance_m', 2)}m-{d.get('position', 'front')}-{stamp}"
    out = WAKE / "robot" / f"{name}.wav"
    out.parent.mkdir(exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.tobytes())
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}}
    m["files"][f"robot/{name}.wav"] = {
        "set": set_, "phrase": key, "text": d.get("text", "Гектор"), "source": "robot", "mic": "spine-ears", "via": "human",
        "speaker": d.get("speaker", "martyn"), "distance_m": float(d.get("distance_m", 2)), "position": d.get("position", "front"),
        "run": "live-page", "emotion": "neutral", "speed": 1.0, "sample_rate": 16000,
        "duration_s": round(len(x) / 16000, 2), "peak_db": round(float(20 * np.log10(np.abs(x).max() / 32768 + 1e-9)), 1),
        "scores": {k: round(v, 2) for k, v in s.scores.items()}, "created": dt.date.today().isoformat(),
    }
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    logger.info(f"live: збережено {name}.wav {len(x)/16000:.2f} с")
    return web.json_response({"ok": True, "file": f"{name}.wav", "duration_s": round(len(x) / 16000, 2)})


async def _auto(request):
    d = await request.json()
    if _scorer:
        _scorer.auto = bool(d.get("auto", True))
        _scorer.force = d.get("force", "") if d.get("force", "") in ("hektor", "hektore", "other") else ""
        _scorer.meta.update({k: d[k] for k in ("speaker", "distance_m", "position") if k in d})
    return web.json_response({"ok": True})


async def _index(request):
    return web.FileResponse(HERE / "live.html")


async def _http():
    global _http_started
    if _http_started:
        return
    app = web.Application()
    app.router.add_get("/", _index)
    app.router.add_get("/events", _events)
    app.router.add_post("/mark", _mark)
    app.router.add_post("/auto", _auto)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PORT).start()
    _http_started = True
    logger.info(f"Живий тест: http://127.0.0.1:{PORT}")


async def bot(runner_args: RunnerArguments):
    global _scorer
    transport = await make_transport(runner_args)
    scorer = Scorer(MODELS, THRESHOLD)
    _scorer = scorer
    logger.info(f"моделі: {scorer.names}, поріг {scorer.threshold}")
    pipeline = Pipeline([transport.input(), scorer])
    worker = PipelineWorker(pipeline, params=PipelineParams(), idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
                            cancel_on_idle_timeout=False)
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        scorer.connected = True
        logger.info("Плата підключилась")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        scorer.connected = False
        logger.info("Плата відключилась")
        await runner.cancel()

    await _http()
    try:
        await runner.run()
    finally:
        logger.info(scorer.summary())


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
