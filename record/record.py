"""Бот-записувач: лише слухає вуха робота і зберігає звук за командою. Без STT, LLM, TTS.

Той самий порт 7860, що й bot.py (плата стукає на нього), тому основного бота на цей час зупинити:
    cd record && uv run record.py -t webrtc --esp32 --host 192.168.0.25

Керування з Mac по HTTP 127.0.0.1:8091 (те саме, що control.py, але інші маршрути):
    POST /rec/start  {"name": "hektor-…-robot-1", "set": "positive", "meta": {...}}
    POST /rec/stop   → {"ok": true, "file": "robot/….wav", "duration_s": 0.9}
    GET  /rec/status → чи є звук, чи йде запис
Клієнт — rerecord.py поруч: грає кліпи в динамік Mac і ставить ці мітки.
"""
from aiohttp import web
from dotenv import load_dotenv
from loguru import logger
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.runner.types import RunnerArguments
from pipecat.workers.runner import WorkerRunner

from recorder import Recorder
from transport import make_transport

load_dotenv(".env.local", override=True)

PORT = 8091
_rec: Recorder | None = None
_http_started = False


async def _start(request):
    if _rec is None:
        return web.json_response({"ok": False, "error": "плата не підключена"}, status=503)
    d = await request.json()
    _rec.start(d["name"], d.get("set", "positive"), d.get("meta", {}))
    return web.json_response({"ok": True})


async def _stop(request):
    if _rec is None:
        return web.json_response({"ok": False, "error": "плата не підключена"}, status=503)
    r = _rec.stop()
    return web.json_response({"ok": r is not None, **(r or {})})


async def _status(request):
    return web.json_response({"connected": _rec is not None, **(_rec.status() if _rec else {})})


async def _http():
    global _http_started
    if _http_started:
        return
    app = web.Application()
    app.router.add_post("/rec/start", _start)
    app.router.add_post("/rec/stop", _stop)
    app.router.add_get("/rec/status", _status)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", PORT).start()
    _http_started = True
    logger.info(f"Записувач: http://127.0.0.1:{PORT}/rec/{{start,stop,status}}")


async def bot(runner_args: RunnerArguments):
    global _rec
    transport = await make_transport(runner_args)
    rec = Recorder()
    pipeline = Pipeline([transport.input(), rec])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        cancel_on_idle_timeout=False,   # робот мовчить годинами — це не привід рвати сесію
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        global _rec
        _rec = rec
        logger.info("Плата підключилась — записувач готовий")

    # Плата після старту шле причину перезавантаження (як у pipecat-server/control.py):
    # 3 = esp_restart (прошивка сама, напр. після «binding request timeout» у libpeer), 6/7 = сторожовий
    # таймер, 9 = просідання живлення. Так видно, хто рве з'єднання кожні кілька хвилин.
    @worker.rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        if message.type == "boot":
            d = message.data or {}
            names = {1: "увімкнення живлення", 3: "програмно (esp_restart)", 4: "збій програми",
                     5: "збій переривань", 6: "сторожовий таймер задачі", 7: "сторожовий таймер",
                     9: "ПРОСІДАННЯ ЖИВЛЕННЯ (brownout)"}
            logger.warning(f"← плата стартувала: {names.get(d.get('reason'), d.get('reason'))}, працює {d.get('uptime_s')} с")

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        global _rec
        if _rec is rec:
            _rec = None
        logger.info("Плата відключилась")
        await runner.cancel()

    await _http()
    await runner.run()


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
