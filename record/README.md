# record/ — listening through the robot's ears

Small [Pipecat](https://github.com/pipecat-ai/pipecat) bots that take the same WebRTC transport the robot uses for its voice assistant, but without STT, LLM or TTS: they only listen to the INMP441 microphone on the ESP32-S3 spine. Everything the detector will ever see passes through this path (mic → Opus → WebRTC), so the dataset's re-recordings, the live recordings and the live test all go through here.

| Script | Role |
|---|---|
| `record.py` + `recorder.py` | Recorder bot. HTTP on `127.0.0.1:8091`: `POST /rec/start`, `POST /rec/stop`, `GET /rec/status`. Saves 16 kHz WAV to `data/robot/` (silence trimmed) or `data/background/` (as is) and writes the provenance record to `data/manifest.json` |
| `rerecord.py` | Client on the laptop: plays dataset clips through the speaker and marks start/stop, so the label comes from the play order, not from recognition; `--live N` records the author saying the phrase on a countdown; `--background` records room noise |
| `live.py` + `live.html` | Live test: scores several models at once (openWakeWord `.onnx` and microWakeWord `manifest.json`) on the stream, one verdict line per utterance, bars in the browser on `127.0.0.1:8092`; can auto-save utterances with a Whisper label |
| `score.py` | Offline: run an openWakeWord model over manifest clips grouped by set × source (originals vs. robot re-recordings) |
| `transport.py` | SmallWebRTC transport params |

```bash
cd record
uv run record.py -t webrtc --esp32 --host <robot-ip>          # terminal 1, wait for "board connected"
uv run rerecord.py --positive 300 --negative 500 --distance 1.5 --position front
uv run rerecord.py --live 20 --text "Гектор" --distance 2
uv run live.py models/v3/manifest.json -t webrtc --esp32 --host <robot-ip>
```

The firmware, the main voice bot and the robot itself live in the main Hektor project (private). This folder is here to show how the live data and the live measurements were obtained.
