# data/ — clip generation and provenance

Scripts that produced the «Гектор» dataset. Clips (`positive/`, `negative/`, `robot/`, `background/`), inbox batches, voice references and the full `manifest.json` are **not** in the repo; `manifest.example.json` shows the record format (16 of 69 380 entries).

Clip format everywhere: WAV, 16 kHz, 16-bit, mono, silence trimmed at both ends. Room, noise and reverb are not baked in; the training pipeline augments.

| Script | What it does | Needs |
|---|---|---|
| `gen_piper.py` | Bulk Piper TTS: Ukrainian voices (+ a few Russian for a hard [ɡ]), random `length_scale` / `noise_scale` / `noise_w` per clip; positives and negatives | `piper-tts`, voices in `data/voices/` |
| `gen_gemini.py` | Gemini TTS positives, 30 voices × 5 models, emotion tags, one request → up to 5 clips split by silence | `GOOGLE_API_KEY` |
| `import_cartesia.py` | Cartesia playground downloads → named clips; phrase guessed by Whisper | `mlx-whisper` |
| `gen_negative_cartesia.py` | Near-homophones («вектор», «гектар», «Віктор»…), commands and everyday phrases in 15 Cartesia voices | `CARTESIA_API_KEY` |
| `gen_firered.py` | Zero-shot clones of a reference voice with FireRedTTS3, many seeds per phrase | FireRedTTS3 checkout |
| `make_refs.py` | Cut ~10 s references for cloning from a long recording, word-aligned by Whisper | `mlx-whisper`, ffmpeg |
| `aug_tempo.py` | 0.6× / 0.72× / 0.85× time-stretch of positives (WSOLA) | `audiomentations` |
| `cut_speech.py` | Long speech recording → single-word / 2–3-word negatives by Whisper word timestamps | `mlx-whisper`, ffmpeg |
| `whisper_check.py` | Mark robot-mic positives where Whisper does not hear the word (`whisper_ok: false`, excluded from training) and background files that contain it | `mlx-whisper` |
| `merge_inbox.py` | Merge a batch from another generator (`inbox/<batch>/data/wake/manifest.<x>.json`) into `manifest.json` | — |

Run everything from the repo root, e.g. `python data/gen_piper.py --per 400`. Keys go in the environment or in `data/.env` (git-ignored).

The ElevenLabs clips were generated outside this repo and merged with `merge_inbox.py`. Recordings through the robot's microphone come from [`../record/`](../record/).
