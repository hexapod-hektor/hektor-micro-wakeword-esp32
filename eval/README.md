# eval/ — device-like evaluation and results

`evaluate.py` (v1, v2) and `eval_v3.py` (v3; adds the live-cmd group, the «Гектор» / «Гекторе» / command kinds, the near-homophone «гектар» subset and the room-ambient FA/h) score the quantised streaming model the way the chip runs it:

1. each test clip is padded with 1 s of −60 dBFS noise before and 0.5 s after;
2. micro-frontend (`pymicro_features`: 40 mel bins, 30 ms window, 10 ms step);
3. streaming int8 model, 3 new frames per call, fresh state per clip;
4. sliding average of the last N outputs (zero-initialised ring buffer, as in ESPHome), windows 5 and 10;
5. detected if the maximum of the average reaches the cutoff; cutoffs 0.5 … 0.97.

False accepts per hour are measured by streaming DiPCo dinner-party conversation (5.335 h, `kahrendt/microwakeword` `dinner_party_eval/testing_ambient`) and the robot's room (0.235 h, five 3-minute files recorded through its own ears, checked with Whisper to contain no wake word). One trigger per rising edge, 1 s refractory. `tech_info()` also counts MACs per inference, probes the minimum tflite-micro arena and times the model on the host.

`results/`:

| File | Content |
|---|---|
| `v1/`, `v2/`, `v3/eval.json` | recall per group × window × cutoff, FA rate on negative clips, FA/h, recommended cutoff, model tech info |
| `v*/table_w5.md`, `table_w10.md` | the same as Markdown tables |
| `v3/compare_v1_v2_v3.md` | all three models on the v3 test set, window 5, at their manifest cutoffs |
| `v3/eval_v1_on_v3.log`, `eval_v2_on_v3.log`, `eval_v3.log` | console output of those runs |
| `v1/eval_room.json` | v1 re-scored with the room ambient as the FA/h set |
| `RESULTS.uk.md` | the full Ukrainian results write-up (per-run tables, cutoff reasoning, caveats) |

Note that v1 and v2 `eval.json` were produced on the v1/v2 test set (9 live clips); the comparison table re-scores them on the v3 test set (22 live clips), which is where the 59 % / 73 % / 95.5 % figures come from.
