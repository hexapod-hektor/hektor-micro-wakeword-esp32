# train/ — from manifest to int8 TFLite

The microWakeWord `basic_training_notebook` rewritten as scripts, run on Google Colab (T4). v3 is the release; v1 and v2 are kept as history. Ukrainian notes with every workaround (datasets 4.x / torchcodec, AudioSet parquet, OOM on validation, uint16 features, Colab background processes): [`TRAINING.uk.md`](TRAINING.uk.md). The research note that preceded the move from openWakeWord to microWakeWord: [`research-2026-10-06.uk.md`](research-2026-10-06.uk.md).

All scripts use `HK_WORK` (default `./work`) as the work directory. `hk_common.py` also reads `HK_MODEL_REPO` / `HK_DATA_REPO` and `HF_TOKEN` for pushing artifacts to Hugging Face; without a token the upload helpers just report failure and continue.

| Step | v1 / v2 | v3 | Notes |
|---|---|---|---|
| 0. Split the dataset by voice | `prepare.py` | `prepare.py` | Test voices never seen in training; live clips alternate train/test; writes `split.json` and `dataset.zip` |
| 1. Clean and lay out clips | `prep_data.py` | `prep_v3.py` | Format check, DC removal, 5 ms fades; v3 replays the v1 negative split so cached features can be reused |
| 2. Corpora + augmented spectrograms | `gen_features.py` | `gen_v3.py` | MIT RIR, AudioSet, FMA, `kahrendt/microwakeword` negatives; SNR −5…20 dB, p(bg) 0.8, p(RIR) 0.7, noise floor −70…−50 dBFS; uint16 RaggedMmap |
| 2b. 1 h DiPCo validation subset | — | `dp1h.py` | Full 9.67 h set OOM-killed the 12 GB VM |
| 3. Train | `train_run.py` | `train_v3.py` | Writes `training_parameters.yaml`, runs `microwakeword.model_train_eval`, syncs checkpoints every 10 min |
| 4. Evaluate | [`../eval/evaluate.py`](../eval/evaluate.py) | [`../eval/eval_v3.py`](../eval/eval_v3.py) | Device-like streaming evaluation |
| 5. Package | `publish.py`, `make_readme.py` | `publish_v3.py` | `model.tflite` + ESPHome `manifest.json` (cutoff, window, arena +10 %), tables, comparison |

`configs/v1..v3/` hold the exact `training_parameters.yaml` and the resolved config of each run (work-dir paths replaced by `${HK_WORK}`), the Keras model summary and the clip counts from the prep step.

Architecture (all runs): MixedNet, pointwise 64×4, kernels [5], [7, 11], [9, 15], [23], first conv 32×5, stride 3, residual connections on; receptive field 204 frames = 2.04 s; input int8 `[1, 3, 40]`, output uint8 `[1, 1]`.
