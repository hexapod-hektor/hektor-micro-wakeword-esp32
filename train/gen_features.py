"""Step 2: background/RIR corpora + augmented spectrogram features (RaggedMmap).

Augmentation = basic_training_notebook defaults with interkelstar's channel changes:
  background SNR -5..20 dB (nb -5..10), p(background) 0.8 (nb 0.75), p(RIR) 0.7 (nb 0.5);
  padding never digital zero: the notebook left-pads with exact zeros, which a real mic never
  delivers (interkelstar: real quiet is 1-10 in feature units, never 0; their ru_stop v2 fired in
  echo-cancelled near-silence). We add a -70..-50 dBFS noise floor to the whole fixed-size clip.
Features are stored as uint16 (value/0.0390625); mWW's loader rescales uint16 itself. Positive
windows are cropped to the newest --keep frames (truncate_start never uses older ones).
Clips are read with scipy (WavClips) instead of mWW's Clips: Clips decodes via datasets.Audio,
which since datasets 4.x requires torchcodec.
"""
import os, sys, glob, argparse, subprocess, shutil, io
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W, tar_and_upload

D, BG, F = f"{W}/data", f"{W}/bg", f"{W}/features"


def sh(cmd):
    print("+", cmd[:300], flush=True)
    subprocess.run(cmd, shell=True, check=True)


def ffmpeg16k(files, out_dir):
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(out_dir, exist_ok=True)

    def one(p):
        o = os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + ".wav")
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", p, "-ac", "1", "-ar", "16000",
                        "-sample_fmt", "s16", o], check=False)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(one, files))
    print(out_dir, len(os.listdir(out_dir)), "wav", flush=True)


def parquet16k(files, out_dir):
    import pyarrow.parquet as pq, soundfile as sf
    from scipy.signal import resample_poly
    from scipy.io import wavfile
    os.makedirs(out_dir, exist_ok=True)
    k = 0
    for p in files:
        for row in pq.read_table(p).to_pylist():
            au = row.get("audio") or {}
            a, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32")
            if a.ndim > 1:
                a = a[:, 0]
            if sr != 16000:
                a = resample_poly(a, 16000, sr)
            name = os.path.splitext(os.path.basename(au.get("path") or f"rir{k}"))[0]
            wavfile.write(f"{out_dir}/{name}.wav", 16000, np.clip(a * 32767, -32768, 32767).astype(np.int16))
            k += 1
    print(out_dir, k, "wav", flush=True)


def download_corpora():
    os.makedirs(BG, exist_ok=True)
    from huggingface_hub import snapshot_download
    if not os.path.isdir(f"{BG}/mit_rirs"):
        p = snapshot_download("davidscripka/MIT_environmental_impulse_responses", repo_type="dataset", local_dir=f"{BG}/_mit")
        aud = [f for f in glob.glob(f"{p}/**/*", recursive=True) if f.lower().endswith((".wav", ".flac"))]
        if aud:
            ffmpeg16k(aud, f"{BG}/mit_rirs")
        else:
            parquet16k(sorted(glob.glob(f"{p}/**/*.parquet", recursive=True)), f"{BG}/mit_rirs")
        shutil.rmtree(f"{BG}/_mit", ignore_errors=True)
    if not os.path.isdir(f"{BG}/audioset_16k"):
        # agkphysics/AudioSet moved from data/bal_train09.tar to parquet shards; 2 shards ~ the old tar
        from huggingface_hub import hf_hub_download
        shards = [hf_hub_download("agkphysics/AudioSet", f"data/bal_train/{i:02d}.parquet", repo_type="dataset",
                                  local_dir=f"{BG}/_as") for i in (8, 9)]
        parquet16k(shards, f"{BG}/audioset_16k")
        shutil.rmtree(f"{BG}/_as", ignore_errors=True)
    if not os.path.isdir(f"{BG}/fma_16k"):
        sh(f"mkdir -p {BG}/fma && cd {BG}/fma && wget -q -O fma_xs.zip https://huggingface.co/datasets/mchl914/fma_xsmall/resolve/main/fma_xs.zip"
           " && unzip -q fma_xs.zip && rm fma_xs.zip")
        ffmpeg16k(glob.glob(f"{BG}/fma/**/*.mp3", recursive=True), f"{BG}/fma_16k")
        shutil.rmtree(f"{BG}/fma", ignore_errors=True)
    neg = f"{W}/negative_datasets"
    os.makedirs(neg, exist_ok=True)
    for name in ["dinner_party", "dinner_party_eval", "no_speech", "speech"]:
        if not os.path.isdir(f"{neg}/{name}"):
            sh(f"wget -q -O {neg}/{name}.zip https://huggingface.co/datasets/kahrendt/microwakeword/resolve/main/{name}.zip"
               f" && unzip -q {neg}/{name}.zip -d {neg} && rm {neg}/{name}.zip")
    from scipy.io import wavfile
    for d in ["audioset_16k", "fma_16k", "mit_rirs"]:
        bad = 0
        for p in glob.glob(f"{BG}/{d}/*.wav"):
            try:
                sr, a = wavfile.read(p)
                assert sr == 16000 and len(a) >= 1600 and np.abs(a).max() >= 10
            except Exception:
                os.remove(p); bad += 1
        print(d, len(glob.glob(f"{BG}/{d}/*.wav")), "ok, removed", bad, flush=True)


class WavClips:
    """Drop-in for microwakeword Clips (no datasets/torchcodec); yields float32 in [-1, 1]."""

    def __init__(self, input_directory):
        self.paths = sorted(glob.glob(f"{input_directory}/*.wav"))

    def _load(self, p):
        from scipy.io import wavfile
        sr, a = wavfile.read(p)
        assert sr == 16000, p
        return a.astype(np.float32) / 32768.0

    def audio_generator(self, split=None, repeat=1):
        for _ in range(repeat):
            for p in self.paths:
                yield self._load(p)

    def get_random_clip(self):
        return self._load(self.paths[np.random.randint(len(self.paths))])


def make_augmenter(duration_s, jitter):
    from microwakeword.audio.augmentation import Augmentation

    class FloorAugmentation(Augmentation):
        def create_fixed_size_clip(self, input_audio):
            out = super().create_fixed_size_clip(input_audio).astype(np.float32)
            floor = 10 ** (np.random.uniform(-70, -50) / 20)
            return out + np.random.normal(0, floor, out.shape).astype(np.float32)

    return FloorAugmentation(
        augmentation_duration_s=duration_s,
        augmentation_probabilities={"SevenBandParametricEQ": 0.1, "TanhDistortion": 0.1, "PitchShift": 0.1,
                                    "BandStopFilter": 0.1, "AddColorNoise": 0.1, "AddBackgroundNoise": 0.8,
                                    "Gain": 1.0, "RIR": 0.7},
        impulse_paths=[f"{BG}/mit_rirs"], background_paths=[f"{BG}/fma_16k", f"{BG}/audioset_16k"],
        background_min_snr_db=-5, background_max_snr_db=20, min_jitter_s=jitter[0], max_jitter_s=jitter[1])


def write_set(name, wav_dir, split, repeat, slide, keep, jitter, duration_s=3.2):
    from microwakeword.audio.spectrograms import SpectrogramGeneration
    from mmap_ninja.ragged import RaggedMmap
    out = f"{F}/{name}/{split}/wakeword_mmap"
    if os.path.exists(f"{out}/.done"):
        print("skip", out); return
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sg = SpectrogramGeneration(clips=WavClips(wav_dir), augmenter=make_augmenter(duration_s, jitter),
                               slide_frames=slide, step_ms=10)
    n = [0]

    def gen():
        for spec in sg.spectrogram_generator(split=None, repeat=repeat):
            if keep:
                spec = spec[-keep:]
            n[0] += 1
            if n[0] % 10000 == 0:
                print(f"  {name}/{split}: {n[0]}", flush=True)
            yield np.clip(np.round(spec / 0.0390625), 0, 65535).astype(np.uint16)

    print(f"== {name}/{split}: {len(os.listdir(wav_dir))} clips x{repeat} slide={slide} keep={keep}", flush=True)
    RaggedMmap.from_generator(out_dir=out, sample_generator=gen(), batch_size=200, verbose=False)
    open(f"{out}/.done", "w").write(str(n[0]))
    print(f"   -> {n[0]} spectrograms", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=210, help="frames kept for positives (>= spectrogram_length 204)")
    ap.add_argument("--live-rep", type=int, default=6, help="repetitions of live clips (others: 2) -> x3")
    ap.add_argument("--live-name", default="pos_live")
    ap.add_argument("--only", default="")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--skip-corpora", action="store_true")
    a = ap.parse_args()
    if not a.skip_corpora:
        download_corpora()
    only = set(filter(None, a.only.split(",")))
    P, N = (0.195, 0.205), (0.0, 0.6)
    sets = [("pos_main", f"{D}/pos_main", "training", 2, 10, a.keep, P),
            (a.live_name, f"{D}/pos_live", "training", a.live_rep, 10, a.keep, P),
            ("pos_main", f"{D}/pos_val", "validation", 1, 10, a.keep, P),
            ("pos_main", f"{D}/test/positive_test", "testing", 1, 1, None, P),
            ("neg_piper", f"{D}/neg_piper", "training", 2, None, None, N),
            ("neg_other", f"{D}/neg_other", "training", 2, None, None, N),
            ("neg_other", f"{D}/neg_val", "validation", 1, None, None, N),
            ("neg_other", f"{D}/test/negative_test", "testing", 1, None, None, N)]
    built = []
    for name, wd, split, rep, slide, keep, jit in sets:
        if only and name not in only:
            continue
        write_set(name, wd, split, rep, slide, keep, jit)
        if name not in built:
            built.append(name)
    if not a.no_upload:
        for name in built:
            tar_and_upload(f"{F}/{name}", f"features_{name}.tar")


if __name__ == "__main__":
    main()
