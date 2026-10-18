"""Shared helpers for the Hektor microWakeWord branch (repo path mww/)."""
import os, re, subprocess, threading, time

# Робоча тека (raw/, data/, features/, trained/ ...). У Colab була /content/hk; тут — $HK_WORK або ./work
W = os.environ.get("HK_WORK", os.path.abspath("work"))
DATA_REPO = os.environ.get("HK_DATA_REPO", "MartynU/hektor-wakeword-data")
MODEL_REPO = os.environ.get("HK_MODEL_REPO", "MartynU/hektor-wakeword-lab")  # приватний архів (до 14.10.2026 — hektor-wakeword)
HF_PREFIX = "mww"
TTS_KEYS = ("cartesia", "elevenlabs", "gemini")


def _s(x):
    return str(x or "").lower()


def pos_group(name, meta):
    """Report group of a positive clip; the most specific channel wins."""
    n = name.lower()
    src, voice = _s(meta.get("source")), _s(meta.get("voice"))
    if n.startswith("live-"):
        return "live"
    if "-robot-" in n or src == "robot":
        return "robot"
    if re.search(r"-t0\.\d+", n) or src == "aug-tempo" or meta.get("tempo") not in (None, ""):
        return "aug-tempo"
    if "firered" in n or src == "firered":
        return "firered"
    if src == "piper" or "-piper-" in n:
        return "piper"
    if src in TTS_KEYS or any(k in n for k in TTS_KEYS):
        return "tts"
    return "other"


def neg_group(name, meta):
    n = name.lower()
    src = _s(meta.get("source"))
    if "-robot-" in n or src == "robot":
        return "robot"
    if n.startswith("neg-speech-martyn") or src.startswith("human"):
        return "live-speech"
    if src == "piper" or "-piper-" in n:
        return "piper"
    if src in TTS_KEYS or any(k in n for k in TTS_KEYS):
        return "tts"
    return "other"


def hf_api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ["HF_TOKEN"])


def hf_upload(local, remote, msg=None):
    """Upload file/folder to MODEL_REPO under mww/<remote>. Never raises."""
    api = hf_api()
    pir = f"{HF_PREFIX}/{remote}".rstrip("/")
    for attempt in range(3):
        try:
            if os.path.isdir(local):
                api.upload_folder(folder_path=local, path_in_repo=pir, repo_id=MODEL_REPO,
                                  commit_message=msg or f"mww: {remote}")
            else:
                api.upload_file(path_or_fileobj=local, path_in_repo=pir, repo_id=MODEL_REPO,
                                commit_message=msg or f"mww: {remote}")
            print(f"[hf] uploaded {local} -> {MODEL_REPO}/{pir}", flush=True)
            return True
        except Exception as e:
            print(f"[hf] upload failed ({attempt+1}/3): {e}", flush=True)
            time.sleep(10)
    return False


def tar_and_upload(src_dir, remote_name):
    tar = f"{W}/_up/{remote_name}"
    os.makedirs(os.path.dirname(tar), exist_ok=True)
    parent, base = os.path.split(src_dir.rstrip("/"))
    subprocess.run(["tar", "-cf", tar, "-C", parent, base], check=True)
    ok = hf_upload(tar, f"work/{remote_name}")
    os.remove(tar)
    return ok


def hf_restore_tar(remote_name, dest_parent):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(MODEL_REPO, f"{HF_PREFIX}/work/{remote_name}", local_dir=f"{W}/_down",
                            token=os.environ["HF_TOKEN"])
    except Exception as e:
        print(f"[hf] no {remote_name} on HF ({type(e).__name__})")
        return False
    os.makedirs(dest_parent, exist_ok=True)
    subprocess.run(["tar", "-xf", p, "-C", dest_parent], check=True)
    os.remove(p)
    print(f"[hf] restored {remote_name}")
    return True


class PeriodicSync(threading.Thread):
    """Uploads the training dir (checkpoints + logs) every period seconds."""

    def __init__(self, train_dir, remote, period=600):
        super().__init__(daemon=True)
        self.train_dir, self.remote, self.period = train_dir, remote, period
        self.stop = threading.Event()

    def run(self):
        while not self.stop.wait(self.period):
            self.push()

    def push(self):
        if os.path.isdir(self.train_dir):
            hf_upload(self.train_dir, self.remote, "mww: checkpoint sync")
