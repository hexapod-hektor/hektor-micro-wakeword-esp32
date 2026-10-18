"""README-mww.md з eval.json обох варіантів."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hk_common import W
G = ["ALL", "live", "robot", "firered", "aug-tempo", "tts", "piper"]
CUT = ["0.5", "0.8", "0.95"]


def load(v):
    p = f"{W}/publish/{v}/eval.json"
    return json.load(open(p)) if os.path.exists(p) else None


def section(v, ev, man, note):
    w = "w" + str(man["micro"]["sliding_window_size"])
    pos, neg = ev["recall_positive_test"], ev["false_accept_rate_negative_test"]
    L = [f"## {v}: {note}", "",
         f"manifest: probability_cutoff **{man['micro']['probability_cutoff']}**, sliding_window_size **{man['micro']['sliding_window_size']}**, tensor_arena_size {man['micro']['tensor_arena_size']}", ""]
    for ww in ("w5", "w10"):
        L += [f"### ковзне вікно {ww[1:]}", "", "| | n | @0.5 | @0.8 | @0.95 |", "|---|---|---|---|---|"]
        for g in G:
            if g in pos:
                L.append(f"| recall {g} | {pos[g]['n']} | " + " | ".join(f"{pos[g][ww][c]*100:.1f}%" for c in CUT) + " |")
        for g, x in neg.items():
            L.append(f"| хибні на negative_test {g} | {x['n']} | " + " | ".join(f"{x[ww][c]*100:.2f}%" for c in CUT) + " |")
        if "dipco_fa_per_hour" in ev:
            L.append(f"| FA/год кімната (testing_ambient, {ev['ambient_hours']} год) | - | " + " | ".join(str(ev['ambient_fa_per_hour'][ww][c]) for c in CUT) + " |")
            L.append(f"| FA/год DiPCo ({ev['dipco_hours']} год) | - | " + " | ".join(str(ev['dipco_fa_per_hour'][ww][c]) for c in CUT) + " |")
        else:
            L.append(f"| FA/год DiPCo ({ev['ambient_hours']} год) | - | " + " | ".join(str(ev['ambient_fa_per_hour'][ww][c]) for c in CUT) + " |")
        L.append("")
    return L


def main():
    t = json.load(open(f"{W}/publish/v1/eval.json"))["tech"]
    L = ["# «Гектор» — microWakeWord (mww/)", "",
         "Паралельна гілка до openWakeWord. Потокова MixedNet (kahrendt/microWakeWord @4665173), int8, для ESPHome micro_wake_word / TFLM на ESP32-S3.",
         "Як тренувалось і всі обходи — TRAINING-mww.md. Файли: mww/v1/, mww/v2/ (model.tflite, manifest.json, training_parameters.yaml, eval.json, logs/); mww/model.tflite + mww/manifest.json — рекомендований варіант.", "",
         "## Параметри для інференсу на платі",
         "- Ознаки: micro-frontend (TFLM micro_speech / pymicro_features): 16 кГц, вікно 30 мс, крок **10 мс**, **40 мел-бінів** (125–7500 Гц), PCAN + noise reduction, log; значення ~0…26 (float), на вхід моделі int8: q = round(x / " + f"{t['input']['scale']:.6f}) + ({t['input']['zero_point']}).",
         f"- Вхід: int8 {t['input']['shape']} — 3 нові кадри (stride 3) → одна інференція кожні **30 мс**; стан згорток — у ресурсних змінних моделі (streaming internal state).",
         f"- Вихід: uint8 {t['output']['shape']}, p = q × {t['output']['scale']}; спрацювання — ковзне середнє останніх N=sliding_window_size виходів (N×30 мс) ≥ probability_cutoff.",
         "- Рецептивне поле моделі: 204 кадри = **2.04 с** (1.5 с clip_duration + 154 кадри valid-padding); модель спрацьовує ≈0.2 с після кінця слова.",
         f"- Обчислення: ≈{t['macs_per_inference']/1000:.0f} тис. MAC на інференцію ≈ {t['macs_per_inference']/0.03/1e6:.2f} MMAC/с; на ESP32-S3 (240 МГц, ESP-NN) це очікувано ~1–3 мс на інференцію, тобто ~3–10 % одного ядра (оцінка з MAC, не замір на платі; на x86 хості — {t['host_cpu_us_per_inference']} мкс).",
         f"- RAM: tensor arena, мінімум за tflite-micro **{t['tflm_arena_min_bytes']} Б** (у manifest +10 %); модель у flash ~85 КБ; плюс micro-frontend ~10–15 КБ і аудіобуфер.",
         ""]
    for v, note in (("v1", "live ×3, усі негативи, вибір ваг за average_viable_recall на DiPCo/chime6 1 год, поріг за DiPCo FA/год ≤ 0.5"),
                    ("v2", "live ×5, без Piper-негативів, validation/testing_ambient = кімната через вуха робота, ваги й поріг за FA/год на кімнаті, 10 000 кроків")):
        ev = load(v)
        if ev:
            L += section(v, ev, json.load(open(f"{W}/publish/{v}/manifest.json")), note)
    L += open(f"{W}/scripts/readme_tail.md").read().splitlines() if os.path.exists(f"{W}/scripts/readme_tail.md") else []
    open(f"{W}/publish/README-mww.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
