# hektor-micro-wakeword-esp32

**[English →](README.md)**

Як з нуля зроблено wake-слово **«Гектор»** для шестиногого робота: синтетичні й живі дані, генерація ознак, три прогони тренування [microWakeWord](https://github.com/OHF-Voice/micro-wake-word) на Colab T4 і замір «як на платі» на голосі автора через мікрофон робота. У цьому репозиторії — **код, конфіги й результати**. Сама модель опублікована на Hugging Face:

**→ [MartynU/hektor-micro-wakeword-esp32](https://huggingface.co/MartynU/hektor-micro-wakeword-esp32)** (реліз `v1.0`, int8 TFLite для ESP32 / ESPHome `micro_wake_word`, 87 КБ)

Аудіо, ознак і ваг тут нема (див. [Що не публікується](#що-не-публікується)).

## Результат

Три прогони тренування, усі переоцінені на одному й тому самому відкладеному тестовому наборі. «Live» — автор каже «Гектор» роботу, звук іде через мікрофон INMP441 робота і Wi-Fi, тобто справжній шлях у роботі. Ковзне вікно 5 (150 мс).

| Прогін | Поріг | Recall live (n=22) | Хибних/год, застільна розмова (5.3 год) | Хибних/год, кімната робота (14 хв) | Хибних на негативах (n=2146) |
|---|---|---|---|---|---|
| v1 базовий | 0.6 | 59.1 % (13/22) | 0.0 | 0.0 | 8.95 % |
| v2 підняти live recall | 0.5 | 72.7 % (16/22) | 42.6 | 4.3 | 38.82 % |
| **v3 → реліз v1.0** | **0.8** | **95.5 % (21/22)** | **1.5** | **0.0** | **4.15 %** |

Джерело: [`eval/results/v3/compare_v1_v2_v3.md`](eval/results/v3/compare_v1_v2_v3.md); усі групи, пороги й вікна — в [`eval/results/v3/eval.json`](eval/results/v3/eval.json).

До microWakeWord проєкт пройшов дев'ять версій [openWakeWord](https://github.com/dscripka/openWakeWord). Синтетичні голоси їм давались, живий голос автора через робота — ні: найкраща з них (v4d) ловила **41 %** тих самих 22 живих кліпів на низькому порозі 0.2 (див. [model card](https://huggingface.co/MartynU/hektor-micro-wakeword-esp32)). Скриптів openWakeWord тут нема; дослідження, після якого стався перехід, — [`train/research-2026-10-06.uk.md`](train/research-2026-10-06.uk.md).

Компроміс порога для v3 (вікно 5):

| Поріг | Recall live | Хибних/год, розмова |
|---|---|---|
| 0.7 | 95.5 % | 3.7 |
| **0.8** | **95.5 %** | **1.5** |
| 0.9 | 90.9 % | 0.4 |
| 0.95 | 86.4 % | 0.0 |

Модель: MixedNet, потокова, int8. 87 376 байтів у flash, мінімальна tensor arena 30.6 КБ (у manifest 33.8 КБ), ≈39 тис. MAC на крок 30 мс, 87 мкс на інференцію на x86-хості. Навантаження на ESP32-S3 оцінене з кількості MAC, на платі ще не заміряне.

## Як використати модель

Той самий файл моделі працює у двох місцях.

### 1. У голосовому боті на Pipecat (перевірено)

Так робот працює зараз: ESP32-S3 шле мікрофон по WebRTC, а модель крутиться на комп'ютері перед STT.
Усі цифри вище виміряні саме так. [`examples/pipecat/wake_gate.py`](examples/pipecat/wake_gate.py) — це
`FrameProcessor`, який пропускає звук далі лише після «Гектор», тримає 300 мс до слова, щоб не обрізати
команду, і закривається після 20 с тиші.

```python
from wake_gate import WakeGate, load_hektor   # pip install pipecat-ai pymicro-wakeword huggingface_hub

pipeline = Pipeline([
    transport.input(),             # 16 кГц моно: PipelineParams(audio_in_sample_rate=16000)
    WakeGate(load_hektor()),       # завантажує модель з Hugging Face, тег v1.0
    stt,
    context_aggregator.user(),
    llm,
    tts,
    transport.output(),
    context_aggregator.assistant(),
])
```

### 2. На самому ESP32 (ще не перевірено)

Модель — int8 потокова мережа TFLite Micro, зроблена під чип, а її `manifest.json` у форматі ESPHome
`micro_wake_word`:

```yaml
micro_wake_word:
  models:
    - model: https://huggingface.co/MartynU/hektor-micro-wakeword-esp32/resolve/main/manifest.json
```

На чипі автор її ще не запускав: пам'ять — з моделі, обчислення оцінені за кількістю MAC.
Приклад для ESP-IDF із заміром часу на ESP32-S3 — у v1.1.

## Пайплайн

```
data/      кліпи: TTS (Piper, Gemini, Cartesia, ElevenLabs), клони голосу (FireRedTTS3),
           темп-аугментація, живі записи через робота, маніфест провенансу
   │
train/     prepare.py  поділ по голосах, не по файлах  →  dataset.zip
           prep_*.py   перевірка формату, прибрати DC, 5 мс fade, теки train/val/test
           gen_*.py    фонові корпуси (MIT RIR, AudioSet, FMA) + аугментовані спектрограми
           train_*.py  training_parameters.yaml → microwakeword.model_train_eval → int8 .tflite
   │
eval/      evaluate.py / eval_v3.py: потокова оцінка як на платі, хибні/год на DiPCo і в кімнаті
   │
record/    Pipecat-боти, що слухають вуха робота: перезапис кліпів, запис живого
           «Гектор», живий тест кількох моделей одночасно
```

### Дані (`data/`)

Кожен кліп має запис у `manifest.json`: джерело, модель, голос, фраза, текст, тривалість, дата, а для записів через робота ще відстань, позиція і перевірка Whisper'ом. [`data/manifest.example.json`](data/manifest.example.json) — 16 із 69 380 записів як приклад формату.

- `gen_piper.py` — масова українська синтетика Piper (7 українських дикторів + кілька російських заради твердого [ґ]), випадковий темп і просодія на кожен кліп; також негативи: майже-омофони й побутові фрази.
- `gen_gemini.py`, `import_cartesia.py` — позитиви хмарних TTS з емоціями; `gen_negative_cartesia.py` — негативи тими самими голосами.
- `gen_firered.py` — zero-shot клони голосу автора (FireRedTTS3) з референсів, нарізаних `make_refs.py`.
- `aug_tempo.py` — прискорення 0.6×, 0.72×, 0.85×, бо живий «Гектор» триває 0.5–0.7 с, а TTS 0.8–1.2 с.
- `cut_speech.py` — довгий запис мови автора, порізаний на негативи по словах за мітками Whisper.
- `whisper_check.py` — кожен позитив через мікрофон робота перевіряється Whisper'ом; де слова не чути, у тренування не йде.
- `merge_inbox.py` — влити партії від інших генераторів у маніфест.

### Тренування (`train/`)

Ноутбук microWakeWord, переписаний у скрипти; обходи Colab описані в [`train/TRAINING.uk.md`](train/TRAINING.uk.md). Прогін v3, що став релізом:

- позитиви: ≈10.3 тис. тренувальних кліпів «Гектор», «Гекторе» і «Гектор, <команда>», із них 70 живих; живі — близько 10 % позитивних вибірок у батчі з потроєним штрафом;
- негативи: ≈10.5 тис. власних кліпів плюс готові спектрограми `speech`, `dinner_party`, `no_speech` з [`kahrendt/microwakeword`](https://huggingface.co/datasets/kahrendt/microwakeword);
- аугментація: фон з SNR −5…20 дБ, імпульсні відгуки кімнат, шумова підлога −70…−50 dBFS замість цифрової тиші, SpecAugment 2×5 по часу й частоті;
- 10 000 кроків, batch 128, lr 0.001, ваги класів 1.5 / 1.0; чекпойнт обрано за ≤ 0.5 хибних/год на фоні кімнати + 1 год застільної розмови, далі за максимальним recall; 38 хвилин на T4.

Конфіги кожного прогону як вони виконувались: [`train/configs/v1..v3/`](train/configs/) (`training_parameters.yaml`, розгорнутий конфіг, схема моделі, підрахунки даних).

### Оцінка (`eval/`)

`evaluate.py` повторює те, що робить плата: 1 с шуму −60 dBFS + кліп + 0.5 с → micro-frontend (40 мел-бінів, вікно 30 мс, крок 10 мс) → потокова int8-модель, 3 кадри на виклик → ковзне середнє останніх N виходів → спрацювання, якщо середнє перевищує поріг. Хибні/год — потоково на 5.3 год застільних розмов DiPCo (одне спрацювання на передній фронт, 1 с рефрактерності) і на 14 хв кімнати робота, записаних через його ж вуха. Результати всіх прогонів — в [`eval/results/`](eval/results/).

### Запис через робота (`record/`)

`record.py` — Pipecat-бот, який лише слухає вуха робота і зберігає звук між HTTP-мітками start/stop. `rerecord.py` грає кліпи набору в динамік і підписує те, що почув робот; той самий клієнт пише живий «Гектор» від автора. `live.py` рахує кілька моделей одночасно на живому потоці й друкує вердикт на кожну репліку. Транспорт, прошивка й решта робота — в основному проєкті (приватний).

## Чого навчили три прогони

1. **Синтетики замало.** v1 добре ловив TTS і 59 % живих кліпів. Різниця — тракт мікрофона, відстань і справжня швидка мова.
2. **Прибрати негативи й валідуватись на тиші — пастка.** v2 викинув Piper-негативи і вибирав чекпойнт за хибними на тихій кімнаті. Live recall виріс, хибні на розмові — з 0 до 42 за годину.
3. **Спрацювало: більше живих даних плюс збалансована валідація.** v3 додав 93 живих записи (70 train, 23 test), повернув негативи, додав 347 важких негативів і валідувався одночасно на шумі кімнати *і* на застільній розмові.

## Обмеження

- Залежність від диктора: живий тест — один голос, 22 кліпи. На невідомих синтетичних голосах recall 62 % на порозі релізу, на хмарних TTS 34 %.
- «Гектор» дається важче за «Гекторе» (55 % проти 78 % на відкладених синтетичних голосах при 0.8).
- Цифра по кімнаті — з 14 хвилин аудіо.
- Час на платі оцінений з MAC; зараз робот ганяє модель на хості через `pymicro-wakeword` на звуку, що стрімиться з ESP32-S3.

## Відтворення

Скрипти — запис того, як робилася модель, а не тренажер «під ключ»: сире аудіо приватне, тож потрібні власні кліпи в тій самій розкладці. Один прогін, `HK_WORK` — робоча тека (типово `./work`):

```bash
pip install -r train/requirements.txt    # плюс microWakeWord @4665173, див. файл
export HK_WORK=$PWD/work HF_TOKEN=...    # HF_TOKEN лише якщо заливаєте артефакти у свій репо на Hub
python train/prepare.py --zip            # manifest.json → dataset.zip, поділ по голосах
#   покласти dataset.zip + manifest.json у $HK_WORK/raw/micro-wake-hector-v3/
python train/prep_v3.py                  # чистка кліпів → $HK_WORK/data/...
python train/gen_v3.py                   # корпуси + аугментовані спектрограми → $HK_WORK/features/
python train/dp1h.py                     # 1 год DiPCo для валідації (обхід RAM у Colab)
python train/train_v3.py --steps 10000   # → $HK_WORK/trained/v3/...stream_state_internal_quant.tflite
python eval/eval_v3.py --tflite $HK_WORK/trained/v3/tflite_stream_state_internal_quant/stream_state_internal_quant.tflite --out $HK_WORK/trained/v3/eval.json
python train/publish_v3.py               # manifest.json, таблиці, завантаження
```

Кожен скрипт з `argparse` відповідає на `--help`. У `data/` і `record/` свої залежності (`data/requirements.txt`, `record/pyproject.toml`).

## Що не публікується

- Аудіо: TTS-кліпи, перезаписи й живі записи автора, фон кімнати. Негативні спектрограми `kahrendt/microwakeword` — CC BY-NC 4.0.
- Ознаки (`.npy`, mmap), чекпойнти, ваги ONNX і TFLite. Випущена модель лежить на Hugging Face під CC BY-NC 4.0 саме через ті негативи.
- Ключі API (Gemini, Cartesia, ElevenLabs, Hugging Face): скрипти читають їх зі змінних середовища.

## Подяки

[microWakeWord](https://github.com/OHF-Voice/micro-wake-word) Кевіна Арендта і його [негативні набори](https://huggingface.co/datasets/kahrendt/microwakeword); [microwakeword-trainer](https://github.com/interkelstar/microwakeword-trainer) за ідеї аугментації (діапазон SNR, шумова підлога, SpecAugment, ваги класів); [pymicro-features](https://github.com/rhasspy/pymicro-features) і [pymicro-wakeword](https://github.com/OHF-Voice/pymicro-wakeword); [Piper](https://github.com/rhasspy/piper); [FireRedTTS3](https://github.com/FireRedTeam/FireRedTTS); [openWakeWord](https://github.com/dscripka/openWakeWord); корпуси MIT RIR, AudioSet, FMA і DiPCo; [Pipecat](https://github.com/pipecat-ai/pipecat) за транспорт до робота; Whisper через `mlx-whisper`.

Код писав Мартин разом із Claude (Anthropic) як парним програмістом: цілі, рішення щодо даних, прослуховування і робот — авторські; значну частину коду накидано з Claude, а далі запущено, зневаджено й перезапущено автором. Коментарі в коді українські, лишені як були.

## Ліцензія

Код: [MIT](LICENSE). Модель і дані мають свої умови (див. вище).
