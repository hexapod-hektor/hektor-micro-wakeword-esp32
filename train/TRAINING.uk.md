# TRAINING-mww — «Гектор» на microWakeWord (гілка mww/, паралельна openWakeWord)

Артефакти openWakeWord не чіпались. Усе описане лежить у mww/ репозиторію MartynU/hektor-wakeword; скрипти — mww/scripts/.

## Середовище
- Colab, Tesla T4 (15 ГБ), 12 ГБ RAM, 2 vCPU; Python 3.13.15, TensorFlow 2.21.0 (GPU), Keras 3.13.2, numpy 2.1.3, ai-edge-litert 2.2.0, pymicro-features 2.0.2, audiomentations 0.43.1, mmap_ninja 0.9.0, datasets 4.8.5, tflite-micro (для заміру арени).
- microWakeWord: github.com/kahrendt/microWakeWord @ 4665173 («Pin dependencies (#98)», 2026-07-06) — той самий HEAD, що й OHF-Voice/micro-wake-word (репозиторій переїхав, «застарілої» версії немає).
- audio-metadata з форку whatsnowplaying@d4ebb23 (як у ноутбуці).
- Ноутбук basic_training_notebook.ipynb переписано у скрипти: prep_data.py → gen_features.py → train_run.py → evaluate.py → publish.py.

## Обходи версій і проблем (хронологічно)
1. Python 3.13 + TF 2.21 + Keras 3: mWW @4665173 працює без патчів (перевірено смоук-тестом: тренування → streaming → int8 tflite → тест). Вихідна модель: вхід int8 [1,3,40], вихід uint8 — як очікує ESPHome.
2. datasets 4.x декодує аудіо лише через torchcodec, а Clips з mWW читає WAV через datasets.Audio. Замість Clips — WavClips (scipy.io.wavfile) з тим самим інтерфейсом (audio_generator, get_random_clip), float32 у [-1, 1].
3. AudioSet: файлу agkphysics/AudioSet/data/bal_train09.tar з ноутбука більше немає (датасет переведено у parquet). Взято data/bal_train/08.parquet + 09.parquet (1000 кліпів), декодовано soundfile → 16 кГц.
4. MIT RIR: snapshot_download + ffmpeg замість datasets.load_dataset(streaming=True) (той самий torchcodec) — 269 RIR.
5. OOM на валідації: перший запуск v1 (18:04 UTC) убитий OOM-killer (exit -9) на кроці 500. mWW ріже validation_ambient (chime6 dev, 9.67 год) на вікна з кроком 100 мс і тримає список + np.array (понад 7 ГБ). Зроблено negative_datasets/dinner_party_eval_1h: validation_ambient = 1 год (один трек, обрізаний), testing_ambient = повний DiPCo 5.33 год. Під час тренування FA/год оцінюється на цій 1 год.
6. Фонові процеси: у Colab "!nohup ... &" тримає клітинку до кінця дочірнього процесу → тренування й оцінка йдуть через subprocess.Popen(start_new_session=True), логи в logs/.
7. Ознаки на диску: uint16 (значення / 0.0390625; mWW сам масштабує uint16 назад), позитиви обрізані до 210 найновіших кадрів (spectrogram_length = 204, truncate_start старіших не використовує): 173k вікон = 2.8 ГБ замість ~9 ГБ. Хак interkelstar «/24.7» для HF-негативів не потрібен — офіційний завантажувач масштабує uint16 сам (×0.0390625).
8. mix-up: у mWW @4665173 mix_up_augmentation_prob читається, але в data.py не застосовується — mix-up фактично немає і в ноутбуці; вимкнено.
9. Секрету HF_TOKEN у ноутбуці спочатку не було — додав власник. Процеси з терміналу Colab токена не бачать (лише ядро) → features_pos_live5.tar дозалито з ядра.
10. Проміжні артефакти одразу на HF: mww/work/features_*.tar (позитиви, негативи, live x3, live x5) і mww/work/train_v1|v2/ (чекпойнти; синхронізація кожні 10 хв і в кінці).

## Крок 1 — дані (prep_data.py)
Вимоги mWW до кліпів: фіксована тривалість не потрібна. Augmentation.augment_clip дописує jitter 0.195–0.205 с тиші В КІНЕЦЬ і доповнює ЗЛІВА до 3.2 с; тренування бере найновіші кадри (truncate_start). Тобто слово має закінчуватися в кінці файлу — модель вчиться спрацьовувати ≈0.2 с після слова. Наші кліпи вже з обрізаною тишею, тож вирівнювання не потрібне. Що зроблено: перевірка формату (усі 24 287 файлів 16 кГц моно int16), прибрано DC, 5 мс fade-in/out на обрізаних краях (інакше клац дає хибний онсет). Довжини позитивів: p99 1.1–1.5 с ≤ clip_duration 1.5 с (один piper-кліп 6.95 с — обрізається з початку).
Розбиття (split.json не змінювався): positive_train → pos_main 8651 / pos_live 9 / pos_val 414 (5 % не-live, лише для вибору чекпойнта); negative_train → neg_piper 7599 / neg_other 2083 (tts 1563, жива мова власника 451, через вуха 69) / neg_val 490. positive_test (2982) і negative_test (2059) — лише фінальна оцінка.
Групи звіту: live (live-hektor-martyn-*), robot (*-robot-*), aug-tempo (*-t0.6/0.72/0.85*, включно з firered-темпами), firered, tts (cartesia/elevenlabs/gemini), piper.

## Крок 2 — фон, кімнати, аугментація (gen_features.py)
Корпуси як у ноутбуці: MIT RIR, AudioSet, FMA xs; готові негативні спектрограми kahrendt/microwakeword (speech, dinner_party, no_speech, dinner_party_eval). Аугментація — дефолти ноутбука з правками interkelstar: SNR фону −5…20 дБ (нб −5…10), p(фон) 0.8 (нб 0.75), p(RIR) 0.7 (нб 0.5), EQ/Tanh/Pitch/BandStop/ColorNoise по 0.1, Gain 1.0; паддинг ніколи не цифровий нуль — шумова підлога −70…−50 dBFS (висновок interkelstar з ru_stop v2). Позитиви: 2 повтори × slide 10 (як у нб); live: 6 повторів (oversampling ×3), у v2 — 10 (×5); наші негативи: 2 повтори, слово закінчується за 0…0.6 с до кінця вікна, truncation random.

## Крок 3 — тренування (train_run.py)
Ноутбук + правки interkelstar (ru_jarvis.yaml): 20 000 кроків (нб 10 000), ваги класів pos 1.5 / neg 1 (нб 1 / 20), SpecAugment 2×5 по часу і 2×5 по частоті (нб 0), lr 0.001, batch 128; mixednet 64×4, ядра [5],[7,11],[9,15],[23], first conv 32×5, stride 3, residual 1,1,1,1 (нб 0,0,0,0). Вибір чекпойнта — дефолт ноутбука: max average_viable_recall (recall, інтегрований по 0…2 FA/год на validation_ambient). Ваги вибірки: pos_main 2.0; live пропорційно кількості (кожна позитивна спектрограма рівноймовірна, тож ×3 / ×5 = кількість повторів); neg_other 4; neg_piper 2 (лише v1); speech 10, dinner_party 10, no_speech 5 (як у нб).
v2 (бо live recall v1 < 50 %): live ×5, без Piper-негативів (лишились TTS, жива мова, через вуха + HF-негативи).
Не перенесено з interkelstar: його власний пайплайн (вікно по центру слова, 50/50 pos/neg, EarlyStopping) — суперечить потоковій логіці mWW «слово в кінці вікна»; FP-mining і щільне віконування реальних хибних спрацювань — у датасеті поки немає записів хибних спрацювань робота.

## Крок 4 — оцінка (evaluate.py)
Як на платі: [1 с шуму −60 dBFS] + кліп + [0.5 с] → micro-frontend (pymicro_features: 40 мел-бінів, вікно 30 мс, крок 10 мс, PCAN/NR) → потокова int8-модель (3 нові кадри = 30 мс на виклик, стан з нуля для кожного кліпу) → ковзне середнє N останніх ймовірностей (кільцевий буфер з нулями, як у ESPHome) → спрацювання, якщо max ≥ поріг. FA/год — потоково на 5.33 год DiPCo (testing_ambient), одне спрацювання на передній фронт, рефрактерність 1 с. tensor_arena — мінімальна арена tflite-micro (бінарний пошук) + 10 %.

## v2 — уточнення власника (06.10, під час прогону)
- Перший запуск v2 (20 000 кроків, ambient = DiPCo/chime6) зупинено на ~8000 кроці за новою вимогою.
- validation_ambient / testing_ambient = фон кімнати через вуха робота (micro-wake-hector/ambient/, 10 WAV по ~3 хв, 16 кГц, без «Гектор» за Whisper): файли 1, 4, 6, 7, 8 → validation (14.8 хв), 2, 3, 5, 9, 10 → testing (14.1 хв); truncation split, sampling_weight 0; ознаки — micro-frontend без аугментації (features_room_ambient.tar).
- Вибір ваг: minimization_metric = ambient_false_positives_per_hour (на кімнатному validation_ambient, поріг 0.5, неструмінно), target 0 → далі max average_viable_recall. Поріг/вікно в manifest — за FA/год на кімнатному testing_ambient (≤ 0.5/год), далі max live recall, далі загальний recall.
- Ознаки позитивів і негативів не перераховувались (взято features_pos_main, features_pos_live5 (live ×5), features_neg_other; без neg_piper). 10 000 кроків.
- DiPCo (5.33 год) лишився у звіті як довідкова FA/год.
- Обхід: ланцюжок «дочекатися кінця тренування» через pgrep -f зациклився (pgrep знаходив власний shell із тим самим рядком) — оцінку v2 запущено вручну.
- Ядро Colab не бачило microwakeword після pip install -e без рестарту → sys.path.insert(0, '/content/microWakeWord') у клітинках ядра (скрипти запускались окремими процесами й цього не потребували).

## v3 — набір micro-wake-hector-v3 (07.10)
- Середовище те саме (TF 2.21.0, Keras 3.13.2, numpy 2.1.3, datasets 4.8.5, pymicro-features 2.0.2, audiomentations 0.43.1, mmap_ninja 0.9.0, ai-edge-litert 2.2.0, mWW @4665173, audio-metadata whatsnowplaying@d4ebb23), Python 3.13.16, T4. Нові обходи не знадобились; скрипти — mww/scripts/*_v3.py, dp1h.py.
- Дані (prep_v3.py): позитиви трьох типів (Гектор / Гекторе / з командою); 93 живих (70 train, з них 3 з командою; 23 test, з них 1 з командою). Групи звіту як раніше + live-cmd окремо (live-*-cmd, phrase cmd або кома в text).
- Негативи: розподіл v1 (neg_piper / neg_other / neg_val) відтворено повтором random.seed(42) після 9065 вибірок позитивів старого zip → 7599 / 2083 / 490, як у логу v1; усі 10 172 старі файли збігаються за іменем і CRC → features_neg_piper.tar і neg_other (training/validation) взяті готові. 347 нових негативів (по 12 на фразу) → окремий набір neg_add_other; вага neg_other 4.0 поділена між neg_other_v3 і neg_add_other пропорційно кількості спектрограм.
- Ознаки (gen_v3.py): перераховано pos_main_v3 (196 040), pos_live_v3 (×25, slide 10 → 17 500), negative_test. Фон кімнати: features_room_ambient.tar перевірено (файли 1,4,6,7,8 → validation, 2,3,5,9,10 → testing, довжини збігаються з manifest) і взято готовим. dinner_party_eval_1h відтворено dp1h.py (найдовший трек chime6 dev, обрізаний до 360 000 кадрів; testing_ambient = повний DiPCo).
- Обхід: download_corpora під час першого запуску gen видалила зламаний RIR (h227_Outside_ParkingLot_1txts.wav), коли augmenter уже мав його у списку → gen перезапущено після завершення завантаження (269 RIR, як у v1).
- Тренування (train_v3.py): конфіг v1 + pos_live w = 2/9 (≈10 % позитивів у батчі), penalty 3.0; neg_piper 1.5, neg_other 4.0, speech 8, dinner_party 6, no_speech 4; ваги класів 1.5/1.0; validation_ambient = кімната + DiPCo 1 год; minimization ambient_false_positives_per_hour ≤ 0.5, далі max average_viable_recall; 10 000 кроків, ~38 хв.
- Оцінка (eval_v3.py): як evaluate.py; v1 і v2 перецінено на тому ж тестовому наборі v3 (logs/eval_v1_on_v3.log, eval_v2_on_v3.log).
