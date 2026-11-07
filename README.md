# SignSpeak - Real-Time Sign Language to Text & Speech Translator

A low-cost, camera-only system that recognises static sign-language gestures from a
standard webcam and converts them into **text** and **speech** in real time.

```
Webcam -> Hand Landmark Detection -> Gesture Classification -> Text Output -> Speech Output
          (MediaPipe, 21 points)     (RF / SVM / MLP)          (buffer)       (pyttsx3, offline)
```

## 1. Setup (one time)

```bash
python -m venv venv
venv\Scripts\activate            # Windows      (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

* Use **Python 3.9 - 3.12**.
* Keep `mediapipe==0.10.14`. Newer MediaPipe releases removed the `mp.solutions.hands`
  API this project uses.
* Linux only: `sudo apt install espeak-ng` (needed by pyttsx3). Windows/macOS work out of the box.

## 2. Run it - three steps

| Step | Command | What it does |
|------|---------|--------------|
| 1 | `python collect_data.py` | Record landmark samples for every sign into `data/landmarks.csv` |
| 2 | `python train_model.py`  | Train + compare 3 models, save the best one and all report charts |
| 3 | `python app.py`          | Launch the live translator |

### Step 1 - collecting data
* Press **SPACE** -> 3-second countdown -> it records 200 samples of the current sign
  (change with `--samples 250`) and moves on to the next sign automatically.
* Click any tile in the grid to jump to a sign. **R** re-records a sign, **N/P** next/previous.
* Progress is saved as you go; you can quit and continue another day.
* **For good accuracy:** plain background, good light, whole hand visible, and *slightly*
  tilt/move the hand while recording. Record from 2-3 distances. Ideally have a friend
  record too (different hand sizes make the model much more robust).
* Not sure how a letter looks? Search "ASL fingerspelling chart".

### Step 2 - training
Produces, in `reports/`: `model_comparison.png`, `confusion_matrix.png`,
`classification_report.txt`, `metrics.json` - ready to paste into your project report.
The best model is saved to `models/signspeak_model.joblib`.

### Step 3 - the live app

| Control | Action |
|---------|--------|
| **Hold a sign ~1 s** | Types it (ring fills up, then it turns green) |
| Hold again ~2 s / change sign | Types a repeated letter (e.g. the "LL" in HELLO) |
| `SPACE` | Adds a space and speaks the word you just spelled |
| `BACKSPACE` / `C` | Delete last character / clear all |
| `S` | Speak the whole sentence |
| `A` | Auto-speak on/off |
| `M` | Switch **TRANSLATE** <-> **PRACTICE** mode |
| `Q` / `ESC` | Quit |

All buttons in the right-hand panel are also clickable.

**Practice mode** turns the app into a learning aid: it shows a target sign and tells you
whether yours is correct (score + streak) - this covers the "educational aid" application in the synopsis.

## 3. How it matches the synopsis

| Synopsis requirement | Where in the project |
|---|---|
| Webcam capture with OpenCV | `app.py`, `collect_data.py` |
| MediaPipe 21-landmark extraction (x, y, z = 63 features, CSV) | `utils.py` (`HandDetector`), `data/landmarks.csv` |
| Self-collected dataset, A-Z + hello/thanks/yes/no, 150-300 samples/sign | `collect_data.py`, `config.py` (200 default) |
| Normalisation relative to the wrist | `utils.normalize_landmarks` (also scale- and left/right-hand-invariant) |
| Random Forest / SVM classifier | `train_model.py` |
| Feedforward neural network as comparison model | `train_model.py` (MLP) |
| Confidence threshold + de-duplication | `SignStabilizer` in `app.py` (smoothing, 80% threshold, hold timer, lock) |
| Text formation | `SentenceBuilder` in `app.py` |
| Offline speech synthesis (pyttsx3) | `Speaker` in `app.py` (own thread, video never freezes) |
| Intuitive interface for live demo | OpenCV dashboard: live skeleton, confidence ring, clickable buttons |

**Not included (marked optional / stretch in the synopsis):** the LSTM for motion gestures and
the Kaggle cross-validation dataset. Both are natural "future work" points.

## 4. Known limitations (good to mention in your report/viva)

* **J and Z** are motion letters in real ASL, and **hello / thanks / yes / no** are also
  motion-based. With a static classifier they are recorded as distinct *static stand-in poses*
  (suggested in the collection screen). The LSTM stretch goal is the proper solution.
* Look-alike signs (e.g. **M/N/T**, **U/V/R**) are the usual confusions; check
  `reports/confusion_matrix.png` and record more samples for the weak ones.
* Accuracy on your own held-out split is higher than it will be for a brand-new person or
  room - that is why recording varied data matters.

## 5. Project structure

```
SignSpeak/
|-- config.py          all settings (signs, thresholds, speeds)
|-- utils.py           MediaPipe wrapper + landmark normalisation
|-- ui.py              OpenCV dashboard drawing toolkit
|-- collect_data.py    step 1 - build the dataset
|-- train_model.py     step 2 - train / compare / evaluate
|-- app.py             step 3 - live translator
|-- requirements.txt
|-- data/  models/  reports/
```

## 6. Troubleshooting

* *"Could not open camera"* -> `python app.py --camera 1`
* *`module 'mediapipe' has no attribute 'solutions'`* -> `pip install mediapipe==0.10.14`
* Wrong letters keep appearing -> raise `CONFIDENCE_THRESHOLD` / `HOLD_SECONDS` in `config.py`
* Typing feels slow -> lower `HOLD_SECONDS` (e.g. 0.6)
* No sound -> check `Speech:` in the status card; install `pyttsx3` (and `espeak-ng` on Linux)
