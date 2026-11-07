"""
SignSpeak - central configuration.

Every tunable value lives here so the three scripts (collect / train / app)
always agree with each other.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
DATA_FILE = DATA_DIR / "landmarks.csv"
MODEL_FILE = MODEL_DIR / "signspeak_model.joblib"

# ----------------------------------------------------------------- vocabulary
# Synopsis: static ASL alphabet (A-Z) + a few common words.
ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
WORDS = ["hello", "thanks", "yes", "no"]
SIGNS = ALPHABET + WORDS

# Hints shown while collecting data. Real ASL "hello/thanks/yes/no" and the
# letters J and Z involve motion, so a distinct *static* pose is used instead
# (the LSTM stretch goal in the synopsis is the proper fix for these).
SIGN_HINTS = {
    "J": "Pinky out, hand tilted ~45 deg (static stand-in for J's hook)",
    "Z": "Index finger pointing forward (static stand-in for Z's stroke)",
    "hello": "Open palm, fingers spread, facing the camera",
    "thanks": "Thumb, index and pinky out ('I-love-you' shape)",
    "yes": "Thumbs-up",
    "no": "Index and middle finger pinched against the thumb",
}

# ----------------------------------------------------------------- camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ----------------------------------------------------------------- data collection
SAMPLES_PER_SIGN = 200        # synopsis: ~150-300 samples per sign
CAPTURE_INTERVAL = 0.06       # seconds between saved samples
COUNTDOWN_SECONDS = 3

# ----------------------------------------------------------------- MediaPipe
MIN_DETECTION_CONF = 0.6
MIN_TRACKING_CONF = 0.5

# ----------------------------------------------------------------- training
TEST_SIZE = 0.20
RANDOM_STATE = 42

# ----------------------------------------------------------------- text formation
CONFIDENCE_THRESHOLD = 0.80   # prediction must be at least this sure
HOLD_SECONDS = 0.90           # hold a sign this long to type it
REPEAT_SECONDS = 2.00         # keep holding to type the same letter twice
SMOOTHING_WINDOW = 7          # majority vote over the last N frames

# ----------------------------------------------------------------- speech
TTS_RATE = 165
TTS_VOLUME = 1.0
