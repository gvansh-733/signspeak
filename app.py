"""
SignSpeak - STEP 3: the real-time translator.

Pipeline (exactly as in the synopsis)
    Webcam -> MediaPipe hand landmarks -> normalise -> classifier
           -> confidence filter + de-duplication -> text -> offline speech (pyttsx3)

Two modes
    TRANSLATE  hold a sign to type it; letters build words, words are spoken
    PRACTICE   learning mode: the app shows a target sign and checks yours

Keys                                 Mouse
    SPACE      add a space (speaks the word)     every button in the right panel
    BACKSPACE  delete last character              is clickable
    S          speak the whole sentence
    C          clear
    A          auto-speak on/off
    M          switch TRANSLATE / PRACTICE
    Q / ESC    quit

Usage
    python app.py
"""
import argparse
import queue
import random
import threading
import time
from collections import Counter, deque

import cv2
import joblib
import numpy as np

import config
import ui
from utils import HandDetector

W, H = 1180, 700
CAM = (20, 84, 760, 570)             # x, y, w, h of the camera view
WINDOW = "SignSpeak - Sign Language to Text & Speech"

BUTTONS = [
    ui.Button("speak", "SPEAK", (800, 488, 84, 40), ui.ACCENT),
    ui.Button("space", "SPACE", (892, 488, 84, 40), ui.ACCENT),
    ui.Button("back", "BACK", (984, 488, 84, 40), ui.WARN),
    ui.Button("clear", "CLEAR", (1076, 488, 84, 40), ui.BAD),
    ui.Button("mode", "MODE", (800, 538, 176, 40), ui.ACCENT2),
    ui.Button("auto", "AUTO", (984, 538, 176, 40), ui.GOOD),
]


# ================================================================= speech
class Speaker:
    """Offline text-to-speech (pyttsx3) running on its own thread so the video never freezes."""

    def __init__(self):
        self.available = True
        self.busy = False
        self._queue = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", config.TTS_RATE)
            engine.setProperty("volume", config.TTS_VOLUME)
        except Exception as exc:                                  # noqa: BLE001
            print(f"[SignSpeak] Text-to-speech unavailable: {exc}")
            self.available = False
            return
        while True:
            text = self._queue.get()
            if text is None:
                return
            self.busy = True
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as exc:                              # noqa: BLE001
                print(f"[SignSpeak] TTS error: {exc}")
            self.busy = False

    def say(self, text):
        text = text.strip()
        if text and self.available:
            self._queue.put(text)

    def close(self):
        self._queue.put(None)


# ================================================================= text formation
class SentenceBuilder:
    """Accumulates recognised signs into text and triggers speech."""

    def __init__(self, speaker):
        self.text = ""
        self.speaker = speaker
        self.auto_speak = True

    def add_sign(self, label):
        if label in config.WORDS:                       # whole-word sign
            if self.text and not self.text.endswith(" "):
                self.text += " "
            self.text += label + " "
            if self.auto_speak:
                self.speaker.say(label)
        else:                                           # fingerspelled letter
            self.text += label.lower()

    def add_space(self):
        if not self.text or self.text.endswith(" "):
            return
        last_word = self.text.split(" ")[-1]
        self.text += " "
        if self.auto_speak:
            self.speaker.say(last_word)

    def backspace(self):
        self.text = self.text[:-1]

    def clear(self):
        self.text = ""

    def speak_all(self):
        self.speaker.say(self.text)


class SignStabilizer:
    """
    Turns noisy per-frame predictions into deliberate key-presses:
      * majority vote over the last N frames (smoothing)
      * confidence threshold
      * the sign must be HELD for HOLD_SECONDS
      * de-duplication: a sign is not typed again until the hand changes sign /
        leaves the view, or it is held for another REPEAT_SECONDS (for "LL")
    update() -> (progress 0..1, committed_label or None)
    """

    def __init__(self):
        self.history = deque(maxlen=config.SMOOTHING_WINDOW)
        self.candidate = None
        self.since = 0.0
        self.locked = None

    def reset(self):
        self.history.clear()
        self.candidate = None
        self.locked = None

    def update(self, label, conf, now):
        confident = label is not None and conf >= config.CONFIDENCE_THRESHOLD
        self.history.append(label if confident else None)
        winner, votes = Counter(self.history).most_common(1)[0]

        if winner is None or votes < len(self.history) // 2 + 1:
            if winner is None:
                self.locked = None                       # hand left -> unlock
            self.candidate = None
            return 0.0, None

        if winner != self.candidate:
            self.candidate, self.since = winner, now
        held = now - self.since

        if winner == self.locked:                         # already typed
            if held >= config.REPEAT_SECONDS:
                self.since = now
                return 1.0, winner
            return 1.0, None

        progress = min(held / config.HOLD_SECONDS, 1.0)
        if progress >= 1.0:
            self.locked, self.since = winner, now
            return 1.0, winner
        return progress, None


# ================================================================= drawing
def draw_camera(cam, hand, label, conf, progress, locked, fps, toast, now):
    if hand is not None:
        ui.draw_hand(cam, hand.points)
        box = ui.hand_bbox(hand.points, bounds=(CAM[2], CAM[3]))
        color = ui.GOOD if locked else ui.ACCENT
        ui.draw_corners(cam, box, color)
        if label is not None:
            tag = f"{label.upper() if len(label) > 1 else label}  {conf * 100:.0f}%"
            tw = ui.text_size(tag, 0.7, 2)[0] + 24
            tx = max(6, min(box[0], CAM[2] - tw - 6))
            ty = max(6, box[1] - 42)
            ui.rounded_rect(cam, (tx, ty), (tx + tw, ty + 34), color, radius=10)
            ui.put_text(cam, tag, (tx + 12, ty + 24), 0.7, ui.BG_BOTTOM, 2)
        ui.chip(cam, f"{hand.handedness.upper()} HAND", 14, CAM[3] - 42, ui.PANEL)
    else:
        pulse = int(6 * np.sin(now * 4)) + 70
        cv2.circle(cam, (CAM[2] // 2, CAM[3] // 2 - 30), pulse, ui.PANEL_LIGHT, 3, ui.AA)
        ui.put_text_centered(cam, "Show your hand to the camera", (CAM[2] // 2, CAM[3] // 2 + 70), 0.8, ui.TEXT, 2)
        ui.put_text_centered(cam, "plain background + good light work best",
                             (CAM[2] // 2, CAM[3] // 2 + 100), 0.5, ui.MUTED, 1)
    ui.chip(cam, f"LIVE   {fps:.0f} FPS", 14, 14, ui.PANEL)
    if now < toast[1]:
        text, _, color = toast
        tw = ui.text_size(text, 0.75, 2)[0]
        ui.chip(cam, text, (CAM[2] - tw - 36) // 2, CAM[3] - 56, color, ui.BG_BOTTOM,
                scale=0.75, h=38, alpha=0.95)


def draw_detected_card(canvas, label, conf, top, progress, locked):
    ui.card(canvas, (800, 84, 360, 176), "DETECTED SIGN")
    center = (884, 184)
    ring_color = ui.GOOD if locked else ui.ACCENT
    ui.ring(canvas, center, 56, progress, ring_color)
    if label is None:
        ui.put_text_centered(canvas, "-", center, 1.6, ui.MUTED, 3)
    elif len(label) == 1:
        ui.put_text_centered(canvas, label, center, 2.6, ui.TEXT, 5)
    else:
        ui.put_text_centered(canvas, label.upper(), center, 0.8, ui.TEXT, 2)

    ui.put_text(canvas, "CONFIDENCE", (964, 138), 0.45, ui.MUTED, 1)
    pct = f"{conf * 100:.0f}%" if label else "--"
    ok = conf >= config.CONFIDENCE_THRESHOLD
    ui.put_text(canvas, pct, (964, 170), 0.95, ui.GOOD if ok else ui.WARN, 2)
    ui.progress_bar(canvas, (964, 180, 180, 8), conf if label else 0, ui.GOOD if ok else ui.WARN)
    ui.put_text(canvas, "ALTERNATIVES", (964, 214), 0.45, ui.MUTED, 1)
    for k, (name, p) in enumerate(top[1:3]):
        shown = name if len(name) == 1 else name.upper()
        ui.put_text(canvas, f"{shown:<7} {p * 100:4.0f}%", (964, 236 + k * 18), 0.5, ui.TEXT, 1)


def draw_sentence_card(canvas, builder, now):
    ui.card(canvas, (800, 270, 360, 206), "SENTENCE")
    lines = ui.wrap_text(builder.text.upper(), 326, 0.8, 2)[-4:]
    if int(now * 2) % 2 == 0:                              # blinking cursor
        lines[-1] += "_"
    for k, line in enumerate(lines):
        ui.put_text(canvas, line, (816, 326 + k * 34), 0.8, ui.TEXT, 2)
    ui.put_text(canvas, f"{len(builder.text)} characters", (816, 466), 0.45, ui.MUTED, 1)


def draw_practice_card(canvas, target, score, streak, feedback, progress, now):
    ui.card(canvas, (800, 270, 360, 206), "PRACTICE MODE - show this sign")
    ui.put_text_centered(canvas, target if len(target) == 1 else target.upper(), (980, 348),
                         3.4 if len(target) == 1 else 1.4, ui.ACCENT, 6 if len(target) == 1 else 3)
    hint = config.SIGN_HINTS.get(target, f"ASL fingerspelling letter '{target}'")
    for k, line in enumerate(ui.wrap_text(hint, 330, 0.45, 1)[:2]):
        ui.put_text_centered(canvas, line, (980, 405 + k * 18), 0.45, ui.MUTED, 1)
    if now < feedback[1]:
        ui.put_text_centered(canvas, feedback[0], (980, 432), 0.65, feedback[2], 2)
    ui.put_text(canvas, f"Score {score}", (816, 466), 0.5, ui.TEXT, 1)
    ui.put_text(canvas, f"Streak {streak}", (1060, 466), 0.5, ui.TEXT, 1)


def draw_status_card(canvas, model_name, accuracy, speaker, fps):
    ui.card(canvas, (800, 588, 360, 66), None)
    ui.put_text(canvas, f"Model: {model_name}  ({accuracy * 100:.1f}% test acc.)", (816, 612), 0.48, ui.TEXT, 1)
    if not speaker.available:
        tts, col = "unavailable", ui.BAD
    elif speaker.busy:
        tts, col = "speaking...", ui.GOOD
    else:
        tts, col = "ready", ui.MUTED
    ui.put_text(canvas, f"Speech: {tts}", (816, 636), 0.48, col, 1)
    ui.put_text(canvas, f"{fps:.0f} FPS", (1090, 636), 0.48, ui.MUTED, 1)


# ================================================================= main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()

    if not config.MODEL_FILE.exists():
        raise SystemExit("[SignSpeak] No trained model found. Run:  python train_model.py")
    bundle = joblib.load(config.MODEL_FILE)
    model, model_name, accuracy = bundle["model"], bundle["name"], bundle["accuracy"]
    classes = list(model.classes_)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"[SignSpeak] Could not open camera {args.camera}. Try --camera 1")

    detector = HandDetector()
    speaker = Speaker()
    builder = SentenceBuilder(speaker)
    stabilizer = SignStabilizer()
    background = ui.make_background(W, H)

    mode = "translate"                           # translate | practice
    target, score, streak = random.choice(classes), 0, 0
    feedback = ("", 0.0, ui.GOOD)
    cooldown_until = 0.0
    toast = ("", 0.0, ui.GOOD)
    fps, last_time = 0.0, time.time()
    mouse = {"pos": (-1, -1), "click": None}

    def on_mouse(event, x, y, flags, param):
        mouse["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    def new_target():
        return random.choice([c for c in classes if c != target] or classes)

    def perform(action, now):
        nonlocal mode, target, score, streak, toast
        if action == "mode":
            mode = "practice" if mode == "translate" else "translate"
            stabilizer.reset()
            if mode == "practice":
                target, score, streak = new_target(), 0, 0
            toast = (f"{mode.upper()} mode", now + 1.5, ui.ACCENT2)
        elif action == "auto":
            builder.auto_speak = not builder.auto_speak
        elif mode == "translate":
            if action == "speak":
                builder.speak_all()
            elif action == "space":
                builder.add_space()
            elif action == "back":
                builder.backspace()
            elif action == "clear":
                builder.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last_time, 1e-6)
        last_time = now

        # ---- 1) capture + 2) landmarks
        cam = ui.fit_frame(cv2.flip(frame, 1), CAM[2], CAM[3])
        hand = detector.detect(cam)

        # ---- 3) normalise + classify
        label, conf, top = None, 0.0, []
        if hand is not None:
            proba = model.predict_proba(hand.features.reshape(1, -1))[0]
            order = np.argsort(proba)[::-1][:3]
            top = [(classes[i], float(proba[i])) for i in order]
            label, conf = top[0]

        # ---- 4) filter + de-duplicate, then 5) text
        progress, committed = stabilizer.update(label, conf, now)
        if committed and now >= cooldown_until:
            if mode == "translate":
                builder.add_sign(committed)
                shown = committed if len(committed) == 1 else committed.upper()
                toast = (f"Added  {shown}", now + 0.9, ui.GOOD)
            elif committed == target:
                score += 1
                streak += 1
                feedback = ("Correct!", now + 1.5, ui.GOOD)
                target = new_target()
                stabilizer.reset()
                cooldown_until = now + 1.5
            else:
                streak = 0
                shown = committed if len(committed) == 1 else committed.upper()
                feedback = (f"That looks like {shown} - try again", now + 1.5, ui.WARN)

        # ---- draw
        draw_camera(cam, hand, label, conf, progress, stabilizer.locked is not None and label == stabilizer.locked,
                    fps, toast, now)
        canvas = background.copy()
        chips = [("TRANSLATE" if mode == "translate" else "PRACTICE", ui.ACCENT2),
                 ("AUTO-SPEAK ON" if builder.auto_speak else "AUTO-SPEAK OFF",
                  ui.GOOD if builder.auto_speak else ui.MUTED)]
        ui.draw_header(canvas, "Real-time Sign Language to Text & Speech", chips)
        ui.paste_rounded(canvas, cam, CAM[0], CAM[1])

        draw_detected_card(canvas, label, conf, top, progress,
                           stabilizer.locked is not None and label == stabilizer.locked)
        if mode == "translate":
            draw_sentence_card(canvas, builder, now)
        else:
            draw_practice_card(canvas, target, score, streak, feedback, progress, now)

        for b in BUTTONS:
            hover = b.contains(*mouse["pos"])
            if b.action == "mode":
                b.draw(canvas, hover, label="MODE: " + mode.upper())
            elif b.action == "auto":
                b.draw(canvas, hover, label="AUTO-SPEAK: " + ("ON" if builder.auto_speak else "OFF"),
                       active=builder.auto_speak)
            else:
                b.draw(canvas, hover, enabled=(mode == "translate"))
        draw_status_card(canvas, model_name, accuracy, speaker, fps)
        ui.put_text_centered(canvas, "Hold a sign to type it   |   SPACE space   |   BACKSPACE delete   |   "
                                     "S speak   |   C clear   |   A auto-speak   |   M mode   |   Q quit",
                             (W // 2, 682), 0.48, ui.MUTED, 1)

        cv2.imshow(WINDOW, canvas)

        # ---- input
        if mouse["click"]:
            for b in BUTTONS:
                if b.contains(*mouse["click"]):
                    perform(b.action, now)
            mouse["click"] = None
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        keymap = {ord(" "): "space", 8: "back", ord("s"): "speak", ord("c"): "clear",
                  ord("a"): "auto", ord("m"): "mode"}
        if key in keymap:
            perform(keymap[key], now)

    speaker.close()
    detector.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
