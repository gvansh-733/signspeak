"""
SignSpeak - STEP 1: collect the dataset.

Records MediaPipe hand-landmark samples (63 features each) for every sign and
stores them in data/landmarks.csv.

Controls
    SPACE  start / stop recording the current sign (3-second countdown)
    N / P  next / previous sign          (or click a tile in the grid)
    R      delete the samples of the current sign and re-record it
    Q/ESC  quit (everything recorded so far is already saved)

Usage
    python collect_data.py
    python collect_data.py --samples 250 --camera 1
"""
import argparse
import csv
import time

import cv2

import config
import ui
from utils import HandDetector, feature_columns

W, H = 1180, 700
CAM = (20, 84, 760, 570)          # x, y, w, h of the camera view
WINDOW = "SignSpeak - Data Collection"

# grid of sign tiles (6 columns)
GRID_X0, GRID_Y0, TILE_W, TILE_H, GAP, COLS = 806, 348, 54, 38, 4, 6


# ----------------------------------------------------------------- csv helpers
def ensure_header():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not config.DATA_FILE.exists():
        with open(config.DATA_FILE, "w", newline="") as f:
            csv.writer(f).writerow(["label"] + feature_columns())


def load_counts():
    counts = {s: 0 for s in config.SIGNS}
    with open(config.DATA_FILE, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row and row[0] in counts:
                counts[row[0]] += 1
    return counts


def append_rows(rows):
    if rows:
        with open(config.DATA_FILE, "a", newline="") as f:
            csv.writer(f).writerows(rows)


def delete_label(label):
    with open(config.DATA_FILE, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], [r for r in rows[1:] if r and r[0] != label]
    with open(config.DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(body)


def tile_rect(i):
    r, c = divmod(i, COLS)
    return (GRID_X0 + c * (TILE_W + GAP), GRID_Y0 + r * (TILE_H + GAP), TILE_W, TILE_H)


# ----------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=config.SAMPLES_PER_SIGN,
                        help="samples to record per sign (default: %(default)s)")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    args = parser.parse_args()
    target = args.samples

    ensure_header()
    counts = load_counts()
    signs = config.SIGNS
    idx = next((i for i, s in enumerate(signs) if counts[s] < target), 0)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    if not cap.isOpened():
        raise SystemExit(f"[SignSpeak] Could not open camera {args.camera}. Try --camera 1")

    detector = HandDetector()
    background = ui.make_background(W, H)

    state = "idle"                 # idle | countdown | recording
    countdown_end = 0.0
    last_capture = 0.0
    rows = []
    toast = ("", 0.0, ui.GOOD)
    mouse = {"pos": (0, 0), "click": None}

    def on_mouse(event, x, y, flags, param):
        mouse["pos"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse["click"] = (x, y)

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)

    def stop_recording(message=None, color=ui.GOOD):
        nonlocal state, rows, toast
        append_rows(rows)
        rows = []
        state = "idle"
        if message:
            toast = (message, time.time() + 2.0, color)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        sign = signs[idx]

        cam = ui.fit_frame(cv2.flip(frame, 1), CAM[2], CAM[3])
        hand = detector.detect(cam)

        # ---- clicks on the sign grid
        if mouse["click"] and state == "idle":
            cx, cy = mouse["click"]
            for i in range(len(signs)):
                x, y, w, h = tile_rect(i)
                if x <= cx <= x + w and y <= cy <= y + h:
                    idx = i
        mouse["click"] = None

        # ---- state machine
        if state == "countdown" and now >= countdown_end:
            state, last_capture = "recording", 0.0
        if state == "recording" and hand is not None and now - last_capture >= config.CAPTURE_INTERVAL:
            rows.append([sign] + [f"{v:.6f}" for v in hand.features])
            counts[sign] += 1
            last_capture = now
            if counts[sign] >= target:
                stop_recording(f"'{sign}' complete - {counts[sign]} samples saved")
                idx = min(idx + 1, len(signs) - 1)

        # ---- draw camera view
        if hand is not None:
            ui.draw_hand(cam, hand.points)
            colour = ui.GOOD if state == "recording" else ui.ACCENT
            ui.draw_corners(cam, ui.hand_bbox(hand.points, bounds=(CAM[2], CAM[3])), colour)
        else:
            ui.put_text_centered(cam, "Show your hand to the camera", (CAM[2] // 2, CAM[3] // 2),
                                 0.9, ui.WARN, 2)
        if state == "recording":
            ui.chip(cam, "REC  %d / %d" % (counts[sign], target), 14, 14, ui.BAD)
            ui.chip(cam, "Slowly tilt / move your hand a little for variety", 14, CAM[3] - 42, ui.PANEL)
        elif state == "countdown":
            ui.put_text_centered(cam, str(int(countdown_end - now) + 1), (CAM[2] // 2, CAM[3] // 2),
                                 7.0, ui.WARN, 12)
            ui.chip(cam, "Get your pose ready", 14, 14, ui.WARN, ui.BG_BOTTOM)
        else:
            ui.chip(cam, "READY - press SPACE to record", 14, 14, ui.PANEL)
        if now < toast[1]:
            ui.chip(cam, toast[0], 14, CAM[3] - 42, toast[2], ui.BG_BOTTOM, alpha=1.0)

        # ---- compose dashboard
        canvas = background.copy()
        total = sum(counts.values())
        ui.draw_header(canvas, "Step 1 - Dataset collection",
                       [(f"{total} samples total", ui.ACCENT)])
        ui.paste_rounded(canvas, cam, CAM[0], CAM[1])

        ui.card(canvas, (800, 84, 360, 220), "CURRENT SIGN")
        big = sign if len(sign) == 1 else sign.upper()
        ui.put_text_centered(canvas, big, (980, 142), 3.2 if len(sign) == 1 else 1.6, ui.TEXT, 5 if len(sign) == 1 else 3)
        hint = config.SIGN_HINTS.get(sign, f"ASL fingerspelling letter '{sign}'")
        for k, line in enumerate(ui.wrap_text(hint, 330, 0.5, 1)[:2]):
            ui.put_text(canvas, line, (816, 206 + k * 18), 0.5, ui.MUTED, 1)
        ui.progress_bar(canvas, (816, 236, 328, 12), counts[sign] / target,
                        ui.GOOD if counts[sign] >= target else ui.ACCENT)
        status = {"idle": "SPACE = record", "countdown": "Starting...", "recording": "Recording..."}[state]
        ui.put_text(canvas, f"{counts[sign]} / {target} samples", (816, 276), 0.55, ui.TEXT, 1)
        ui.put_text(canvas, status, (1010, 276), 0.55, ui.ACCENT, 1)

        ui.card(canvas, (800, 312, 360, 250), "ALL SIGNS  (click a tile to jump)")
        for i, s in enumerate(signs):
            x, y, w, h = tile_rect(i)
            frac = counts[s] / target
            fill = ui.PANEL_LIGHT if frac == 0 else (ui.GOOD if frac >= 1 else ui.WARN)
            ui.rounded_rect(canvas, (x, y), (x + w, y + h), fill, radius=8)
            tc = ui.TEXT if frac == 0 else ui.BG_BOTTOM
            label = s
            ui.put_text_centered(canvas, label, (x + w // 2, y + h // 2), 0.55 if len(s) == 1 else 0.38, tc, 1)
            if i == idx:
                ui.rounded_rect(canvas, (x - 2, y - 2), (x + w + 2, y + h + 2), ui.ACCENT2, radius=9, thickness=2)

        ui.card(canvas, (800, 572, 360, 82), None)
        ui.put_text(canvas, "Tip: keep the whole hand in view, good light,", (816, 603), 0.45, ui.MUTED, 1)
        ui.put_text(canvas, "plain background. Record from a few distances.", (816, 625), 0.45, ui.MUTED, 1)

        ui.put_text_centered(canvas, "SPACE record/stop   |   N next   |   P previous   |   R redo sign   |   Q quit",
                             (W // 2, 682), 0.5, ui.MUTED, 1)

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            if state == "idle":
                if counts[sign] >= target:
                    toast = ("Already complete - press R to redo this sign", now + 2.5, ui.WARN)
                else:
                    state, countdown_end = "countdown", now + config.COUNTDOWN_SECONDS
            elif state == "recording":
                stop_recording(f"Saved {counts[sign]} samples for '{sign}'")
            else:
                state = "idle"
        elif key == ord("n") and state == "idle":
            idx = min(idx + 1, len(signs) - 1)
        elif key == ord("p") and state == "idle":
            idx = max(idx - 1, 0)
        elif key == ord("r") and state == "idle":
            delete_label(sign)
            counts[sign] = 0
            toast = (f"Cleared '{sign}' - ready to re-record", now + 2.0, ui.WARN)

    stop_recording()
    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"[SignSpeak] Dataset saved to {config.DATA_FILE}  ({sum(counts.values())} samples)")


if __name__ == "__main__":
    main()
