"""
SignSpeak - tiny OpenCV UI toolkit.

Gives the OpenCV window a modern dashboard look: gradient background, rounded
cards, progress rings, clickable buttons, glowing hand skeleton.
All colours are BGR tuples (OpenCV convention).
"""
from typing import List, Tuple

import cv2
import numpy as np

from utils import FINGERTIPS, HAND_CONNECTIONS

# ----------------------------------------------------------------- palette
BG_TOP, BG_BOTTOM = (38, 26, 18), (20, 14, 10)
PANEL = (56, 42, 32)
PANEL_LIGHT = (78, 62, 48)
ACCENT = (214, 200, 0)        # teal / cyan
ACCENT2 = (190, 100, 255)     # pink
GOOD = (120, 215, 90)         # green
WARN = (50, 180, 255)         # amber
BAD = (95, 95, 240)           # red
TEXT = (245, 245, 245)
MUTED = (170, 155, 140)

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA


# ----------------------------------------------------------------- text
def text_size(text: str, scale: float = 0.6, thickness: int = 1) -> Tuple[int, int]:
    (w, h), _ = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h


def put_text(img, text, org, scale=0.6, color=TEXT, thickness=1):
    cv2.putText(img, text, org, FONT, scale, color, thickness, AA)


def put_text_centered(img, text, center, scale=0.6, color=TEXT, thickness=1):
    w, h = text_size(text, scale, thickness)
    put_text(img, text, (int(center[0] - w / 2), int(center[1] + h / 2)), scale, color, thickness)


def wrap_text(text: str, max_width: int, scale=0.8, thickness=2) -> List[str]:
    """Word-wrap `text` to `max_width` pixels (long words are split)."""
    lines, current = [], ""
    for word in text.split(" "):
        trial = word if not current else f"{current} {word}"
        if text_size(trial, scale, thickness)[0] <= max_width:
            current = trial
            continue
        if current:
            lines.append(current)
        while text_size(word, scale, thickness)[0] > max_width and len(word) > 1:
            k = len(word)
            while k > 1 and text_size(word[:k], scale, thickness)[0] > max_width:
                k -= 1
            lines.append(word[:k])
            word = word[k:]
        current = word
    lines.append(current)
    return lines


# ----------------------------------------------------------------- shapes
def make_background(w: int, h: int) -> np.ndarray:
    """Vertical gradient canvas."""
    top = np.array(BG_TOP, dtype=np.float32)
    bottom = np.array(BG_BOTTOM, dtype=np.float32)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    column = top * (1 - t) + bottom * t                      # (h, 1, 3)
    return np.repeat(column, w, axis=1).astype(np.uint8)


def rounded_rect(img, p1, p2, color, radius=14, thickness=-1):
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, AA)


def card(img, rect, title=None, color=PANEL):
    x, y, w, h = rect
    rounded_rect(img, (x, y), (x + w, y + h), color, radius=16)
    if title:
        put_text(img, title, (x + 16, y + 26), 0.5, MUTED, 1)


def progress_bar(img, rect, value, color=ACCENT):
    x, y, w, h = rect
    rounded_rect(img, (x, y), (x + w, y + h), PANEL_LIGHT, radius=h // 2)
    fill = int(w * max(0.0, min(1.0, value)))
    if fill > h:
        rounded_rect(img, (x, y), (x + fill, y + h), color, radius=h // 2)


def ring(img, center, radius, value, color=ACCENT, thickness=9):
    """Circular progress indicator, starts at 12 o'clock."""
    cv2.circle(img, center, radius, PANEL_LIGHT, thickness, AA)
    value = max(0.0, min(1.0, value))
    if value > 0.01:
        cv2.ellipse(img, center, (radius, radius), -90, 0, 360 * value, color, thickness, AA)


def chip(img, text, x, y, color=PANEL, text_color=TEXT, scale=0.5, pad=12, h=28, alpha=0.8):
    """Semi-transparent pill label. Returns its width."""
    tw, th = text_size(text, scale)
    w = tw + 2 * pad
    ih, iw = img.shape[:2]
    x2, y2 = min(x + w, iw), min(y + h, ih)
    if x2 <= x or y2 <= y:
        return w
    roi = img[y:y2, x:x2]
    fill = roi.copy()
    rounded_rect(fill, (0, 0), (w - 1, h - 1), color, radius=h // 2)
    cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0, roi)
    put_text(img, text, (x + pad, y + h // 2 + th // 2), scale, text_color, 1)
    return w


# ----------------------------------------------------------------- hand overlay
def draw_hand(img, points):
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, points[a], points[b], (40, 40, 40), 5, AA)      # shadow
        cv2.line(img, points[a], points[b], ACCENT, 2, AA)
    for i, p in enumerate(points):
        if i in FINGERTIPS:
            cv2.circle(img, p, 7, ACCENT2, -1, AA)
            cv2.circle(img, p, 9, TEXT, 1, AA)
        else:
            cv2.circle(img, p, 4, TEXT, -1, AA)


def hand_bbox(points, pad=24, bounds=None):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, y1, x2, y2 = min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad
    if bounds is not None:
        bw, bh = bounds
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(bw - 1, x2), min(bh - 1, y2)
    return x1, y1, x2, y2


def draw_corners(img, box, color=ACCENT, length=26, thickness=3):
    """Sci-fi style corner brackets around a bounding box."""
    x1, y1, x2, y2 = box
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * length, cy), color, thickness, AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * length), color, thickness, AA)


# ----------------------------------------------------------------- layout helpers
def fit_frame(frame, w, h):
    """Centre-crop `frame` to the w:h aspect ratio, then resize."""
    fh, fw = frame.shape[:2]
    target = w / h
    if fw / fh > target:
        new_w = int(fh * target)
        x0 = (fw - new_w) // 2
        frame = frame[:, x0:x0 + new_w]
    else:
        new_h = int(fw / target)
        y0 = (fh - new_h) // 2
        frame = frame[y0:y0 + new_h]
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)


def paste_rounded(canvas, img, x, y, radius=18):
    """Paste `img` onto `canvas` with rounded corners."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    rounded_rect(mask, (0, 0), (w - 1, h - 1), 255, radius=radius)
    roi = canvas[y:y + h, x:x + w]
    roi[mask > 0] = img[mask > 0]


def draw_header(canvas, subtitle, chips=()):
    """App title bar. `chips` is a list of (text, colour) drawn right-aligned."""
    w = canvas.shape[1]
    rounded_rect(canvas, (20, 14), (64, 58), ACCENT, radius=12)
    put_text_centered(canvas, "S", (42, 36), 1.0, BG_BOTTOM, 3)
    put_text(canvas, "SignSpeak", (80, 36), 0.95, TEXT, 2)
    put_text(canvas, subtitle, (80, 54), 0.45, MUTED, 1)
    x = w - 20
    for text, color in reversed(list(chips)):
        cw = text_size(text, 0.5)[0] + 24
        x -= cw
        chip(canvas, text, x, 22, color, BG_BOTTOM, alpha=1.0)
        x -= 8


# ----------------------------------------------------------------- button
class Button:
    def __init__(self, action: str, label: str, rect, color=ACCENT):
        self.action, self.label, self.rect, self.color = action, label, rect, color

    def contains(self, px: int, py: int) -> bool:
        x, y, w, h = self.rect
        return x <= px <= x + w and y <= py <= y + h

    def draw(self, img, hover=False, enabled=True, label=None, active=False):
        x, y, w, h = self.rect
        if not enabled:
            fill, txt = PANEL, MUTED
        elif active:
            fill, txt = self.color, BG_BOTTOM
        else:
            fill = tuple(min(255, c + 25) for c in PANEL_LIGHT) if hover else PANEL_LIGHT
            txt = TEXT
        rounded_rect(img, (x, y), (x + w, y + h), fill, radius=12)
        if enabled and not active:
            rounded_rect(img, (x, y), (x + w, y + h), self.color, radius=12, thickness=1)
        put_text_centered(img, label or self.label, (x + w // 2, y + h // 2), 0.5, txt, 1)
