import json
import os
import string
from collections import defaultdict, deque

import cv2
import numpy as np
import mediapipe as mp

# ========== WHERE TO SAVE/LOAD ==========
BASE_DIR = r"yourpath"
os.makedirs(BASE_DIR, exist_ok=True)
REFS_PATH = os.path.join(BASE_DIR, "refs.json")
TRANSCRIPT_PATH = os.path.join(BASE_DIR, "transcript.txt")
# ========================================

mp_hands = mp.solutions.hands

# -------- tuning ----------
PRED_SMOOTH_FRAMES = 9        # how many frames to smooth prediction
MIN_SAMPLES_TO_TRUST = 2      # per letter before usable
CONFIDENCE_THRESHOLD = 0.55   # lower = easier to accept a prediction
AUTO_HOLD_FRAMES = 8          # frames the same letter must persist to auto-append
AUTO_COOLDOWN_FRAMES = 8      # frames to wait after appending one letter
FONT = cv2.FONT_HERSHEY_SIMPLEX
# --------------------------

# --------- landmark utilities ----------
def landmarks_to_vec(hand_landmarks):
    """Flatten 21 (x,y) -> (42,), normalized: centered on wrist + scale-invariant."""
    pts = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark], dtype=np.float32)  # (21,2)
    pts -= pts[0]  # center on wrist (index 0)
    scale = np.linalg.norm(pts, axis=1).max()
    if scale < 1e-6:
        scale = 1.0
    pts /= scale
    return pts.flatten()

def dist(a, b):
    return float(np.linalg.norm(a - b))

# --------- refs load/save ----------
def load_refs(path):
    if not os.path.exists(path):
        return defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    refs = defaultdict(list)
    for k, lst in data.items():
        refs[k] = [np.array(v, dtype=np.float32) for v in lst]
    return refs

def save_refs(path, refs):
    data = {k: [v.tolist() for v in lst] for k, lst in refs.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

# --------- classifier ----------
def classify(vec, refs):
    best_label, best_d = None, float("inf")
    for label, samples in refs.items():
        if len(samples) < MIN_SAMPLES_TO_TRUST:
            continue
        d = min(dist(vec, s) for s in samples)
        if d < best_d:
            best_label, best_d = label, d
    if best_label is None:
        return None, float("inf"), 0
    if best_d > CONFIDENCE_THRESHOLD:
        return None, best_d, len(refs.get(best_label, []))
    return best_label, best_d, len(refs.get(best_label, []))

# --------- ui helpers ----------
def draw_panel(frame, x, y, w, h, alpha=0.35):
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_bar(frame, x, y, w, h, val, color_ok=(60, 200, 60), color_bad=(40, 40, 40)):
    val = max(0.0, min(1.0, val))
    filled = int(w * val)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color_bad, -1)
    cv2.rectangle(frame, (x, y), (x + filled, y + h), color_ok, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (220, 220, 220), 1)

def get_current_word(text_chars):
    s = "".join(text_chars)
    parts = s.split(" ")
    return parts[-1] if parts else ""

def draw_hints(frame, refs_loaded, auto_on):
    lines = [
        "Controls:",
        "  A–Z: record sample for that letter (3–10 per letter recommended)",
        "  S: append current letter    SPACE: add space",
        "  C: clear all text          B/Backspace: delete last char",
        "  W: save refs.json          T: save transcript.txt",
        f"  TAB: toggle auto-append  [{'ON' if auto_on else 'OFF'}]",
        f"Refs: {REFS_PATH}",
    ]
    y = 90
    for t in lines:
        cv2.putText(frame, t, (20, y), FONT, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
        y += 24
    cv2.putText(frame, "Letters with samples: " + ", ".join(sorted([k for k in refs_loaded if refs_loaded[k]])),
                (20, y), FONT, 0.6, (200, 210, 255), 1, cv2.LINE_AA)

def main():
    refs = load_refs(REFS_PATH)
    print(f"[INFO] Using folder: {BASE_DIR}")
    print(f"[INFO] Loaded references for: {sorted([k for k in refs.keys() if refs[k]])}")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    text_buffer = []  # your sentence as a list of chars
    pred_history = deque(maxlen=PRED_SMOOTH_FRAMES)
    last_stable = None
    same_count = 0
    cooldown = 0
    auto_append = True

    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            # background panels
            draw_panel(frame, 0, 0, frame.shape[1], 80, alpha=0.6)          # top bar
            draw_panel(frame, 10, 85, frame.shape[1]-20, 170, alpha=0.35)    # help panel
            draw_panel(frame, frame.shape[1]-300, 270, 280, 230, alpha=0.35) # right status
            draw_panel(frame, 10, frame.shape[0]-110, frame.shape[1]-20, 100, alpha=0.45)  # bottom text panel

            current_pred = None
            best_d = float("inf")

            if result.multi_hand_landmarks:
                hand_lms = result.multi_hand_landmarks[0]
                vec = landmarks_to_vec(hand_lms)
                p, d, _ = classify(vec, refs)
                current_pred, best_d = p, d

                mp_draw.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # prediction smoothing
            pred_history.append(current_pred)
            smoothed = None
            if pred_history:
                vals = [v for v in pred_history if v is not None]
                smoothed = max(vals, key=vals.count) if vals else None

            # big prediction card
            big = smoothed if smoothed is not None else "?"
            cv2.putText(frame, f"Prediction: {big}",
                        (20, 50), FONT, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Prediction: {big}",
                        (20, 50), FONT, 1.1, (20, 20, 20), 1, cv2.LINE_AA)

            # confidence bar (invert distance: smaller d = higher conf)
            conf = 1.0 - min(1.0, best_d / max(1e-6, CONFIDENCE_THRESHOLD))
            cv2.putText(frame, "Confidence", (frame.shape[1]-285, 300),
                        FONT, 0.7, (230, 230, 230), 1, cv2.LINE_AA)
            draw_bar(frame, frame.shape[1]-285, 325, 250, 22, conf)

            # auto-append logic (builds words automatically)
            if cooldown > 0:
                cooldown -= 1

            if smoothed is not None:
                if smoothed == last_stable:
                    same_count += 1
                else:
                    last_stable = smoothed
                    same_count = 1

                if auto_append and cooldown == 0 and same_count >= AUTO_HOLD_FRAMES:
                    text_buffer.append(smoothed)
                    cooldown = AUTO_COOLDOWN_FRAMES
                    same_count = 0  # reset so it requires a fresh hold
            else:
                last_stable = None
                same_count = 0

            # sentence / current word
            sentence = "".join(text_buffer)
            current_word = get_current_word(text_buffer)

            cv2.putText(frame, "Current word:",
                        (20, frame.shape[0]-75), FONT, 0.8, (200, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, current_word if current_word else "-",
                        (230, frame.shape[0]-75), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # full text line
            cv2.putText(frame, "Text:",
                        (20, frame.shape[0]-30), FONT, 0.8, (200, 230, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, sentence if sentence else "(empty)",
                        (110, frame.shape[0]-30), FONT, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            # hints + status
            draw_hints(frame, refs, auto_append)

            # show
            cv2.imshow("Simple ASL (record-your-own)", frame)

            # keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                save_refs(REFS_PATH, refs)
                if sentence:
                    with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
                        f.write(sentence + "\n")
                break

            # record sample for letter (A-Z)
            if key in [ord(c) for c in string.ascii_letters]:
                ch = chr(key).upper()
                if ch in string.ascii_uppercase:
                    if result.multi_hand_landmarks:
                        vec = landmarks_to_vec(result.multi_hand_landmarks[0])
                        refs[ch].append(vec)
                        print(f"[INFO] recorded sample for {ch}. total = {len(refs[ch])}")
                    else:
                        print("[WARN] no hand to record; hold the sign in frame and try again.")

            # append current prediction (manual)
            if key == ord('s'):
                if smoothed is not None:
                    text_buffer.append(smoothed)

            # space & enter → space
            if key == 32 or key == 13:
                if text_buffer and text_buffer[-1] != ' ':
                    text_buffer.append(' ')

            # backspace (8 or 127) and 'b'
            if key in (8, 127) or key == ord('b'):
                if text_buffer:
                    text_buffer.pop()

            # clear
            if key == ord('c'):
                text_buffer.clear()

            # save refs
            if key == ord('w'):
                save_refs(REFS_PATH, refs)
                print(f"[INFO] saved {REFS_PATH}")

            # save transcript
            if key == ord('t'):
                with open(TRANSCRIPT_PATH, "a", encoding="utf-8") as f:
                    f.write("".join(text_buffer) + "\n")
                print(f"[INFO] saved text to {TRANSCRIPT_PATH}")

            # toggle auto-append
            if key == 9:  # TAB
                auto_append = not auto_append
                print(f"[INFO] Auto-append: {auto_append}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
