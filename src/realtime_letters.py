
# Realtime sign-to-text with UI styled like your screenshot (Live text large green, cyan consensus, etc.)




import time, yaml, json
from collections import deque, Counter
from pathlib import Path
import cv2
from ultralytics import YOLO

# ========== CONFIG (tweak if needed) ==========
MODEL = "runs/detect/letters_run/weights/best.pt"
DEVICE = 0
IMG_SZ = 640

# consensus params
WINDOW_SECONDS = 1.0
REQUIRED_CONSENSUS_LETTER = 5
COOLDOWN_AFTER_COMMIT = 1.2
DETECT_CONF = 0.25

# filters for ignoring big/face boxes (normalized)
MIN_BOX_AREA = 0.001
MAX_BOX_AREA = 0.40
MIN_Y_CENTER = 0.05
MAX_Y_CENTER = 0.85

# Colors (match your screenshot)
COLOR_LIVE_GREEN = (0, 200, 0)        # Live text
COLOR_CONSENSUS_CYAN = (200, 230, 240) # consensus (cyan-ish)
COLOR_DETECT_MS = (220, 220, 220)     # light detect ms
COLOR_COOLDOWN_RED = (0, 0, 255)      # red cooldown
COLOR_ACCEPT = (0, 200, 0)
COLOR_IGNORE = (0, 0, 255)

# ==============================================

# load class names
yaml_p = Path("data/letters.yaml")
if yaml_p.exists():
    names = yaml.safe_load(yaml_p.read_text()).get("names")
else:
    cj = Path("classes.json")
    if cj.exists():
        j = json.loads(cj.read_text())
        inv = {v:k for k,v in j.items()}
        names = [inv[i] for i in range(max(inv.keys())+1)]
    else:
        names = [chr(ord("A")+i).lower() for i in range(26)]

NAMES = [str(x).lower() for x in names]
print("Loaded class names (sample):", NAMES[:8])

# model and video
model = YOLO(MODEL)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

# state
history = deque(maxlen=1000)  # (label, timestamp)
last_commit_time = 0.0
COMPOSED = ""

def add_detection(label):
    history.append((label, time.time()))

def counts_in_window(window):
    cutoff = time.time() - window
    return Counter([lbl for (lbl,ts) in history if ts >= cutoff])

def commit_letter(letter):
    global COMPOSED, last_commit_time
    COMPOSED += letter
    last_commit_time = time.time()
    print("[COMMIT]", letter, "->", COMPOSED)

def bbox_norm_and_area(box, frame_shape):
    x1,y1,x2,y2 = box
    h,w = frame_shape[:2]
    bw = max(0.0, (x2-x1)/w)
    bh = max(0.0, (y2-y1)/h)
    xc = ((x1+x2)/2)/w
    yc = ((y1+y2)/2)/h
    return xc, yc, bw, bh, bw*bh

print("Ready. Press q to quit. SPACE=space BACKSPACE=delete")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t0 = time.time()
    # run detection with a low-enough conf to inspect many boxes and filter ourselves
    results = model.predict(source=frame, imgsz=IMG_SZ, conf=0.10, device=DEVICE, verbose=False)
    r = results[0]

    boxes = []
    cls_ids = []
    confs = []
    if hasattr(r, "boxes") and len(r.boxes):
        try:
            arr = r.boxes.xyxy.cpu().numpy()
            cls_arr = r.boxes.cls.cpu().numpy()
            conf_arr = r.boxes.conf.cpu().numpy()
            for i in range(len(arr)):
                boxes.append(arr[i].tolist())
                cls_ids.append(int(cls_arr[i]))
                confs.append(float(conf_arr[i]))
        except Exception:
            for b in r.boxes:
                boxes.append(b.xyxy.cpu().numpy()[0].tolist())
                cls_ids.append(int(b.cls.cpu().numpy()[0]))
                confs.append(float(b.conf.cpu().numpy()[0]))

    # filter & decide accepted vs ignored (for UI + consensus)
    accepted_labels = []
    ignored_debug = []
    for box, cid, conf in zip(boxes, cls_ids, confs):
        name = NAMES[cid] if 0 <= cid < len(NAMES) else str(cid)
        lname = name.lower()
        xc,yc,w_rel,h_rel,area = bbox_norm_and_area(box, frame.shape)
        # apply size / y_center filters
        if area < MIN_BOX_AREA or area > MAX_BOX_AREA or yc < MIN_Y_CENTER or yc > MAX_Y_CENTER:
            ignored_debug.append((name, conf, area, yc))
            continue
        # require confidence threshold to count for consensus
        if conf >= DETECT_CONF and len(lname)==1 and lname.isalpha():
            accepted_labels.append(lname.upper())
        else:
            # if detection passes spatial filters but low conf or non-letter, ignore for consensus
            ignored_debug.append((name, conf, area, yc))

    # add accepted to history (counts)
    now = time.time()
    for lbl in accepted_labels:
        history.append((lbl, now))

    # consensus and commit logic
    cnts = counts_in_window(WINDOW_SECONDS)
    top_label, top_count = (None, 0)
    if cnts:
        top_label, top_count = cnts.most_common(1)[0]
    if top_label and (time.time() - last_commit_time) >= COOLDOWN_AFTER_COMMIT:
        if top_count >= REQUIRED_CONSENSUS_LETTER:
            # avoid immediate repeat
            last_char = COMPOSED[-1] if COMPOSED else None
            if last_char != top_label:
                commit_letter(top_label)
                history.clear()

    t1 = time.time()

    # Draw UI 
    disp = frame.copy()
    h,w = disp.shape[:2]

    # draw boxes: accepted green, ignored red (thin)
    for box, cid, conf in zip(boxes, cls_ids, confs):
        name = NAMES[cid] if 0 <= cid < len(NAMES) else str(cid)
        xc,yc,w_rel,h_rel,area = bbox_norm_and_area(box, frame.shape)
        x1,y1,x2,y2 = map(int, box)
        # determine acceptance status
        accepted_flag = False
        # quick check: if this label (char) is in accepted_labels now and conf>=DETECT_CONF -> mark accepted
        if len(name)==1 and name.isalpha() and conf >= DETECT_CONF and (name.upper() in accepted_labels):
            accepted_flag = True
        # color
        col = COLOR_ACCEPT if accepted_flag else COLOR_IGNORE
        thickness = 2 if accepted_flag else 1
        cv2.rectangle(disp, (x1,y1), (x2,y2), col, thickness)
        # label/conf
        cv2.putText(disp, f"{name} {conf:.2f}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2 if accepted_flag else 1)

    # Top-left: large "Live text:" in green (like screenshot)
    cv2.putText(disp, "Live text:", (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.6, COLOR_LIVE_GREEN, 4)
    # below it the composed text (smaller)
    lines = COMPOSED.split("\n")[-3:] if COMPOSED else [""]
    y0 = 34 + 40
    for L in lines:
        cv2.putText(disp, L, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_LIVE_GREEN, 2)
        y0 += 30

    # Consensus (cyan) text
    cv2.putText(disp, f"Consensus(last {WINDOW_SECONDS:.1f}s):", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CONSENSUS_CYAN, 2)
    # show top 3 counts
    ycons = 158
    for lbl, c in cnts.most_common(3):
        cv2.putText(disp, f"{lbl}:{c}", (12, ycons), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_CONSENSUS_CYAN, 2)
        ycons += 26

    # Detect ms (light) and Cooldown (red)
    ms = int((t1 - t0) * 1000)
    cv2.putText(disp, f"Detect ms: {ms}", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_DETECT_MS, 2)
    cooldown_left = max(0.0, COOLDOWN_AFTER_COMMIT - (time.time() - last_commit_time))
    cv2.putText(disp, f"Cooldown: {cooldown_left:.2f}s", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_COOLDOWN_RED, 2)

    # bottom hint
    cv2.putText(disp, "Keys: SPACE=space  BACKSPACE=delete  ENTER=newline  Q=quit", (10, h-16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("Sign-to-Text (filtered)", disp)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == 32:  # spacebar
        COMPOSED += " "
        history.clear()
        last_commit_time = time.time()
    if key == 8 or key == 127:  # delete/backspace
        COMPOSED = COMPOSED[:-1]
        history.clear()
        last_commit_time = time.time()
    if key == 13 or key == 10:  # enter
        COMPOSED += "\n"
        history.clear()
        last_commit_time = time.time()

cap.release()
cv2.destroyAllWindows
