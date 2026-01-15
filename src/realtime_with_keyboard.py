# realtime_with_keyboard_ui.py
# Realtime ASL letters detector with boxes + confidences + keyboard controls
# Save and run: python realtime_with_keyboard_ui.py

import cv2, time, yaml, json
from pathlib import Path
from collections import deque, Counter
from ultralytics import YOLO

# ---------- CONFIG ----------
MODEL = "runs/detect/letters_run/weights/best.pt"  # change if different
DEVICE = 0             # 0 for GPU # or "cpu"
IMG_SZ = 640

# detection -> commit (auto append letters)
WINDOW_SECONDS = 1.0       # consensus window
COMMIT_THRESHOLD = 5       # number of detections in window to commit a letter
COOLDOWN = 1.0             # seconds after a commit before next auto commit
DETECT_CONF = 0.25         # per-box confidence for counting toward consensus

# UI tuning
BOX_COLOR = (200, 120, 20)     # blue-ish for boxes
TEXT_COLOR = (0, 200, 0)       # green for Live text
DBG_COLOR = (220,220,0)        # yellow for debug text

# ----------------------------

# load class names
yaml_p = Path("data/letters.yaml")
if yaml_p.exists():
    data = yaml.safe_load(yaml_p.read_text())
    NAMES = data.get("names", [])
else:
    # fallback to classes.json order if present
    cj = Path("classes.json")
    if cj.exists():
        j = json.loads(cj.read_text())
        # classes.json may be name->id mapping; we need names ordered by id
        # attempt to invert mapping
        inv = {v:k for k,v in j.items()}
        NAMES = [inv[i] for i in range(max(inv.keys())+1)]
    else:
        # last fallback A..Z
        NAMES = [chr(ord("A")+i).lower() for i in range(26)]

NAMES = [str(x) for x in NAMES]
print("Loaded class names (first few):", NAMES[:8])

model = YOLO(MODEL)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

# detection history: elements are (label, timestamp)
history = deque(maxlen=1000)
last_commit_time = 0.0
COMPOSED = ""

def add_detection(label):
    history.append((label, time.time()))

def window_counts(window_seconds):
    cutoff = time.time() - window_seconds
    return Counter([lbl for (lbl,ts) in history if ts >= cutoff])

def commit_letter(letter):
    global COMPOSED, last_commit_time
    COMPOSED += letter
    last_commit_time = time.time()
    print("[AUTO COMMIT] ", letter, " -> ", COMPOSED)

def draw_box_with_label(img, box, label, conf, color):
    # box: [x1,y1,x2,y2] may be floats
    x1,y1,x2,y2 = [int(v) for v in box]
    cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
    txt = f"{label} {conf:.2f}"
    # text background
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, txt, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

print("Ready. Controls: SPACE (space), BACKSPACE/DEL (delete), ENTER (newline), q (quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # predict
    results = model.predict(source=frame, imgsz=IMG_SZ, conf=0.10, device=DEVICE, verbose=False)
    r = results[0]

    raw_boxes = []
    raw_labels = []
    raw_confs = []

    # extract boxes, classes, confs (work with both tensor and list shapes)
    if hasattr(r, "boxes") and len(r.boxes):
        try:
            # prefer batches where xyxy, cls, conf are tensors
            arr = r.boxes.xyxy.cpu().numpy()        # shape (N,4)
            cls_arr = r.boxes.cls.cpu().numpy()     # shape (N,)
            conf_arr = r.boxes.conf.cpu().numpy()   # shape (N,)
            for i in range(len(arr)):
                raw_boxes.append(arr[i].tolist())
                cid = int(cls_arr[i])
                raw_labels.append(NAMES[cid] if cid < len(NAMES) else str(cid))
                raw_confs.append(float(conf_arr[i]))
        except Exception:
            # fallback iteration
            for b in r.boxes:
                xy = b.xyxy.cpu().numpy()[0].tolist()
                cid = int(float(b.cls.cpu().numpy()[0]))
                confv = float(b.conf.cpu().numpy()[0])
                raw_boxes.append(xy); raw_labels.append(NAMES[cid] if cid < len(NAMES) else str(cid)); raw_confs.append(confv)

    # draw all boxes with label & conf
    disp = frame.copy()
    for box,label,conf in zip(raw_boxes, raw_labels, raw_confs):
        draw_box_with_label(disp, box, label, conf, BOX_COLOR)

    # consider boxes for letter consensus: letter = single alpha char
    for box,label,conf in zip(raw_boxes, raw_labels, raw_confs):
        lname = label.lower()
        if len(lname) == 1 and lname.isalpha() and conf >= DETECT_CONF:
            # count this detection toward consensus
            add_detection(lname.upper())

    # consensus check & auto-commit
    counts = window_counts(WINDOW_SECONDS)
    if counts:
        top, ct = counts.most_common(1)[0]
        now = time.time()
        if ct >= COMMIT_THRESHOLD and (now - last_commit_time) >= COOLDOWN:
            # avoid repeating same char immediately
            last_char = COMPOSED[-1] if len(COMPOSED) else None
            if last_char != top:
                commit_letter(top)
                history.clear()

    # UI overlays
    #  - Live text
    y0 = 30
    for line in COMPOSED.split("\n")[-3:]:  # show last 3 lines if multiline
        cv2.putText(disp, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 1.0, TEXT_COLOR, 2)
        y0 += 36

    # debug: show consensus counts top 3
    cc = counts.most_common(4)
    y = 150
    cv2.putText(disp, "Consensus (recent):", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, DBG_COLOR, 1)
    for lbl,cnt in cc:
        y += 22
        cv2.putText(disp, f"{lbl}:{cnt}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, DBG_COLOR, 2)

    # controls hint
    h = 20
    cv2.putText(disp, "Keys: SPACE=space  BACKSPACE/DEL=delete  ENTER=newline  Q=quit", (10, disp.shape[0]-h), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

    cv2.imshow("ASL Realtime (boxes + keyboard)", disp)

    key = cv2.waitKey(1) & 0xFF

    # keyboard actions
    if key == ord("q"):
        break
    # spacebar ascii 32
    if key == 32:
        COMPOSED += " "
        print("[KEY] SPACE ->", COMPOSED)
        last_commit_time = time.time()
        history.clear()
    # backspace is usually 8; sometimes 127 depending on OS
    if key == 8 or key == 127:
        COMPOSED = COMPOSED[:-1]
        print("[KEY] DELETE ->", COMPOSED)
        last_commit_time = time.time()
        history.clear()
    # enter (carriage return) typically 13
    if key == 13 or key == 10:
        COMPOSED += "\n"
        print("[KEY] ENTER -> newline")
        last_commit_time = time.time()
        history.clear()

cap.release()
cv2.destroyAllWindows()
