# debug_tune_realtime.py
# Debug & interactive tuner for realtime sign detection filters.
# Run, see raw detections, and adjust thresholds live with keyboard keys.
#
# Keys:
# 1/2: CONF_THRESH -0.05/+0.05
# 3/4: MIN_BOX_AREA -0.001/+0.001
# 5/6: MAX_BOX_AREA -0.01/+0.01
# 7/8: MIN_Y_CENTER -0.02/+0.02
# 9/0: MAX_Y_CENTER -0.02/+0.02
# -/= : REQUIRED_CONSENSUS_LETTER -/+ 1 (for notes)
# s: save screenshot; c: clear composed test string; q: quit
#
import time
from pathlib import Path
from collections import deque, Counter

import cv2
from ultralytics import YOLO
import yaml

# ---------- initial parameters (you will tune interactively) ----------
MODEL = "runs/detect/letters_run/weights/best.pt"  # replace if needed, or "yolov8n.pt"
DEVICE = 0
IMG_SZ = 640

CONF_THRESH = 0.2
MIN_BOX_AREA = 0.001
MAX_BOX_AREA = 0.3
MIN_Y_CENTER = 0.5
MAX_Y_CENTER = 0.9
REQUIRED_CONSENSUS_LETTER = 4
# ----------------------------------------------------------------------
p = Path("data/letters.yaml")
if p.exists():
    names = yaml.safe_load(p.read_text()).get("names")
else:
    names = [chr(ord("A") + i) for i in range(26)] + ["delete", "space"]
def normalize_name(n):
    if not n: return None
    low = n.lower()
    if low in ("delete","space"): return low
    if len(n)==1 and n.isalpha(): return n.upper()
    if n[0].isalpha(): return n[0].upper()
    return n
NAMES = [normalize_name(x) for x in names]

print("Using model:", MODEL)
print("Class names sample:", NAMES[:6], "...", NAMES[-2:])

# load model
model = YOLO(MODEL)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Cannot open camera")

snap_i = 0
composed_test = ""
history = deque(maxlen=500)

def bbox_norm_and_area(box, frame_shape):
    x1,y1,x2,y2 = box
    h,w = frame_shape[:2]
    x1 = max(0,min(w,x1)); x2 = max(0,min(w,x2))
    y1 = max(0,min(h,y1)); y2 = max(0,min(h,y2))
    bw = (x2-x1)/w
    bh = (y2-y1)/h
    xc = ((x1+x2)/2)/w
    yc = ((y1+y2)/2)/h
    area = bw*bh
    return xc,yc,bw,bh,area

def process_frame(frame, conf_thresh):
    res = model.predict(source=frame, imgsz=IMG_SZ, conf=conf_thresh, device=DEVICE, verbose=False)
    r = res[0]
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
    return boxes, cls_ids, confs

print("Debugger started. Press q to quit. Tune values with keys shown in source.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    t0 = time.time()
    boxes, cls_ids, confs = process_frame(frame, CONF_THRESH)
    t1 = time.time()
    accepted = []
    ignored = []
    # Evaluate each detection against filters
    for box, cid, conf in zip(boxes, cls_ids, confs):
        if cid < 0 or cid >= len(NAMES):
            label = f"id{cid}"
        else:
            label = NAMES[cid]
        xc,yc,bw,bh,area = bbox_norm_and_area(box, frame.shape)
        passed = True
        reason = ""
        if area < MIN_BOX_AREA:
            passed = False; reason = f"area<{MIN_BOX_AREA:.4f}"
        if area > MAX_BOX_AREA:
            passed = False; reason = f"area>{MAX_BOX_AREA:.4f}"
        if yc < MIN_Y_CENTER:
            passed = False; reason = f"yc<{MIN_Y_CENTER:.2f}"
        if yc > MAX_Y_CENTER:
            passed = False; reason = f"yc>{MAX_Y_CENTER:.2f}"
        info = dict(label=label, cid=cid, conf=conf, area=area, yc=yc, reason=reason)
        if passed:
            accepted.append((box, info))
        else:
            ignored.append((box, info))
    # add accepted labels to history for quick stats
    now = time.time()
    for _,info in accepted:
        history.append((info['label'], now))

    # Draw boxes: accepted green, ignored red; annotate conf, area, yc
    disp = frame.copy()
    for box, info in ignored:
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,0,255), 1)
        txt = f"{info['label']} {info['conf']:.2f} a={info['area']:.3f} yc={info['yc']:.2f}"
        cv2.putText(disp, txt, (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
    for box, info in accepted:
        x1,y1,x2,y2 = map(int, box)
        cv2.rectangle(disp, (x1,y1), (x2,y2), (0,200,0), 2)
        txt = f"{info['label']} {info['conf']:.2f} a={info['area']:.3f} yc={info['yc']:.2f}"
        cv2.putText(disp, txt, (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2)

    # show UI: current params
    ui_lines = [
        f"CONF_THRESH: {CONF_THRESH:.2f}  (1/2 change)",
        f"MIN_BOX_AREA: {MIN_BOX_AREA:.4f}  (3/4 change)",
        f"MAX_BOX_AREA: {MAX_BOX_AREA:.3f}  (5/6 change)",
        f"MIN_Y_CENTER: {MIN_Y_CENTER:.2f}  (7/8 change)",
        f"MAX_Y_CENTER: {MAX_Y_CENTER:.2f}  (9/0 change)",
        f"REQ_CONS_LETTER: {REQUIRED_CONSENSUS_LETTER}  (-/= change)",
        f"Accepted:{len(accepted)}  Ignored:{len(ignored)}  RawBoxes:{len(boxes)}",
        f"Detect ms: {(t1-t0)*1000:.0f}"
    ]
    y = 20
    for line in ui_lines:
        cv2.putText(disp, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        y += 24

    cv2.imshow("DEBUG TUNE - press q to quit", disp)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("1"):
        CONF_THRESH = max(0.05, CONF_THRESH - 0.05)
    elif key == ord("2"):
        CONF_THRESH = min(0.95, CONF_THRESH + 0.05)
    elif key == ord("3"):
        MIN_BOX_AREA = max(0.000, MIN_BOX_AREA - 0.001)
    elif key == ord("4"):
        MIN_BOX_AREA = MIN_BOX_AREA + 0.001
    elif key == ord("5"):
        MAX_BOX_AREA = max(0.01, MAX_BOX_AREA - 0.01)
    elif key == ord("6"):
        MAX_BOX_AREA = min(0.9, MAX_BOX_AREA + 0.01)
    elif key == ord("7"):
        MIN_Y_CENTER = max(0.0, MIN_Y_CENTER - 0.02)
    elif key == ord("8"):
        MIN_Y_CENTER = min(0.9, MIN_Y_CENTER + 0.02)
    elif key == ord("9"):
        MAX_Y_CENTER = max(0.0, MAX_Y_CENTER - 0.02)
    elif key == ord("0"):
        MAX_Y_CENTER = min(1.0, MAX_Y_CENTER + 0.02)
    elif key == ord("-"):
        REQUIRED_CONSENSUS_LETTER = max(1, REQUIRED_CONSENSUS_LETTER - 1)
    elif key == ord("=") or key == ord("+"):
        REQUIRED_CONSENSUS_LETTER = REQUIRED_CONSENSUS_LETTER + 1
    elif key == ord("s"):
        fname = f"debug_snap_{snap_i}.jpg"
        cv2.imwrite(fname, disp)
        print("Saved", fname)
        snap_i += 1
    elif key == ord("c"):
        composed_test = ""
    # loop continues

cap.release()
cv2.destroyAllWindows()
