import os, base64, threading, time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from flask import Flask, request, jsonify, render_template
from pyngrok import ngrok

app = Flask(__name__)

# ── MediaPipe Tasks API setup ──────────────────────────────────────
# Must point to the downloaded face_landmarker.task file
MODEL_PATH = "face_landmarker.task"  # update if using Drive: '/content/drive/MyDrive/folder/face_landmarker.task'

assert os.path.exists(MODEL_PATH), f"❌ Model not found: {os.path.abspath(MODEL_PATH)}"
print(f"✅ Model found ({os.path.getsize(MODEL_PATH):,} bytes)")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1,
    min_face_detection_confidence=0.3,   # lowered for better detection
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

# Create ONE landmarker instance and reuse it (thread-safe with lock)
landmarker = vision.FaceLandmarker.create_from_options(options)
print("✅ FaceLandmarker ready")

# ── Landmark indices ───────────────────────────────────────────────
NOSE_TIP  = 4
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── Tunable constants ──────────────────────────────────────────────
EAR_CLOSED_THRESH = 0.21
BLINK_HOLD_S      = 0.8
BLINK_COOLDOWN_S  = 1.0
DEAD_ZONE         = 0.15
PORT              = 5000

# ── State ──────────────────────────────────────────────────────────
lock = threading.Lock()
blink_start       = None
blink_triggered   = False
blink_cooldown_at = 0.0
calib_done        = False
calib_dir         = None
calib_samples     = {d: [] for d in ['center','left','right','up','down']}
calib_ranges      = {'cx':0.5,'cy':0.5,'left_x':0.38,'right_x':0.62,'up_y':0.42,'down_y':0.58}


# ── EAR computation ────────────────────────────────────────────────
def ear(lm, idx):
    pts = np.array([[lm[i].x, lm[i].y] for i in idx])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2 * C + 1e-6)


# ── Head direction mapping ─────────────────────────────────────────
def head_dir(nx, ny):
    r = calib_ranges
    cx, cy = r["cx"], r["cy"]
    xr = max(r["right_x"] - cx, cx - r["left_x"], 0.01)
    yr = max(r["down_y"]  - cy, cy - r["up_y"],   0.01)
    rx, ry = (nx - cx) / xr, (ny - cy) / yr
    ha = abs(rx) > DEAD_ZONE
    va = abs(ry) > DEAD_ZONE
    if not ha and not va: return "center"
    v = "up"    if va and ry < 0 else "down"  if va else "center"
    h = "left"  if ha and rx < 0 else "right" if ha else "center"
    return f"{v}-{h}"


# ── Core frame processor ───────────────────────────────────────────
# BUG THAT WAS HERE: old code used mp.face_mesh.process(rgb) which is
# the LEGACY API. Since we created a FaceLandmarker with the Tasks API,
# we must call landmarker.detect(mp_image) instead.
def process(bgr):
    global blink_start, blink_triggered, blink_cooldown_at
    now = time.time()

    # Convert BGR (OpenCV) → RGB (MediaPipe requirement)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Wrap in MediaPipe Image object — required by Tasks API
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # ✅ CORRECT: use the Tasks API landmarker, not mp.face_mesh
    with lock:
        result = landmarker.detect(mp_image)

    # No face found
    if not result.face_landmarks:
        with lock:
            blink_start = None
        return {
            "face_detected": False,
            "direction": "center",
            "ear": 0,
            "blink_hold": 0,
            "blink_triggered": False,
            "head_x": 0.5,
            "head_y": 0.5
        }

    # Tasks API returns NormalizedLandmark objects in result.face_landmarks[0]
    lm = result.face_landmarks[0]  # list of NormalizedLandmark

    nx = lm[NOSE_TIP].x
    ny = lm[NOSE_TIP].y
    avg_ear = (ear(lm, LEFT_EYE) + ear(lm, RIGHT_EYE)) / 2

    with lock:
        closed = avg_ear < EAR_CLOSED_THRESH
        fired  = False
        if closed:
            if blink_start is None:
                blink_start = now
            hold = now - blink_start
            if hold >= BLINK_HOLD_S and now >= blink_cooldown_at and not blink_triggered:
                fired = blink_triggered = True
                blink_cooldown_at = now + BLINK_COOLDOWN_S
        else:
            blink_start    = None
            blink_triggered = False
        hold = (now - blink_start) if blink_start else 0.0

    direction = head_dir(nx, ny) if calib_done else "center"

    return {
        "face_detected":   True,
        "direction":       direction,
        "ear":             round(avg_ear, 4),
        "blink_hold":      round(hold, 3),
        "blink_triggered": fired,
        "head_x":          round(nx, 4),
        "head_y":          round(ny, 4)
    }


def decode(url):
    if "," in url:
        url = url.split(",", 1)[1]
    arr = np.frombuffer(base64.b64decode(url), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ── Routes ─────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def proc():
    f = decode(request.json.get("frame", ""))
    if f is None:
        return jsonify({"face_detected": False, "direction": "center",
                        "ear": 0, "blink_hold": 0, "blink_triggered": False,
                        "head_x": 0.5, "head_y": 0.5})
    return jsonify(process(f))

@app.route("/calibrate_step", methods=["POST"])
def calib_step():
    global calib_dir
    calib_dir = request.json.get("step")
    print(f"  Calibrating: {calib_dir}")
    return jsonify({"ok": True})

@app.route("/process_calib", methods=["POST"])
def proc_calib():
    global calib_dir
    d         = request.json
    f         = decode(d.get("frame", ""))
    direction = d.get("dir") or calib_dir
    if f is None or direction not in calib_samples:
        return jsonify({"ok": False})

    rgb      = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    with lock:
        result = landmarker.detect(mp_image)

    if result.face_landmarks:
        n = result.face_landmarks[0][NOSE_TIP]
        calib_samples[direction].append((n.x, n.y))
        print(f"  [{direction}] sample {len(calib_samples[direction])}: ({n.x:.3f}, {n.y:.3f})")

    return jsonify({"ok": True})

@app.route("/calibrate_done", methods=["POST"])
def calib_done_route():
    global calib_done, calib_ranges

    def avg(d, axis):
        s = calib_samples.get(d, [])
        return float(np.mean([x[axis] for x in s])) if s else None

    cx = avg("center", 0) or 0.5
    cy = avg("center", 1) or 0.5
    calib_ranges = {
        "cx":      cx,
        "cy":      cy,
        "left_x":  avg("left",  0) or cx - 0.12,
        "right_x": avg("right", 0) or cx + 0.12,
        "up_y":    avg("up",    1) or cy - 0.08,
        "down_y":  avg("down",  1) or cy + 0.08,
    }
    calib_done = True
    print("✅ Calibration done:", calib_ranges)
    return jsonify({"ok": True, "ranges": calib_ranges})


# ── Start ──────────────────────────────────────────────────────────
ngrok.kill()  # clear any existing tunnels first
ngrok.set_auth_token("YOUR_NGROK_TOKEN_HERE")  # get free at ngrok.com

threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False),
    daemon=True
).start()
time.sleep(2)

url = ngrok.connect(PORT)
print(f"""
{"="*60}
  PUBLIC URL: {url}
{"="*60}
  1. Open the URL in Chrome or Firefox
  2. Allow camera access
  3. Complete 5-step calibration
  4. Use head movement to highlight buttons
  5. Hold blink 0.8s to select
{"="*60}
""")