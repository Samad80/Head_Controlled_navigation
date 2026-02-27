"""
Face Detection Diagnostic Script
Run each cell one by one to pinpoint exactly where detection fails.
"""

# ─────────────────────────────────────────────
# CELL 1 — Check versions and imports
# ─────────────────────────────────────────────
import mediapipe as mp
import cv2
import numpy as np
print(f"MediaPipe version : {mp.__version__}")
print(f"OpenCV version    : {cv2.__version__}")
print(f"NumPy version     : {np.__version__}")


# ─────────────────────────────────────────────
# CELL 2 — Verify model file is valid
# ─────────────────────────────────────────────
import os

MODEL_PATH = 'face_landmarker.task'   # ← update if using Drive path

assert os.path.exists(MODEL_PATH), f"❌ Model not found at: {os.path.abspath(MODEL_PATH)}"

size = os.path.getsize(MODEL_PATH)
print(f"File size: {size:,} bytes")

# A valid face_landmarker.task is ~3–5 MB
assert size > 1_000_000, f"❌ File too small ({size} bytes) — likely corrupt or incomplete download"
print("✅ Model file looks valid")


# ─────────────────────────────────────────────
# CELL 3 — Test with a KNOWN GOOD image (no webcam needed)
# Downloads a clear frontal face photo and runs detection on it.
# If this fails, the problem is your model/setup, not your webcam.
# ─────────────────────────────────────────────
import urllib.request

# Download a royalty-free test face image
test_image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg"
# Use a real human face instead:
test_image_url = "https://thispersondoesnotexist.com/"  # AI-generated face, no copyright

try:
    req = urllib.request.Request(test_image_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=5) as response:
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        test_img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    print(f"✅ Test image downloaded: {test_img_bgr.shape}")
except Exception as e:
    print(f"⚠️  Could not download test image: {e}")
    print("Creating a synthetic test instead — upload your own face photo as 'test_face.jpg'")
    test_img_bgr = None


# ─────────────────────────────────────────────
# CELL 4 — Run MediaPipe on the test image
# ─────────────────────────────────────────────
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,  # note: "facial" not "face"
    num_faces=1,
    min_face_detection_confidence=0.3,   # ← lowered for testing
    min_face_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

landmarker = vision.FaceLandmarker.create_from_options(options)
print("✅ FaceLandmarker created successfully")

# If you have a local face image, use it:
# test_img_bgr = cv2.imread('test_face.jpg')

if test_img_bgr is not None:
    # IMPORTANT: MediaPipe needs RGB, OpenCV loads BGR — must convert!
    rgb = cv2.cvtColor(test_img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)

    if result.face_landmarks:
        print(f"✅ Face detected! Found {len(result.face_landmarks)} face(s)")
        print(f"   First face has {len(result.face_landmarks[0])} landmarks")
    else:
        print("❌ No face detected in test image")
        print("   → The model loaded but detection confidence may be too high,")
        print("     or the image format is wrong. See CELL 5 for more checks.")


# ─────────────────────────────────────────────
# CELL 5 — Test with YOUR webcam frame (Colab)
# Captures a single frame from webcam via JavaScript
# ─────────────────────────────────────────────
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import io
from PIL import Image

def capture_webcam_frame():
    """Captures a single frame from the webcam using Colab's JS bridge."""
    js = Javascript('''
        async function capture() {
            const div = document.createElement('div');
            const video = document.createElement('video');
            const canvas = document.createElement('canvas');

            // Request webcam
            const stream = await navigator.mediaDevices.getUserMedia({video: true});
            document.body.appendChild(div);
            div.appendChild(video);
            video.srcObject = stream;
            await video.play();

            // Wait for video to be ready
            await new Promise(r => setTimeout(r, 2000));

            // Capture frame
            canvas.width  = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            // Stop stream
            stream.getTracks().forEach(t => t.stop());
            div.remove();

            return canvas.toDataURL('image/jpeg', 0.9);
        }
        capture();
    ''')
    display(js)
    data_url = eval_js('capture()')
    # Decode base64 image
    header, encoded = data_url.split(',', 1)
    img_bytes = b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    frame_rgb = np.array(pil_img)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    return frame_rgb, frame_bgr

print("📷 Capturing webcam frame — allow camera access when prompted...")

try:
    frame_rgb, frame_bgr = capture_webcam_frame()
    print(f"✅ Frame
