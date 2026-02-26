# 🧠 Head-Controlled Communication App

A web-based assistive communication tool for children with limited mobility. Uses only a webcam — no special hardware required. The child controls the interface entirely through **head movements** and **sustained blinks**.

---

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Setup](#setup)
- [Running the App](#running-the-app)
- [Calibration](#calibration)
- [Using the App](#using-the-app)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Configuration](#configuration)

---

## How It Works

| Input | Action |
|---|---|
| Move head left/right/up/down | Moves the on-screen cursor |
| Hold blink for 0.8 seconds | Clicks the highlighted button |
| Short/normal blinks | Ignored (won't trigger anything) |
| ESC key | Exits the app |

The app uses **MediaPipe Face Mesh** to track 478 facial landmarks in real time. The nose tip landmark is used for cursor position, and Eye Aspect Ratio (EAR) is calculated from 6-point eye contours to detect sustained blinks.

---

## Requirements

### Python
- Python 3.10 or higher

### Libraries
Install all dependencies with:

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
mediapipe==0.10.32
opencv-python
flask
pyngrok
numpy
```

### Model File
Download the MediaPipe face landmark model (required — ~4MB):

```bash
# Option A — wget
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Option B — curl
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Option C — paste this URL directly into your browser
# https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

Place `face_landmarker.task` in the **same folder as `app.py`**.

---

## Setup

### Running Locally (Recommended)

```
your_project_folder/
├── app.py
├── face_landmarker.task        ← downloaded model
├── requirements.txt
└── templates/
    └── index.html
```

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000 in Chrome or Firefox
```

---

### Running in Google Colab

**Step 1 — Upload files to Google Drive**

Upload both files into the same Drive folder:
- `app.py`
- `face_landmarker.task`

**Step 2 — Mount Drive and install dependencies**

```python
from google.colab import drive
drive.mount('/content/drive')

!pip install mediapipe opencv-python flask pyngrok numpy
```

**Step 3 — Update the model path in app.py**

```python
MODEL_PATH = '/content/drive/MyDrive/your_folder_name/face_landmarker.task'
```

**Step 4 — Set your ngrok token**

Get a free token at [ngrok.com](https://ngrok.com) → sign up → Dashboard → Your Authtoken.

```python
ngrok.set_auth_token("your_token_here")
```

**Step 5 — Run**

```python
%run app.py
# Copy the printed PUBLIC URL and open it in your browser
```

---

## Calibration

When the app first loads, it runs a **5-step calibration** to learn the range of the user's head movement. This makes tracking accurate regardless of how the person is positioned in front of the camera.

| Step | Instruction | What it measures |
|---|---|---|
| 1 | Look straight at the screen | Center reference point |
| 2 | Turn head slightly left | Left boundary |
| 3 | Turn head slightly right | Right boundary |
| 4 | Tilt head slightly up | Upper boundary |
| 5 | Tilt head slightly down | Lower boundary |

**Tips for good calibration:**
- Sit at a comfortable, natural distance from the webcam (50–80 cm)
- Use the natural range of motion — don't over-exaggerate movements
- Keep movements small and controlled
- Make sure lighting is even on the face (avoid strong backlight)
- Hold each position steady for 2–3 seconds while samples are collected

---

## Using the App

After calibration, the communication board appears with 6 large buttons:

| Button | Speaks |
|---|---|
| 💧 Water | "Water" |
| 🚽 Bathroom | "Bathroom" |
| ✅ Yes | "Yes" |
| ❌ No | "No" |
| 🤕 Pain | "Pain" |
| 📞 Call Mom | "Call Mom" |

**To select a button:**
1. Move your head to point the cursor at the button — it will highlight
2. Hold your eyes closed for 0.8 seconds
3. The button will flash and the word will be spoken aloud

**After a selection** there is a 1-second cooldown before another selection can be made — this prevents accidental double-triggers.

---

## Troubleshooting

### ❌ "Model not found" error
The `face_landmarker.task` file is missing or in the wrong location. Download it (see [Requirements](#requirements)) and place it in the same folder as `app.py`.

### ❌ Face not detected
- Make sure you are using the **Tasks API** consistently — do not mix with the legacy `mp.face_mesh` API
- Ensure frames are converted **BGR → RGB** before passing to MediaPipe
- Ensure the frame is wrapped in `mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)`
- Try lowering confidence thresholds to `0.3` in the options
- Check lighting — face should be well-lit and clearly visible
- Verify your webcam works with a simple OpenCV test:
  ```python
  import cv2
  cap = cv2.VideoCapture(0)
  ret, frame = cap.read()
  print("Camera OK:", ret, frame.shape)
  cap.release()
  ```

### ❌ Blink detection too sensitive / not sensitive enough
Adjust `EAR_CLOSED_THRESH` in the configuration section:
- **Increase** (e.g. 0.25) if blinks aren't being detected
- **Decrease** (e.g. 0.18) if normal blinks are triggering clicks

### ❌ Cursor moves too fast / too slow
Adjust `HEAD_SENSITIVITY` and `DEAD_ZONE` in the configuration section.

### ❌ ngrok "5 tunnel limit" error
```python
from pyngrok import ngrok
ngrok.kill()   # run this before connecting a new tunnel
```

### ❌ App works but speech doesn't play
Speech is handled in the browser via the **Web Speech API** (`speechSynthesis`). Make sure:
- You are using Chrome or Firefox (not Safari)
- Your system volume is not muted
- The browser tab is not muted

---

## Project Structure

```
project/
├── app.py                  # Flask backend — MediaPipe tracking, blink detection
├── face_landmarker.task    # MediaPipe model file (download separately)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── templates/
    └── index.html          # Frontend — camera capture, UI, calibration flow
```

---

## Configuration

All tunable parameters are at the top of `app.py`:

| Variable | Default | Description |
|---|---|---|
| `EAR_CLOSED_THRESH` | `0.21` | Eye Aspect Ratio below this = eye closed |
| `BLINK_HOLD_S` | `0.8` | Seconds eyes must stay closed to trigger click |
| `BLINK_COOLDOWN_S` | `1.0` | Seconds between allowed selections |
| `DEAD_ZONE` | `0.15` | Fraction of range treated as center (reduces jitter) |
| `PORT` | `5000` | Flask server port |

---

## How the Key Algorithms Work

### Eye Aspect Ratio (EAR)
```
EAR = (||p2−p6|| + ||p3−p5||) / (2 × ||p1−p4||)
```
Six landmarks form a contour around each eye. When the eye is open, the vertical distances (p2–p6, p3–p5) are large relative to the horizontal distance (p1–p4). When closed, EAR drops sharply. Both eyes are averaged to reduce noise.

### Head Tracking
The nose tip landmark (index 4) gives a normalized (x, y) position from 0.0 to 1.0. After calibration, this is mapped against the measured range for each direction. A dead zone in the center prevents small tremors from moving the cursor.

### Calibration Mapping
During calibration, 10–15 samples are averaged for each direction (center, left, right, up, down). These averages become the boundary values. Live nose position is then linearly interpolated against these boundaries to produce cursor coordinates.

---

## License

MIT License — free to use, modify, and distribute.