# Virtual Painter (fluff bluff) ヾ(≧▽≦*)o

A gesture-controlled virtual painting app where you **draw fluffy cloud strokes in mid-air** using your hand.
Built with OpenCV + MediaPipe, enhanced with smoothing, fake 3D depth, glow, motion dynamics, and video recording.

Wave your hands. Paint clouds. Feel powerful.  

---

## What is this? （⊙ｏ⊙）

This project turns your webcam into a **cloud painting canvas**.

You draw by moving your **index finger**, select colors with gestures, add shine and floating effects, and even **record your artwork as a video** — all without touching a mouse or keyboard.

Think:
- Soft, fluffy cloud strokes
- Organic motion
- Gesture-only controls
- Real-time visual feedback

Basically: *hand tracking meets aesthetic chaos*.

---

## Features (+_+)?

- Cloud-like brush strokes (soft, fluffy, layered)
- Motion smoothing (no jittery lines)
- Fake 3D depth + highlights
- Gesture-based modes (no UI clutter)
- Video recording via hand gestures
- Canvas clear & save-ready
- Works in real time with a webcam

---

## Controls →_→ 

```

Controls:

* Index finger           : Draw clouds
* Index + Middle finger  : Select color
* Thumb only             : Make em shine!
* Thumb + Index          : Make em float up!
* 4 fingers              : Start recording
* 5 fingers              : Stop recording
* Press 'c'              : Clear canvas
* Press 'v'              : Quit

```

(Yes, your hands are now the UI.)

---

## Project Structure 

```

.
├── paint.py               # Main application
├── handdetectmodule.py    # Hand tracking logic
├── headers/               # UI header images
├── recordings/            # Saved recordings (auto-created)
├── requirements.txt       # As I learn from past mistakes
├── .gitignore
└── README.md

````

---

## Requirements

- Python 3.10 or 3.11 (important)
- Webcam
- Libraries:
  - opencv-python
  - mediapipe
  - numpy

> Note: Python 3.10 or 3.11 is recommended. MediaPipe may not work correctly on newer Python versions.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows

Install dependencies:

pip install -r requirements.txt
````

---

## How to Run

```bash
python paint.py
```

Make sure:

* Your webcam is connected
* You’re in good lighting (you look cute anyway, yes I CAN SEE YOU!! no, i can't)
* Your hands are visible

---

## Recording Output

When you show **4 fingers**, recording starts.
When you show **5 fingers**, it stops.

Videos are saved automatically in:

```
/recordings
```

Format: `.mp4`

---

## Known Notes

* Performance depends on your camera resolution
* Cloud brush is intentionally heavy for softness
* Best experience in moderate lighting
* Dark backgrounds make clouds pop more

---

## Why this exists

Coz I want it to

Also:

* Gesture-based interfaces are fun
* Creative coding is therapy
* Clouds are elite

---

## Future Ideas (maybe, maybe not)

* Perlin noise cloud turbulence
* Pressure-sensitive strokes
* Background mood modes
* Export PNG artwork
* Multiplayer chaos (lol)

---

## Final words

If this crashes: my bad.
If this works: you're welcome.

Happy painting  ( •̀ ω •́ )✧

Feel free to use this, fork it, maybe give it a star (hihi)

```
