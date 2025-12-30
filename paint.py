import cv2
import time
import os
import numpy as np
import handdetectmodule as hdm

RECORD_DIR = "recordings"
os.makedirs(RECORD_DIR, exist_ok=True)

# Cloudy
CLOUD_PUFFS = 4  # puffiness 
CLOUD_SIZE_VARIATION = 0.3  # size variation factor
CLOUD_OPACITY = 0.6  # oppacity of cloud layers

# Animation 
SHINE_DURATION = 2.0  
FLOAT_SPEED = 10
TURBULENCE_STRENGTH = 2  

# Color palette 
COLOR_PALETTE = [
    (180, 150, 255),  # Soft Pink
    (190, 240, 200),  # Soft Green
    (255, 230, 180),  # Soft Blue
    (200, 250, 255),  # Soft Yellow
    (255, 180, 240),  # Soft Purple
    (0, 0, 0)         # Eraser (black)
]


BRUSH_THICKNESS = 25  
ERASER_THICKNESS = 60

SMOOTHING_BUFFER_SIZE = 5  
smoothing_buffer_x = []
smoothing_buffer_y = []

# Animation 
shine_active = False
shine_start_time = 0
float_active = False
float_offset = 0

# Global Variables 
recording = False
video_writer = None
record_start_time = None
color = COLOR_PALETTE[0]
xp, yp = 0, 0

def perlin_noise(x, y, seed=0):
    """Simple Perlin-like noise function for turbulence"""
    # Simplified noise using sine waves only
    noise = (np.sin(x * 0.05 + seed) + np.sin(y * 0.05 + seed * 2)) * 0.5
    return noise

def apply_turbulence(canvas, strength=2):
    """Apply optimized Perlin noise turbulence"""
    h, w = canvas.shape[:2]

    # Skip if canvas is mostly empty
    if np.sum(canvas) < 1000:
        return canvas

    seed = int(time.time() * 10) % 1000

    y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    noise_x = np.sin(x_coords * 0.05 + seed) * strength
    noise_y = np.sin(y_coords * 0.05 + seed * 2) * strength

    map_x = (x_coords + noise_x).astype(np.float32)
    map_y = (y_coords + noise_y).astype(np.float32)

    result = cv2.remap(canvas, map_x, map_y, cv2.INTER_LINEAR)

    return result

def apply_shine(canvas, intensity=0.5):
    """Optimized shine effect"""
    # Skip if canvas is mostly empty
    if np.sum(canvas) < 1000:
        return canvas
    
    shine = canvas.astype(np.float32)
    shine = np.clip(shine * (1 + intensity * 0.3), 0, 255).astype(np.uint8)
    
    # add a few sparkles
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    
    sparkle = np.zeros_like(canvas)
    h, w = canvas.shape[:2]
    
    num_sparkles = 50
    for _ in range(num_sparkles):
        sx = np.random.randint(0, w)
        sy = np.random.randint(0, h)
        if mask[sy, sx] > 0:
            cv2.circle(sparkle, (sx, sy), 1, (255, 255, 200), -1)
    
    result = cv2.addWeighted(shine, 0.8, sparkle, 0.2, 0)
    return result

def apply_float(canvas, offset):
    h, w = canvas.shape[:2]
    
    # upward movement
    M = np.array([[1, 0, 0], [0, 1, -offset]], dtype=np.float32)
    floated = cv2.warpAffine(canvas, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    return floated

def smooth_point(x, y, buffer_x, buffer_y, buffer_size):
    """Apply moving average smoothing to coordinates"""
    buffer_x.append(x)
    buffer_y.append(y)
    
    if len(buffer_x) > buffer_size:
        buffer_x.pop(0)
        buffer_y.pop(0)
    
    avg_x = int(sum(buffer_x) / len(buffer_x))
    avg_y = int(sum(buffer_y) / len(buffer_y))
    
    return avg_x, avg_y

def draw_cloud_stroke(canvas, p1, p2, color, thickness):
    """Optimized cloud-like stroke"""
    r, g, b = color
    
    # Calculate distance
    distance = int(np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2))
    
    if distance < 1:
        distance = 1
  
    num_puffs = max(2, min(distance // 8, 8))  # Cap at 8 puffs
    
    for i in range(num_puffs):
        t = i / max(num_puffs - 1, 1)
        center_x = int(p1[0] + (p2[0] - p1[0]) * t)
        center_y = int(p1[1] + (p2[1] - p1[1]) * t)
        
        for layer in range(2):
            layer_opacity = 0.25 - (layer * 0.08)
            layer_size = thickness + (layer * 8)
            
            overlay = canvas.copy()
            
            for puff in range(5):
                angle = (puff / 4) * 2 * np.pi
                puff_distance = layer_size * 0.3
                
                puff_x = int(center_x + np.cos(angle) * puff_distance)
                puff_y = int(center_y + np.sin(angle) * puff_distance)
                
                size_var = 1 + (np.random.rand() - 0.5) * CLOUD_SIZE_VARIATION
                puff_radius = int(layer_size * size_var * 0.5)
                
                cloud_color = (
                    min(255, int(r + (255 - r) * 0.3)),
                    min(255, int(g + (255 - g) * 0.3)),
                    min(255, int(b + (255 - b) * 0.3))
                )
                
                cv2.circle(overlay, (puff_x, puff_y), puff_radius, cloud_color, -1, cv2.LINE_AA)
            
            cv2.addWeighted(overlay, layer_opacity, canvas, 1 - layer_opacity, 0, canvas)
    
    for i in range(max(1, num_puffs // 3)):
        t = (i * 3) / max(num_puffs - 1, 1)
        high_x = int(p1[0] + (p2[0] - p1[0]) * t)
        high_y = int(p1[1] + (p2[1] - p1[1]) * t)
        cv2.circle(canvas, (high_x - 2, high_y - 2), thickness // 4, (255, 255, 255), -1)

def start_recording(frame):
    """Start video recording"""
    global video_writer, recording, record_start_time
    if recording:
        return
    
    h, w, _ = frame.shape
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(RECORD_DIR, f"drawing_{timestamp}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(filename, fourcc, 30, (w, h))
    recording = True
    record_start_time = time.time()
    print(f"[REC] look in the camera, it is recording!! → {filename}")

def stop_recording():
    """Stop video recording"""
    global video_writer, recording
    if not recording:
        return
    
    video_writer.release()
    video_writer = None
    recording = False
    print("[REC] bye bye!")

def main():
    global color, xp, yp, recording, video_writer, smoothing_buffer_x, smoothing_buffer_y
    global shine_active, shine_start_time, float_active, float_offset
    
    # Load header image
    header_path = os.path.join("headers", "full.png")
    if not os.path.exists(header_path):
        print(f"Warning: Header image not found at {header_path}")
        header = None
    else:
        header = cv2.imread(header_path)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Read first frame to get dimensions
    success, img = cap.read()
    if not success or img is None:
        raise RuntimeError("Failed to grab frame from camera. Please check your webcam. Oops!")
    
    frame_height, frame_width = img.shape[:2]
    
    # Initialize canvas
    imgcanvas = np.zeros((frame_height, frame_width, 3), np.uint8)
    
    # Calculate header height (30% of screen for better drawing area)
    if header is not None:
        header_height = int(frame_height * 0.15)  # Reduced from 0.4 to 0.15
        header = cv2.resize(header, (frame_width, header_height))
    else:
        header_height = 0
    
    # Initialize hand detector
    detector = hdm.Handdetect(detectconfi=0.85)
    
    print("Virtual Painter girlies!!!")
    print("Controls (coz dumb me forgets stuff):")
    print("- Index finger: Draw clouds")
    print("- Index + Middle finger: Select color")
    print("- Thumb only: Make em shine!")
    print("- 3 fingers: Make em float up!")
    print("- 4 fingers: Start recording")
    print("- 5 fingers: Stop recording")
    print("- Press 'c' to clear canvas")
    print("- Press 'v' to quit")
    
    last_time = time.time()
    
    while True:
        # Capture frame
        success, img = cap.read()
        if not success:
            print("Failed to grab frame, oops")
            break
        
        # Calculate delta time for smooth animations
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Flip for mirror effect
        img = cv2.flip(img, 1)
        
        # Detect hands
        img = detector.findhands(img)
        lmlist = detector.findposition(img, False)
        
        if len(lmlist) != 0:
            # Get fingertip positions
            x1, y1 = lmlist[8][1:]  # Index finger
            x2, y2 = lmlist[12][1:]  # Middle finger
            
            fingers = detector.fingersup()
            finger_count = sum(fingers)
            
            # Thumb → SHINE 
            if fingers[0] and not fingers[1] and finger_count == 1:
                if not shine_active:
                    shine_active = True
                    shine_start_time = current_time
                    print("look at the clouds, look how they shine for you....")
                xp, yp = 0, 0
                smoothing_buffer_x.clear()
                smoothing_buffer_y.clear()
            
            # Three fingers up → FLOAT 
            elif finger_count == 3:
                if not float_active:
                    float_active = True
                    float_offset = 0
                    print("Clouds floating up up up up")
                xp, yp = 0, 0
                smoothing_buffer_x.clear()
                smoothing_buffer_y.clear()
            
            # 4 fingers → start recording
            elif finger_count == 4 and not recording:
                start_recording(img)
            
            # 5 fingers → stop recording
            elif finger_count == 5 and recording:
                stop_recording()
            
            # Selection mode: 2 fingers up (middle + index)
            elif fingers[1] and fingers[2]:
                xp, yp = 0, 0
                cv2.rectangle(img, (x1, y1-30), (x2, y2+30), (255, 255, 0), -1)
                
                # Color selection
                if header is not None and y1 < header_height:
                    section_width = frame_width // len(COLOR_PALETTE)
                    for idx, col in enumerate(COLOR_PALETTE):
                        if idx * section_width < x1 < (idx + 1) * section_width:
                            color = col
                            break
                    cv2.rectangle(img, (x1, y1-30), (x2, y2+30), color, -1)
                
                # Reset effects when selecting color
                shine_active = False
                float_active = False
            
            # Draw mode: only index finger up
            elif fingers[1] and not fingers[2] and finger_count == 1:
                # Allow drawing on entire screen (including over header if needed)
                # Apply smoothing to finger position
                smooth_x, smooth_y = smooth_point(x1, y1, smoothing_buffer_x, smoothing_buffer_y, SMOOTHING_BUFFER_SIZE)
                
                cv2.circle(img, (smooth_x, smooth_y), 15, (255, 255, 0), -1)
                
                if xp == 0 and yp == 0:
                    xp, yp = smooth_x, smooth_y
                
                if color == (0, 0, 0):  # Eraser
                    cv2.line(img, (xp, yp), (smooth_x, smooth_y), color, ERASER_THICKNESS)
                    cv2.line(imgcanvas, (xp, yp), (smooth_x, smooth_y), color, ERASER_THICKNESS)
                else:  # Draw with cloud effect
                    draw_cloud_stroke(imgcanvas, (xp, yp), (smooth_x, smooth_y), color, BRUSH_THICKNESS)
                
                xp, yp = smooth_x, smooth_y
            else:
                # Reset smoothing buffer when not drawing
                smoothing_buffer_x.clear()
                smoothing_buffer_y.clear()
            
            # Reset drawing when no fingers up
            if finger_count == 0:
                xp, yp = 0, 0
                smoothing_buffer_x.clear()
                smoothing_buffer_y.clear()
        
        # Apply shine effect if active
        display_canvas = imgcanvas.copy()
        
        if shine_active:
            elapsed = current_time - shine_start_time
            if elapsed < SHINE_DURATION:
                # Simplified pulsing shine
                intensity = 0.3 + 0.1 * np.sin(elapsed * 3)
                display_canvas = apply_shine(display_canvas, intensity)
            else:
                shine_active = False
        
        # Apply float effect if active (limit update rate)
        if float_active:
            float_offset += FLOAT_SPEED * dt
            
            # Only apply turbulence every 3rd frame to reduce lag
            if int(current_time * 10) % 3 == 0:
                display_canvas = apply_turbulence(display_canvas, TURBULENCE_STRENGTH)
            
            # Float upward
            display_canvas = apply_float(display_canvas, int(float_offset))
            
            # Update main canvas with floated version
            imgcanvas = display_canvas.copy()
        
        # Blend canvas with video feed
        imggray = cv2.cvtColor(display_canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(imggray, 60, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        
        img = cv2.bitwise_and(img, imgInv)
        img = cv2.bitwise_or(img, display_canvas)
        
        # Overlay header
        if header is not None:
            img[0:header_height, 0:frame_width] = header
        
        # Show recording indicator
        if recording and video_writer is not None:
            video_writer.write(img)
            cv2.circle(img, (30, 30), 10, (0, 0, 255), -1)
            cv2.putText(img, "REC", (50, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Show effect indicators
        if shine_active:
            cv2.putText(img, "SHINE", (frame_width - 150, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        if float_active:
            cv2.putText(img, "FLOAT", (frame_width - 150, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        
        # Display
        cv2.namedWindow("Virtual Painter", cv2.WINDOW_NORMAL)
        cv2.imshow("Virtual Painter", img)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Exit on 'v' key
        if key == ord('v'):
            if recording:
                stop_recording()
            break
        
        # Clear canvas on 'c' key
        if key == ord('c'):
            imgcanvas = np.zeros((frame_height, frame_width, 3), np.uint8)
            xp, yp = 0, 0
            smoothing_buffer_x.clear()
            smoothing_buffer_y.clear()
            shine_active = False
            float_active = False
            float_offset = 0
            print("Canvas cleared! drawings gone!")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()