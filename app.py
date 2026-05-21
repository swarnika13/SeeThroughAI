import cv2
import torch
import pyttsx3
import threading
import tkinter as tk
from PIL import Image, ImageTk
import queue
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)
model.conf = 0.5  # Confidence threshold

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Store last spoken time for each object to avoid repeating too fast
last_spoken = {}

# TTS function
def speak(text):
    print(f"Speaking: {text}")  # Optional debug print
    engine.say(text)
    engine.runAndWait()

# Start capturing video
cap = cv2.VideoCapture(0)

# GUI Setup
root = tk.Tk()
root.title("SeeThroughAI - Real-Time Object Detection for Blind Users")
label = tk.Label(root)
label.pack()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Object detection function
def detect_objects():
    ret, frame = cap.read()
    if not ret:
        root.after(10, detect_objects)
        return

    results = model(frame)
    annotated_frame = frame.copy()
    now = time.time()

    directions = []
    detected = results.xyxy[0]

    zone_colors = [(0, 255, 0), (255, 255, 0), (0, 0, 255)]  # BGR for Left, Center, Right
    zone_width = frame_width // 3

    # Draw zone lines
    cv2.line(annotated_frame, (zone_width, 0), (zone_width, frame.shape[0]), zone_colors[0], 2)
    cv2.line(annotated_frame, (2 * zone_width, 0), (2 * zone_width, frame.shape[0]), zone_colors[2], 2)

    messages_to_speak = []

    for *xyxy, conf, cls in detected:
        x1, y1, x2, y2 = map(int, xyxy)
        label_name = model.names[int(cls)]
        center_x = (x1 + x2) // 2

        # Determine direction
        if center_x < zone_width:
            direction = "left"
            color = zone_colors[0]
        elif center_x < 2 * zone_width:
            direction = "center"
            color = zone_colors[1]
        else:
            direction = "right"
            color = zone_colors[2]

        directions.append((label_name, direction))

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_frame, f"{label_name} ({direction})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        key = f"{label_name}_{direction}"
        if key not in last_spoken or now - last_spoken[key] > 4:
            messages_to_speak.append(f"{label_name} on your {direction}")
            last_spoken[key] = now

    # Speak the messages sequentially
    for message in messages_to_speak:
        speak(message)
        time.sleep(2)

    # Convert frame to ImageTk format
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Schedule the next frame update
    root.after(5, detect_objects)

# Start the detection loop
detect_objects()

# Start the GUI event loop
root.mainloop()

# Release resources when GUI closes
cap.release()
cv2.destroyAllWindows()
