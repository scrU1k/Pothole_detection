import cv2
import time
import os
from ultralytics import YOLO

# --- 1. SETUP ---
openvino_model = 'best_openvino_model/'

# Verify the OpenVINO model exists
if not os.path.exists(openvino_model):
    print("ERROR: Could not find 'best_openvino_model/'. Please run app.py first to optimize the model.")
    exit()

WINDOW_NAME = "Pothole Dashcam Stream"
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

print("Loading Pothole Detection Model (Live Stream Mode)...")
model = YOLO(openvino_model)

# --- 2. CAMERA INITIALIZATION ---
# '0' is usually the default laptop camera. 
# If you plug in a USB dashcam/webcam, you might need to change this to 1 or 2.
cap = cv2.VideoCapture(0) # Change 0 to 1, 2, or 3 for phone feed to appear

if not cap.isOpened():
    print("ERROR: Cannot open camera. Check if it is plugged in or being used by another app (like Zoom).")
    exit()

# Optional: Try to force the camera to a high resolution (like 720p)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n" + "="*40)
print("LIVE STREAM ACTIVE")
print("Press 'q' on your keyboard to quit.")
print("="*40)

# --- 3. LIVE PROCESSING LOOP ---
# Variables to calculate real-time Frames Per Second (FPS)
prev_time = 0

while True:
    # Grab the current frame from the camera
    success, frame = cap.read()
    
    if not success:
        print("Failed to grab frame from camera. Exiting...")
        break

    # Run YOLO detection on the live frame
    # We lower confidence slightly for live video since motion blur can affect accuracy
    results = model(frame, conf=0.25, verbose=False) 

    # Draw the bounding boxes on the frame
    annotated_frame = frame.copy()
    
    # 2. Loop through every single pothole the AI found in this frame
    for box in results[0].boxes:
        # Extract the exact pixel coordinates of the bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Draw a RED rectangle (BGR: 0 Blue, 0 Green, 194 Red)
        # The '3' at the end is the line thickness
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 194), 3)
        
        # Optional: Add custom text right above the box
        cv2.putText(annotated_frame, "POTHOLE", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 194), 2)

    cv2.imshow(WINDOW_NAME, annotated_frame)
    
    # Calculate and display the FPS on the video feed
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    # Draw a black box for the FPS counter background
    cv2.rectangle(annotated_frame, (10, 10), (180, 50), (0, 0, 0), -1)
    # Overlay the FPS text in green
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the live feed
    cv2.imshow("Pothole Dashcam Stream", annotated_frame)

    # Check if key is pressed to quit (wait 1 millisecond)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        print("Shutting down live stream...")
        break

    # Check if the "X" button was clicked
    # getWindowProperty returns -1 or 0 if the window is closed
    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
        print("Window closed by user. Shutting down...")
        break

# --- 4. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print("System Offline.")