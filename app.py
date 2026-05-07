import cv2
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# --- 1. SETUP AND OPTIMIZATION ---

pytorch_model = 'best.pt'
openvino_model = 'best_openvino_model/'

# Automatically optimize the YOLO26 model for Intel hardware on the first run
if not os.path.exists(openvino_model):
    if not os.path.exists(pytorch_model):
        print(f"ERROR: Cannot find {pytorch_model}. Make sure it is in this folder.")
        exit()
        
    print("First run detected. Converting YOLO26 model to Intel OpenVINO format...")
    temp_model = YOLO(pytorch_model)
    # YOLO26's DFL removal makes this export highly efficient
    temp_model.export(format='openvino', imgsz=640)
    print("Optimization complete!")

# Load the optimized Intel model
print("Loading Pothole Detection Model...")
model = YOLO(openvino_model)

# --- 2. GUI SETUP ---

# Initialize Tkinter but hide the main, empty window
# We only want to use its native file dialog box
root = tk.Tk()
root.withdraw() 
# Keeps the dialog window on top of other apps
root.attributes('-topmost', True) 

# --- 3. MAIN EXECUTION LOOP ---

while True:
    print("\n" + "="*30)
    print("POTHOLE DETECTION SYSTEM")
    print("="*30)
    user_input = input("Press 'Enter' to upload an image, or type 'q' to quit: ")
    
    if user_input.lower() == 'q':
        break
        
    print("Opening file explorer...")
    
    # Open the native OS file picker
    file_path = filedialog.askopenfilename(
        title="Select Road Image to Test",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
    )
    
    # If the user selected a file (and didn't hit cancel)
    if file_path:
        filename = os.path.basename(file_path)
        print(f"\nProcessing: {filename}")
        
        # Load the image using OpenCV
        image = cv2.imread(file_path)
        if image is None:
            print("ERROR: Could not read image file. It might be corrupted.")
            continue
            
        # Run the YOLO26 detection
        # conf=0.25 means it will draw boxes for anything it is 25%+ sure is a pothole
        results = model(image, conf=0.25)
        
        # Plot the bounding boxes onto the image
        annotated_image = results[0].plot()
        
        # Display the result in a new window
        window_name = f"Result: {filename} (Press ANY KEY on your keyboard to close)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, annotated_image)
        
        print("Detection complete. Look at the pop-up window.")
        print("Press ANY KEY while the image window is active to close it.")
        
        # Pause the script until the user presses a key
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
        
    else:
        print("\nNo file was selected.")

# Clean up before shutting down
root.destroy()
print("\nSystem shut down successfully.")
