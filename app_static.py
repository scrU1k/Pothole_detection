import cv2
import os
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

# --- 1. SETUP AND OPTIMIZATION ---

pytorch_model = 'best.pt'
openvino_model = 'best_openvino_model/'

# Automatically optimize the model for Intel hardware if not done yet
if not os.path.exists(openvino_model):
    if not os.path.exists(pytorch_model):
        print(f"ERROR: Cannot find {pytorch_model}. Make sure it is in this folder.")
        exit()
        
    print("First run detected. Converting model to Intel OpenVINO format...")
    temp_model = YOLO(pytorch_model)
    temp_model.export(format='openvino', imgsz=640)
    print("Optimization complete!")

# Load the optimized Intel model
print("Loading Pothole Detection Model...")
model = YOLO(openvino_model)

# --- 2. GUI SETUP ---

# Initialize Tkinter but hide the main window
root = tk.Tk()
root.withdraw() 
root.attributes('-topmost', True) 

# --- 3. MAIN EXECUTION LOOP ---
conf = 0.25     # Confidence is 0.25, i.e it will mark all those potholes its is 25%+ sure about
while True:
    print("\n" + "="*30)
    print("POTHOLE BATCH PROCESSOR")
    print("="*30)
    user_input = input("Press 'Enter' to select a FOLDER of images, 'I' to enter a SINGLE IMAGE or type 'q' to QUIT: ")

    if user_input.lower() == 'q':
        break
    
    # --- OPTION 2: SINGLE IMAGE PROCESSING ---
    elif user_input.lower() == 'i':
        print("\nOpening file explorer for Single Image...")
        
        file_path = filedialog.askopenfilename(
            title="Select a Single Road Image to Test",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if file_path:
            filename = os.path.basename(file_path)
            print(f"Processing: {filename}")
            
            image = cv2.imread(file_path)
            if image is None:
                print("ERROR: Could not read image file. It might be corrupted.")
                continue
                
            # Run YOLO detection for a single image
            results = list(model(image, conf))
            annotated_image = results[0].plot()
            
            # Display popup window
            window_name = f"Result: {filename} (Press ANY KEY to close)"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.imshow(window_name, annotated_image)
            
            print("Detection complete. Look at the pop-up window.")
            print("Press ANY KEY while the image window is active to close it.")
            
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
        else:
            print("No file was selected.")
    
    # --- OPTION 3: FOLDER BATCH PROCESSING ---
    # An empty string '' means the user just pressed the Enter key
    elif user_input == '':
        print("\nOpening file explorer for Folder Batch Processing...")
        
        folder_path = filedialog.askdirectory(
            title="Select Folder Containing Road Images"
        )
        
        if folder_path:
            print(f"Processing all images in: {folder_path}")
            print("Please wait. Bounding boxes are being drawn and saved...")
            
            # Run YOLO batch prediction and save quietly
            results = model.predict(source=folder_path, conf=0.25, save=True)
            
            print("\n" + "*"*40)
            print("BATCH PROCESSING COMPLETE!")
            print("To view your results, open the 'runs/detect/predict' folder in your project directory.")
            print("*"*40)
        else:
            print("No folder was selected.")
            
    # --- CATCH INVALID INPUT ---
    else:
        print("\nInvalid input. Please just press 'Enter', type 'I', or type 'q'.")

# Clean up before shutting down
root.destroy()
print("\nSystem shut down successfully.")