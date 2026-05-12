# Pothole_detection
Pothole detection system using YOLOv26 Small model

Features:
1. Fast and lightweight model, especially optimized for Intel CPUs (openvino model).
2. Removing openvino optimization, defaults to NVIDIA CUDA GPUs,can be implemented for hardware using OPNNX model.
3. Can process a batch of images(folder) at once or a single image.
4. Multiple image formats supported such as .jpg, .jpeg, .png, .bmp, .webp
5. Supports live detection by connecting to a dashcam,a smartphone camera (third-party software needed),or the device's webcam.
 
