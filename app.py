import cv2
import numpy as np
import pyvirtualcam
from PIL import Image
import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf

_original_depthwise_init = tf.keras.layers.DepthwiseConv2D.__init__

def _patched_depthwise_init(self, *args, **kwargs):
    kwargs.pop('groups', None) 
    _original_depthwise_init(self, *args, **kwargs)

tf.keras.layers.DepthwiseConv2D.__init__ = _patched_depthwise_init

np.set_printoptions(suppress=True)

try:
    model = tf.keras.models.load_model("keras_model.h5", compile=False)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

class_names = []
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

image_map = {
    0: "neutral.jpeg",     
    1: "smiling.jpg",      
    2: "surprised.jpeg",   
    3: "thinking.jpg",     
    4: "schocked.png"      
}

loaded_images = {}
for idx, img_path in image_map.items():
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            # Resize to 640x480
            img = cv2.resize(img, (640, 480))
            loaded_images[idx] = img
        else:
            print(f"Warning: Could not load image {img_path}")
    else:
        print(f"Warning: Image file {img_path} not found")

# Create a default image if none load
if not loaded_images:
    default_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(default_img, "No images loaded", (150, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for i in range(5):
        loaded_images[i] = default_img.copy()

# Open the webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

# Get camera properties
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting virtual camera...")
print("This will create a virtual camera that Discord can use.")
print("Press 'q' to quit")
print("\nExpression mapping:")
for idx, name in enumerate(class_names):
    print(f"  {name} -> {image_map.get(idx, 'N/A')}")

# Initialize with neutral expression
current_expression = 0
frame_count = 0
prediction_interval = 5  # Predict every 5 frames for performance

try:
    # Create virtual camera with pyvirtualcam
    with pyvirtualcam.Camera(width=640, height=480, fps=30) as vcam:
        print(f'\nVirtual camera device: {vcam.device}')
        print('Virtual camera is now active! You can select it in Discord.')
        
        while True:
            # Capture frame-by-frame
            success, image = camera.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Only run prediction every N frames for better performance
            if frame_count % prediction_interval == 0:
                # Prepare image for model
                # Resize to 224x224 for the model
                resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
                
                # Convert BGR to RGB
                rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
                # Make the image a numpy array and reshape it
                image_array = np.asarray(rgb_image, dtype=np.float32).reshape(1, 224, 224, 3)
                
                # Normalize the image array
                image_array = (image_array / 127.5) - 1
                
                # Predict
                prediction = model.predict(image_array, verbose=0)
                index = np.argmax(prediction)
                current_expression = index
                confidence = prediction[0][index]
                
                # Print prediction
                class_name = class_names[index] if index < len(class_names) else "Unknown"
                print(f"Expression: {class_name} (Confidence: {confidence:.2f})")
            
            frame_count += 1
            
            # Get the image corresponding to the detected expression
            display_image = loaded_images.get(current_expression, loaded_images[0])
            
            # Add text overlay showing current expression
            overlay_image = display_image.copy()
            class_name = class_names[current_expression] if current_expression < len(class_names) else "Unknown"
            cv2.putText(overlay_image, f"Expression: {class_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Convert BGR to RGB for virtual camera
            rgb_frame = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            
            # Send to virtual camera
            vcam.send(rgb_frame)
            
            # Also display in a window for preview
            cv2.imshow('Virtual Camera Preview', overlay_image)
            
            # Wait for key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
except Exception as e:
    print(f"Error: {e}")
    print("\nNote: If you see 'No module named pyvirtualcam', install it with:")
    print("  pip install pyvirtualcam")
    print("\nOn macOS, you also need to install OBS and enable OBS Virtual Camera:")
    print("  1. Install OBS: https://obsproject.com/")
    print("  2. Start OBS")
    print("  3. Go to Tools -> Start Virtual Camera")
    print("  4. Then run this script")
finally:
    # Release everything
    camera.release()
    cv2.destroyAllWindows()
    print("\nVirtual camera stopped.")
