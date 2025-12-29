"""
Test version without virtual camera - just shows the detection in a window
Use this if you don't have OBS installed yet
"""
import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings

# Fix for Teachable Machine model compatibility
import tensorflow as tf

# Monkey-patch DepthwiseConv2D to ignore 'groups' parameter
_original_depthwise_init = tf.keras.layers.DepthwiseConv2D.__init__

def _patched_depthwise_init(self, *args, **kwargs):
    kwargs.pop('groups', None)
    _original_depthwise_init(self, *args, **kwargs)

tf.keras.layers.DepthwiseConv2D.__init__ = _patched_depthwise_init

# Disable scientific notation
np.set_printoptions(suppress=True)

# Load the model with safe_mode disabled for Keras 3
try:
    # Try loading with options to skip strict mode
    model = tf.keras.models.load_model(
        "keras_model.h5", 
        compile=False,
        safe_mode=False
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("\nTrying to load with custom loading...")
    try:
        # Alternative: Load using legacy format
        import tensorflow.keras.models
        with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}):
            model = tensorflow.keras.models.load_model("keras_model.h5", compile=False)
        print("✓ Model loaded with custom scope")
    except Exception as e2:
        print(f"Failed: {e2}")
        print("\nPlease re-export your model from Teachable Machine")
        exit(1)

# Load the labels
class_names = []
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Map label indices to image files
image_map = {
    0: "neutral.jpeg",      # Neutral
    1: "smiling.jpg",       # Smiling
    2: "surprised.jpeg",    # Surprise
    3: "thinking.jpg",      # Thinking
    4: "schocked.png"       # Shocked
}

# Load all images and resize them to 640x480
loaded_images = {}
for idx, img_path in image_map.items():
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            # Resize to 640x480
            img = cv2.resize(img, (640, 480))
            loaded_images[idx] = img
            print(f"✓ Loaded {img_path}")
        else:
            print(f"✗ Could not load image {img_path}")
    else:
        print(f"✗ Image file {img_path} not found")

# Open the webcam
camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

# Get camera properties
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*60)
print("SPLIT-SCREEN EXPRESSION DETECTOR")
print("="*60)
print("Left: Your webcam | Right: Expression image")
print("Press 'q' to quit\n")

# Initialize with neutral expression
current_expression = 0
frame_count = 0
prediction_interval = 5  # Predict every 5 frames

try:
    while True:
        # Capture frame-by-frame
        success, webcam_frame = camera.read()
        if not success:
            print("Failed to grab frame")
            break
        
        # Only run prediction every N frames for better performance
        if frame_count % prediction_interval == 0:
            # Prepare image for model
            resized = cv2.resize(webcam_frame, (224, 224), interpolation=cv2.INTER_AREA)
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image_array = np.asarray(rgb_image, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1
            
            # Predict
            prediction = model.predict(image_array, verbose=0)
            index = np.argmax(prediction)
            current_expression = index
            confidence = prediction[0][index]
            
            # Print prediction
            class_name = class_names[index] if index < len(class_names) else "Unknown"
            print(f"Expression: {class_name:12} Confidence: {confidence:.2%}")
        
        frame_count += 1
        
        # Get the image corresponding to the detected expression
        if current_expression in loaded_images:
            expression_image = loaded_images[current_expression].copy()
        else:
            # Create a blank image if no image available
            expression_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(expression_image, "No image", (250, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Resize webcam frame to match expression image height
        webcam_display = cv2.resize(webcam_frame, (640, 480))
        
        # Add text labels
        class_name = class_names[current_expression] if current_expression < len(class_names) else "Unknown"
        cv2.putText(webcam_display, "Your Webcam", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(expression_image, f"Expression: {class_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Create split screen - side by side
        split_screen = np.hstack((webcam_display, expression_image))
        
        # Display
        cv2.imshow('Expression Detection - Split Screen', split_screen)
        
        # Wait for key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
except KeyboardInterrupt:
    print("\nStopped by user")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
finally:
    camera.release()
    cv2.destroyAllWindows()
    print("\nTest complete!")
