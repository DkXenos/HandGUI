"""
Master Expression Detection App
Features:
- Split-screen expression detector (Webcam | Expression Image)
- Teachable Machine model support (Keras/TensorFlow)
- Monkey-patched DepthwiseConv2D for compatibility
"""
import cv2
import numpy as np
import os
import warnings
import tensorflow as tf

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Fix for Teachable Machine model compatibility
# Monkey-patch DepthwiseConv2D to ignore 'groups' parameter which sometimes causes issues in newer TF versions
_original_depthwise_init = tf.keras.layers.DepthwiseConv2D.__init__

def _patched_depthwise_init(self, *args, **kwargs):
    kwargs.pop('groups', None)
    _original_depthwise_init(self, *args, **kwargs)

tf.keras.layers.DepthwiseConv2D.__init__ = _patched_depthwise_init

# Disable scientific notation
np.set_printoptions(suppress=True)

def load_expression_model(model_path="keras_model.h5"):
    """Load the model with specific compatibility fixes."""
    try:
        model = tf.keras.models.load_model(
            model_path, 
            compile=False,
            safe_mode=False
        )
        print(f"✓ Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nTrying alternative loading with custom scope...")
        try:
            with tf.keras.utils.custom_object_scope({'DepthwiseConv2D': tf.keras.layers.DepthwiseConv2D}):
                model = tf.keras.models.load_model(model_path, compile=False)
            print("✓ Model loaded with custom scope")
            return model
        except Exception as e2:
            print(f"Failed to load model: {e2}")
            return None

def main():
    # 1. Initialization
    model = load_expression_model("keras_model.h5")
    if model is None:
        print("Required model file 'keras_model.h5' not found or failed to load.")
        return

    # Load labels
    try:
        with open("labels.txt", "r") as f:
            class_names = [line.strip().split(' ', 1)[-1] if ' ' in line else line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Required labels file 'labels.txt' not found.")
        return

    # Map label indices to available image files
    image_map = {
        0: "neutral.jpeg",
        1: "smiling.jpg",
        2: "surprised.jpeg",
        3: "thinking.jpg",
        4: "schocked.png"
    }

    # Load and resize expression images
    loaded_images = {}
    for idx, img_path in image_map.items():
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (640, 480))
                loaded_images[idx] = img
                print(f"✓ Loaded {img_path}")
            else:
                print(f"✗ Could not read {img_path}")
        else:
            print(f"✗ Image {img_path} missing")

    # 2. Camera Setup
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not access webcam.")
        return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "="*60)
    print("EXPRESSION DETECTOR - MASTER VERSION")
    print("="*60)
    print("Controls: Press 'q' to Quit")
    print("Detection active...")

    # 3. Main Loop
    current_expression = 0
    frame_count = 0
    prediction_interval = 5  # Optimize by predicting every N frames

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                break
            
            # Prediction Logic
            if frame_count % prediction_interval == 0:
                # Pre-process for model (224x224 RGB normalized)
                resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                input_data = np.asarray(rgb, dtype=np.float32).reshape(1, 224, 224, 3)
                input_data = (input_data / 127.5) - 1
                
                prediction = model.predict(input_data, verbose=0)
                current_expression = np.argmax(prediction)
                confidence = prediction[0][current_expression]
                
                label = class_names[current_expression] if current_expression < len(class_names) else "Unknown"
                print(f"\rExpression: {label:12} | Confidence: {confidence:.2%}", end="", flush=True)

            frame_count += 1

            # GUI Construction
            webcam_display = cv2.resize(frame, (640, 480))
            
            if current_expression in loaded_images:
                expr_display = loaded_images[current_expression].copy()
            else:
                expr_display = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(expr_display, "Image Not Found", (180, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Annotations
            label = class_names[current_expression] if current_expression < len(class_names) else "???"
            cv2.putText(webcam_display, "WEBCAM", (20, 40), 1, 2, (0, 255, 0), 2)
            cv2.putText(expr_display, f"DETECTED: {label}", (20, 40), 1, 2, (0, 255, 0), 2)

            # Combine and Show
            canvas = np.hstack((webcam_display, expr_display))
            cv2.imshow('HandGUI Expression Detector', canvas)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("\nCleaning up...")
        camera.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
