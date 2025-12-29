"""
Split-screen expression detector with tf-keras for better compatibility
+ MediaPipe hand and face detection with indicators
"""
import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tf_keras as keras
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

np.set_printoptions(suppress=True)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  
    (0, 5), (5, 6), (6, 7), (7, 8),  
    (0, 9), (9, 10), (10, 11), (11, 12),  
    (0, 13), (13, 14), (14, 15), (15, 16),  
    (0, 17), (17, 18), (18, 19), (19, 20),  
    (5, 9), (9, 13), (13, 17)  
]

def draw_hand_landmarks(image, hand_landmarks):
    """Draw hand landmarks on the image using OpenCV"""
    h, w, c = image.shape
    
    
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        start_landmark = hand_landmarks[start_idx]
        end_landmark = hand_landmarks[end_idx]
        
        start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
        end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
        
        cv2.line(image, start_point, end_point, (0, 255, 0), 2)
    
    
    for landmark in hand_landmarks:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
        cv2.circle(image, (x, y), 5, (0, 255, 255), 2)
    
    return image


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
hand_detector = vision.HandLandmarker.create_from_options(options)

print("Loading model...")
try:

    model = keras.models.load_model("keras_model.h5", compile=False)
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"❌ Error loading model: {e}\n")
    print("Let's test the images first with keyboard controls")
    print("Run: python3 simple_test.py")
    exit(1)

class_names = []
with open("labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

image_map = {
    0: "neutral.jpeg",
    1: "surprised.jpeg",
    2: "thinking.jpg",
    3: "smiling.jpg",
    4: "schocked.png"
}

loaded_images = {}
for idx, img_path in image_map.items():
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (640, 480))
            loaded_images[idx] = img
            print(f"✓ Loaded {img_path}")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    print("Error: Could not open webcam")
    exit()

camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\n" + "="*60)
print("SPLIT-SCREEN EXPRESSION DETECTOR + HAND TRACKING")
print("="*60)
print("Left: Your webcam with hand indicators")
print("Right: Expression image")
print("The image will change based on your detected expression!")
print("Press 'q' to quit\n")

current_expression = 0
frame_count = 0
prediction_interval = 5 

try:
    while True:
        success, webcam_frame = camera.read()
        if not success:
            print("Failed to grab frame")
            break
        
    
        if frame_count % prediction_interval == 0:
        
            resized = cv2.resize(webcam_frame, (224, 224), interpolation=cv2.INTER_AREA)
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            image_array = np.asarray(rgb_image, dtype=np.float32).reshape(1, 224, 224, 3)
            image_array = (image_array / 127.5) - 1
            
            
            prediction = model.predict(image_array, verbose=0)
            index = np.argmax(prediction)
            current_expression = index
            confidence = prediction[0][index]
            
            
            class_name = class_names[index] if index < len(class_names) else "Unknown"
            print(f"Expression: {class_name:12} Confidence: {confidence:.2%}")
        
        frame_count += 1
        
        
        if current_expression in loaded_images:
            expression_image = loaded_images[current_expression].copy()
        else:
            expression_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(expression_image, "No image", (250, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        
        webcam_display = cv2.resize(webcam_frame, (640, 480))
        
        
        webcam_display = cv2.flip(webcam_display, 1)
        
        
        rgb_frame = cv2.cvtColor(webcam_display, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        
        try:
            hand_results = hand_detector.detect(mp_image)
            if hand_results.hand_landmarks:
                for hand_landmarks in hand_results.hand_landmarks:
                    webcam_display = draw_hand_landmarks(webcam_display, hand_landmarks)
        except Exception as e:
            pass  
        
        
        class_name = class_names[current_expression] if current_expression < len(class_names) else "Unknown"
        cv2.putText(webcam_display, "Your Webcam", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(expression_image, f"Expression: {class_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        
        split_screen = np.hstack((webcam_display, expression_image))
        
        
        cv2.imshow('Expression Detection - Split Screen', split_screen)
        
        
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
    hand_detector.close()
    cv2.destroyAllWindows()
    print("\nDone!")
