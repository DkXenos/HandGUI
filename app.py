"""
Simple split-screen expression detector without model loading issues
Uses opencv-based face detection as a fallback while we fix the model
"""
import cv2
import numpy as np
import os

# Map expression indices to image files
image_map = {
    0: "neutral.jpeg",
    1: "surprised.jpeg",
    2: "thinking.jpg",
    3: "smiling.jpg",
    4: "schocked.png"
}

# Load all images
loaded_images = {}
for idx, img_path in image_map.items():
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, (640, 480))
            loaded_images[idx] = img
            print(f"✓ Loaded {img_path}")

# For now, let's use keyboard controls to test the images
print("\n" + "="*60)
print("SPLIT-SCREEN EXPRESSION TESTER")
print("="*60)
print("Left: Your webcam | Right: Expression image")
print("\nKEYBOARD CONTROLS:")
print("  0 = Neutral")
print("  1 = Smiling")
print("  2 = Surprise")
print("  3 = Thinking")
print("  4 = Shocked")
print("  q = Quit\n")

labels = ["0 Neutral", "1 Surprise", "2 Thinking", "3 Smiling", "4 Shocked"]

# Open webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_expression = 0

try:
    while True:
        success, webcam_frame = camera.read()
        if not success:
            break
        
        # Resize webcam
        webcam_display = cv2.resize(webcam_frame, (640, 480))
        
        # Get expression image
        if current_expression in loaded_images:
            expression_image = loaded_images[current_expression].copy()
        else:
            expression_image = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add labels
        cv2.putText(webcam_display, "Your Webcam", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        expression_name = labels[current_expression].split(" ", 1)[1]
        cv2.putText(expression_image, f"Expression: {expression_name}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Create split screen
        split_screen = np.hstack((webcam_display, expression_image))
        
        # Display
        cv2.imshow('Expression Tester - Press 0-4 to change expression', split_screen)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key >= ord('0') and key <= ord('4'):
            current_expression = key - ord('0')
            print(f"Switched to: {labels[current_expression]}")
            
except KeyboardInterrupt:
    print("\nStopped")
finally:
    camera.release()
    cv2.destroyAllWindows()
