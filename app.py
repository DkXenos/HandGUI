import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pyautogui
import math
import time

# Disable pyautogui fail-safe
pyautogui.FAILSAFE = False


class HandHUD:
    """Futuristic AR-style Hand Controller with HUD overlay"""
    
    def __init__(self):
        # Initialize hand detector
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        # HUD Colors (Neon Cyan/Blue)
        self.color_primary = (255, 255, 0)  # Cyan (BGR)
        self.color_secondary = (255, 200, 0)  # Light Blue
        self.color_accent = (0, 255, 255)  # Yellow
        
        # Smoothing variables
        self.smooth_factor = 0.5
        self.prev_center = None
        self.prev_rotation = 0
        
        # Mode state
        self.current_mode = 0
        self.mode_names = ["Idle", "Navigation", "Volume", "Brightness", "Alt-Tab", "Keyboard Dial"]
        
        # Pinch state
        self.pinch_threshold = 0.05
        self.last_pinch_time = 0
        self.pinch_cooldown = 0.5
        
        # System control state
        self.last_volume_time = 0
        self.last_brightness_time = 0
        self.volume_cooldown = 0.2
        self.alt_tab_active = False
        
        # Keyboard dial
        self.alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.last_typed_time = 0
        
    def count_fingers(self, hand_landmarks):
        """Count extended fingers"""
        fingers = []
        
        # Thumb (check if tip is further from wrist than IP joint)
        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        thumb_mcp = hand_landmarks[2]
        
        # Thumb extended if tip is further from MCP than IP
        thumb_extended = math.dist([thumb_tip.x, thumb_tip.y], [thumb_mcp.x, thumb_mcp.y]) > \
                        math.dist([thumb_ip.x, thumb_ip.y], [thumb_mcp.x, thumb_mcp.y])
        fingers.append(thumb_extended)
        
        # Other fingers (check if tip is above PIP joint)
        for finger_idx in range(1, 5):
            tip_idx = 4 * finger_idx + 4
            pip_idx = 4 * finger_idx + 2
            
            tip = hand_landmarks[tip_idx]
            pip = hand_landmarks[pip_idx]
            
            # Finger extended if tip is above PIP (lower y value in image coordinates)
            fingers.append(tip.y < pip.y)
        
        return sum(fingers)
    
    def calculate_palm_rotation(self, hand_landmarks):
        """Calculate palm roll angle using Index MCP (5) and Pinky MCP (17)"""
        index_mcp = hand_landmarks[5]
        pinky_mcp = hand_landmarks[17]
        
        # Calculate vector and angle
        dx = pinky_mcp.x - index_mcp.x
        dy = pinky_mcp.y - index_mcp.y
        
        angle = math.degrees(math.atan2(dy, dx))
        
        # Smooth rotation
        if self.prev_rotation is not None:
            angle = self.prev_rotation + self.smooth_factor * (angle - self.prev_rotation)
        self.prev_rotation = angle
        
        return angle
    
    def detect_pinch(self, hand_landmarks, frame_shape):
        """Detect pinch gesture between thumb and index finger"""
        h, w = frame_shape[:2]
        
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        
        # Calculate normalized distance
        dist = math.sqrt((thumb_tip.x - index_tip.x)**2 + (thumb_tip.y - index_tip.y)**2)
        
        return dist < self.pinch_threshold
    
    def get_smooth_center(self, hand_landmarks, frame_shape):
        """Get smoothed center position at Landmark 9 (Middle MCP)"""
        h, w = frame_shape[:2]
        
        center_landmark = hand_landmarks[9]
        current_center = (int(center_landmark.x * w), int(center_landmark.y * h))
        
        if self.prev_center is None:
            self.prev_center = current_center
        
        # Apply smoothing
        smooth_x = int(self.prev_center[0] + self.smooth_factor * (current_center[0] - self.prev_center[0]))
        smooth_y = int(self.prev_center[1] + self.smooth_factor * (current_center[1] - self.prev_center[1]))
        self.prev_center = (smooth_x, smooth_y)
        
        return self.prev_center
    
    def draw_mode_indicator(self, overlay, center, mode, rotation=0):
        """Draw mode-specific geometric shapes"""
        
        if mode == 1:  # Point (Navigation)
            cv2.circle(overlay, center, 15, self.color_primary, -1)
            cv2.circle(overlay, center, 20, self.color_primary, 2)
            
        elif mode == 2:  # Line (Volume)
            length = 80
            angle_rad = math.radians(rotation)
            end_x = int(center[0] + length * math.cos(angle_rad))
            end_y = int(center[1] + length * math.sin(angle_rad))
            cv2.line(overlay, center, (end_x, end_y), self.color_primary, 4)
            cv2.circle(overlay, center, 8, self.color_accent, -1)
            cv2.circle(overlay, (end_x, end_y), 8, self.color_accent, -1)
            
        elif mode == 3:  # Triangle (Brightness)
            size = 60
            pts = self.get_rotated_polygon(center, 3, size, rotation)
            cv2.polylines(overlay, [pts], True, self.color_primary, 3)
            
        elif mode == 4:  # Square (Alt-Tab)
            size = 50
            pts = self.get_rotated_polygon(center, 4, size, rotation)
            cv2.polylines(overlay, [pts], True, self.color_primary, 3)
            
        elif mode == 5:  # Pentagon (Keyboard Dial)
            size = 60
            pts = self.get_rotated_polygon(center, 5, size, rotation)
            cv2.polylines(overlay, [pts], True, self.color_primary, 3)
            
            # Draw letter indicator
            letter_idx = int((rotation % 360) / (360 / 26)) % 26
            letter = self.alphabet[letter_idx]
            cv2.putText(overlay, letter, (center[0] - 15, center[1] + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.color_accent, 3)
    
    def get_rotated_polygon(self, center, sides, radius, rotation):
        """Generate rotated polygon points"""
        pts = []
        angle_step = 360 / sides
        
        for i in range(sides):
            angle = math.radians(rotation + i * angle_step - 90)
            x = int(center[0] + radius * math.cos(angle))
            y = int(center[1] + radius * math.sin(angle))
            pts.append([x, y])
        
        return np.array(pts, np.int32)
    
    def draw_finger_tips(self, overlay, hand_landmarks, frame_shape, finger_count):
        """Draw glowing indicators on fingertips"""
        h, w = frame_shape[:2]
        
        # Finger tip indices: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
        tip_indices = [4, 8, 12, 16, 20]
        
        for i, tip_idx in enumerate(tip_indices[:finger_count]):
            tip = hand_landmarks[tip_idx]
            x, y = int(tip.x * w), int(tip.y * h)
            
            # Glowing effect
            cv2.circle(overlay, (x, y), 12, self.color_primary, -1)
            cv2.circle(overlay, (x, y), 18, self.color_secondary, 2)
    
    def handle_system_actions(self, mode, rotation, pinched):
        """Execute system-level actions based on mode and gestures"""
        current_time = time.time()
        
        if mode == 2:  # Volume Control
            if current_time - self.last_volume_time > self.volume_cooldown:
                # Normalize rotation to 0-360
                norm_rotation = rotation % 360
                
                if pinched:
                    if 45 <= norm_rotation <= 135:
                        pyautogui.press('volumeup')
                        self.last_volume_time = current_time
                    elif 225 <= norm_rotation <= 315:
                        pyautogui.press('volumedown')
                        self.last_volume_time = current_time
                        
        elif mode == 4:  # Alt-Tab Switcher
            if pinched and not self.alt_tab_active:
                pyautogui.keyDown('alt')
                pyautogui.press('tab')
                self.alt_tab_active = True
            elif not pinched and self.alt_tab_active:
                pyautogui.keyUp('alt')
                self.alt_tab_active = False
                
        elif mode == 5:  # Keyboard Dial
            if pinched and current_time - self.last_typed_time > 0.5:
                letter_idx = int((rotation % 360) / (360 / 26)) % 26
                letter = self.alphabet[letter_idx]
                pyautogui.write(letter.lower())
                self.last_typed_time = current_time
    
    def draw_hud_info(self, frame, mode, finger_count, rotation, pinched):
        """Draw HUD information panel"""
        info_y = 30
        
        # Mode info
        mode_text = f"MODE: {self.mode_names[mode]} ({finger_count} fingers)"
        cv2.putText(frame, mode_text, (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_primary, 2)
        
        # Rotation info
        rotation_text = f"Rotation: {int(rotation)}°"
        cv2.putText(frame, rotation_text, (10, info_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_secondary, 2)
        
        # Pinch status
        pinch_text = f"Pinch: {'ACTIVE' if pinched else 'Inactive'}"
        color = (0, 255, 0) if pinched else (128, 128, 128)
        cv2.putText(frame, pinch_text, (10, info_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to Quit", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def process_frame(self, frame):
        """Main processing pipeline for each frame"""
        h, w = frame.shape[:2]
        
        # Create transparent overlay
        overlay = np.zeros_like(frame)
        
        # Convert to RGB and detect hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            
            # Count fingers and determine mode
            finger_count = self.count_fingers(hand_landmarks)
            self.current_mode = finger_count
            
            # Calculate palm rotation
            rotation = self.calculate_palm_rotation(hand_landmarks)
            
            # Get smoothed center position
            center = self.get_smooth_center(hand_landmarks, frame.shape)
            
            # Detect pinch
            pinched = self.detect_pinch(hand_landmarks, frame.shape)
            
            # Draw fingertips
            self.draw_finger_tips(overlay, hand_landmarks, frame.shape, finger_count)
            
            # Draw mode indicator
            self.draw_mode_indicator(overlay, center, finger_count, rotation)
            
            # Handle system actions
            self.handle_system_actions(finger_count, rotation, pinched)
            
            # Draw HUD info
            self.draw_hud_info(frame, finger_count, finger_count, rotation, pinched)
        
        # Blend overlay with frame for glowing effect
        frame = cv2.addWeighted(frame, 1, overlay, 0.6, 0)
        
        return frame
    
    def run(self):
        """Main application loop"""
        cap = cv2.VideoCapture(0)
        
        print("=" * 60)
        print("🚀 FUTURISTIC AR HAND CONTROLLER ACTIVATED")
        print("=" * 60)
        print("\n📋 MODES:")
        print("  1 Finger  → Navigation (Point)")
        print("  2 Fingers → Volume Control (Line)")
        print("  3 Fingers → Brightness Control (Triangle)")
        print("  4 Fingers → Alt-Tab Switcher (Square)")
        print("  5 Fingers → Keyboard Dial (Pentagon)")
        print("\n🎮 CONTROLS:")
        print("  • Rotate palm to control features")
        print("  • Pinch (thumb + index) to activate")
        print("  • Press 'Q' to quit")
        print("=" * 60)
        
        fps_time = time.time()
        fps_counter = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break
            
            # Flip frame for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Calculate FPS
            fps_counter += 1
            if time.time() - fps_time > 1.0:
                fps = fps_counter
                fps_counter = 0
                fps_time = time.time()
                
                # Display FPS
                cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('AR Hand Controller', frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.detector.close()
        print("\n✅ Hand Controller terminated successfully")


if __name__ == "__main__":
    controller = HandHUD()
    controller.run()


