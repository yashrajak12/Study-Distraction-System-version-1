import cv2
import mediapipe as mp
import numpy as np
import time

class StudyTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.start_time = None
        self.running = False

        # Counters
        self.eye_moves = 0
        self.blinks = 0
        self.hand_moves = 0
        self.head_moves = 0

        # For movement comparisons - Pupil tracking
        self.prev_left_pupil = None
        self.prev_right_pupil = None
        self.prev_nose_tip = None  # For head movement
        self.last_blink_time = 0

        # Frame counter for movement detection
        self.frame_count = 0
        self.movement_check_interval = 5  # Check every 5 frames

        # Thresholds (in pixels)
        self.pupil_move_threshold = 4  # Lower = more sensitive
        self.head_move_threshold = 15  # Lower = more sensitive
        self.blink_threshold = 0.20

        # Gesture detection
        self.gesture_start_time = None
        self.last_gesture = None
        self.gesture_hold_duration = 1.0

        self.summary_shown = False
        self.end_time = None

    #     Done Till now

    def _get_pupil_position(self, landmarks, eye_indices, w, h):
        """Get pupil center from iris landmarks (more accurate than eye average)."""
        # MediaPipe provides iris landmarks for better pupil tracking
        # Left iris: 468, 469, 470, 471, 472
        # Right iris: 473, 474, 475, 476, 477

        # Calculate center of eye region
        eye_points = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        center = np.mean(eye_points, axis=0)
        return center

    def _eye_aspect_ratio(self, landmarks, eye_indices, w, h):
        """Calculate Eye Aspect Ratio for blink detection."""
        # Get vertical eye landmarks
        top = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices[:3]], axis=0)
        bottom = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices[3:]], axis=0)
        left = (landmarks[eye_indices[6]].x * w, landmarks[eye_indices[6]].y * h)
        right = (landmarks[eye_indices[7]].x * w, landmarks[eye_indices[7]].y * h)

        vert = np.linalg.norm(np.array(top) - np.array(bottom))
        hori = np.linalg.norm(np.array(left) - np.array(right))

        return vert / hori if hori > 0 else 0

    def _count_extended_fingers(self, hand_landmarks):
        """Count how many fingers are extended."""
        lm = hand_landmarks.landmark  # 21 landmarks

        fingers = {
            'thumb': (4, 3),
            'index': (8, 6),
            'middle': (12, 10),
            'ring': (16, 14),
            'pinky': (20, 18)
        }

        extended_count = 0

        # Check thumb
        if abs(lm[fingers['thumb'][0]].x - lm[fingers['thumb'][1]].x) > 0.05:
            extended_count += 1

        # Check other fingers
        for finger in ['index', 'middle', 'ring', 'pinky']:
            tip_idx, pip_idx = fingers[finger]
            if lm[tip_idx].y < lm[pip_idx].y - 0.02:
                extended_count += 1

        return extended_count

    def _detect_gesture(self, hand_landmarks):
        """Detect open palm or closed fist."""
        extended = self._count_extended_fingers(hand_landmarks)

        if extended >= 4:
            return "open_palm"
        elif extended <= 1:
            return "fist"
        return None

    def _handle_gesture(self, current_gesture):
        """Handle gesture with hold duration."""
        current_time = time.time()

        if current_gesture != self.last_gesture:
            self.gesture_start_time = current_time
            self.last_gesture = current_gesture
            return

        if current_gesture is None:
            self.gesture_start_time = None
            self.last_gesture = None
            return

        if self.gesture_start_time is None:
            self.gesture_start_time = current_time
            return

        hold_time = current_time - self.gesture_start_time

        # Start session
        if current_gesture == "open_palm" and hold_time >= self.gesture_hold_duration:
            if not self.running:
                self.running = True
                self.start_time = time.time()
                self.summary_shown = False

                # Reset everything
                self.eye_moves = 0
                self.blinks = 0
                self.hand_moves = 0
                self.head_moves = 0
                self.prev_left_pupil = None
                self.prev_right_pupil = None
                self.prev_nose_tip = None
                self.frame_count = 0

                print("✅ Session STARTED!")
                self.gesture_start_time = None

        # Stop session
        elif current_gesture == "fist" and hold_time >= self.gesture_hold_duration:
            if self.running:
                self.running = False
                self.end_time = time.time()
                self.summary_shown = True
                print("⏹️ Session STOPPED!")
                self.gesture_start_time = None

    def generate_frames(self):
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("ERROR: Camera nahi khul raha!")
            return

        print("✅ Camera successfully opened!")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Frame nahi mil raha!")
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_res = self.face_mesh.process(rgb)
            hand_res = self.hands.process(rgb)

            # --- Gesture Detection ---
            current_gesture = None

            if hand_res.multi_hand_landmarks:
                for hand in hand_res.multi_hand_landmarks:
                    gesture = self._detect_gesture(hand)
                    if gesture:
                        current_gesture = gesture

                        # Draw hand landmarks
                        for pt in hand.landmark:
                            x, y = int(pt.x * w), int(pt.y * h)
                            cv2.circle(frame, (x, y), 3, (255, 0, 255), -1)

                        # Show gesture
                        if gesture == "open_palm":
                            cv2.putText(frame, "OPEN PALM DETECTED", (10, h - 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            if not self.running:
                                cv2.putText(frame, "Hold to START...", (10, h - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        elif gesture == "fist":
                            cv2.putText(frame, "FIST DETECTED", (10, h - 60),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if self.running:
                                cv2.putText(frame, "Hold to STOP...", (10, h - 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            self._handle_gesture(current_gesture)

            # --- Session Running ---
            if self.running and not self.summary_shown:
                cv2.putText(frame, "SESSION RUNNING", (w - 250, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if face_res.multi_face_landmarks:
                    lm = face_res.multi_face_landmarks[0].landmark

                    # Eye landmark indices
                    left_eye_indices = [33, 160, 158, 133, 153, 144, 173, 157]
                    right_eye_indices = [362, 385, 387, 263, 373, 380, 398, 384]

                    # Get iris/pupil positions (MediaPipe iris landmarks)
                    left_iris = [468, 469, 470, 471, 472]
                    right_iris = [473, 474, 475, 476, 477]

                    # Calculate pupil centers
                    if len(lm) > max(left_iris + right_iris):  # Check if iris landmarks available
                        left_pupil = np.mean([(lm[i].x * w, lm[i].y * h) for i in left_iris], axis=0)
                        right_pupil = np.mean([(lm[i].x * w, lm[i].y * h) for i in right_iris], axis=0)
                    else:
                        # Fallback to eye center
                        left_pupil = self._get_pupil_position(lm, left_eye_indices, w, h)
                        right_pupil = self._get_pupil_position(lm, right_eye_indices, w, h)

                    # Draw eye outline (green)
                    for idx in left_eye_indices + right_eye_indices:
                        x, y = int(lm[idx].x * w), int(lm[idx].y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                    # Draw pupil centers (RED - larger)
                    cv2.circle(frame, (int(left_pupil[0]), int(left_pupil[1])), 5, (0, 0, 255), -1)
                    cv2.circle(frame, (int(right_pupil[0]), int(right_pupil[1])), 5, (0, 0, 255), -1)

                    # ===== EYE MOVEMENT DETECTION =====
                    self.frame_count += 1
                    if self.frame_count % self.movement_check_interval == 0:
                        if self.prev_left_pupil is not None and self.prev_right_pupil is not None:
                            # Calculate movement distance
                            left_dist = np.linalg.norm(np.array(left_pupil) - np.array(self.prev_left_pupil))
                            right_dist = np.linalg.norm(np.array(right_pupil) - np.array(self.prev_right_pupil))
                            avg_dist = (left_dist + right_dist) / 2

                            if avg_dist > self.pupil_move_threshold:
                                self.eye_moves += 1
                                print(f"👁️ Eye movement detected! Distance: {avg_dist:.2f}")

                        self.prev_left_pupil = left_pupil.copy()
                        self.prev_right_pupil = right_pupil.copy()

                    # ===== HEAD MOVEMENT DETECTION =====
                    # Use nose tip for head movement
                    nose_tip = np.array([lm[1].x * w, lm[1].y * h])  # Nose tip landmark

                    # Draw nose tip (BLUE)
                    cv2.circle(frame, (int(nose_tip[0]), int(nose_tip[1])), 5, (255, 0, 0), -1)

                    if self.frame_count % self.movement_check_interval == 0:
                        if self.prev_nose_tip is not None:
                            head_dist = np.linalg.norm(nose_tip - self.prev_nose_tip)

                            if head_dist > self.head_move_threshold:
                                self.head_moves += 1
                                print(f"🗣️ Head movement detected! Distance: {head_dist:.2f}")

                        self.prev_nose_tip = nose_tip.copy()

                    # ===== BLINK DETECTION =====
                    left_ear = self._eye_aspect_ratio(lm, left_eye_indices, w, h)
                    right_ear = self._eye_aspect_ratio(lm, right_eye_indices, w, h)
                    avg_ear = (left_ear + right_ear) / 2

                    if avg_ear < self.blink_threshold and time.time() - self.last_blink_time > 0.3:
                        self.blinks += 1
                        self.last_blink_time = time.time()
                        print(f"👁️ Blink detected! Count: {self.blinks}")

                # Display stats
                elapsed = int(time.time() - self.start_time)
                mins = elapsed // 60
                secs = elapsed % 60

                cv2.putText(frame, f"Time: {mins}m {secs}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Eyes: {self.eye_moves} | Head: {self.head_moves} | Blinks: {self.blinks}",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # --- Summary Screen ---
            elif self.summary_shown:
                total_time = int(self.end_time - self.start_time)
                mins = total_time // 60
                secs = total_time % 60

                # Calculate focus score
                distraction_score = self.eye_moves + self.head_moves
                focus_score = max(0, 100 - distraction_score)

                # Black overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (30, 80), (w - 30, h - 80), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                summary = [
                    "=== SESSION SUMMARY ===",
                    f"Duration: {mins}m {secs}s",
                    f"Eye Movements: {self.eye_moves}",
                    f"Head Movements: {self.head_moves}",
                    f"Blinks: {self.blinks}",
                    f"Focus Score: {focus_score}/100",
                    "",
                    "Show OPEN PALM to restart!"
                ]

                y = 130
                for line in summary:
                    cv2.putText(frame, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 255, 0), 2)
                    y += 45

            # --- Waiting to Start ---
            else:
                cv2.putText(frame, "Show OPEN PALM (5 fingers) to START", (50, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Encode and yield frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        cap.release()