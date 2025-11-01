#  face_tracker previous files


# import cv2
# import mediapipe as mp
#
# class FaceTracker:
#     def __init__(self):
#         self.mp_face = mp.solutions.face_mesh
#         self.mp_hands = mp.solutions.hands
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.mp_styles = mp.solutions.drawing_styles
#
#         # Initialize both modules
#         self.face = self.mp_face.FaceMesh(refine_landmarks=True, max_num_faces=1)
#         self.hands = self.mp_hands.Hands(max_num_hands=2)
#
#     def generate_frames(self):
#         cap = cv2.VideoCapture(0)
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break
#
#             # Convert to RGB
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#             # Process face + hands
#             face_results = self.face.process(rgb)
#             hand_results = self.hands.process(rgb)
#
#             # Draw results on original frame
#             frame.flags.writeable = True
#
#             # ---- Draw eyes only (from face landmarks) ----
#             if face_results.multi_face_landmarks:
#                 for face_landmarks in face_results.multi_face_landmarks:
#                     # Eye landmark indexes for left & right eyes (MediaPipe reference)
#                     left_eye_idx = [33, 133, 160, 159, 158, 157, 173]
#                     right_eye_idx = [362, 263, 387, 386, 385, 384, 398]
#
#                     for i in left_eye_idx:
#                         x = int(face_landmarks.landmark[i].x * frame.shape[1])
#                         y = int(face_landmarks.landmark[i].y * frame.shape[0])
#                         cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#
#                     for i in right_eye_idx:
#                         x = int(face_landmarks.landmark[i].x * frame.shape[1])
#                         y = int(face_landmarks.landmark[i].y * frame.shape[0])
#                         cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#
#             # ---- Draw hands skeleton (if visible) ----
#             if hand_results.multi_hand_landmarks:
#                 for hand_landmarks in hand_results.multi_hand_landmarks:
#                     self.mp_drawing.draw_landmarks(
#                         frame,
#                         hand_landmarks,
#                         self.mp_hands.HAND_CONNECTIONS,
#                         self.mp_styles.get_default_hand_landmarks_style(),
#                         self.mp_styles.get_default_hand_connections_style(),
#                     )
#
#             # Encode frame
#             _, buffer = cv2.imencode('.jpg', frame)
#             yield buffer.tobytes()


# eye movement count

# import cv2
# import mediapipe as mp
# import numpy as np
# import time
#
# class FaceTracker:
#     def __init__(self, camera_index=0):
#         self.face_mesh = mp.solutions.face_mesh.FaceMesh(
#             refine_landmarks=True,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5,
#             max_num_faces=1
#         )
#         self.camera_index = camera_index
#
#         # Movement counters
#         self.eye_move_count = 0
#         self.blink_count = 0
#         self.last_blink_time = 0
#         self.prev_eye_center = None
#         self.start_time = time.time()
#
#         # Thresholds (tune as needed)
#         self.movement_threshold = 15   # pixels
#         self.blink_threshold = 0.20    # EAR ratio
#
#     def _eye_aspect_ratio(self, landmarks, eye_indices, w, h):
#         """Calculate Eye Aspect Ratio (EAR) to detect blinks."""
#         # upper & lower eye points for vertical distance
#         top = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices[:3]], axis=0)
#         bottom = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices[3:]], axis=0)
#         left = (landmarks[eye_indices[6]].x * w, landmarks[eye_indices[6]].y * h)
#         right = (landmarks[eye_indices[7]].x * w, landmarks[eye_indices[7]].y * h)
#         # distances
#         vert_dist = np.linalg.norm(np.array(top) - np.array(bottom))
#         hori_dist = np.linalg.norm(np.array(left) - np.array(right))
#         return vert_dist / hori_dist if hori_dist != 0 else 0
#
#     def generate_frames(self):
#         cap = cv2.VideoCapture(self.camera_index)
#         if not cap.isOpened():
#             raise RuntimeError("Webcam not found!")
#
#         while True:
#             success, frame = cap.read()
#             if not success:
#                 break
#
#             frame = cv2.flip(frame, 1)
#             h, w = frame.shape[:2]
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             res = self.face_mesh.process(rgb)
#
#             if res.multi_face_landmarks:
#                 face_landmarks = res.multi_face_landmarks[0].landmark
#
#                 # --- Eye landmark sets ---
#                 left_eye = [33, 160, 158, 133, 153, 144, 173, 157]
#                 right_eye = [362, 385, 387, 263, 373, 380, 398, 384]
#
#                 # Calculate centers
#                 left_center = np.mean([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in left_eye], axis=0)
#                 right_center = np.mean([(face_landmarks[i].x * w, face_landmarks[i].y * h) for i in right_eye], axis=0)
#                 eye_center = ((left_center[0] + right_center[0]) / 2, (left_center[1] + right_center[1]) / 2)
#
#                 # Draw eyes
#                 for idx in left_eye + right_eye:
#                     x, y = int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)
#                     cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#
#                 # --- Eye Movement detection ---
#                 if self.prev_eye_center is not None:
#                     movement = np.linalg.norm(np.array(eye_center) - np.array(self.prev_eye_center))
#                     if movement > self.movement_threshold:
#                         self.eye_move_count += 1
#
#                 self.prev_eye_center = eye_center
#
#                 # --- Blink detection (EAR) ---
#                 left_ear = self._eye_aspect_ratio(face_landmarks, left_eye, w, h)
#                 right_ear = self._eye_aspect_ratio(face_landmarks, right_eye, w, h)
#                 avg_ear = (left_ear + right_ear) / 2
#
#                 if avg_ear < self.blink_threshold:
#                     if time.time() - self.last_blink_time > 0.5:
#                         self.blink_count += 1
#                         self.last_blink_time = time.time()
#
#                 # --- Display info ---
#                 elapsed = int(time.time() - self.start_time)
#                 cv2.putText(frame, f"Eye Movements: {self.eye_move_count}", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#                 cv2.putText(frame, f"Blinks: {self.blink_count}", (10, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
#                 cv2.putText(frame, f"Time: {elapsed}s", (10, 90),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 255), 2)
#
#                 # Status based on movement frequency
#                 status = "FOCUSED" if self.eye_move_count < (elapsed / 10) else "DISTRACTED"
#                 color = (0, 255, 0) if status == "FOCUSED" else (0, 0, 255)
#                 cv2.putText(frame, f"Status: {status}", (10, 120),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#
#             _, buffer = cv2.imencode('.jpg', frame)
#             yield buffer.tobytes()
#
#         cap.release()


# 👁️ Eye movement count,
# ✋ Hand gestures count,
# 🙂 Head movement count,
# ⏱️ Timer start–stop system hoga,