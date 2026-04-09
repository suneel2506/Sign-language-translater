"""
Gesture Detection Module
========================
Uses the new MediaPipe Tasks API (HandLandmarker) to detect hand landmarks
and classify simple gestures based on finger positions (no ML training needed).

Requires the model file:  hand_landmarker.task
Download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
"""

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

# ── MediaPipe Tasks API imports ──────────────────────────────────────
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarksConnections = mp.tasks.vision.HandLandmarksConnections
RunningMode = mp.tasks.vision.RunningMode
draw_landmarks = mp.tasks.vision.drawing_utils.draw_landmarks
DrawingSpec = mp.tasks.vision.drawing_utils.DrawingSpec

# Path to the .task model file (same directory as this script)
_MODEL_PATH = str(Path(__file__).parent / "hand_landmarker.task")


class GestureDetector:
    """Real-time hand gesture detection using MediaPipe Tasks API."""

    # ── MediaPipe Landmark Indices ──
    THUMB_TIP = 4
    THUMB_IP = 3
    INDEX_TIP = 8
    INDEX_PIP = 6
    MIDDLE_TIP = 12
    MIDDLE_PIP = 10
    RING_TIP = 16
    RING_PIP = 14
    PINKY_TIP = 20
    PINKY_PIP = 18
    WRIST = 0

    # ── Gesture Definitions ──
    # Each gesture maps to a finger-state pattern: [thumb, index, middle, ring, pinky]
    # 1 = finger extended, 0 = finger closed
    GESTURES = {
        "Hello":     [1, 1, 1, 1, 1],  # Open palm – all fingers up
        "Yes":       [1, 0, 0, 0, 0],  # Thumbs up
        "No":        [0, 1, 0, 0, 0],  # Index finger only
        "Thank You": [1, 0, 0, 0, 1],  # Thumb + pinky (shaka / hang loose)
        "Stop":      [0, 0, 0, 0, 0],  # Closed fist
    }

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.5,
    ):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

        # Drawing styles
        self._landmark_style = DrawingSpec(color=(0, 255, 130), thickness=2, circle_radius=3)
        self._connection_style = DrawingSpec(color=(200, 180, 255), thickness=2)

    # ── Private Helpers ──────────────────────────────────────────────

    def _get_finger_states(self, landmarks, handedness_label: str) -> list[int]:
        """Return a list of 5 ints (0/1) indicating which fingers are extended."""
        fingers = []

        # Thumb – uses x-axis comparison (direction depends on handedness)
        if handedness_label == "Right":
            fingers.append(
                1 if landmarks[self.THUMB_TIP].x > landmarks[self.THUMB_IP].x else 0
            )
        else:  # Left hand
            fingers.append(
                1 if landmarks[self.THUMB_TIP].x < landmarks[self.THUMB_IP].x else 0
            )

        # Index → Pinky – uses y-axis (tip above PIP = extended)
        tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        pips = [self.INDEX_PIP, self.MIDDLE_PIP, self.RING_PIP, self.PINKY_PIP]
        for tip, pip_ in zip(tips, pips):
            fingers.append(1 if landmarks[tip].y < landmarks[pip_].y else 0)

        return fingers

    def _match_gesture(self, finger_states: list[int]) -> str | None:
        """Return the gesture name if finger_states matches a known pattern."""
        for name, pattern in self.GESTURES.items():
            if finger_states == pattern:
                return name
        return None

    def _draw_hand_landmarks(self, frame: np.ndarray, landmarks) -> np.ndarray:
        """Draw hand landmarks and connections on the frame manually using OpenCV."""
        h, w, _ = frame.shape

        # Convert normalised landmarks to pixel coordinates
        points = {}
        for idx, lm in enumerate(landmarks):
            px, py = int(lm.x * w), int(lm.y * h)
            points[idx] = (px, py)

        # Draw connections
        for connection in HandLandmarksConnections.HAND_CONNECTIONS:
            start_idx = connection.start
            end_idx = connection.end
            if start_idx in points and end_idx in points:
                cv2.line(frame, points[start_idx], points[end_idx],
                         (200, 180, 255), 2, cv2.LINE_AA)

        # Draw landmark dots
        for idx, (px, py) in points.items():
            cv2.circle(frame, (px, py), 4, (0, 255, 130), -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    # ── Public API ───────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str | None]:
        """
        Process a single BGR frame.

        Returns
        -------
        frame : np.ndarray
            The (flipped) frame with hand-landmarks drawn.
        gesture : str | None
            Recognized gesture name, or None.
        """
        # Mirror the frame so it feels natural (selfie-mode)
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect hand landmarks
        result = self.landmarker.detect(mp_image)

        gesture = None

        if result.hand_landmarks and result.handedness:
            for hand_lms, hand_cls in zip(result.hand_landmarks, result.handedness):
                # Draw landmarks on the BGR frame
                frame = self._draw_hand_landmarks(frame, hand_lms)

                # Determine handedness
                handedness_label = hand_cls[0].category_name  # "Left" or "Right"

                # Detect finger states and match gesture
                finger_states = self._get_finger_states(hand_lms, handedness_label)
                gesture = self._match_gesture(finger_states)

        return frame, gesture

    def release(self):
        """Release MediaPipe resources."""
        self.landmarker.close()
