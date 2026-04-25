# ============================================================
# EMMA - Emotional Mental Management Assistant (Unified File)
# Mode A: 
#   python EMOBOT_gpt.py               → Runs camera emotion detection
#   streamlit run EMOBOT_gpt.py -- --chat → Runs Streamlit chatbot
# ============================================================

# -----------------------------
# 1. IMPORTS
# -----------------------------
import cv2
import numpy as np
import threading
import requests
import json
from datetime import datetime
import time
import os
import warnings
from collections import deque
from typing import List, Dict, Tuple, Optional, Any
import queue

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---- Optional Libraries ----
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    HAS_TF = True
except:
    HAS_TF = False

try:
    from deepface import DeepFace
    HAS_DF = True
except:
    HAS_DF = False

try:
    import mediapipe as mp
    HAS_MP = True
except:
    HAS_MP = False
    mp = None

try:
    import pyttsx3
    HAS_TTS = True
except:
    HAS_TTS = False


# -----------------------------
# 2. CONFIG
# -----------------------------
class CONFIG:
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720

    EMOTION_INTERVAL = 0.15
    EMOTION_MIN_FACE_SIZE = 100
    MAX_FACES = 3

    EMOTION_HISTORY = 12
    EMOTION_CONF_THRESHOLD = 0.45

    DEBUG = True
    USE_THREADING = True

    OLLAMA_MODEL = "llama3.2:3b"
    OLLAMA_URL = "http://localhost:11434/api/chat"
    OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

    # UI Colors
    BG = (15, 20, 30)
    EMOTION_COLOR = {
        "happy": (100, 220, 150),
        "sad": (120, 150, 255),
        "angry": (100, 120, 255),
        "surprise": (255, 200, 100),
        "fear": (200, 120, 255),
        "disgust": (150, 200, 100),
        "neutral": (160, 180, 200),
    }

# -----------------------------
# 3. EMOTION STATE FILE
# -----------------------------
EMOTION_JSON = "emotion_state.json"


def write_emotion_state(emotion: str, confidence: float, timestamp=None):
    if timestamp is None:
        timestamp = time.time()
    try:
        with open(EMOTION_JSON, "w") as f:
            json.dump({
                "emotion": emotion,
                "confidence": float(confidence),
                "timestamp": float(timestamp)
            }, f)
    except Exception as e:
        print(f"[WARN] Failed to write emotion_state.json: {e}")


# -----------------------------
# 4. DEBUG LOGGER
# -----------------------------
def debug(msg):
    if CONFIG.DEBUG:
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[{now}] {msg}")
# ============================================================
# PART 2/6 — AdvancedEmotionDetector (Complete)
# ============================================================

class AdvancedEmotionDetector:
    """
    Ensemble emotion detector using DeepFace (if available) and MediaPipe landmarks.
    Returns standardized dict: {'emotion': str, 'confidence': float, 'all_scores': {...}, 'source': str}
    """

    def __init__(self):
        self.frame_counter = 0
        self.emotion_history: Dict[int, deque] = {}
        self.conf_history: Dict[int, deque] = {}
        self.current_emotions: Dict[int, Dict] = {}
        self.last_detect_time: Dict[int, float] = {}

        # Weights when combining models
        self.weights = {
            "deepface": 0.7 if HAS_DF else 0.0,
            "mediapipe": 0.3 if HAS_MP else 0.0,
        }

        # initialize mediapipe face mesh if available
        self.face_mesh = None
        if HAS_MP:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=CONFIG.MAX_FACES,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                debug("MediaPipe FaceMesh initialized")
            except Exception as e:
                debug(f"MediaPipe init failed: {e}")

        # sanity: ensure at least one method weight is non-zero
        if sum(self.weights.values()) == 0:
            # fallback to a basic detector behavior
            self.weights["basic"] = 1.0
            debug("No advanced detectors available - falling back to basic mode")

    # -------------------------
    def preprocess_face(self, face: np.ndarray) -> np.ndarray:
        """Apply CLAHE + denoise to improve model input (non-destructive)."""
        try:
            if face is None or face.size == 0:
                return face
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            stacked = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            blended = cv2.addWeighted(stacked, 0.7, face, 0.3, 0)
            return blended
        except Exception as e:
            debug(f"Preprocess failed: {e}")
            return face

    # -------------------------
    def analyze_emotion(self, face_crop: np.ndarray, face_id: int = 0) -> Dict:
        """
        Top-level API: accepts a face crop (BGR), returns smoothed emotion dict.
        Uses frame skipping for performance.
        """
        try:
            if face_crop is None or face_crop.size == 0:
                return self._get_cached_emotion(face_id)

            h, w = face_crop.shape[:2]
            if h < CONFIG.EMOTION_MIN_FACE_SIZE or w < CONFIG.EMOTION_MIN_FACE_SIZE:
                return self._get_cached_emotion(face_id)

            self.frame_counter += 1
            # skip some frames to save CPU
            if self.frame_counter % 2 != 0:
                return self._get_cached_emotion(face_id)

            pre = self.preprocess_face(face_crop)
            results = []

            # DeepFace
            if HAS_DF and self.weights.get("deepface", 0) > 0:
                df_res = self._detect_deepface(pre)
                if df_res:
                    results.append(("deepface", df_res))

            # MediaPipe
            if HAS_MP and self.weights.get("mediapipe", 0) > 0:
                mp_res = self._detect_mediapipe(pre)
                if mp_res:
                    results.append(("mediapipe", mp_res))

            # If no advanced results, use basic
            if not results:
                results.append(("basic", self._detect_basic(pre)))

            # Combine if multiple
            if len(results) > 1:
                combined = self._ensemble_combine(results)
            else:
                combined = results[0][1]

            # smoothing + caching
            smoothed = self._apply_advanced_smoothing(face_id, combined)
            self.current_emotions[face_id] = smoothed
            self.last_detect_time[face_id] = time.time()
            return smoothed

        except Exception as e:
            debug(f"analyze_emotion error: {e}")
            return self._get_cached_emotion(face_id)

    # -------------------------
    def _detect_deepface(self, face_crop: np.ndarray) -> Optional[Dict]:
        """Run DeepFace emotion analyze if available."""
        try:
            if not HAS_DF:
                return None
            img = cv2.resize(face_crop, (224, 224))
            # DeepFace returns dict or list depending on version/config
            res = DeepFace.analyze(img, actions=["emotion"], enforce_detection=False, silent=True, detector_backend="opencv")
            if isinstance(res, list) and res:
                res = res[0]
            emotions = res.get("emotion", {}) if isinstance(res, dict) else {}
            if not emotions:
                return None
            # normalize
            total = sum(emotions.values()) if sum(emotions.values()) != 0 else 1.0
            normalized = {k.lower(): v / total for k, v in emotions.items()}
            dominant = max(normalized.items(), key=lambda x: x[1])
            return {
                "emotion": dominant[0],
                "confidence": float(min(dominant[1], 0.98)),
                "all_scores": normalized,
                "source": "deepface"
            }
        except Exception as e:
            debug(f"DeepFace detection error: {e}")
            return None

    # -------------------------
    def _detect_mediapipe(self, face_crop: np.ndarray) -> Optional[Dict]:
        """Estimate emotion from facial geometry using MediaPipe landmarks."""
        try:
            if not HAS_MP or self.face_mesh is None:
                return None
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            if not results or not results.multi_face_landmarks:
                return None
            # use first face's landmarks
            lms = results.multi_face_landmarks[0].landmark
            h, w = face_crop.shape[:2]
            features = self._extract_enhanced_features(lms, w, h)
            emo, conf, all_scores = self._advanced_classify(features)
            return {
                "emotion": emo,
                "confidence": conf,
                "all_scores": all_scores,
                "source": "mediapipe"
            }
        except Exception as e:
            debug(f"MediaPipe detect error: {e}")
            return None

    # -------------------------
    def _extract_enhanced_features(self, landmarks, w: int, h: int) -> Dict:
        """Convert normalized landmarks to geometric features used for heuristic classification."""
        try:
            pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
            def safe_indices(idx_list):
                return pts[idx_list] if max(idx_list) < len(pts) else pts[np.arange(len(idx_list))] * 0

            # eyes
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            left_eye = safe_indices(left_eye_idx)
            right_eye = safe_indices(right_eye_idx)
            left_eye_h = max(left_eye[:,1]) - min(left_eye[:,1]) if left_eye.size else 0.0
            right_eye_h = max(right_eye[:,1]) - min(right_eye[:,1]) if right_eye.size else 0.0
            left_eye_w = max(left_eye[:,0]) - min(left_eye[:,0]) if left_eye.size else 1.0
            right_eye_w = max(right_eye[:,0]) - min(right_eye[:,0]) if right_eye.size else 1.0

            eye_openness = (left_eye_h + right_eye_h) / (2.0 * h + 1e-6)
            eye_aspect = ((left_eye_h/left_eye_w) + (right_eye_h/right_eye_w)) / 2.0

            # mouth
            mouth_outer_idx = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
            mouth_inner_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
            mouth_outer = safe_indices(mouth_outer_idx)
            mouth_inner = safe_indices(mouth_inner_idx)
            mouth_w = (max(mouth_outer[:,0]) - min(mouth_outer[:,0])) if mouth_outer.size else 1.0
            mouth_h_outer = (max(mouth_outer[:,1]) - min(mouth_outer[:,1])) if mouth_outer.size else 0.0
            mouth_h_inner = (max(mouth_inner[:,1]) - min(mouth_inner[:,1])) if mouth_inner.size else 0.0

            mouth_openness = mouth_h_outer / (h + 1e-6)
            mouth_aspect = mouth_h_outer / (mouth_w + 1e-6)
            mouth_tension = mouth_h_inner / (mouth_h_outer + 1e-6) if mouth_h_outer > 0 else 0.5

            # mouth lift / symmetry
            left_corner = pts[61] if 61 < len(pts) else np.array([0,0])
            right_corner = pts[291] if 291 < len(pts) else np.array([0,0])
            nose_base = pts[2] if 2 < len(pts) else np.array([0,0])
            avg_corner_y = (left_corner[1] + right_corner[1]) / 2.0
            mouth_lift = (nose_base[1] - avg_corner_y) / (h + 1e-6)
            left_lift = (nose_base[1] - left_corner[1]) if left_corner is not None else 0.0
            right_lift = (nose_base[1] - right_corner[1]) if right_corner is not None else 0.0
            mouth_symmetry = 1.0 - abs(left_lift - right_lift) / (h + 1e-6)

            # brows
            left_brow_idx = [70, 63, 105, 66, 107]
            right_brow_idx = [300, 293, 334, 296, 336]
            left_brow = safe_indices(left_brow_idx)
            right_brow = safe_indices(right_brow_idx)
            left_brow_center = np.mean(left_brow[:,1]) if left_brow.size else 0.0
            right_brow_center = np.mean(right_brow[:,1]) if right_brow.size else 0.0
            left_eye_center = np.mean(left_eye[:,1]) if left_eye.size else 0.0
            right_eye_center = np.mean(right_eye[:,1]) if right_eye.size else 0.0

            brow_raise = ((left_eye_center - left_brow_center) + (right_eye_center - right_brow_center)) / (2.0 * h + 1e-6)
            inner_brow_dist = abs(pts[70][0] - pts[300][0]) if 70 < len(pts) and 300 < len(pts) else 0.0
            brow_furrow = inner_brow_dist / (w + 1e-6)

            # jaw tension
            jaw_left = pts[172] if 172 < len(pts) else np.array([0,0])
            jaw_right = pts[397] if 397 < len(pts) else np.array([0,0])
            jaw_width = abs(jaw_right[0] - jaw_left[0]) if jaw_left is not None and jaw_right is not None else w
            jaw_tension = jaw_width / (w + 1e-6)

            features = {
                "eye_openness": float(eye_openness),
                "eye_aspect_ratio": float(eye_aspect),
                "mouth_openness": float(mouth_openness),
                "mouth_aspect_ratio": float(mouth_aspect),
                "mouth_tension": float(mouth_tension),
                "mouth_lift": float(mouth_lift),
                "mouth_symmetry": float(np.clip(mouth_symmetry, 0.0, 1.0)),
                "brow_raise": float(brow_raise),
                "brow_furrow": float(brow_furrow),
                "jaw_tension": float(jaw_tension),
            }
            return features
        except Exception as e:
            debug(f"Feature extraction error: {e}")
            # return reasonable defaults
            return {
                "eye_openness": 0.04, "eye_aspect_ratio": 0.3,
                "mouth_openness": 0.02, "mouth_aspect_ratio": 0.3,
                "mouth_tension": 0.5, "mouth_lift": 0.0,
                "mouth_symmetry": 1.0, "brow_raise": 0.03,
                "brow_furrow": 0.15, "jaw_tension": 0.5
            }

    # -------------------------
    def _advanced_classify(self, f: Dict) -> Tuple[str, float, Dict]:
        """
        Heuristic classifier over geometric features to produce per-emotion scores.
        Returns (dominant_emotion, confidence, all_scores_dict)
        """
        try:
            scores = {}

            # HAPPY heuristics
            happy = 0.0
            if f["mouth_lift"] < -0.012:
                happy += min(0.4, abs(f["mouth_lift"]) * 30)
            if f["mouth_symmetry"] > 0.85:
                happy += 0.15
            if f["eye_aspect_ratio"] < 0.28:
                happy += 0.2
            if f["brow_raise"] > 0.025:
                happy += 0.1
            scores["happy"] = min(happy, 0.95)

            # SAD
            sad = 0.0
            if f["mouth_lift"] > 0.012:
                sad += min(0.35, f["mouth_lift"] * 28)
            if f["brow_raise"] > 0.04:
                sad += 0.25
            if f["eye_openness"] < 0.035:
                sad += 0.15
            if f["mouth_tension"] > 0.65:
                sad += 0.1
            scores["sad"] = min(sad, 0.90)

            # ANGRY
            angry = 0.0
            if f["brow_raise"] < 0.025:
                angry += 0.3
            if f["brow_furrow"] < 0.13:
                angry += 0.25
            if f["mouth_tension"] < 0.4:
                angry += 0.2
            if f["jaw_tension"] > 0.52:
                angry += 0.15
            scores["angry"] = min(angry, 0.90)

            # SURPRISE
            surprise = 0.0
            if f["eye_openness"] > 0.055:
                surprise += 0.35
            if f["brow_raise"] > 0.05:
                surprise += 0.3
            if f["mouth_openness"] > 0.045:
                surprise += 0.25
            if f["jaw_tension"] < 0.48:
                surprise += 0.1
            scores["surprise"] = min(surprise, 0.92)

            # FEAR
            fear = 0.0
            if f["eye_openness"] > 0.048:
                fear += 0.25
            if f["brow_raise"] > 0.045:
                fear += 0.25
            if f["mouth_openness"] > 0.03 and f["mouth_tension"] > 0.6:
                fear += 0.2
            if f["mouth_symmetry"] < 0.8:
                fear += 0.15
            scores["fear"] = min(fear, 0.88)

            # DISGUST
            disgust = 0.0
            if f["brow_furrow"] < 0.14:
                disgust += 0.2
            if f["jaw_tension"] > 0.55:
                disgust += 0.25
            if f["mouth_lift"] > 0.008:
                disgust += 0.25
            scores["disgust"] = min(disgust, 0.85)

            # NEUTRAL fallback
            neutral = 0.0
            if (0.032 < f["eye_openness"] < 0.050 and
                -0.008 < f["mouth_lift"] < 0.008 and
                0.025 < f["brow_raise"] < 0.038 and
                0.45 < f["mouth_tension"] < 0.6):
                neutral = 0.7
            scores["neutral"] = min(neutral, 0.80)

            total = sum(scores.values()) + 1e-8
            normalized = {k: (v / total) for k, v in scores.items()}

            dominant = max(normalized.items(), key=lambda x: x[1])
            return dominant[0], float(min(dominant[1], 0.95)), normalized
        except Exception as e:
            debug(f"advanced_classify error: {e}")
            return "neutral", 0.5, {"neutral": 1.0}

    # -------------------------
    def _detect_basic(self, face_crop: np.ndarray) -> Dict:
        return {"emotion": "neutral", "confidence": 0.45, "all_scores": {"neutral": 1.0}, "source": "basic"}

    # -------------------------
    def _ensemble_combine(self, results: List[Tuple[str, Dict]]) -> Dict:
        """Combine 'all_scores' from multiple detectors using configured weights."""
        try:
            combined = {}
            total_w = 0.0
            for method, res in results:
                w = self.weights.get(method, 0.5)
                total_w += w
                scores = res.get("all_scores", {})
                for emo, sc in scores.items():
                    combined[emo] = combined.get(emo, 0.0) + sc * w
            if total_w > 0:
                combined = {k: v / total_w for k, v in combined.items()}
            dominant = max(combined.items(), key=lambda x: x[1])
            return {"emotion": dominant[0], "confidence": float(dominant[1]), "all_scores": combined, "source": "ensemble"}
        except Exception as e:
            debug(f"ensemble_combine error: {e}")
            # fallback to first result
            return results[0][1]

    # -------------------------
    def _apply_advanced_smoothing(self, face_id: int, result: Dict) -> Dict:
        """Temporal smoothing using deque of previous emotions and confidence-weighted voting."""
        try:
            if face_id not in self.emotion_history:
                self.emotion_history[face_id] = deque(maxlen=CONFIG.EMOTION_HISTORY)
                self.conf_history[face_id] = deque(maxlen=CONFIG.EMOTION_HISTORY)

            self.emotion_history[face_id].append(result.get("emotion", "neutral"))
            self.conf_history[face_id].append(result.get("confidence", 0.5))

            hist = list(self.emotion_history[face_id])
            confs = list(self.conf_history[face_id])

            if len(hist) < 4:
                return result

            weights = {}
            for i, (emo, conf) in enumerate(zip(hist, confs)):
                recency = (i + 1) / len(hist)
                w = recency * conf
                weights[emo] = weights.get(emo, 0.0) + w

            if not weights:
                return result

            dominant = max(weights.items(), key=lambda x: x[1])
            total_w = sum(weights.values()) + 1e-8
            smoothed_conf = dominant[1] / total_w
            # decay if differs from current detection
            if dominant[0] != result.get("emotion"):
                smoothed_conf *= 0.92

            # require consistency to fully switch
            count = sum(1 for e in hist if e == dominant[0])
            if count >= 3:
                return {"emotion": dominant[0], "confidence": float(min(smoothed_conf, 0.97)), "all_scores": result.get("all_scores", {}), "smoothed": True}

            return result
        except Exception as e:
            debug(f"smoothing error: {e}")
            return result

    # -------------------------
    def _get_cached_emotion(self, face_id: int) -> Dict:
        """Return cached emotion if available, apply light time decay on confidence."""
        if face_id in self.current_emotions:
            cached = dict(self.current_emotions[face_id])
            last = self.last_detect_time.get(face_id, time.time())
            dt = time.time() - last
            if dt > 1.0:
                decay = max(0.3, 1.0 - (dt - 1.0) * 0.1)
                cached["confidence"] = float(cached.get("confidence", 0.5) * decay)
            return cached
        return {"emotion": "neutral", "confidence": 0.5, "all_scores": {"neutral": 1.0}}
# ============================================================
# PART 3/6 — FaceTracker, FaceDetector, MessageManager, Chat Engine
# ============================================================

# -------------------------
# Face Tracker
# -------------------------
class FaceTracker:
    def __init__(self):
        self.tracks: Dict[int, Dict] = {}
        self.next_id = 0
        self.iou_threshold = 0.3
        self.max_lost = 3.0  # seconds before dropping

    def update(self, detected: List[Dict]) -> List[Dict]:
        """
        detected: list of {'bbox': (x,y,w,h), 'center': (cx,cy), 'confidence': float}
        returns: matched list with 'tracked_id' assigned
        """
        now = time.time()
        matched = []
        unmatched = list(detected)

        # remove old tracks
        for tid in list(self.tracks.keys()):
            if now - self.tracks[tid]['last_seen'] > self.max_lost:
                del self.tracks[tid]

        # match existing tracks
        for tid, t in list(self.tracks.items()):
            best = None
            best_score = 0.0
            for face in unmatched:
                dist = np.linalg.norm(np.array(t['center']) - np.array(face['center']))
                dist_score = max(0.0, 1.0 - dist / 200.0)
                iou = self._iou(t['bbox'], face['bbox'])
                score = 0.6 * iou + 0.4 * dist_score
                if score > best_score and score > self.iou_threshold:
                    best_score = score
                    best = face
            if best is not None:
                best['tracked_id'] = tid
                self.tracks[tid].update({
                    'center': best['center'],
                    'bbox': best['bbox'],
                    'last_seen': now
                })
                matched.append(best)
                unmatched.remove(best)

        # create new tracks for remaining
        for face in unmatched[: CONFIG.MAX_FACES - len(matched)]:
            face['tracked_id'] = self.next_id
            self.tracks[self.next_id] = {
                'center': face['center'],
                'bbox': face['bbox'],
                'last_seen': now
            }
            self.next_id += 1
            matched.append(face)

        return matched

    def _iou(self, b1: Tuple, b2: Tuple) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)
        if xb <= xa or yb <= ya:
            return 0.0
        inter = (xb - xa) * (yb - ya)
        union = w1 * h1 + w2 * h2 - inter
        return inter / (union + 1e-6)

# -------------------------
# Face Detector (MediaPipe + Cascade fallback)
# -------------------------
class FaceDetector:
    def __init__(self):
        self.detectors = {}
        self.tracker = FaceTracker()
        self.frame_skip = 0
        self.last_faces = []

        # setup mediapipe face detection if available
        if HAS_MP:
            try:
                mp_fd = mp.solutions.face_detection
                self.detectors['mediapipe'] = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.5)
                debug("MediaPipe face detector ready")
            except Exception as e:
                debug(f"MediaPipe face detector init error: {e}")

        # haarcascade fallback
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.detectors['cascade'] = cascade
                debug("Haarcascade face detector ready")
        except Exception as e:
            debug(f"Haarcascade init error: {e}")

        if not self.detectors:
            debug("No face detectors available. Face detection will be disabled.")

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Returns list of faces: {'bbox': (x,y,w,h), 'center': (cx,cy), 'confidence': float}
        """
        if frame is None or frame.size == 0:
            return []

        self.frame_skip += 1
        # small frame skip for CPU saving
        if self.frame_skip % 2 != 0:
            return self.last_faces

        faces = []

        # MediaPipe detection
        if 'mediapipe' in self.detectors:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = self.detectors['mediapipe'].process(rgb)
                if res and res.detections:
                    h, w = frame.shape[:2]
                    for d in res.detections[: CONFIG.MAX_FACES]:
                        bbox = d.location_data.relative_bounding_box
                        x = int(max(0, bbox.xmin * w))
                        y = int(max(0, bbox.ymin * h))
                        bw = int(min(bbox.width * w, w - x))
                        bh = int(min(bbox.height * h, h - y))
                        if bw >= CONFIG.EMOTION_MIN_FACE_SIZE and bh >= CONFIG.EMOTION_MIN_FACE_SIZE:
                            faces.append({
                                'bbox': (x, y, bw, bh),
                                'center': (x + bw // 2, y + bh // 2),
                                'confidence': float(d.score[0]) if d.score else 0.8
                            })
            except Exception as e:
                debug(f"MediaPipe detection error: {e}")

        # Cascade fallback
        if not faces and 'cascade' in self.detectors:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = self.detectors['cascade'].detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                                                  minSize=(CONFIG.EMOTION_MIN_FACE_SIZE, CONFIG.EMOTION_MIN_FACE_SIZE))
                for (x, y, w, h) in rects[: CONFIG.MAX_FACES]:
                    faces.append({
                        'bbox': (x, y, w, h),
                        'center': (x + w // 2, y + h // 2),
                        'confidence': 0.7
                    })
            except Exception as e:
                debug(f"Cascade detection error: {e}")

        tracked = self.tracker.update(faces)
        self.last_faces = tracked
        return tracked

# -------------------------
# Message Manager
# -------------------------
class MessageManager:
    def __init__(self, max_messages=30):
        self.messages: List[Dict] = []
        self.lock = threading.Lock()
        self.max = max_messages

    def add(self, role: str, content: str):
        with self.lock:
            self.messages.append({
                'role': role,
                'content': content,
                'ts': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            if len(self.messages) > self.max:
                self.messages.pop(0)

    def get(self) -> List[Dict]:
        with self.lock:
            return list(self.messages)

    def get_for_api(self) -> List[Dict]:
        with self.lock:
            return [{'role': m['role'], 'content': m['content']} for m in self.messages]

# -------------------------
# ProfessionalAIChatEngine (Ollama integration + threading)
# -------------------------
class ProfessionalAIChatEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = True
        self.ollama_ok = self._check_ollama()
        if CONFIG.USE_THREADING:
            self.worker = threading.Thread(target=self._worker, daemon=True)
            self.worker.start()

    def _check_ollama(self) -> bool:
        try:
            r = requests.get(CONFIG.OLLAMA_URL.replace("/api/chat", ""), timeout=4)
            if r.status_code in (200, 404):
                debug("Ollama reachable")
                return True
            return True
        except Exception as e:
            debug(f"Ollama check failed: {e}")
            return False

    def send(self, messages: List[Dict], emotion_context: Dict = None):
        self.queue.put((messages, emotion_context))

    def get_response(self) -> Optional[str]:
        try:
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None

    def _worker(self):
        while self.running:
            try:
                messages, emotion_context = self.queue.get(timeout=0.1)
                debug("Chat worker: generating response")
                resp = self._generate(messages, emotion_context)
                self.response_queue.put(resp)
                debug("Chat worker: response queued")
            except queue.Empty:
                continue
            except Exception as e:
                debug(f"Chat worker error: {e}")
                self.response_queue.put("⚠ Chat engine error occurred.")

    def _generate(self, messages: List[Dict], emotion_context: Dict = None) -> str:
        """
        Build system prompt with emotion context and call Ollama.
        """
        try:
            base = ("You are EMMA, a warm supportive assistant. Keep responses concise (2-3 sentences). "
                    "Use empathy, offer one practical tip, and finish with encouragement.\n\n")
            if emotion_context:
                emo = emotion_context.get('emotion', 'neutral')
                conf = int(100 * float(emotion_context.get('confidence', 0.5)))
                mapping = {
                    "sad": f"User seems sad ({conf}%). Be gentle and offer one small coping step.",
                    "fear": f"User seems anxious ({conf}%). Provide calming reassurance and a brief grounding step.",
                    "angry": f"User seems frustrated ({conf}%). Acknowledge feelings and de-escalate.",
                    "happy": f"User seems happy ({conf}%). Match the positive tone and celebrate briefly.",
                    "surprise": f"User seems surprised ({conf}%). Be curious and encourage engagement.",
                    "disgust": f"User seems uncomfortable ({conf}%). Validate and be gentle.",
                    "neutral": f"User seems neutral ({conf}%). Be warm and helpful."
                }
                base += mapping.get(emo, mapping['neutral'])

            system = {"role": "system", "content": base}
            api_messages = [system] + messages

            return self._call_ollama(api_messages)
        except Exception as e:
            debug(f"_generate error: {e}")
            return "⚠ Failed to generate response."

    # Ollama API call (chat endpoint)
    def _call_ollama(self, messages: List[Dict]) -> str:
        try:
            if not self.ollama_ok:
                self.ollama_ok = self._check_ollama()
                if not self.ollama_ok:
                    return "⚠ Ollama not reachable. Start with 'ollama serve'."

            payload = {
                "model": CONFIG.OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.75, "num_predict": 180, "top_p": 0.9}
            }
            r = requests.post(CONFIG.OLLAMA_URL, json=payload, timeout=60)
            if r.status_code == 200:
                data = r.json()
                return data.get("message", {}).get("content", "") or self._call_ollama_generate(messages)
            elif r.status_code == 404:
                return f"⚠ Model {CONFIG.OLLAMA_MODEL} not installed. Run: ollama pull {CONFIG.OLLAMA_MODEL}"
            else:
                return f"⚠ Ollama error HTTP {r.status_code}"
        except requests.exceptions.Timeout:
            return "⚠ Ollama request timed out."
        except requests.exceptions.ConnectionError:
            self.ollama_ok = False
            return "⚠ Cannot connect to Ollama (connection error)."
        except Exception as e:
            debug(f"_call_ollama exception: {e}")
            return f"⚠ Ollama call failed: {e}"

    # Ollama generate fallback
    def _call_ollama_generate(self, messages: List[Dict]) -> str:
        try:
            prompt_parts = []
            for m in messages:
                role = m.get("role")
                content = m.get("content", "")
                prompt_parts.append(f"{role.upper()}: {content}")
            prompt = "\n\n".join(prompt_parts) + "\n\nASSISTANT:"
            payload = {"model": CONFIG.OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.75, "num_predict": 180}}
            r = requests.post(CONFIG.OLLAMA_GENERATE_URL, json=payload, timeout=60)
            if r.status_code == 200:
                data = r.json()
                # new Ollama generate may return 'response' key
                return data.get("response", "") or "⚠ Empty generate response."
            return f"⚠ Generate endpoint returned HTTP {r.status_code}"
        except Exception as e:
            debug(f"_call_ollama_generate error: {e}")
            return f"⚠ Ollama generate failed: {e}"

    def stop(self):
        self.running = False
# ============================================================
# PART 4/6 — TTS Engine, Medium UI Renderer, Main App
# ============================================================

# -------------------------
# TTS Engine (pyttsx3-based, threaded)
# -------------------------
class TTSEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.lock = threading.Lock()
        self.engine = None
        if HAS_TTS:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)
                if CONFIG.USE_THREADING:
                    self.worker = threading.Thread(target=self._worker, daemon=True)
                    self.worker.start()
                debug("TTS initialized")
            except Exception as e:
                debug(f"TTS init failed: {e}")
                self.engine = None
        else:
            debug("pyttsx3 not available — TTS disabled")

    def speak(self, text: str):
        if not text or text.strip() == "":
            return
        # only speak first meaningful sentence
        first = text.split('.')
        if first:
            s = first[0].strip()
            if len(s) > 3:
                try:
                    self.queue.put(s)
                except Exception as e:
                    debug(f"TTS queue put failed: {e}")

    def _worker(self):
        while self.running:
            try:
                text = self.queue.get(timeout=0.1)
                if self.engine and text:
                    with self.lock:
                        try:
                            self.engine.say(text)
                            self.engine.runAndWait()
                        except Exception as e:
                            debug(f"TTS play error: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                debug(f"TTS worker error: {e}")

    def stop(self):
        self.running = False
        try:
            if self.engine:
                self.engine.stop()
        except Exception:
            pass

# -------------------------
# Medium UI renderer (keeps header + video + overlays + simple chat area)
# -------------------------
class ProfessionalUI:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.header_h = 90
        self.input_h = 70
        # video area width (approx)
        self.video_w = int(self.width * 0.66)
        self.chat_w = self.width - self.video_w

    def base(self):
        return np.full((self.height, self.width, 3), CONFIG.BG, dtype=np.uint8)

    def draw_header(self, frame: np.ndarray):
        cv2.rectangle(frame, (0, 0), (self.width, self.header_h), (10, 15, 25), -1)
        cv2.putText(frame, "EMMA — Emotional Mental Management Assistant", (20, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press ENTER to send (in console), ESC/q to quit", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 220), 1, cv2.LINE_AA)
        return frame

    def draw_video_panel(self, frame: np.ndarray, video_frame: np.ndarray, faces: List[Dict]):
        y0 = self.header_h
        # video background
        cv2.rectangle(frame, (0, y0), (self.video_w, self.height - self.input_h), (25, 30, 40), -1)

        if video_frame is not None and video_frame.size > 0:
            vh, vw = video_frame.shape[:2]
            panel_h = self.height - self.header_h - self.input_h
            scale = min(self.video_w / vw, panel_h / vh) * 0.95
            nw, nh = int(vw * scale), int(vh * scale)
            if nw > 0 and nh > 0:
                resized = cv2.resize(video_frame, (nw, nh))
                x_off = (self.video_w - nw) // 2
                y_off = y0 + (panel_h - nh) // 2
                frame[y_off:y_off+nh, x_off:x_off+nw] = resized

                # draw overlays scaled accordingly
                for f in faces:
                    bx, by, bw, bh = f.get('bbox', (0,0,0,0))
                    emotion = f.get('emotion', 'neutral')
                    conf = f.get('confidence', 0.0)
                    # scale bbox relative to placed video
                    sx = int(bx * scale) + x_off
                    sy = int(by * scale) + y_off
                    sw = int(bw * scale)
                    sh = int(bh * scale)
                    color = CONFIG.EMOTION_COLOR.get(emotion, (200,200,200))
                    cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), color, 2)
                    label = f"{emotion.upper()} {int(conf*100)}%"
                    cv2.putText(frame, label, (sx, sy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        # border
        cv2.rectangle(frame, (0, y0), (self.video_w, self.height - self.input_h), (60, 80, 100), 2)
        return frame

    def draw_chat_panel(self, frame: np.ndarray, messages: List[Dict], processing: bool = False):
        x0 = self.video_w
        y0 = self.header_h
        cv2.rectangle(frame, (x0, y0), (self.width, self.height - self.input_h), (18, 22, 30), -1)
        cv2.putText(frame, "Chat (Streamlit runs separately)", (x0 + 16, y0 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 230, 255), 1, cv2.LINE_AA)

        # draw up to last 6 messages
        y = y0 + 50
        max_msgs = 6
        show_msgs = messages[-max_msgs:]
        for m in show_msgs:
            role = m.get('role', '')
            content = m.get('content', '')
            ts = m.get('ts', '')
            badge = "YOU" if role == 'user' else "EMMA"
            color = (100, 140, 200) if role == 'user' else (100, 220, 180)
            cv2.rectangle(frame, (x0+12, y-18), (self.width-12, y+34), (10,10,10), -1)
            cv2.putText(frame, f"{badge} [{ts}]", (x0+18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
            # wrap content (naive)
            lines = []
            words = content.split()
            cur = ""
            for w in words:
                if len(cur) + len(w) + 1 < 40:
                    cur = (cur + " " + w).strip()
                else:
                    lines.append(cur)
                    cur = w
            if cur:
                lines.append(cur)
            yy = y + 18
            for ln in lines[:3]:
                cv2.putText(frame, ln, (x0+18, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200,220,240), 1, cv2.LINE_AA)
                yy += 18
            y = yy + 8

        # processing indicator
        if processing:
            cv2.putText(frame, "Processing...", (x0+18, self.height - self.input_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,180,0), 1, cv2.LINE_AA)

        # border
        cv2.rectangle(frame, (x0, y0), (self.width, self.height - self.input_h), (80, 90, 110), 2)
        return frame

    def draw_input_box(self, frame: np.ndarray, input_text: str):
        y0 = self.height - self.input_h
        cv2.rectangle(frame, (0, y0), (self.width, self.height), (12, 16, 22), -1)
        cv2.putText(frame, "Type (console) and press ENTER to send:", (12, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190,200,210), 1, cv2.LINE_AA)
        display = input_text[-70:] if input_text else ""
        cv2.putText(frame, display, (12, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220,240,255), 1, cv2.LINE_AA)
        return frame

# -------------------------
# Main EMMA app
# -------------------------
class EnhancedLanguageLearningApp:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.emotion_detector = AdvancedEmotionDetector()
        self.message_manager = MessageManager()
        self.chat_engine = ProfessionalAIChatEngine()
        self.tts = TTSEngine()
        self.camera = None
        self.ui = None
        self.is_running = False
        self.input_text = ""
        self.processing_response = False
        self.last_emotion_time = 0
        self.dominant_emotion = "neutral"
        self.dominant_confidence = 0.5

    def start(self):
        debug("Starting EMMA app...")
        self.camera = cv2.VideoCapture(CONFIG.CAMERA_INDEX)
        if not self.camera.isOpened():
            print("❌ Failed to open camera. Exiting.")
            return
        # set camera properties
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        # read frame to set UI dimensions
        ret, frame = self.camera.read()
        if not ret:
            print("❌ Cannot read from camera. Exiting.")
            return
        h, w = frame.shape[:2]
        self.ui = ProfessionalUI(w, h)
        # welcome message
        welcome = "Hi — I'm EMMA. I'm here to listen. Type and press ENTER."
        self.message_manager.add('assistant', welcome)
        self.is_running = True
        self.run()

    def run(self):
        window_name = "EMMA - Emotional Mental Management Assistant"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        fps_count = 0
        fps_time = time.time()
        fps = 0
        last_response_check = time.time()

        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break

            fps_count += 1
            if time.time() - fps_time >= 1.0:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()

            faces = self.face_detector.detect(frame)
            faces_data = []

            # emotion detection section (periodic)
            now = time.time()
            if now - self.last_emotion_time >= CONFIG.EMOTION_INTERVAL:
                emotion_scores = {}
                for f in faces:
                    bx, by, bw, bh = f.get('bbox', (0,0,0,0))
                    pad = 12
                    x1 = max(0, bx - pad)
                    y1 = max(0, by - pad)
                    x2 = min(frame.shape[1], bx + bw + pad)
                    y2 = min(frame.shape[0], by + bh + pad)
                    crop = frame[y1:y2, x1:x2]
                    fid = f.get('tracked_id', 0)
                    res = self.emotion_detector.analyze_emotion(crop, fid)
                    faces_data.append({'bbox': f['bbox'], 'emotion': res.get('emotion','neutral'), 'confidence': res.get('confidence',0.0)})
                    emo = res.get('emotion','neutral')
                    conf = res.get('confidence', 0.0)
                    emotion_scores.setdefault(emo, []).append(conf)

                if emotion_scores:
                    weighted = {emo: sum(confs) / len(confs) for emo, confs in emotion_scores.items()}
                    dom = max(weighted.items(), key=lambda x: x[1])
                    self.dominant_emotion = dom[0]
                    self.dominant_confidence = dom[1]
                    # write emotion to disk for Streamlit to read
                    try:
                        write_emotion_state(self.dominant_emotion, self.dominant_confidence, time.time())
                    except Exception as e:
                        debug(f"Failed to write emotion state: {e}")

                self.last_emotion_time = now
            else:
                # use cached emotions to display
                for f in faces:
                    fid = f.get('tracked_id', 0)
                    cached = self.emotion_detector._get_cached_emotion(fid)
                    faces_data.append({'bbox': f['bbox'], 'emotion': cached.get('emotion','neutral'), 'confidence': cached.get('confidence',0.0)})

            # check for response available from chat engine
            if self.processing_response:
                resp = self.chat_engine.get_response()
                if resp:
                    debug("Received chat response")
                    self.message_manager.add('assistant', resp)
                    # speak via TTS
                    try:
                        if HAS_TTS:
                            self.tts.speak(resp)
                    except Exception as e:
                        debug(f"TTS speak failed: {e}")
                    self.processing_response = False

            # render UI
            ui_frame = self.ui.base()
            ui_frame = self.ui.draw_header(ui_frame)
            ui_frame = self.ui.draw_video_panel(ui_frame, frame, faces_data)
            ui_frame = self.ui.draw_chat_panel(ui_frame, self.message_manager.get(), self.processing_response)
            ui_frame = self.ui.draw_input_box(ui_frame, self.input_text)
            cv2.putText(ui_frame, f"FPS: {fps}", (10, self.ui.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,200,220), 1, cv2.LINE_AA)

            cv2.imshow(window_name, ui_frame)
            key = cv2.waitKey(1) & 0xFF

            # Input handling:
            # ESC / q -> quit
            if key == 27 or key == ord('q'):
                self.is_running = False
                break
            # ENTER -> send message if not already processing
            elif key == 13:
                if self.input_text.strip() and not self.processing_response:
                    user_msg = self.input_text.strip()
                    self.message_manager.add('user', user_msg)
                    # prepare emotion context
                    emo_ctx = {'emotion': self.dominant_emotion, 'confidence': self.dominant_confidence}
                    # send to chat engine (non-blocking)
                    self.chat_engine.send(self.message_manager.get_for_api(), emo_ctx)
                    self.processing_response = True
                    self.input_text = ""
            # backspace
            elif key == 8:
                self.input_text = self.input_text[:-1]
            # printable chars
            elif key != 255 and 32 <= key <= 126:
                if len(self.input_text) < 300:
                    self.input_text += chr(key)

        # shutdown
        self.stop()

    def stop(self):
        debug("Shutting down EMMA app...")
        self.is_running = False
        try:
            if self.camera:
                self.camera.release()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.chat_engine.stop()
        except Exception:
            pass
        try:
            self.tts.stop()
        except Exception:
            pass
        debug("Shutdown complete.")
# ============================================================
# PART 5/6 — Streamlit Chatbot (wrapped inside run_streamlit_chatbot)
# ============================================================

# ============================================================
# EMMA - Emotional Mental Management Assistant (FIXED VERSION)
# Replace the run_streamlit_chatbot() function with this version
# ============================================================

def run_streamlit_chatbot():
    """
    Enhanced Streamlit UI for EMMA with futuristic design and fixed state management.
    Launch with: streamlit run EMOBOT_gpt.py -- --chat
    """

    import streamlit as st
    from gtts import gTTS
    from io import BytesIO
    from pathlib import Path
    import time
    import json

    # Config
    OLLAMA_URL = CONFIG.OLLAMA_URL
    OLLAMA_MODEL = CONFIG.OLLAMA_MODEL
    EMOTION_PATH = EMOTION_JSON
    MEMORY_PATH = "chat_memory.json"

    # -------------------------
    # Utilities
    # -------------------------
    def load_emotion_state():
        try:
            if Path(EMOTION_PATH).exists():
                with open(EMOTION_PATH, "r") as f:
                    data = json.load(f)
                    return {
                        "emotion": data.get("emotion", "neutral"),
                        "confidence": float(data.get("confidence", 0.5)),
                        "timestamp": float(data.get("timestamp", time.time()))
                    }
        except Exception as e:
            st.sidebar.error(f"Failed to read emotion state: {e}")
        return {"emotion": "neutral", "confidence": 0.5, "timestamp": time.time()}

    def load_memory():
        try:
            if Path(MEMORY_PATH).exists():
                with open(MEMORY_PATH, "r") as f:
                    return json.load(f)
        except Exception:
            return []
        return []

    def save_memory(mem):
        try:
            with open(MEMORY_PATH, "w") as f:
                json.dump(mem, f, indent=2)
        except Exception as e:
            st.sidebar.error(f"Failed to save memory: {e}")

    def speak_bytes(text: str):
        try:
            tts = gTTS(text=text, lang="en")
            bio = BytesIO()
            tts.write_to_fp(bio)
            bio.seek(0)
            return bio.read()
        except Exception as e:
            st.sidebar.error(f"TTS error: {e}")
            return None

    def build_system_prompt(emotion: str, confidence: float) -> str:
        c = int(confidence * 100)
        mapping = {
            "sad": f"User appears sad ({c}%). Be gentle, empathetic, and offer one small coping step.",
            "fear": f"User appears anxious ({c}%). Offer calming reassurance and a short grounding technique.",
            "angry": f"User appears angry ({c}%). Acknowledge frustration and offer de-escalation.",
            "happy": f"User appears happy ({c}%). Match the tone and celebrate briefly.",
            "surprise": f"User appears surprised ({c}%). Be curious and encourage engagement.",
            "disgust": f"User appears uncomfortable ({c}%). Validate discomfort and be gentle.",
            "neutral": f"User appears neutral ({c}%). Be warm and helpful."
        }
        base = (
            "You are EMMA, an empathetic emotional-support assistant. Keep responses concise (2-3 sentences). "
            "Start with empathy, offer one practical suggestion, and end with encouragement or a question.\n\n"
        )
        base += mapping.get(emotion, mapping["neutral"])
        base += "\n\nIf the user expresses thoughts of self-harm or immediate danger, encourage seeking professional help and emergency contacts."
        return base

    def call_ollama(messages: list) -> str:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.75, "num_predict": 180, "top_p": 0.9}
            }
            r = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if r.status_code == 200:
                data = r.json()
                return data.get("message", {}).get("content", "") or ""
            elif r.status_code == 404:
                return f"⚠ Model {OLLAMA_MODEL} not installed on Ollama (HTTP 404)."
            else:
                return f"⚠ Ollama returned HTTP {r.status_code}"
        except requests.exceptions.ConnectionError:
            return "⚠ Could not connect to Ollama. Is 'ollama serve' running?"
        except Exception as e:
            return f"⚠ Ollama call failed: {e}"

    # -------------------------
    # Advanced Futuristic UI Styling
    # -------------------------
    st.set_page_config(
        page_title="EMMA - Emotional Intelligence Chat",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for futuristic, professional design
    st.markdown("""
    <style>
    /* Main background with gradient */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        color: #e8f4ff;
    }
    
    /* Glassmorphism effect for containers */
    .glass-container {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* User message styling */
    .user-message {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(99, 102, 241, 0.15));
        border-left: 4px solid #3b82f6;
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        backdrop-filter: blur(5px);
        animation: slideInRight 0.3s ease-out;
    }
    
    /* Assistant message styling */
    .assistant-message {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(5, 150, 105, 0.15));
        border-left: 4px solid #10b981;
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        backdrop-filter: blur(5px);
        animation: slideInLeft 0.3s ease-out;
    }
    
    /* Message header */
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
        font-weight: 600;
        font-size: 14px;
    }
    
    .user-badge {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 10px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .assistant-badge {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 10px;
        font-size: 12px;
        font-weight: bold;
    }
    
    .timestamp {
        color: #94a3b8;
        font-size: 11px;
        opacity: 0.7;
    }
    
    .message-content {
        color: #e2e8f0;
        line-height: 1.6;
        font-size: 15px;
    }
    
    /* Emotion indicator */
    .emotion-card {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.2), rgba(168, 85, 247, 0.2));
        border-radius: 16px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(139, 92, 246, 0.3);
        margin: 10px 0;
    }
    
    .emotion-icon {
        font-size: 48px;
        margin-bottom: 10px;
        animation: pulse 2s ease-in-out infinite;
    }
    
    .emotion-label {
        font-size: 24px;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 2px;
        background: linear-gradient(135deg, #8b5cf6, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }
    
    .confidence-bar {
        background: rgba(255, 255, 255, 0.1);
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin-top: 10px;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #8b5cf6, #a855f7);
        transition: width 0.3s ease;
        box-shadow: 0 0 10px rgba(139, 92, 246, 0.5);
    }
    
    /* Animations */
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(10, 14, 39, 0.95);
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Text input styling */
    .stTextArea>div>div>textarea {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #e8f4ff;
        font-size: 15px;
        padding: 12px;
    }
    
    .stTextArea>div>div>textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    
    /* Metrics styling */
    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    
    h3 {
        color: #cbd5e1;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
    }
    </style>
    """, unsafe_allow_html=True)

    # -------------------------
    # Session State Initialization
    # -------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_memory()
    
    if "message_sent" not in st.session_state:
        st.session_state.message_sent = False

    # -------------------------
    # Header
    # -------------------------
    st.markdown("<h1>🧠 EMMA - Emotional Intelligence Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 16px; margin-top: -10px;'>Advanced emotion-aware conversation powered by AI</p>", unsafe_allow_html=True)

    # -------------------------
    # Sidebar
    # -------------------------
    with st.sidebar:
        st.markdown("### ⚙️ System Configuration")
        
        with st.expander("🔧 Ollama Settings", expanded=False):
            st.code(f"URL: {OLLAMA_URL}", language="text")
            st.code(f"Model: {OLLAMA_MODEL}", language="text")
        
        with st.expander("💾 Storage", expanded=False):
            st.code(f"Memory: {MEMORY_PATH}", language="text")
            st.code(f"Emotion: {EMOTION_PATH}", language="text")
        
        st.markdown("---")
        st.markdown("### 🎛️ Controls")
        
        auto_speak = st.checkbox("🔊 Auto-speak replies", value=True, help="Uses Google TTS to speak responses")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                save_memory([])
                st.success("Chat cleared!")
                st.rerun()
        
        st.markdown("---")
        st.markdown("### 📊 Statistics")
        st.metric("Total Messages", len(st.session_state.chat_history))
        
        st.markdown("---")
        st.markdown("""
        <div style='font-size: 12px; color: #64748b; line-height: 1.6;'>
        <b>💡 Tip:</b> Run the camera app separately:<br>
        <code>python EMOBOT_gpt.py</code><br><br>
        EMMA reads real-time emotions and adapts responses accordingly.
        </div>
        """, unsafe_allow_html=True)

    # -------------------------
    # Main Layout
    # -------------------------
    left_col, right_col = st.columns([3, 1])

    # Right column - Emotion Display
    with right_col:
        st.markdown("### 🎭 Live Emotion Feed")
        
        emotion_state = load_emotion_state()
        emotion = emotion_state["emotion"]
        confidence = emotion_state["confidence"]
        timestamp = emotion_state["timestamp"]
        
        # Emotion emoji mapping
        emotion_emojis = {
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "fear": "😰",
            "surprise": "😲",
            "disgust": "😖",
            "neutral": "😐"
        }
        
        emoji = emotion_emojis.get(emotion, "🤔")
        
        st.markdown(f"""
        <div class="emotion-card">
            <div class="emotion-icon">{emoji}</div>
            <div class="emotion-label">{emotion}</div>
            <div style="color: #94a3b8; font-size: 14px; margin-top: 5px;">
                Confidence: {int(confidence * 100)}%
            </div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
            </div>
            <div style="color: #64748b; font-size: 11px; margin-top: 10px;">
                Updated: {time.strftime('%H:%M:%S', time.localtime(timestamp))}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("💡 Camera app updates this in real-time")

    # Left column - Chat Interface
    with left_col:
        st.markdown("### 💬 Conversation")
        
        # Chat history container
        chat_container = st.container()
        with chat_container:
            if not st.session_state.chat_history:
                st.markdown("""
                <div style='text-align: center; padding: 40px; color: #64748b;'>
                    <div style='font-size: 48px; margin-bottom: 10px;'>👋</div>
                    <div style='font-size: 18px; font-weight: 600;'>Welcome to EMMA</div>
                    <div style='font-size: 14px; margin-top: 5px;'>Start a conversation to begin...</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for item in st.session_state.chat_history:
                    role = item.get("role", "user")
                    content = item.get("content", "")
                    ts = item.get("ts", "")
                    
                    if role == "user":
                        st.markdown(f"""
                        <div class="user-message">
                            <div class="message-header">
                                <span class="user-badge">YOU</span>
                                <span class="timestamp">{ts}</span>
                            </div>
                            <div class="message-content">{content}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="assistant-message">
                            <div class="message-header">
                                <span class="assistant-badge">EMMA</span>
                                <span class="timestamp">{ts}</span>
                            </div>
                            <div class="message-content">{content}</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("---")
        
        # Input form
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Type your message...",
                height=100,
                placeholder="Share your thoughts with EMMA...",
                key="user_input_field",
                label_visibility="collapsed"
            )
            
            submit_col, _ = st.columns([1, 3])
            with submit_col:
                submitted = st.form_submit_button("Send 📤", use_container_width=True)

        # Process message
        if submitted and user_input and user_input.strip():
            # Refresh emotion state before processing
            emotion_state = load_emotion_state()
            emotion = emotion_state["emotion"]
            confidence = emotion_state["confidence"]

            # Build system prompt with emotion context
            system_prompt = build_system_prompt(emotion, confidence)

            # Prepare messages for Ollama
            messages = [{"role": "system", "content": system_prompt}]
            for mem in st.session_state.chat_history:
                messages.append({
                    "role": mem.get("role", "user"),
                    "content": mem.get("content", "")
                })
            messages.append({"role": "user", "content": user_input.strip()})

            # Generate response
            with st.spinner("🤔 EMMA is thinking..."):
                reply = call_ollama(messages)

            # Save to history
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input.strip(),
                "ts": ts
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": reply,
                "ts": ts
            })
            save_memory(st.session_state.chat_history)

            # Auto-speak if enabled
            if auto_speak and reply and not reply.startswith("⚠"):
                audio = speak_bytes(reply)
                if audio:
                    st.audio(audio, format="audio/mp3", autoplay=True)

            # Rerun to display new messages
            st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #64748b; font-size: 13px; padding: 20px;'>
        <b>EMMA</b> - Emotional Mental Management Assistant | 
        Powered by Ollama & Advanced Emotion Detection | 
        <a href='https://github.com' style='color: #3b82f6; text-decoration: none;'>View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

    st.title("💙 EMMA — Emotion-aware Chat (Streamlit)")
    st.write("This chat reads the latest emotion from your EMMA camera app (`emotion_state.json`) and includes that emotion in the system prompt sent to Ollama.")

    # sidebar
    with st.sidebar:
        st.header("Settings")
        st.write("Ollama URL:")
        st.code(OLLAMA_URL)
        st.write("Model:")
        st.code(OLLAMA_MODEL)
        st.markdown("---")
        st.write("Memory file:")
        st.code(MEMORY_PATH)
        st.markdown("---")
        st.write("Run the EMMA camera app (python EMOBOT_gpt.py) so it writes `emotion_state.json`.")

    # session state memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_memory()

    left, right = st.columns([3, 1])
    with right:
        st.markdown("### Emotion feed")
        emotion_state = load_emotion_state()
        st.metric("Emotion", emotion_state["emotion"].upper(), delta=f"{int(emotion_state['confidence']*100)}%")
        st.write(f"Last update: {time.ctime(emotion_state['timestamp'])}")
        if st.button("Refresh emotion"):
            emotion_state = load_emotion_state()
            st.rerun()

        st.markdown("---")
        st.markdown("### Controls")
        auto_speak = st.checkbox("Auto-speak replies (gTTS)", value=True)
        if st.button("Clear memory"):
            st.session_state.chat_history = []
            save_memory(st.session_state.chat_history)
            st.success("Memory cleared.")
            st.rerun()


    with left:
        st.markdown("### Conversation")
        chat_container = st.container()
        with chat_container:
            for item in st.session_state.chat_history:
                role = item.get("role", "user")
                content = item.get("content", "")
                ts = item.get("ts", "")
                if role == "user":
                    st.markdown(f"<div class='user'><b>You</b> <span class='small'>{ts}</span><br>{content}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot'><b>EMMA</b> <span class='small'>{ts}</span><br>{content}</div>", unsafe_allow_html=True)

        st.markdown("---")
        user_input = st.text_area("Type your message", height=90, key="input_area")
        send_col, clear_col = st.columns([1,1])
        with send_col:
            send = st.button("Send")
        with clear_col:
            clear = st.button("Clear input", on_click=lambda: st.session_state.update({"input_area": ""}))

        if send and user_input.strip():
            # read freshest emotion
            emotion_state = load_emotion_state()
            emotion = emotion_state["emotion"]
            confidence = emotion_state["confidence"]

            system_prompt = build_system_prompt(emotion, confidence)

            messages = [{"role": "system", "content": system_prompt}]
            for mem in st.session_state.chat_history:
                messages.append({"role": mem.get("role", "user"), "content": mem.get("content", "")})
            messages.append({"role": "user", "content": user_input})

            with st.spinner("Generating reply..."):
                reply = call_ollama(messages)

            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            # append user + assistant
            st.session_state.chat_history.append({"role": "user", "content": user_input, "ts": ts})
            st.session_state.chat_history.append({"role": "assistant", "content": reply, "ts": ts})
            save_memory(st.session_state.chat_history)

            # auto speak
            if auto_speak and reply and not reply.startswith("⚠"):
                audio = speak_bytes(reply)
                if audio:
                    st.audio(audio, format="audio/mp3")

            # clear input and rerun to show new messages
            st.session_state.input_area = ""
            st.rerun()


    st.markdown("---")
    st.markdown("**How it works**: Run the EMMA camera app (python EMOBOT_gpt.py). It writes `emotion_state.json`. This Streamlit UI reads that file when you send a message and includes the emotion in the system prompt. Chat memory is saved to `chat_memory.json` in the same folder.")

    # keep Streamlit function alive (nothing more to do)
    return
if __name__ == "__main__":
    import sys

    # If user wants Streamlit mode
    if "--chat" in sys.argv or "chat" in sys.argv:
        print("\n==============================================")
        print("      EMMA STREAMLIT CHATBOT MODE")
        print("==============================================")
        print("Launching Streamlit chatbot UI...")
        print("Make sure your EMMA camera app is also running.")
        print("Reading emotion from: emotion_state.json\n")
        run_streamlit_chatbot()

    else:
        # Default: Camera Emotion Detection Mode
        print("\n==============================================")
        print("      EMMA CAMERA + EMOTION DETECTION")
        print("==============================================")
        print("Running camera window with live advanced emotion detection.")
        print("Streamlit chatbot can be opened separately using:")
        print("   streamlit run EMOBOT_gpt.py -- --chat")
        print("==============================================\n")

        try:
            app = EnhancedLanguageLearningApp()
            app.start()
        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user.")
        except Exception as e:
            print(f"\n❌ FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()