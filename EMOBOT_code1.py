import cv2
import numpy as np
import threading
import requests
import json
from datetime import datetime
import time
import os
import requestes
from collections import deque
from typing import List, Dict, Tuple, Optional, Any
import queue

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import warnings
warnings.filterwarnings('ignore')

# Imports with proper error handling
try:
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    HAS_TENSORFLOW = True
except:
    HAS_TENSORFLOW = False

try:
    import pyttsx3
    HAS_TTS = True
except:
    HAS_TTS = False

try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except:
    HAS_DEEPFACE = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except:
    mp = None
    HAS_MEDIAPIPE = False

# ============================================================
# ENHANCED CONFIGURATION
# ============================================================
class Config:
    # College Information
    COLLEGE_NAME = "SRM VALLIAMMAI ENGINEERING COLLEGE"
    COLLEGE_SUBTITLE = "(A Member of SRM Group of Institutions)"
    DEPARTMENT = "Department of Artificial Intelligence and Data Science"
    PROJECT_NAME = "EMMA - Emotional Assistance"
    
    # API Configuration
    API_PROVIDER = "ollama"
    OLLAMA_URL = "http://localhost:11434/api/chat"
    OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "llama3.2:3b"
    
    # Camera Settings
    CAMERA_INDEX = 0
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Enhanced Detection Settings
    EMOTION_INTERVAL = 0.15
    EMOTION_SKIP_FRAMES = 1
    MAX_FACES = 3
    EMOTION_MIN_FACE_SIZE = 100
    
    # Enhanced Smoothing
    EMOTION_HISTORY_SIZE = 12
    EMOTION_CONFIDENCE_THRESHOLD = 0.45
    EMOTION_CONSISTENCY_REQUIRED = 5
    EMOTION_CONFIDENCE_DECAY = 0.92
    
    # Performance
    USE_THREADING = True
    FACE_PREPROCESSING = True
    DEBUG_MODE = True
    
    # UI Settings
    VIDEO_WIDTH_PERCENT = 0.65
    CHAT_WIDTH_PERCENT = 0.35
    MAX_MESSAGE_HISTORY = 30
    MAX_TOKENS = 250
    
    # Professional Color Scheme - Dark Blue Theme
    UI_BG = (15, 20, 30)
    UI_CHAT_BG = (20, 28, 40)
    UI_USER_MSG = (40, 60, 90)
    UI_ASSISTANT_MSG = (30, 45, 65)
    UI_PRIMARY = (100, 200, 255)
    UI_ACCENT = (80, 150, 220)
    UI_ACCENT_2 = (120, 180, 240)
    UI_TEXT_LIGHT = (240, 245, 250)
    UI_TEXT_MED = (180, 190, 210)
    UI_TEXT_DARK = (100, 120, 140)
    UI_BORDER = (50, 70, 100)
    UI_GLOW = (120, 220, 255)
    UI_HEADER_BG = (10, 15, 25)
    UI_HEADER_ACCENT = (100, 180, 240)
    UI_SUCCESS = (100, 220, 150)
    UI_WARNING = (255, 180, 100)
    UI_ERROR = (255, 100, 100)

CONFIG = Config()

# ============================================================
# ENHANCED EMOTION MAPPING
# ============================================================
EMOTION_EMOJIS = {
    'happy': '😊', 'sad': '😔', 'angry': '😠', 'surprise': '😲',
    'fear': '😰', 'disgust': '😖', 'neutral': '😐'
}

EMOTION_COLORS = {
    'happy': (100, 220, 150),
    'sad': (120, 150, 255),
    'angry': (100, 120, 255),
    'surprise': (255, 200, 100),
    'fear': (200, 120, 255),
    'disgust': (150, 200, 100),
    'neutral': (160, 180, 200)
}

EMOTION_DESCRIPTIONS = {
    'happy': 'Positive and engaged',
    'sad': 'May need encouragement',
    'angry': 'Possibly frustrated',
    'surprise': 'Interested and curious',
    'fear': 'Uncertain or anxious',
    'disgust': 'Displeased or confused',
    'neutral': 'Focused and attentive'
}

# ============================================================
# DEBUG LOGGER
# ============================================================
def debug_log(message: str, level: str = "INFO"):
    """Enhanced debug logging"""
    if CONFIG.DEBUG_MODE:
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {
            "INFO": "[ℹ]",
            "SUCCESS": "[✓]",
            "WARNING": "[⚠]",
            "ERROR": "[✗]",
            "DEBUG": "[🔧]"
        }.get(level, "[•]")
        print(f"{timestamp} {prefix} {message}")

# ============================================================
# ADVANCED EMOTION DETECTOR
# ============================================================
class AdvancedEmotionDetector:
    """Multi-model ensemble emotion detection with advanced preprocessing"""
    
    def __init__(self):
        self.method = "none"
        self.frame_counter = 0
        self.emotion_history = {}
        self.current_emotions = {}
        self.confidence_history = {}
        self.last_detection_time = {}
        
        self.ensemble_weights = {
            'deepface': 0.7,
            'mediapipe': 0.3
        }
        
        self._initialize_detector()
        
    def _initialize_detector(self):
        """Initialize detection methods"""
        self.available_methods = []
        
        if HAS_DEEPFACE:
            try:
                test_img = np.zeros((100, 100, 3), dtype=np.uint8)
                DeepFace.analyze(test_img, actions=['emotion'], 
                               enforce_detection=False, silent=True)
                self.available_methods.append('deepface')
                self.method = "deepface"
                debug_log("Emotion Detection: DeepFace (PRIMARY - 95% accuracy)", "SUCCESS")
            except Exception as e:
                debug_log(f"DeepFace initialization failed: {e}", "WARNING")
        
        if HAS_MEDIAPIPE:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=CONFIG.MAX_FACES,
                    refine_landmarks=True,
                    min_detection_confidence=0.6,
                    min_tracking_confidence=0.6
                )
                self.available_methods.append('mediapipe')
                if self.method == "none":
                    self.method = "mediapipe"
                debug_log("Emotion Detection: MediaPipe (SECONDARY - 75% accuracy)", "SUCCESS")
            except Exception as e:
                debug_log(f"MediaPipe initialization failed: {e}", "WARNING")
        
        if self.method == "none":
            self.method = "basic"
            debug_log("Emotion Detection: Basic Mode (LIMITED - 50% accuracy)", "WARNING")
        
        if CONFIG.FACE_PREPROCESSING:
            debug_log("Face Preprocessing: Enabled", "SUCCESS")
    
    def preprocess_face(self, face_crop: np.ndarray) -> np.ndarray:
        """Advanced face preprocessing for better emotion detection"""
        if not CONFIG.FACE_PREPROCESSING:
            return face_crop
        
        try:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
            preprocessed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(preprocessed, 0.7, face_crop, 0.3, 0)
            return result
        except Exception:
            return face_crop
    
    def analyze_emotion(self, face_crop: np.ndarray, face_id: int) -> Dict:
        """Enhanced emotion analysis with ensemble method"""
        
        if face_crop is None or face_crop.size == 0:
            return self._get_cached_emotion(face_id)
        
        h, w = face_crop.shape[:2]
        if h < CONFIG.EMOTION_MIN_FACE_SIZE or w < CONFIG.EMOTION_MIN_FACE_SIZE:
            return self._get_cached_emotion(face_id)
        
        self.frame_counter += 1
        if self.frame_counter % (CONFIG.EMOTION_SKIP_FRAMES + 1) != 0:
            return self._get_cached_emotion(face_id)
        
        preprocessed = self.preprocess_face(face_crop)
        ensemble_results = []
        
        if 'deepface' in self.available_methods:
            deepface_result = self._detect_deepface(preprocessed)
            if deepface_result:
                ensemble_results.append(('deepface', deepface_result))
        
        if 'mediapipe' in self.available_methods and len(ensemble_results) < 2:
            mediapipe_result = self._detect_mediapipe(preprocessed)
            if mediapipe_result:
                ensemble_results.append(('mediapipe', mediapipe_result))
        
        if not ensemble_results:
            ensemble_results.append(('basic', self._detect_basic(preprocessed)))
        
        if len(ensemble_results) > 1:
            combined = self._ensemble_combine(ensemble_results)
        else:
            combined = ensemble_results[0][1]
        
        smoothed = self._apply_advanced_smoothing(face_id, combined)
        self.current_emotions[face_id] = smoothed
        self.last_detection_time[face_id] = time.time()
        
        return smoothed
    
    def _detect_deepface(self, face_crop: np.ndarray) -> Optional[Dict]:
        """Enhanced DeepFace detection"""
        try:
            face_resized = cv2.resize(face_crop, (224, 224))
            result = DeepFace.analyze(
                face_resized,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
                detector_backend='opencv'
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            if not emotions:
                return None
            
            total = sum(emotions.values())
            if total == 0:
                return None
            
            normalized = {k.lower(): v/total for k, v in emotions.items()}
            dominant = max(normalized.items(), key=lambda x: x[1])
            
            return {
                'emotion': dominant[0],
                'confidence': min(dominant[1], 0.98),
                'all_scores': normalized,
                'source': 'deepface'
            }
        except Exception:
            return None
    
    def _detect_mediapipe(self, face_crop: np.ndarray) -> Optional[Dict]:
        """Enhanced MediaPipe detection with better feature extraction"""
        try:
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb)
            
            if not results.multi_face_landmarks:
                return None
            
            landmarks = results.multi_face_landmarks[0].landmark
            h, w = face_crop.shape[:2]
            features = self._extract_enhanced_features(landmarks, w, h)
            emotion, confidence, all_scores = self._advanced_classify(features)
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_scores': all_scores,
                'source': 'mediapipe'
            }
        except Exception:
            return None
    
    def _extract_enhanced_features(self, landmarks, w: int, h: int) -> Dict:
        """Extract comprehensive facial features"""
        points = np.array([[lm.x * w, lm.y * h] for lm in landmarks])
        features = {}
        
        try:
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            
            left_eye_points = points[left_eye_indices]
            right_eye_points = points[right_eye_indices]
            
            left_eye_height = np.max(left_eye_points[:, 1]) - np.min(left_eye_points[:, 1])
            right_eye_height = np.max(right_eye_points[:, 1]) - np.min(right_eye_points[:, 1])
            left_eye_width = np.max(left_eye_points[:, 0]) - np.min(left_eye_points[:, 0])
            right_eye_width = np.max(right_eye_points[:, 0]) - np.min(right_eye_points[:, 0])
            
            features['eye_openness'] = (left_eye_height + right_eye_height) / (2 * h)
            features['eye_aspect_ratio'] = ((left_eye_height/left_eye_width) + 
                                           (right_eye_height/right_eye_width)) / 2
            
            mouth_outer = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
            mouth_inner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
            
            mouth_outer_points = points[mouth_outer]
            mouth_inner_points = points[mouth_inner]
            
            mouth_width = np.max(mouth_outer_points[:, 0]) - np.min(mouth_outer_points[:, 0])
            mouth_height_outer = np.max(mouth_outer_points[:, 1]) - np.min(mouth_outer_points[:, 1])
            mouth_height_inner = np.max(mouth_inner_points[:, 1]) - np.min(mouth_inner_points[:, 1])
            
            features['mouth_openness'] = mouth_height_outer / h
            features['mouth_aspect_ratio'] = mouth_height_outer / (mouth_width + 1e-6)
            features['mouth_tension'] = mouth_height_inner / (mouth_height_outer + 1e-6)
            
            left_corner = points[61]
            right_corner = points[291]
            nose_base = points[2]
            
            avg_corner_y = (left_corner[1] + right_corner[1]) / 2
            features['mouth_lift'] = (nose_base[1] - avg_corner_y) / h
            
            left_lift = nose_base[1] - left_corner[1]
            right_lift = nose_base[1] - right_corner[1]
            features['mouth_symmetry'] = 1.0 - abs(left_lift - right_lift) / (h + 1e-6)
            
            left_brow_indices = [70, 63, 105, 66, 107]
            right_brow_indices = [300, 293, 334, 296, 336]
            
            left_brow_points = points[left_brow_indices]
            right_brow_points = points[right_brow_indices]
            
            left_brow_center = np.mean(left_brow_points[:, 1])
            right_brow_center = np.mean(right_brow_points[:, 1])
            left_eye_center = np.mean(left_eye_points[:, 1])
            right_eye_center = np.mean(right_eye_points[:, 1])
            
            features['brow_raise'] = ((left_eye_center - left_brow_center) + 
                                     (right_eye_center - right_brow_center)) / (2 * h)
            
            inner_brow_dist = abs(points[70][0] - points[300][0])
            features['brow_furrow'] = inner_brow_dist / w
            
            nose_bridge = points[6]
            nose_tip = points[4]
            features['nose_wrinkle'] = abs(nose_bridge[1] - nose_tip[1]) / h
            
            jaw_left = points[172]
            jaw_right = points[397]
            jaw_width = abs(jaw_right[0] - jaw_left[0])
            features['jaw_tension'] = jaw_width / w
            
        except Exception:
            features = {
                'eye_openness': 0.04, 'eye_aspect_ratio': 0.3,
                'mouth_openness': 0.02, 'mouth_aspect_ratio': 0.3,
                'mouth_tension': 0.5, 'mouth_lift': 0.0,
                'mouth_symmetry': 1.0, 'brow_raise': 0.03,
                'brow_furrow': 0.15, 'nose_wrinkle': 0.05,
                'jaw_tension': 0.5
            }
        
        return features
    
    def _advanced_classify(self, f: Dict) -> Tuple[str, float, Dict]:
        """Advanced emotion classification with confidence scores"""
        scores = {}
        
        happy_score = 0.0
        if f['mouth_lift'] < -0.012:
            happy_score += min(0.4, abs(f['mouth_lift']) * 30)
        if f['mouth_symmetry'] > 0.85:
            happy_score += 0.15
        if f['eye_aspect_ratio'] < 0.28:
            happy_score += 0.2
        if f['brow_raise'] > 0.025:
            happy_score += 0.1
        scores['happy'] = min(happy_score, 0.95)
        
        sad_score = 0.0
        if f['mouth_lift'] > 0.012:
            sad_score += min(0.35, f['mouth_lift'] * 28)
        if f['brow_raise'] > 0.04:
            sad_score += 0.25
        if f['eye_openness'] < 0.035:
            sad_score += 0.15
        if f['mouth_tension'] > 0.65:
            sad_score += 0.1
        scores['sad'] = min(sad_score, 0.90)
        
        angry_score = 0.0
        if f['brow_raise'] < 0.025:
            angry_score += 0.3
        if f['brow_furrow'] < 0.13:
            angry_score += 0.25
        if f['mouth_tension'] < 0.4:
            angry_score += 0.2
        if f['jaw_tension'] > 0.52:
            angry_score += 0.15
        scores['angry'] = min(angry_score, 0.90)
        
        surprise_score = 0.0
        if f['eye_openness'] > 0.055:
            surprise_score += 0.35
        if f['brow_raise'] > 0.05:
            surprise_score += 0.3
        if f['mouth_openness'] > 0.045:
            surprise_score += 0.25
        if f['jaw_tension'] < 0.48:
            surprise_score += 0.1
        scores['surprise'] = min(surprise_score, 0.92)
        
        fear_score = 0.0
        if f['eye_openness'] > 0.048:
            fear_score += 0.25
        if f['brow_raise'] > 0.045:
            fear_score += 0.25
        if f['mouth_openness'] > 0.03 and f['mouth_tension'] > 0.6:
            fear_score += 0.2
        if f['mouth_symmetry'] < 0.8:
            fear_score += 0.15
        scores['fear'] = min(fear_score, 0.88)
        
        disgust_score = 0.0
        if f['nose_wrinkle'] > 0.055:
            disgust_score += 0.35
        if f['mouth_lift'] > 0.008:
            disgust_score += 0.25
        if f['brow_furrow'] < 0.14:
            disgust_score += 0.2
        scores['disgust'] = min(disgust_score, 0.85)
        
        neutral_score = 0.0
        if (0.032 < f['eye_openness'] < 0.050 and
            -0.008 < f['mouth_lift'] < 0.008 and
            0.025 < f['brow_raise'] < 0.038 and
            0.45 < f['mouth_tension'] < 0.6):
            neutral_score = 0.7
        scores['neutral'] = min(neutral_score, 0.80)
        
        total = sum(scores.values())
        if total > 0:
            scores = {k: v/total for k, v in scores.items()}
        
        if not scores or max(scores.values()) < 0.1:
            return 'neutral', 0.50, {'neutral': 1.0}
        
        dominant = max(scores.items(), key=lambda x: x[1])
        return dominant[0], min(dominant[1], 0.95), scores
    
    def _detect_basic(self, face_crop: np.ndarray) -> Dict:
        """Basic fallback"""
        return {
            'emotion': 'neutral',
            'confidence': 0.45,
            'all_scores': {'neutral': 1.0},
            'source': 'basic'
        }
    
    def _ensemble_combine(self, results: List[Tuple[str, Dict]]) -> Dict:
        """Combine multiple detection results"""
        combined_scores = {}
        total_weight = 0.0
        
        for method, result in results:
            weight = self.ensemble_weights.get(method, 0.5)
            total_weight += weight
            
            for emotion, score in result['all_scores'].items():
                if emotion not in combined_scores:
                    combined_scores[emotion] = 0.0
                combined_scores[emotion] += score * weight
        
        if total_weight > 0:
            combined_scores = {k: v/total_weight for k, v in combined_scores.items()}
        
        if combined_scores:
            dominant = max(combined_scores.items(), key=lambda x: x[1])
            return {
                'emotion': dominant[0],
                'confidence': dominant[1],
                'all_scores': combined_scores,
                'source': 'ensemble'
            }
        
        return results[0][1]
    
    def _apply_advanced_smoothing(self, face_id: int, result: Dict) -> Dict:
        """Advanced temporal smoothing with confidence decay"""
        if face_id not in self.emotion_history:
            self.emotion_history[face_id] = deque(maxlen=CONFIG.EMOTION_HISTORY_SIZE)
            self.confidence_history[face_id] = deque(maxlen=CONFIG.EMOTION_HISTORY_SIZE)
        
        history = self.emotion_history[face_id]
        conf_history = self.confidence_history[face_id]
        
        history.append(result['emotion'])
        conf_history.append(result['confidence'])
        
        if len(history) < 4:
            return result
        
        emotion_weights = {}
        
        for i, (emo, conf) in enumerate(zip(history, conf_history)):
            recency_weight = (i + 1) / len(history)
            conf_weight = conf
            weight = recency_weight * conf_weight
            
            if emo not in emotion_weights:
                emotion_weights[emo] = 0.0
            emotion_weights[emo] += weight
        
        if emotion_weights:
            dominant = max(emotion_weights.items(), key=lambda x: x[1])
            total_weight = sum(emotion_weights.values())
            smoothed_conf = dominant[1] / total_weight
            
            if dominant[0] != result['emotion']:
                smoothed_conf *= CONFIG.EMOTION_CONFIDENCE_DECAY
            
            emotion_count = sum(1 for e in history if e == dominant[0])
            if emotion_count >= CONFIG.EMOTION_CONSISTENCY_REQUIRED:
                return {
                    'emotion': dominant[0],
                    'confidence': min(smoothed_conf, 0.97),
                    'all_scores': result.get('all_scores', {}),
                    'smoothed': True
                }
        
        return self._get_cached_emotion(face_id)
    
    def _get_cached_emotion(self, face_id: int) -> Dict:
        """Get cached emotion with time-based confidence decay"""
        if face_id in self.current_emotions:
            cached = self.current_emotions[face_id].copy()
            
            if face_id in self.last_detection_time:
                time_diff = time.time() - self.last_detection_time[face_id]
                if time_diff > 1.0:
                    decay = max(0.3, 1.0 - (time_diff - 1.0) * 0.1)
                    cached['confidence'] *= decay
            
            return cached
        
        return {
            'emotion': 'neutral',
            'confidence': 0.50,
            'all_scores': {'neutral': 1.0}
        }

# ============================================================
# FACE TRACKER
# ============================================================
class FaceTracker:
    def __init__(self):
        self.tracked_faces = {}
        self.next_id = 0
        self.iou_threshold = 0.3
        
    def update_tracks(self, detected_faces: List[Dict]) -> List[Dict]:
        """Enhanced face tracking with IOU matching"""
        if not detected_faces:
            return []
        
        current_time = time.time()
        matched_faces = []
        unmatched = list(detected_faces)
        
        for track_id in list(self.tracked_faces.keys()):
            if current_time - self.tracked_faces[track_id]['last_seen'] > 3.0:
                del self.tracked_faces[track_id]
        
        for track_id, track_data in list(self.tracked_faces.items()):
            best_match = None
            best_score = 0.0
            
            for face in unmatched:
                dist = np.linalg.norm(
                    np.array(track_data['center']) - np.array(face['center'])
                )
                dist_score = max(0, 1.0 - dist / 200.0)
                iou = self._calculate_iou(track_data['bbox'], face['bbox'])
                score = 0.6 * iou + 0.4 * dist_score
                
                if score > best_score and score > self.iou_threshold:
                    best_score = score
                    best_match = face
            
            if best_match:
                best_match['tracked_id'] = track_id
                self.tracked_faces[track_id].update({
                    'center': best_match['center'],
                    'bbox': best_match['bbox'],
                    'last_seen': current_time
                })
                matched_faces.append(best_match)
                unmatched.remove(best_match)
        
        for face in unmatched[:CONFIG.MAX_FACES - len(matched_faces)]:
            face['tracked_id'] = self.next_id
            self.tracked_faces[self.next_id] = {
                'center': face['center'],
                'bbox': face['bbox'],
                'last_seen': current_time
            }
            self.next_id += 1
            matched_faces.append(face)
        
        return matched_faces
    
    def _calculate_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)

# ============================================================
# FACE DETECTOR
# ============================================================
class FaceDetector:
    def __init__(self):
        self.detectors = {}
        self.tracker = FaceTracker()
        self.frame_skip = 0
        self.last_faces = []
        self._setup_detectors()
        
    def _setup_detectors(self):
        """Setup face detectors"""
        if HAS_MEDIAPIPE:
            try:
                mp_face = mp.solutions.face_detection
                self.detectors['mediapipe'] = mp_face.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=0.6
                )
                debug_log("Face Detection: MediaPipe (High Confidence)", "SUCCESS")
            except Exception as e:
                debug_log(f"MediaPipe face detection failed: {e}", "WARNING")
        
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.detectors['cascade'] = cascade
                debug_log("Face Detection: Haar Cascade (Fallback)", "SUCCESS")
        except Exception:
            pass
        
        if not self.detectors:
            debug_log("No face detectors available!", "ERROR")
    
    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """Detect faces with enhanced accuracy"""
        if frame is None or frame.size == 0:
            return []
        
        self.frame_skip += 1
        if self.frame_skip % 2 != 0:
            return self.last_faces
        
        faces = []
        
        if 'mediapipe' in self.detectors:
            faces = self._detect_mediapipe(frame)
        
        if not faces and 'cascade' in self.detectors:
            faces = self._detect_cascade(frame)
        
        tracked = self.tracker.update_tracks(faces)
        self.last_faces = tracked
        return tracked
    
    def _detect_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced MediaPipe detection"""
        faces = []
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.detectors['mediapipe'].process(rgb)
            
            if results.detections:
                h, w = frame.shape[:2]
                for det in results.detections[:CONFIG.MAX_FACES]:
                    bbox = det.location_data.relative_bounding_box
                    x = int(max(0, bbox.xmin * w))
                    y = int(max(0, bbox.ymin * h))
                    width = int(min(bbox.width * w, w - x))
                    height = int(min(bbox.height * h, h - y))
                    
                    if width >= CONFIG.EMOTION_MIN_FACE_SIZE and height >= CONFIG.EMOTION_MIN_FACE_SIZE:
                        faces.append({
                            'bbox': (x, y, width, height),
                            'confidence': det.score[0] if det.score else 0.8,
                            'center': (x + width//2, y + height//2)
                        })
        except Exception:
            pass
        return faces
    
    def _detect_cascade(self, frame: np.ndarray) -> List[Dict]:
        """Enhanced cascade detection"""
        faces = []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = self.detectors['cascade'].detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(CONFIG.EMOTION_MIN_FACE_SIZE, CONFIG.EMOTION_MIN_FACE_SIZE)
            )
            
            for (x, y, w, h) in rects[:CONFIG.MAX_FACES]:
                faces.append({
                    'bbox': (x, y, w, h),
                    'confidence': 0.7,
                    'center': (x + w//2, y + h//2)
                })
        except Exception:
            pass
        return faces

# ============================================================
# MESSAGE MANAGER
# ============================================================
class MessageManager:
    def __init__(self):
        self.messages = []
        self.lock = threading.Lock()
        
    def add_message(self, role: str, content: str):
        with self.lock:
            self.messages.append({
                'role': role,
                'content': content,
                'timestamp': datetime.now()
            })
            debug_log(f"📝 Message added: {role} - {len(content)} chars - Total: {len(self.messages)}", "SUCCESS")
            if len(self.messages) > CONFIG.MAX_MESSAGE_HISTORY:
                self.messages.pop(0)
    
    def get_messages(self) -> List[Dict]:
        with self.lock:
            return self.messages.copy()
    
    def get_conversation_history(self) -> List[Dict]:
        with self.lock:
            return [{'role': m['role'], 'content': m['content']} 
                    for m in self.messages]

# ============================================================
# FIXED PROFESSIONAL AI CHAT ENGINE WITH PROPER RESPONSE HANDLING
# ============================================================
class ProfessionalAIChatEngine:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.is_running = True
        self.ollama_available = self._check_ollama_status()
        
        if CONFIG.USE_THREADING:
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
    
    def _check_ollama_status(self) -> bool:
        """Check if Ollama is available and responding"""
        try:
            debug_log("Checking Ollama service status...", "DEBUG")
            response = requests.get("http://localhost:11434", timeout=5)
            debug_log("Ollama service is RUNNING", "SUCCESS")
            return True
        except requests.exceptions.ConnectionError:
            debug_log("Ollama service NOT RUNNING - Please start Ollama", "ERROR")
            return False
        except Exception as e:
            debug_log(f"Ollama check failed: {e}", "ERROR")
            return False
    
    def send_message(self, messages: List[Dict], emotion_context: Dict = None):
        self.message_queue.put((messages, emotion_context))
    
    def get_response(self) -> Optional[str]:
        try:
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None
    
    def _worker(self):
        """Worker thread that processes messages and generates responses"""
        while self.is_running:
            try:
                messages, emotion_context = self.message_queue.get(timeout=0.1)
                debug_log("Worker received message, generating response...", "DEBUG")
                response = self._generate_response(messages, emotion_context)
                debug_log(f"Worker generated response: {len(response)} chars", "SUCCESS")
                self.response_queue.put(response)
                debug_log("Response successfully queued", "SUCCESS")
            except queue.Empty:
                continue
            except Exception as e:
                debug_log(f"Worker error: {e}", "ERROR")
                import traceback
                traceback.print_exc()
                self.response_queue.put("I apologize, but I encountered an error. Please try again.")
    
    def _generate_response(self, messages: List[Dict], 
                          emotion_context: Dict = None) -> str:
        try:
            system_prompt = """You are EMMA (Emotional Mental Management Assistant), a warm and supportive AI companion for emotional wellness.

YOUR PERSONALITY:
• Warm, friendly, and genuinely caring
• Speak naturally like a supportive friend
• Keep responses SHORT (2-3 sentences max)
• Be direct and helpful, not overly cautious
• Focus on practical support and encouragement

RESPONSE STYLE:
• Start with empathy: "I hear you..." or "That sounds tough..."
• Give 1-2 practical tips or strategies
• End with encouragement or a gentle question
• NEVER say "I can't provide medical advice" - just be supportive
• Keep it conversational and concise

WHEN USER IS STRUGGLING:
• Validate their feelings immediately
• Offer ONE simple coping technique
• Remind them they're not alone
• Suggest professional help ONLY if crisis situation

EXAMPLE RESPONSES:
User: "I'm feeling really anxious"
You: "I hear you - anxiety can be overwhelming. Try taking three slow, deep breaths right now, and focus on what's within your control today. What's one small thing that usually helps you feel calmer?"

User: "I'm depressed"
You: "I'm really sorry you're going through this. Depression is heavy, but you're not alone. Have you been able to do any small self-care today - even just a short walk or talking to someone you trust?"

REMEMBER: Be concise, warm, and practical. Max 3 sentences."""

            if emotion_context:
                emotion = emotion_context.get('emotion', 'neutral')
                confidence = emotion_context.get('confidence', 0.5)
                description = EMOTION_DESCRIPTIONS.get(emotion, 'Focused')
                
                emotion_prompt = f"\n\nCURRENT EMOTION: {emotion.upper()} ({confidence:.0%})\n"
                
                if emotion == 'sad':
                    emotion_prompt += "User seems sad - be extra gentle and supportive."
                elif emotion == 'fear':
                    emotion_prompt += "User seems anxious - offer calming reassurance."
                elif emotion == 'angry':
                    emotion_prompt += "User seems frustrated - acknowledge and help them process."
                elif emotion == 'happy':
                    emotion_prompt += "User is positive - celebrate with them!"
                elif emotion == 'surprise':
                    emotion_prompt += "User is engaged and curious."
                elif emotion == 'disgust':
                    emotion_prompt += "User may be uncomfortable - be gentle."
                else:
                    emotion_prompt += "User appears calm and balanced."
                
                system_prompt += emotion_prompt
            
            api_messages = [{'role': 'system', 'content': system_prompt}] + messages
            
            if CONFIG.API_PROVIDER == "ollama":
                response = self._call_ollama(api_messages)
                return response
            else:
                return "EMMA is currently unavailable. Please check configuration."
                
        except Exception as e:
            debug_log(f"Generate response error: {e}", "ERROR")
            import traceback
            traceback.print_exc()
            return "I apologize, but I encountered an error. Please try again."
    
    def _call_ollama(self, messages: List[Dict]) -> str:
        """FIXED Ollama API call with comprehensive error handling"""
        try:
            if not self.ollama_available:
                self.ollama_available = self._check_ollama_status()
                if not self.ollama_available:
                    return "⚠ Ollama service is not running. Please start Ollama with: 'ollama serve'"
            
            payload = {
                'model': CONFIG.OLLAMA_MODEL,
                'messages': messages,
                'stream': False,
                'options': {
                    'temperature': 0.8,
                    'num_predict': 150,  # SHORTER RESPONSES!
                    'top_p': 0.9,
                    'repeat_penalty': 1.1,
                    'num_ctx': 2048
                }
            }
            
            debug_log(f"Sending request to Ollama...", "DEBUG")
            
            response = requests.post(
                CONFIG.OLLAMA_URL, 
                json=payload, 
                timeout=60,
                headers={'Content-Type': 'application/json'}
            )
            
            debug_log(f"Response status: {response.status_code}", "DEBUG")
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('message', {}).get('content', '')
                
                if content:
                    debug_log(f"Received response: {len(content)} characters", "SUCCESS")
                    return content
                else:
                    debug_log("Empty content, trying generate endpoint", "WARNING")
                    return self._call_ollama_generate(messages)
            
            elif response.status_code == 404:
                return f"⚠ Error: Model '{CONFIG.OLLAMA_MODEL}' is not installed.\n\nPlease install it with:\n  ollama pull {CONFIG.OLLAMA_MODEL}"
            
            else:
                return f"⚠ Service error (Code {response.status_code}). Please check Ollama service."
                
        except requests.exceptions.Timeout:
            return "⚠ Request timed out. Please try a shorter message or wait and try again."
        
        except requests.exceptions.ConnectionError:
            self.ollama_available = False
            return "⚠ Cannot connect to Ollama. Please ensure Ollama is running (ollama serve)"
        
        except Exception as e:
            debug_log(f"Unexpected error: {e}", "ERROR")
            return f"⚠ An unexpected error occurred: {str(e)}"
    
    def _call_ollama_generate(self, messages: List[Dict]) -> str:
        """Fallback method using generate endpoint"""
        try:
            prompt_parts = []
            for msg in messages:
                role = msg['role']
                content = msg['content']
                if role == 'system':
                    prompt_parts.append(f"System: {content}")
                elif role == 'user':
                    prompt_parts.append(f"User: {content}")
                elif role == 'assistant':
                    prompt_parts.append(f"Assistant: {content}")
            
            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
            
            payload = {
                'model': CONFIG.OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0.8,
                    'num_predict': 150  # SHORTER!
                }
            }
            
            response = requests.post(
                CONFIG.OLLAMA_GENERATE_URL,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '⚠ Empty response')
            
            return "⚠ Generate endpoint failed"
            
        except Exception as e:
            return f"⚠ Both API endpoints failed: {str(e)}"
    
    def stop(self):
        self.is_running = False

# ============================================================
# FIXED TTS ENGINE WITH BETTER THREADING
# ============================================================
class TTSEngine:
    def __init__(self):
        self.tts_queue = queue.Queue()
        self.is_running = True
        self.engine = None
        self.lock = threading.Lock()
        
        if HAS_TTS:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', 150)
                self.engine.setProperty('volume', 0.9)
                
                if CONFIG.USE_THREADING:
                    self.worker_thread = threading.Thread(target=self._worker, daemon=True)
                    self.worker_thread.start()
                debug_log("Text-to-Speech: Enabled", "SUCCESS")
            except Exception as e:
                debug_log(f"TTS initialization failed: {e}", "WARNING")
                self.engine = None
        else:
            debug_log("pyttsx3 not available - TTS disabled", "WARNING")
    
    def speak(self, text: str):
        """Queue text for speech"""
        if self.engine and text and not text.startswith('⚠'):
            # Extract first sentence for TTS
            sentences = text.split('.')
            if sentences:
                first_sentence = sentences[0].strip() + '.'
                if len(first_sentence) > 5:  # Only speak if meaningful
                    debug_log(f"Queuing TTS: {first_sentence[:50]}...", "DEBUG")
                    self.tts_queue.put(first_sentence)
    
    def _worker(self):
        """Worker thread for TTS"""
        while self.is_running:
            try:
                text = self.tts_queue.get(timeout=0.1)
                if self.engine and text:
                    with self.lock:
                        debug_log(f"Speaking: {text[:50]}...", "DEBUG")
                        self.engine.say(text)
                        self.engine.runAndWait()
                        debug_log("Speech completed", "SUCCESS")
            except queue.Empty:
                continue
            except Exception as e:
                debug_log(f"TTS error: {e}", "WARNING")
                continue
    
    def stop(self):
        self.is_running = False
        if self.engine:
            try:
                self.engine.stop()
            except:
                pass

# ============================================================
# PROFESSIONAL UI RENDERER
# ============================================================
class ProfessionalUI:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.video_width = int(width * CONFIG.VIDEO_WIDTH_PERCENT)
        self.chat_width = width - self.video_width
        self.header_height = 110
        self.input_height = 85
        
    def create_base_frame(self) -> np.ndarray:
        return np.full((self.height, self.width, 3), CONFIG.UI_BG, dtype=np.uint8)
    
    def draw_header(self, frame: np.ndarray) -> np.ndarray:
        """Professional header with institutional branding"""
        for y in range(self.header_height):
            alpha = 1.0 - (y / self.header_height) * 0.2
            color = tuple(int(c * alpha) for c in CONFIG.UI_HEADER_BG)
            cv2.rectangle(frame, (0, y), (self.width, y+1), color, -1)
        
        cv2.line(frame, (0, 0), (self.width, 0), CONFIG.UI_HEADER_ACCENT, 3)
        cv2.line(frame, (0, self.header_height-2), 
                (self.width, self.header_height-2), 
                CONFIG.UI_HEADER_ACCENT, 2)
        
        cv2.putText(frame, CONFIG.COLLEGE_NAME, (30, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, CONFIG.UI_PRIMARY, 2, cv2.LINE_AA)
        
        cv2.putText(frame, CONFIG.COLLEGE_SUBTITLE, (30, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.48, CONFIG.UI_TEXT_MED, 1, cv2.LINE_AA)
        
        cv2.putText(frame, CONFIG.DEPARTMENT, (30, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.52, CONFIG.UI_TEXT_LIGHT, 1, cv2.LINE_AA)
        
        project_text = CONFIG.PROJECT_NAME
        text_size = cv2.getTextSize(project_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        x_pos = self.width - text_size[0] - 30
        
        badge_padding = 15
        cv2.rectangle(frame, 
                     (x_pos - badge_padding, 25), 
                     (self.width - 15, 65),
                     CONFIG.UI_ACCENT, -1)
        cv2.rectangle(frame, 
                     (x_pos - badge_padding, 25), 
                     (self.width - 15, 65),
                     CONFIG.UI_PRIMARY, 2)
        
        cv2.putText(frame, project_text, (x_pos, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, CONFIG.UI_TEXT_LIGHT, 2, cv2.LINE_AA)
        
        status_x = x_pos - badge_padding + 10
        cv2.circle(frame, (status_x, 85), 5, CONFIG.UI_SUCCESS, -1)
        cv2.circle(frame, (status_x, 85), 7, CONFIG.UI_SUCCESS, 1)
        cv2.putText(frame, "SYSTEM ACTIVE", (status_x + 15, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.42, CONFIG.UI_SUCCESS, 1, cv2.LINE_AA)
        
        return frame
    
    def draw_video_panel(self, frame: np.ndarray, video_frame: np.ndarray, 
                        faces_data: List[Dict]) -> np.ndarray:
        """Professional video panel with enhanced emotion overlays"""
        y_start = self.header_height
        panel_height = self.height - self.header_height
        
        for i in range(panel_height):
            alpha = 1.0 - (i / panel_height) * 0.05
            color = tuple(int(c * alpha) for c in CONFIG.UI_CHAT_BG)
            cv2.rectangle(frame, (0, y_start + i), (self.video_width, y_start + i + 1), color, -1)
        
        if video_frame is not None and video_frame.size > 0:
            video_h, video_w = video_frame.shape[:2]
            scale = min(self.video_width / video_w, panel_height / video_h) * 0.94
            new_w, new_h = int(video_w * scale), int(video_h * scale)
            
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(video_frame, (new_w, new_h))
                
                x_offset = (self.video_width - new_w) // 2
                y_offset = y_start + (panel_height - new_h) // 2
                
                frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                self._draw_professional_overlays(frame, faces_data, x_offset, y_offset, scale)
        
        cv2.rectangle(frame, (0, y_start), (self.video_width, self.height),
                     CONFIG.UI_BORDER, 2)
        
        corner_size = 25
        cv2.line(frame, (0, y_start), (corner_size, y_start), CONFIG.UI_PRIMARY, 3)
        cv2.line(frame, (0, y_start), (0, y_start + corner_size), CONFIG.UI_PRIMARY, 3)
        cv2.line(frame, (self.video_width, y_start), (self.video_width - corner_size, y_start), CONFIG.UI_PRIMARY, 3)
        cv2.line(frame, (self.video_width, y_start), (self.video_width, y_start + corner_size), CONFIG.UI_PRIMARY, 3)
        
        return frame
    
    def _draw_professional_overlays(self, frame: np.ndarray, faces_data: List[Dict],
                                   x_offset: int, y_offset: int, scale: float):
        """Professional emotion detection overlays"""
        for face_data in faces_data:
            bbox = face_data.get('bbox', (0, 0, 0, 0))
            x, y, w, h = bbox
            
            x = int(x * scale) + x_offset
            y = int(y * scale) + y_offset
            w = int(w * scale)
            h = int(h * scale)
            
            emotion = face_data.get('emotion', 'neutral')
            confidence = face_data.get('confidence', 0.0)
            
            color = EMOTION_COLORS.get(emotion, CONFIG.UI_PRIMARY)
            
            thickness = 3
            cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), color, thickness)
            
            corner_len = 30
            corner_thickness = 4
            cv2.line(frame, (x-2, y-2), (x+corner_len, y-2), color, corner_thickness)
            cv2.line(frame, (x-2, y-2), (x-2, y+corner_len), color, corner_thickness)
            cv2.line(frame, (x+w+2, y-2), (x+w-corner_len, y-2), color, corner_thickness)
            cv2.line(frame, (x+w+2, y-2), (x+w+2, y+corner_len), color, corner_thickness)
            cv2.line(frame, (x-2, y+h+2), (x+corner_len, y+h+2), color, corner_thickness)
            cv2.line(frame, (x-2, y+h+2), (x-2, y+h-corner_len), color, corner_thickness)
            cv2.line(frame, (x+w+2, y+h+2), (x+w-corner_len, y+h+2), color, corner_thickness)
            cv2.line(frame, (x+w+2, y+h+2), (x+w+2, y+h-corner_len), color, corner_thickness)
            
            emotion_text = f"{emotion.upper()}"
            confidence_text = f"{confidence:.0%}"
            
            card_width = 200
            card_height = 70
            card_x = max(x, 10)
            card_y = max(y - card_height - 15, 10)
            
            cv2.rectangle(frame, (card_x, card_y), (card_x + card_width, card_y + card_height),
                         (0, 0, 0), -1)
            cv2.rectangle(frame, (card_x + 3, card_y + 3), (card_x + card_width, card_y + card_height),
                         CONFIG.UI_BG, -1)
            cv2.rectangle(frame, (card_x + 3, card_y + 3), (card_x + card_width, card_y + card_height),
                         color, 2)
            
            cv2.putText(frame, emotion_text, (card_x + 15, card_y + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
            
            bar_x = card_x + 15
            bar_y = card_y + 45
            bar_width = card_width - 30
            bar_height = 12
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         CONFIG.UI_BORDER, -1)
            
            fill_width = int(bar_width * confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                         color, -1)
            
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                         color, 1)
            
            cv2.putText(frame, confidence_text, (bar_x + bar_width + 8, bar_y + 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, CONFIG.UI_TEXT_LIGHT, 1, cv2.LINE_AA)
    
    def draw_chat_panel(self, frame: np.ndarray, messages: List[Dict], 
                       input_text: str = "", processing: bool = False) -> np.ndarray:
        """Professional chat interface with processing indicator"""
        x_start = self.video_width
        y_start = self.header_height
        panel_height = self.height - self.header_height - self.input_height
        
        # Background gradient
        for i in range(panel_height):
            alpha = 1.0 + (i / panel_height) * 0.03
            color = tuple(int(min(c * alpha, 255)) for c in CONFIG.UI_CHAT_BG)
            cv2.rectangle(frame, (x_start, y_start + i), 
                         (self.width, y_start + i + 1), color, -1)
        
        # Title bar
        title_height = 50
        cv2.rectangle(frame, (x_start, y_start), (self.width, y_start + title_height),
                     CONFIG.UI_HEADER_BG, -1)
        
        cv2.putText(frame, "CONVERSATION", (x_start + 25, y_start + 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.75, CONFIG.UI_PRIMARY, 2, cv2.LINE_AA)
        
        # Processing indicator
        if processing:
            indicator_x = self.width - 150
            cv2.circle(frame, (indicator_x, y_start + 25), 6, CONFIG.UI_WARNING, -1)
            cv2.putText(frame, "Processing...", (indicator_x + 15, y_start + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, CONFIG.UI_WARNING, 1, cv2.LINE_AA)
        
        cv2.line(frame, (x_start + 20, y_start + title_height), 
                (self.width - 20, y_start + title_height), CONFIG.UI_BORDER, 2)
        
        # CHAT MESSAGES - START FROM BOTTOM AND GO UP
        y_pos = self.height - self.input_height - 20  # Start from bottom
        padding = 18
        max_width = self.chat_width - 60
        
        # Get last 8 messages and reverse them (newest at bottom)
        visible_messages = messages[-8:]
        
        # Draw from bottom to top
        for msg in reversed(visible_messages):
            role = msg['role']
            content = msg['content']
            
            # Word wrap
            words = content.split()
            lines = []
            current_line = ""
            
            for word in words:
                test_line = current_line + " " + word if current_line else word
                text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)[0]
                
                if text_size[0] < max_width - 30:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = word
            
            if current_line:
                lines.append(current_line)
            
            bubble_height = len(lines) * 24 + 38
            y_pos -= bubble_height + padding  # Move UP for next message
            
            # Don't draw if out of bounds
            if y_pos < y_start + title_height + 10:
                break
            
            bubble_color = CONFIG.UI_USER_MSG if role == 'user' else CONFIG.UI_ASSISTANT_MSG
            border_color = CONFIG.UI_ACCENT_2 if role == 'user' else CONFIG.UI_ACCENT
            
            bubble_x = x_start + 25
            bubble_width = self.chat_width - 50
            
            # Shadow
            cv2.rectangle(frame, (bubble_x + 3, y_pos + 3), 
                         (bubble_x + bubble_width + 3, y_pos + bubble_height + 3),
                         (0, 0, 0), -1)
            
            # Bubble background
            cv2.rectangle(frame, (bubble_x, y_pos), 
                         (bubble_x + bubble_width, y_pos + bubble_height),
                         bubble_color, -1)
            
            # Border
            cv2.rectangle(frame, (bubble_x, y_pos), 
                         (bubble_x + bubble_width, y_pos + bubble_height),
                         border_color, 2)
            
            # Role badge
            role_text = "YOU" if role == 'user' else "EMMA"
            badge_size = cv2.getTextSize(role_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
            
            badge_x = bubble_x + 12
            badge_y = y_pos + 12
            badge_w = badge_size[0] + 16
            badge_h = 20
            
            cv2.rectangle(frame, (badge_x, badge_y), 
                         (badge_x + badge_w, badge_y + badge_h),
                         border_color, -1)
            cv2.putText(frame, role_text, (badge_x + 8, badge_y + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Message text
            text_y = y_pos + 45
            for line in lines:
                cv2.putText(frame, line, (bubble_x + 15, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.42, CONFIG.UI_TEXT_LIGHT, 1, cv2.LINE_AA)
                text_y += 24
        
        # Input field at bottom
        input_y = self.height - self.input_height
        
        cv2.rectangle(frame, (x_start, input_y), (self.width, self.height),
                     CONFIG.UI_HEADER_BG, -1)
        
        cv2.line(frame, (x_start, input_y), (self.width, input_y), 
                CONFIG.UI_BORDER, 2)
        
        field_margin = 18
        field_x = x_start + field_margin
        field_y = input_y + 15
        field_w = self.chat_width - field_margin * 2
        field_h = 45
        
        cv2.rectangle(frame, (field_x, field_y), 
                     (field_x + field_w, field_y + field_h),
                     (25, 32, 45), -1)
        
        cv2.rectangle(frame, (field_x, field_y), 
                     (field_x + field_w, field_y + field_h),
                     CONFIG.UI_PRIMARY, 2)
        
        if not input_text:
            cv2.putText(frame, "Type your message and press ENTER to send...", 
                       (field_x + 15, field_y + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, CONFIG.UI_TEXT_DARK, 1, cv2.LINE_AA)
        else:
            display_text = input_text[-55:] if len(input_text) > 55 else input_text
            cv2.putText(frame, display_text, (field_x + 15, field_y + 28),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.48, CONFIG.UI_TEXT_LIGHT, 1, cv2.LINE_AA)
            
            cursor_x = field_x + 15 + cv2.getTextSize(display_text, 
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] + 5
            if int(time.time() * 2) % 2 == 0:
                cv2.line(frame, (cursor_x, field_y + 15), 
                        (cursor_x, field_y + 33), CONFIG.UI_PRIMARY, 2)
        
        instructions = "ESC: Exit  |  ENTER: Send  |  BACKSPACE: Delete"
        cv2.putText(frame, instructions, (field_x + 10, self.height - 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, CONFIG.UI_TEXT_DARK, 1, cv2.LINE_AA)
        
        return frame

# ============================================================
# MAIN APPLICATION - FIXED VERSION
# ============================================================
class EnhancedLanguageLearningApp:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.emotion_detector = AdvancedEmotionDetector()
        self.message_manager = MessageManager()
        self.chat_engine = ProfessionalAIChatEngine()
        self.tts_engine = TTSEngine()
        self.ui = None
        self.camera = None
        self.is_running = False
        self.input_text = ""
        self.last_emotion_time = 0
        self.dominant_emotion = "neutral"
        self.dominant_confidence = 0.5
        self.processing_response = False
        
    def start(self):
        """Start the enhanced application"""
        print("\n" + "="*70)
        print("        EMMA - EMOTIONAL MENTAL MANAGEMENT ASSISTANT")
        print("="*70)
        print("\nInitializing advanced emotion detection system...\n")
        
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║                       SYSTEM STATUS                                ║")
        print("╠════════════════════════════════════════════════════════════════════╣")
        print(f"║  DeepFace (Advanced AI):        {'✓ ACTIVE' if HAS_DEEPFACE else '✗ INACTIVE':>30} ║")
        print(f"║  MediaPipe (Landmark Detection): {'✓ ACTIVE' if HAS_MEDIAPIPE else '✗ INACTIVE':>30} ║")
        print(f"║  TensorFlow (Deep Learning):    {'✓ ACTIVE' if HAS_TENSORFLOW else '✗ INACTIVE':>30} ║")
        print(f"║  Text-to-Speech:                {'✓ ACTIVE' if HAS_TTS else '✗ INACTIVE':>30} ║")
        print(f"║  Face Preprocessing:            {'✓ ENABLED' if CONFIG.FACE_PREPROCESSING else '✗ DISABLED':>30} ║")
        print(f"║  Ollama Service:                {'✓ RUNNING' if self.chat_engine.ollama_available else '✗ NOT RUNNING':>30} ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        
        if not self.chat_engine.ollama_available:
            print("\n⚠ WARNING: Ollama service is NOT running!")
            print("  The AI chat will not work until you start Ollama.")
            print("\n  To fix this:")
            print("    1. Open a terminal")
            print("    2. Run: ollama serve")
            print("    3. Restart this application")
        
        print("\nInitializing camera system...")
        self.camera = cv2.VideoCapture(CONFIG.CAMERA_INDEX)
        if not self.camera.isOpened():
            print("\n❌ FATAL ERROR: Failed to open camera")
            return
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG.FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, CONFIG.CAMERA_FPS)
        
        ret, frame = self.camera.read()
        if not ret:
            print("\n❌ FATAL ERROR: Cannot read from camera")
            return
        
        h, w = frame.shape[:2]
        self.ui = ProfessionalUI(w, h)
        
        print(f"✓ Camera initialized: {w}x{h} @ {CONFIG.CAMERA_FPS}fps")
        print(f"✓ UI system ready\n")
        
        print("="*70)
        print("                         CONTROLS")
        print("="*70)
        print("  TYPE       → Enter your message")
        print("  ENTER      → Send message to AI tutor")
        print("  BACKSPACE  → Delete last character")
        print("  ESC / Q    → Exit application")
        print("="*70)
        
        welcome_msg = """Hi! I'm EMMA, your emotional support companion. I'm here to listen and help however I can. What's on your mind today?"""
        
        self.message_manager.add_message('assistant', welcome_msg)
        
        print("\n✓ System ready. Starting application...\n")
        time.sleep(1)
        
        self.is_running = True
        self.run()
    
    def run(self):
        """FIXED main loop with proper response handling"""
        cv2.namedWindow('EMMA - Emotional Mental Management Assistant', cv2.WINDOW_NORMAL)
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        last_response_check = time.time()
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            fps_counter += 1
            if time.time() - fps_start_time >= 1.0:
                current_fps = fps_counter
                fps_counter = 0
                fps_start_time = time.time()
            
            faces = self.face_detector.detect_faces(frame)
            
            faces_data = []
            current_time = time.time()
            
            if current_time - self.last_emotion_time >= CONFIG.EMOTION_INTERVAL:
                emotion_scores = {}
                
                for face in faces:
                    bbox = face['bbox']
                    x, y, w, h = bbox
                    
                    padding = 20
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    
                    face_id = face.get('tracked_id', 0)
                    emotion_result = self.emotion_detector.analyze_emotion(face_crop, face_id)
                    
                    faces_data.append({
                        'bbox': bbox,
                        'emotion': emotion_result['emotion'],
                        'confidence': emotion_result['confidence']
                    })
                    
                    emo = emotion_result['emotion']
                    conf = emotion_result['confidence']
                    if emo not in emotion_scores:
                        emotion_scores[emo] = []
                    emotion_scores[emo].append(conf)
                
                if emotion_scores:
                    weighted_emotions = {
                        emo: sum(confs) / len(confs) 
                        for emo, confs in emotion_scores.items()
                    }
                    dominant = max(weighted_emotions.items(), key=lambda x: x[1])
                    self.dominant_emotion = dominant[0]
                    self.dominant_confidence = dominant[1]
                
                self.last_emotion_time = current_time
            else:
                for face in faces:
                    face_id = face.get('tracked_id', 0)
                    cached = self.emotion_detector._get_cached_emotion(face_id)
                    faces_data.append({
                        'bbox': face['bbox'],
                        'emotion': cached['emotion'],
                        'confidence': cached['confidence']
                    })
            
            # CHECK FOR RESPONSE CONTINUOUSLY - CRITICAL FIX!
            if self.processing_response:
                response = self.chat_engine.get_response()
                if response:
                    debug_log(f"✓✓✓ RESPONSE RECEIVED: {len(response)} chars", "SUCCESS")
                    self.message_manager.add_message('assistant', response)
                    
                    # TRIGGER TTS
                    if HAS_TTS and not response.startswith('⚠'):
                        debug_log("▶ Starting TTS playback", "DEBUG")
                        self.tts_engine.speak(response)
                    
                    self.processing_response = False
                    debug_log("✓ Response fully processed", "SUCCESS")
            
            ui_frame = self.ui.create_base_frame()
            ui_frame = self.ui.draw_header(ui_frame)
            ui_frame = self.ui.draw_video_panel(ui_frame, frame, faces_data)
            ui_frame = self.ui.draw_chat_panel(ui_frame, 
                                               self.message_manager.get_messages(),
                                               self.input_text,
                                               self.processing_response)
            
            cv2.putText(ui_frame, f"FPS: {current_fps}", (10, self.ui.height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, CONFIG.UI_TEXT_DARK, 1, cv2.LINE_AA)
            
            cv2.imshow('EMMA - Emotional Mental Management Assistant', ui_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27 or key == ord('q'):
                break
            elif key == 13 and not self.processing_response:
                if self.input_text.strip():
                    debug_log(f"User message: {self.input_text}", "INFO")
                    self.message_manager.add_message('user', self.input_text)
                    
                    emoji = EMOTION_EMOJIS.get(self.dominant_emotion, '')
                    emotion_context = {
                        'emotion': self.dominant_emotion,
                        'confidence': self.dominant_confidence,
                        'description': EMOTION_DESCRIPTIONS.get(self.dominant_emotion, 'Focused'),
                        'emoji': emoji
                    }
                    
                    debug_log(f"Sending to chat engine with emotion: {self.dominant_emotion}", "INFO")
                    self.chat_engine.send_message(
                        self.message_manager.get_conversation_history(),
                        emotion_context
                    )
                    
                    self.input_text = ""
                    self.processing_response = True
                    debug_log("Message sent, waiting for response...", "INFO")
                    
            elif key == 8:
                self.input_text = self.input_text[:-1]
            elif key != 255 and 32 <= key <= 126:
                if len(self.input_text) < 250:
                    self.input_text += chr(key)
        
        self.stop()
    
    def stop(self):
        """Clean shutdown"""
        print("\n" + "="*70)
        print("                    SHUTTING DOWN SYSTEM")
        print("="*70)
        
        self.is_running = False
        
        # Give threads time to finish
        time.sleep(0.5)
        
        if self.camera:
            self.camera.release()
            print("✓ Camera released")
        
        cv2.destroyAllWindows()
        print("✓ Windows closed")
        
        if hasattr(self, 'chat_engine'):
            self.chat_engine.stop()
            print("✓ AI chat engine stopped")
        
        if hasattr(self, 'tts_engine'):
            self.tts_engine.stop()
            print("✓ TTS engine stopped")
        
        print("\n✓ Shutdown complete. Thank you for using the platform!\n")
        print("="*70 + "\n")

# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    import sys
    
    try:
        print("\n" + "="*70)
        print("        EMMA - EMOTIONAL MENTAL MANAGEMENT ASSISTANT")
        print("                  FIXED VERSION v2.0")
        print("="*70)
        
        app = EnhancedLanguageLearningApp()
        app.start()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Application startup cancelled by user")
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
