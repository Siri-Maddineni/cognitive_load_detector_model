"""Facial feature extraction - 21 features"""
import cv2
import numpy as np
from collections import deque


class FaceAnalyzer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.frame_count = 0
        self.face_detected_count = 0
        self.blink_count = 0
        self.eye_closure_history = deque(maxlen=30)

    def extract_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        features = {
            'AU01': 0.0, 'AU02': 0.0, 'AU04': 0.0, 'AU05': 0.0,
            'AU06': 0.0, 'AU07': 0.0, 'AU09': 0.0, 'AU12': 0.0,
            'AU14': 0.0, 'AU15': 0.0, 'AU17': 0.0, 'AU20': 0.0,
            'AU23': 0.0, 'AU25': 0.0, 'AU26': 0.0, 'AU45': 0.0,
            'pose_Rx': 0.0, 'pose_Ry': 0.0, 'pose_Rz': 0.0,
            'gaze_stability': 0.5, 'head_stability': 0.5,
            'face_detected': 0, 'face_lost_ratio': 0,
            'blink_rate': 10, 'eye_closure': 0.2
        }
        self.frame_count += 1
        if len(faces) > 0:
            features['face_detected'] = 1
            self.face_detected_count += 1
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y + h, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            features['AU05'] = min(1.0, w * h / 50000)
            features['AU12'] = 0.3 if len(eyes) >= 2 else 0.1
            if len(eyes) < 2:
                features['eye_closure'] = 0.6
                features['blink_rate'] = 25
            else:
                features['eye_closure'] = 0.2
                features['blink_rate'] = 12
            img_h, img_w = frame.shape[:2]
            features['pose_Ry'] = (x + w / 2 - img_w / 2) / img_w
            features['pose_Rx'] = (y + h / 2 - img_h / 2) / img_h
            features['gaze_stability'] = 0.8 if len(eyes) >= 2 else 0.4
        features['face_lost_ratio'] = 1 - (self.face_detected_count / max(self.frame_count, 1))
        features['head_stability'] = 1 - abs(features['pose_Ry'])
        self.eye_closure_history.append(features['eye_closure'])
        return features

    def get_feature_vector(self, features):
        keys = ['AU01', 'AU02', 'AU04', 'AU05', 'AU06', 'AU07', 'AU09',
                'AU12', 'AU14', 'AU15', 'AU17', 'AU20', 'AU23', 'AU25',
                'AU26', 'AU45', 'pose_Rx', 'pose_Ry', 'pose_Rz',
                'gaze_stability', 'head_stability']
        return [features.get(k, 0.0) for k in keys]

    def reset(self):
        self.frame_count = 0
        self.face_detected_count = 0
        self.blink_count = 0
        self.eye_closure_history.clear()