"""ADHD and Dyslexia risk analyzer"""
import numpy as np


class ADHDDyslexiaAnalyzer:

    def analyze_adhd_risk(self, cursor_features_list, interaction_features_list):
        if not cursor_features_list:
            return 0.1
        jitters = [c[1] for c in cursor_features_list if len(c) > 1]
        avg_jitter = np.mean(jitters) if jitters else 0
        jitter_score = min(1.0, avg_jitter / 50)
        response_times = [i[0] for i in interaction_features_list if i]
        if len(response_times) > 1:
            consistency_score = min(1.0, np.std(response_times) / 5)
        else:
            consistency_score = 0.3
        correction_rates = [i[4] for i in interaction_features_list if len(i) > 4]
        avg_correction = np.mean(correction_rates) if correction_rates else 0
        correction_score = min(1.0, avg_correction * 2)
        adhd_risk = (jitter_score * 0.4 + consistency_score * 0.35 + correction_score * 0.25)
        return min(0.95, adhd_risk)

    def analyze_dyslexia_risk(self, face_features_list, interaction_features_list):
        if not interaction_features_list:
            return 0.1
        read_to_answer = [i[6] for i in interaction_features_list if len(i) > 6]
        avg_read_time = np.mean(read_to_answer) if read_to_answer else 0
        reading_score = min(1.0, avg_read_time / 10)
        correction_rates = [i[4] for i in interaction_features_list if len(i) > 4]
        avg_correction = np.mean(correction_rates) if correction_rates else 0
        correction_score = min(1.0, avg_correction * 2)
        gaze_stability = [f.get('gaze_stability', 0.5) for f in face_features_list]
        avg_gaze = np.mean(gaze_stability) if gaze_stability else 0.5
        gaze_score = 1 - avg_gaze
        dyslexia_risk = (reading_score * 0.4 + correction_score * 0.35 + gaze_score * 0.25)
        return min(0.95, dyslexia_risk)

    def get_recommendations(self, adhd_risk, dyslexia_risk):
        recs = []
        if adhd_risk > 0.6:
            recs.append("🔹 Provide structured breaks every 15-20 minutes")
            recs.append("🔹 Use timer-based task segmentation")
            recs.append("🔹 Minimize visual distractions on screen")
        elif adhd_risk > 0.3:
            recs.append("🔸 Encourage note-taking to maintain focus")
            recs.append("🔸 Use progress indicators")
        if dyslexia_risk > 0.6:
            recs.append("🔹 Offer text-to-speech options")
            recs.append("🔹 Use dyslexia-friendly fonts")
            recs.append("🔹 Provide extended time for reading tasks")
        elif dyslexia_risk > 0.3:
            recs.append("🔸 Highlight key terms in questions")
            recs.append("🔸 Allow verbal responses when possible")
        if not recs:
            recs.append("✅ No significant indicators detected. Continue current approach.")
        return recs