"""M1-M5 Model Architectures"""


class ModelPredictor:
    def __init__(self):
        self.fusion_weights = {'facial': 0.45, 'cursor': 0.30, 'interaction': 0.25}

    def predict_m1(self, face_features, response_time, difficulty):
        load = difficulty
        if face_features.get('eye_closure', 0) > 0.4:
            load = min(2, load + 1)
        return load

    def predict_m2(self, cursor_features, response_time, difficulty):
        load = difficulty
        if len(cursor_features) > 1 and cursor_features[1] > 30:
            load = min(2, load + 1)
        return load

    def predict_m3(self, interaction_features, response_time, difficulty):
        load = difficulty
        if response_time > 7:
            load = min(2, load + 1)
        return load

    def predict_m4(self, all_features, response_time, difficulty):
        loads = []
        if 'face_features' in all_features:
            loads.append(self.predict_m1(all_features['face_features'], response_time, difficulty))
        if 'cursor_features' in all_features:
            loads.append(self.predict_m2(all_features['cursor_features'], response_time, difficulty))
        if 'interaction_features' in all_features:
            loads.append(self.predict_m3(all_features['interaction_features'], response_time, difficulty))
        return int(round(sum(loads) / len(loads))) if loads else difficulty

    def predict_m5(self, face, cursor, interaction, rt, diff, face_ok):
        m1 = self.predict_m1(face, rt, diff)
        m2 = self.predict_m2(cursor, rt, diff)
        m3 = self.predict_m3(interaction, rt, diff)
        fw = 0.45 if face_ok else 0.25
        cw = 0.35
        iw = 0.25 if rt < 10 else 0.50
        t = fw + cw + iw
        w = {'facial': fw / t, 'cursor': cw / t, 'interaction': iw / t}
        wl = m1 * w['facial'] + m2 * w['cursor'] + m3 * w['interaction']
        return int(round(wl)), w

    def get_all_predictions(self, face, cursor, interaction, rt, diff, face_ok):
        m1 = self.predict_m1(face, rt, diff)
        m2 = self.predict_m2(cursor, rt, diff)
        m3 = self.predict_m3(interaction, rt, diff)
        all_f = {'face_features': face, 'cursor_features': cursor, 'interaction_features': interaction}
        m4 = self.predict_m4(all_f, rt, diff)
        m5, w = self.predict_m5(face, cursor, interaction, rt, diff, face_ok)
        conf = 0.5 + 0.3 * (1 - min(1.0, rt / 10))
        return {
            'M1: FAU': {'level': m1, 'confidence': conf},
            'M2: Cursor': {'level': m2, 'confidence': conf},
            'M3: Interaction': {'level': m3, 'confidence': conf},
            'M4: Early': {'level': m4, 'confidence': conf},
            'M5: Late⭐': {'level': m5, 'confidence': conf, 'modality_weights': w},
        }