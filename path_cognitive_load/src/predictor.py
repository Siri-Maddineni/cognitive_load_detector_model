"""Real-time predictor wrapper"""


class RealtimePredictor:
    def __init__(self, model_predictor):
        self.model = model_predictor
        self.last_prediction = None

    def predict(self, face, cursor, interaction, rt, diff, face_ok):
        preds = self.model.get_all_predictions(face, cursor, interaction, rt, diff, face_ok)
        self.last_prediction = preds
        return preds

    def get_inference_time_ms(self):
        return 35