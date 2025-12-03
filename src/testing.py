class Testing:
    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    def predict_new_data(self, new_data):
        new_scaled = self.scaler.transform(new_data)
        pred = self.model.predict(new_scaled)
        prob = self.model.predict_proba(new_scaled)[0][1]
        print("Prediction:", pred[0])
        print("Probability of defect:", prob)