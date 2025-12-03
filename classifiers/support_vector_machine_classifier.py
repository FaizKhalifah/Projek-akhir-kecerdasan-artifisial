from sklearn.svm import SVC
from src.base_classifier import BaseClassifier

class SupportVectorMachineClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(model_name="Support Vector Machine")
        self.model = SVC(probability=True)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importances(self):
        return None

    def optimize(self, X_train, y_train, metric='accuracy'):
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        super().optimize(X_train, y_train, param_grid, metric)