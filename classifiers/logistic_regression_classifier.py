from sklearn.linear_model import LogisticRegression
from src.base_classifier import BaseClassifier

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(model_name="Logistic Regression")
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importances(self):
        return self.model.coef_[0]

    def optimize(self, X_train, y_train, metric='accuracy'):
        param_grid = {
            'C': [0.1, 1, 10],
            'max_iter': [100, 200, 300]
        }
        super().optimize(X_train, y_train, param_grid, metric)