from sklearn.tree import DecisionTreeClassifier as DTC
from src.base_classifier import BaseClassifier

class DecisionTreeClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(model_name="Decision Tree")
        self.model = DTC()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def get_feature_importances(self):
        return self.model.feature_importances_

    def optimize(self, X_train, y_train, metric='accuracy'):
        param_grid = {
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        super().optimize(X_train, y_train, param_grid, metric)