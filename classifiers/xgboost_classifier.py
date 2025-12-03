from xgboost import XGBClassifier
from src.base_classifier import BaseClassifier

class XGBoostClassifier(BaseClassifier):
    def __init__(self):
        super().__init__(model_name="XGBoost")
        self.model = XGBClassifier(eval_metric="logloss")

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
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        super().optimize(X_train, y_train, param_grid, metric)