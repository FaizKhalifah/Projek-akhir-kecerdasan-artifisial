import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
from sklearn.model_selection import GridSearchCV

class BaseClassifier:
    def __init__(self, model_name):
        self.model_name = model_name

    def train(self, X_train, y_train):
        raise NotImplementedError("Subclass must implement abstract method")

    def evaluate(self, X_test, y_test, metric='accuracy', output_folder='outputs', show_plots=False):
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)[:, 1]

        report = {
            'model_name': self.model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }

        print(f"\n\nModel: {self.model_name}")
        if metric == 'accuracy':
            print("Accuracy:", report['accuracy'])
        elif metric == 'f1_score':
            print("F1-score:", report['f1_score'])
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("MCC:", report['mcc'])
        print("Cohen Kappa:", report['kappa'])
        print("ROC-AUC:", report['roc_auc'])

        # Ensure the output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title(f"ROC Curve - {self.model_name}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(os.path.join(output_folder, f"roc_curve_{self.model_name}.png"))
        if show_plots:
            plt.show()

        # Precision Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        print("Average Precision (AP):", ap)
        plt.figure()
        plt.plot(recall, precision)
        plt.title(f"Precision Recall Curve - {self.model_name}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(output_folder, f"precision_recall_curve_{self.model_name}.png"))
        if show_plots:
            plt.show()

        return report

    def predict(self, X):
        raise NotImplementedError("Subclass must implement abstract method")

    def predict_proba(self, X):
        raise NotImplementedError("Subclass must implement abstract method")

    def get_feature_importances(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def feature_importances(self, feature_names, output_folder='outputs', show_plots=True):
        importances = self.get_feature_importances()
        if importances is not None:
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names, importances)
            plt.title(f"Feature Importance - {self.model_name}")
            plt.savefig(os.path.join(output_folder, f"feature_importances_{self.model_name}.png"))
            if show_plots:
                plt.show()
        else:
            print(f"Feature importances are not available for {self.model_name}")
    
    def optimize(self, X_train, y_train, param_grid, metric='accuracy'):
        scoring = metric if metric in ['accuracy', 'f1'] else 'accuracy'
        grid_search = GridSearchCV(self.model, param_grid, cv=5, scoring=scoring)
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f"Best parameters for {self.model_name}: {grid_search.best_params_}")