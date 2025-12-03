from src.base_classifier import BaseClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.metrics import matthews_corrcoef, cohen_kappa_score
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier

import pandas as pd
import os

class Training:
    def __init__(self, X, y, classifiers, optimize=False, metric='accuracy', output_file='evaluation_report.csv', output_folder='outputs', show_plots=False):
        self.X = X
        self.y = y
        self.scaler = StandardScaler()
        self.classifiers = classifiers
        self.optimize = optimize
        self.metric = metric
        self.output_file = output_file
        self.output_folder = output_folder
        self.show_plots = show_plots

    def train_test_split(self):
        X_scaled = self.scaler.fit_transform(self.X)
        return train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

    def train_models(self, X_train, y_train):
        for clf in self.classifiers:
            if self.optimize:
                clf.optimize(X_train, y_train, metric=self.metric)
            clf.train(X_train, y_train)

    def evaluate_models(self, X_test, y_test):
        reports = []
        for clf in self.classifiers:
            report = clf.evaluate(X_test, y_test, metric=self.metric, output_folder=self.output_folder, show_plots=self.show_plots)
            reports.append(report)
            clf.feature_importances(self.X.columns, output_folder=self.output_folder, show_plots=self.show_plots)
        self.save_to_csv(reports)

    def save_to_csv(self, reports):
        # Ensure the output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # Clear the existing file if it exists
        output_path = os.path.join(self.output_folder, self.output_file)
        if os.path.exists(output_path):
            os.remove(output_path)

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(reports)
        
        # Save the DataFrame to a CSV file
        df.to_csv(output_path, index=False)

    def train_ensemble(self, X_train, y_train):
        voting_clf = VotingClassifier(
            estimators=[(clf.model_name, clf.model) for clf in self.classifiers],
            voting='soft'
        )
        voting_clf.fit(X_train, y_train)
        return voting_clf

    def evaluate_ensemble(self, X_test, y_test, voting_clf):
        y_pred = voting_clf.predict(X_test)
        y_prob = voting_clf.predict_proba(X_test)[:, 1]

        report = {
            'model_name': 'Ensemble',
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred),
            'kappa': cohen_kappa_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }

        print("\n\nEnsemble Model Evaluation")
        if self.metric == 'accuracy':
            print("Accuracy:", report['accuracy'])
        elif self.metric == 'f1_score':
            print("F1-score:", report['f1_score'])
        print("\nClassification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("MCC:", report['mcc'])
        print("Cohen Kappa:", report['kappa'])
        print("ROC-AUC:", report['roc_auc'])

        # Ensure the output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.title("ROC Curve - Ensemble")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(os.path.join(self.output_folder, "roc_curve_ensemble.png"))
        if self.show_plots:
            plt.show()

        # Precision Recall Curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        print("Average Precision (AP):", ap)
        plt.figure()
        plt.plot(recall, precision)
        plt.title("Precision Recall Curve - Ensemble")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(os.path.join(self.output_folder, "precision_recall_curve_ensemble.png"))
        if self.show_plots:
            plt.show()

        # Append the ensemble report to the CSV file
        df = pd.DataFrame([report])
        df.to_csv(os.path.join(self.output_folder, self.output_file), index=False, mode='a', header=not os.path.exists(os.path.join(self.output_folder, self.output_file)))