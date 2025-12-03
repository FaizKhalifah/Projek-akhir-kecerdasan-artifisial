
from classifiers.random_forest_classifier import RandomForestClassifier
from classifiers.decision_tree_classifier import DecisionTreeClassifier
from classifiers.logistic_regression_classifier import LogisticRegressionClassifier
from classifiers.support_vector_machine_classifier import SupportVectorMachineClassifier
from classifiers.xgboost_classifier import XGBoostClassifier
from src.data_loader import DataLoader
from src.preprocessing import PreProcessing
from src.training import Training

if __name__ == "__main__":
    # Load data
    data_loader = DataLoader("data/software_defect.csv")
    df = data_loader.load_data()

    # Preprocess data
    preprocessor = PreProcessing(df)
    df_cleaned = preprocessor.clean_data()
    df_normalized = preprocessor.normalize_data()
    df_balanced = preprocessor.handle_imbalanced_data()
    X, y = preprocessor.split_features_and_labels()

    # Initialize classifiers
    xgb_model = XGBoostClassifier()
    lr_model = LogisticRegressionClassifier()
    svm_model = SupportVectorMachineClassifier()
    rf_model = RandomForestClassifier()
    dt_model = DecisionTreeClassifier()

    # Initialize Training class with classifiers, optimization option, metric, output file, output folder, and show_plots option
    training = Training(X, y, classifiers=[xgb_model, lr_model, svm_model, rf_model, dt_model], optimize=False, metric='f1_score', output_file='evaluation_report.csv', output_folder='outputs', show_plots=False)

    # Train-test split
    X_train, X_test, y_train, y_test = training.train_test_split()

    # Train and optionally optimize individual models
    training.train_models(X_train, y_train)

    # Evaluate individual models
    training.evaluate_models(X_test, y_test)

    # Train ensemble model
    voting_clf = training.train_ensemble(X_train, y_train)

    # Evaluate ensemble model
    training.evaluate_ensemble(X_test, y_test, voting_clf)