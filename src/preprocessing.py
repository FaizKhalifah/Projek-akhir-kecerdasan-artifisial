import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

class PreProcessing:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def clean_data(self):
        self.df = self.df.replace("?", np.nan)
        self.df = self.df.apply(pd.to_numeric)
        self.df = self.df.fillna(self.df.median(numeric_only=True))
        print("After cleaning:", self.df.shape)
        return self.df
    
    def handle_imbalanced_data(self):
        X = self.df.drop(columns=["defects"])
        y = self.df["defects"].astype(int)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        self.df = pd.concat([X_resampled, y_resampled], axis=1)
        print("After handling imbalance:", self.df.shape)
        return self.df

    def split_features_and_labels(self):
        X = self.df.drop(columns=["defects"])
        y = self.df["defects"].astype(int)
        return X, y