#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:24:21 2023

@author: gabriel
"""

import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

from sklearn.datasets import make_classification


class FeatureSelector:
    def __init__(self, dataframe):
        self.df = dataframe

    def calculate_iterations(
        self, n_features, base_iterations=100, multiplier=50
    ):
        """
        Calculate the number of iterations based on the number of features.
        """
        n_iterations = base_iterations + multiplier * np.log(n_features)
        return int(n_iterations)

    def tree_based_feature_importance(self, n_select_features, rf_params=None):
        """
        Unsupervised Feature Selection with Random Forests (UFSRF)
        """
        # Number of features to select
        n_features = n_select_features

        # Calculate the number of iterations based on the total features
        n_iterations = self.calculate_iterations(self.df.shape[1])

        # Placeholder for feature importances across all iterations
        aggregated_importances = np.zeros(self.df.shape[1])

        # Ensure all data is float type
        df_float = self.df.astype(float)

        # Setting default RandomForest parameters if none provided
        if rf_params is None:
            rf_params = {
                "n_estimators": 150,
                "max_depth": 10,
                "random_state": 42,
            }

        for _ in range(n_iterations):
            # Generating random discrete class labels for classification
            random_labels = np.random.randint(0, 2, df_float.shape[0])
            model = RandomForestClassifier(**rf_params)
            model.fit(df_float, random_labels)
            aggregated_importances += model.feature_importances_

        # Averaging feature importances over all iterations
        aggregated_importances /= n_iterations

        # Sorting features based on their aggregated importance scores
        important_features = df_float.columns[
            np.argsort(aggregated_importances)[-n_features:]
        ]

        return df_float[important_features]


if __name__ == "__main__":

    X, y = make_classification(
        n_samples=200,
        n_features=1000,
        n_informative=100,
        n_redundant=900,
        random_state=42,
    )
    df = pd.DataFrame(X)

    # Experiment 1
    print("Experiment 1:")
    selector = FeatureSelector(df)
    selected_features_df = selector.tree_based_feature_importance(
        n_select_features=100
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    auc_scorer = make_scorer(
        roc_auc_score, greater_is_better=True, needs_proba=True
    )

    full_dataset_auc = cross_val_score(
        clf, df, y, cv=10, scoring=auc_scorer
    ).mean()
    reduced_features_auc = cross_val_score(
        clf, selected_features_df, y, cv=10, scoring=auc_scorer
    ).mean()

    print(f"AUC using full_dataset_auc: {full_dataset_auc}")
    print(f"AUC using reduced_features_auc: {reduced_features_auc}")
    print("----------------------------------\n")

    # Experiment 2
    print("Experiment 2:")
    selected_features_df = selector.tree_based_feature_importance(
        n_select_features=300
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    auc_scorer = make_scorer(
        roc_auc_score, greater_is_better=True, needs_proba=True
    )

    full_dataset_auc = cross_val_score(
        clf, df, y, cv=10, scoring=auc_scorer
    ).mean()
    reduced_features_auc = cross_val_score(
        clf, selected_features_df, y, cv=10, scoring=auc_scorer
    ).mean()

    print(f"AUC using full_dataset_auc: {full_dataset_auc}")
    print(f"AUC using reduced_features_auc: {reduced_features_auc}")
