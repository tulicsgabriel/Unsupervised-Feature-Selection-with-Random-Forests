# Feature Selector

## Description
The `feature_selector.py` script provides a set of tools for unsupervised feature selection based on tree-based models, primarily Random Forests. It leverages the power of random labeling to gauge the importance of features consistently.

## Usage
To use the Feature Selector, follow these steps:
1. Import the necessary classes/functions from the script.
2. Instantiate the `FeatureSelector` class with your dataset.
3. Use the `tree_based_feature_importance` method to perform feature selection.

## Methods
- `calculate_iterations`: Determines the number of iterations based on the number of features.
- `tree_based_feature_importance`: Performs feature selection by training a RandomForest with random labels.

## Parameters
- `n_select_features`: Number of features to retain after selection.
- `rf_params`: Parameters for the RandomForest classifier.

## Dependencies
- pandas
- numpy
- scikit-learn

## Example
```python
from feature_selector import FeatureSelector
selector = FeatureSelector(dataframe)
selected_features = selector.tree_based_feature_importance(n_select_features=100)
