"""Validation functions for decision tree models."""
import numpy as np
from decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

# Sample dataset to train and validate the tree
X = np.array([
    [False, "steep", "high"],
    [True, "moderate", "low"],
    [True, "steep", "medium"],
    [False, "steep", "medium"],
    [False, "flat", "high"],
    [True, "steep", "highest"],
    [True, "steep", "high"]
])
X_encoded = np.column_stack([
    X[:, 0] == "True",
    X[:, 1] == "flat",
    X[:, 1] == "moderate",
    X[:, 1] == "steep",
    X[:, 2] == "high",
    X[:, 2] == "highest",
    X[:, 2] == "low",
    X[:, 2] == "medium"
])
y = np.array(["chapparal", "riparian", "riparian", "chapparal", "conifer", "conifer", "chapparal"])
y_encoded = np.array([0 if label == "chapparal" else 1 if label == "riparian" else 2 for label in y])

# Test DecisionTreeClassifier with both criteria
for criterion in ["gini", "entropy"]:
    # Create the model
    model = DecisionTreeClassifier(max_depth=3, min_samples_split=2, criterion=criterion)
    # Train the model
    model.fit(X_encoded, y_encoded)
    # Make predictions
    predictions = model.predict(X_encoded)
    # Evaluate the model
    accuracy = model.eval(X_encoded, y_encoded)

    # Print the results
    print(f"DecisionTreeClassifier with criterion '{criterion}':")
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy}")
    print("-" * 50)

# Test DecisionTreeRegressor with both criteria
for criterion in ["mse", "mae"]:
    # Create the model
    model = DecisionTreeRegressor(max_depth=2, min_samples_split=2, criterion=criterion)
    # Train the model
    model.fit(X_encoded, y_encoded)
    # Make predictions
    predictions = model.predict(X_encoded)
    # Evaluate the model
    r2_score = model.eval(X_encoded, y_encoded)

    # Print the results
    print(f"DecisionTreeRegressor with criterion '{criterion}':")
    print(f"Predictions: {np.round(predictions, 4)}")
    print(f"R^2 Score: {r2_score}")
    print("-" * 50)