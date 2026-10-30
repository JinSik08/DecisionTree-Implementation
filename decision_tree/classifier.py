"""Classification tree skeleton."""
from .node import Node
from .base import BaseDecisionTree
import numpy as np

class DecisionTreeClassifier(BaseDecisionTree):
    """Tree variant for classification.

    Parameters:
        max_depth (int or None): See ``BaseDecisionTree``.
        min_samples_split (int): See ``BaseDecisionTree``.
        criterion (str): Supported values: 'gini', 'entropy'.

    Each hook below documents what it receives and must return; see
    ``BaseDecisionTree`` for the full contract.
    """
    # Criterion must be one of these values ('gini', 'entropy')
    _valid_criteria = ("gini", "entropy")

    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        super().__init__(max_depth, min_samples_split, criterion)

    def _build_tree(self, X, y, depth):
        """Recursively grow the tree.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Current depth (root is 0).

        Returns:
            Node: Leaf (when a stopping rule triggers) or internal node with
            two subtrees built from the best split.
        """
        # Check stopping conditions: max depth reached, min samples split, or pure node
        if (self.max_depth is not None and depth >= self.max_depth) or len(y) < self.min_samples_split or self._impurity(y) == 0:
            return Node(value = self._leaf_value(y))

        # Find the best split
        best_split = self._best_split(X, y)
        
        # If no valid split is found, return a leaf node
        if best_split is None:
            return Node(value = self._leaf_value(y))

        feature_index, threshold = best_split

        # Split the data into left and right subsets
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        # Build the left and right subtrees recursively
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        # Return the current node with its subtrees
        return Node(feature_index = feature_index, threshold = threshold, left = left_subtree, right = right_subtree)
        
    def _impurity(self, y):
        """Impurity of ``y`` per ``self.criterion``: Gini impurity 1 - sum(p_k^2) or entropy -sum(p_k * log2(p_k)), where p_k is the fraction of samples of class k.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity (0 if pure).
        """
        # An empty node is considered pure (no samples to split)
        if len(y) == 0:
            return 0.0
        
        # Calculate the proportions of each class in y
        _, counts = np.unique(y, return_counts = True)
        probabilities = counts / len(y)

        # Calculate Gini impurity or entropy based on the criterion
        if self.criterion == "gini":
            return 1 - np.sum(probabilities ** 2)
        elif self.criterion == "entropy":
            return -np.sum(probabilities * np.log2(probabilities))

        # If the criterion is not valid, raise an error
        raise ValueError(f"Invalid criterion: {self.criterion}. Supported values are 'gini' and 'entropy'.")

    def _best_split(self, X, y):
        """Search all features/thresholds for the largest impurity decrease.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)``, or
            None if no split reduces impurity.
        """
        # Calculate the impurity before the split
        parent_impurity = self._impurity(y)

        best_gain = 0
        best_split = None

        # Skip splitting if the node is pure or has too few samples
        if parent_impurity == 0 or len(y) < self.min_samples_split:
            return None
        
        # Examine each feature for potential splits
        for feature_index in range(X.shape[1]):
            # Get unique values for the feature
            values = np.unique(X[:, feature_index])

            # Skip if there is only one unique value (no split possible)
            if len(values) < 2:
                continue

            # Consider thresholds between unique values
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                # Split the data based on the threshold
                left_mask = X[:, feature_index] <= threshold
                right_mask = X[:, feature_index] > threshold

                # Skip if either side has empty samples (no split)
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                y_left = y[left_mask]
                y_right = y[right_mask]

                # Calculate the weighted impurity after the split
                weighted_impurity = (len(y_left) * self._impurity(y_left) + len(y_right) * self._impurity(y_right)) / len(y)

                # Calculate the gain from this split
                gain = parent_impurity - weighted_impurity

                # Update the best split if this gain is better
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold)

        return best_split

    def _leaf_value(self, y):
        """Return the majority (most frequent) class of y.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The prediction for the leaf.
        """
        # Get the unique classes and their counts
        classes, counts = np.unique(y, return_counts = True)

        # Return the class with the highest count
        highest_count_class = classes[np.argmax(counts)]
        
        return highest_count_class

    def _score(self, y_true, y_pred):
        """Metric for ``eval``: accuracy = fraction of correct predictions.

        Args:
            y_true (ndarray, shape (n_samples,)): True targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        # Calculate the number of correct predictions
        correct_predictions = np.sum(y_true == y_pred)

        # Return the accuracy as the fraction of correct predictions
        accuracy = correct_predictions / len(y_true)
        
        return accuracy
