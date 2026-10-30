"""Base class for decision trees with a scikit-learn like API."""
from abc import ABC, abstractmethod
import numpy as np

class BaseDecisionTree(ABC):
    """Skeleton shared by every tree variant (classification, regression, ...).

    Hyperparameters are set in ``__init__``; ``fit``, ``predict`` and ``eval``
    follow the scikit-learn conventions. Subclasses implement the abstract
    hooks; the TODOs are left to the students.

    Parameters:
        max_depth (int or None): Maximum depth of the tree. The root has depth
            0. ``None`` means nodes grow until they are pure or smaller than
            ``min_samples_split``.
        min_samples_split (int): Minimum number of samples a node needs to be
            considered for splitting. Must be >= 2.
        criterion (str): Name of the impurity measure used to choose splits
            (e.g. 'gini', 'entropy', 'mse'). Valid values depend on the
            subclass.

    Attributes:
        root (Node or None): Root of the fitted tree; ``None`` before ``fit``.
    """

    def __init__(self, max_depth=None, min_samples_split=2, criterion=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def get_params(self):
        """Return the hyperparameters.

        Returns:
            dict: Maps 'max_depth', 'min_samples_split' and 'criterion' to
            their current values. Add new hyperparameters here.
        """
        return {
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "criterion": self.criterion,
        }

    def set_params(self, **params):
        """Update hyperparameters by keyword, e.g. ``set_params(max_depth=5)``.

        Args:
            **params: Hyperparameter names and new values.

        Returns:
            BaseDecisionTree: ``self``, to allow chaining.

        Raises:
            ValueError: If a name is not a known hyperparameter.
        """
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self

    def fit(self, X, y):
        """Build the tree from training data.

        TODO (students): implement this method. It should
          1. validate the hyperparameters (e.g. ``max_depth`` is None or a
             positive int, ``min_samples_split >= 2``, ``criterion`` is
             supported) and raise ``ValueError`` otherwise;
          2. convert ``X`` and ``y`` to arrays and check that they have the
             same number of samples;
          3. call ``self._build_tree(X, y, depth=0)`` and store the returned
             node in ``self.root``;
          4. return ``self``.

        Args:
            X (array-like, shape (n_samples, n_features)): Training features.
            y (array-like, shape (n_samples,)): Targets (class labels for
                classification, real values for regression).

        Returns:
            BaseDecisionTree: ``self`` (the fitted estimator).
        """
        # max_depth must be None, 0, or a positive integer
        if self.max_depth is not None and (not isinstance(self.max_depth, int) or self.max_depth < 0):
                raise ValueError("max_depth must be None, 0, or a positive integer.")
        
        # min_samples_split must be an integer >= 2
        if not isinstance(self.min_samples_split, int) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2.")
        
        # criterion must be one of the valid criteria defined in the subclass
        valid_criteria = getattr(self, "_valid_criteria", [])
        if valid_criteria is not None and self.criterion not in valid_criteria:
            raise ValueError(f"criterion must be one of {valid_criteria}.")

        # Convert X and y to arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Check that X and y have the same number of samples
        if X.ndim != 2: # X with 2 dimensions (samples, features)
            raise ValueError("X must be a 2D array.")
        if y.ndim != 1: # y with 1 dimension (targets)
            raise ValueError("y must be a 1D array.")
        if X.shape[0] != y.shape[0]: # Check that the number of samples in X and y are the same
            raise ValueError("X and y must have the same number of samples.")
        if X.shape[0] == 0 or X.shape[1] == 0: # Check that training data has at least one sample and one feature
            raise ValueError("X must have at least one sample and one feature.")

        # Build the tree
        self.root = self._build_tree(X, y, depth=0)

        return self

    def predict(self, X):
        """Predict a target for each row of ``X``.

        TODO (students): implement this method. It should raise an error if
        the model has not been fitted (``self.root is None``) and, for each
        sample, walk from ``self.root``: go left when
        ``x[feature_index] <= threshold`` else right, until a leaf is
        reached, then return that leaf's ``value``.

        Args:
            X (array-like, shape (n_samples, n_features)): Samples to predict.

        Returns:
            array, shape (n_samples,): Predicted target for every sample.
        """
        # The model must be fitted before making predictions
        if self.root is None:
            raise ValueError("The model has not been fitted yet. Please call 'fit' before 'predict'.")

        # Convert X to an array
        X = np.asarray(X)

        # Check that X is a 2D array
        if X.ndim != 2:
            raise ValueError("X must be a 2D array.")

        # Make predictions
        predictions = []
        for sample in X:
            node = self.root
            # Walk down the tree until a leaf is reached (left if smaller than or equal to threshold, right otherwise)
            while not node.is_leaf:
                if sample[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            # Append the value of the leaf node to predictions if a leaf was reached
            predictions.append(node.value)

        return np.array(predictions)

    def eval(self, X, y):
        """Score the model on ``(X, y)``.

        TODO (students): implement this method. It should call
        ``self.predict(X)`` and return ``self._score(y, y_pred)``.

        Args:
            X (array-like, shape (n_samples, n_features)): Samples to evaluate.
            y (array-like, shape (n_samples,)): True targets for ``X``.

        Returns:
            float: Metric value (accuracy for classification, R^2 for
            regression; higher is better).
        """
        # Convert y to arrays
        y = np.asarray(y)

        # Check that y has 1 dimension
        if y.ndim != 1:
            raise ValueError("y must be a 1D array.")

        # Make predictions
        y_pred = self.predict(X)

        # Check that the number of predictions matches the number of true targets
        if y_pred.shape[0] != y.shape[0]:
            raise ValueError("The number of predictions does not match the number of true targets.")

        # Compute and return the score
        return self._score(y, y_pred)

    @abstractmethod
    def _build_tree(self, X, y, depth):
        """Recursively grow the tree and return its root ``Node``.

        TODO (students): create a leaf (``Node(value=self._leaf_value(y))``)
        when a stopping condition holds: ``depth`` reached ``max_depth``,
        fewer than ``min_samples_split`` samples, ``y`` is pure, or
        ``_best_split`` finds no useful split. Otherwise split the data on the
        returned (feature, threshold) and recurse on both parts with
        ``depth + 1``.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at this node.
            y (ndarray, shape (n_samples,)): Targets at this node.
            depth (int): Depth of this node (root is 0).

        Returns:
            Node: Subtree built from this data.
        """
        pass

    @abstractmethod
    def _impurity(self, y):
        """Compute the impurity of a set of targets.

        TODO (students): implement the measure selected by ``self.criterion``
        (e.g. Gini or entropy for classification, MSE or MAE for regression).
        Lower means more homogeneous; it is used to score candidate splits.

        Args:
            y (ndarray, shape (n_samples,)): Targets of a node.

        Returns:
            float: Impurity of ``y`` (0 for a perfectly pure node).
        """
        pass

    @abstractmethod
    def _best_split(self, X, y):
        """Find the split that most reduces impurity.

        TODO (students): for each feature and candidate threshold (e.g.
        midpoints between sorted unique values), partition ``y`` into left
        (``x <= threshold``) and right, compute the weighted impurity
        ``(n_l * imp_l + n_r * imp_r) / n`` and keep the split with the
        largest gain over ``self._impurity(y)``. Skip splits leaving a side
        empty.

        Args:
            X (ndarray, shape (n_samples, n_features)): Features at the node.
            y (ndarray, shape (n_samples,)): Targets at the node.

        Returns:
            tuple (int, float) or None: ``(feature_index, threshold)`` of the
            best split, or ``None`` if no split improves impurity.
        """
        pass

    @abstractmethod
    def _leaf_value(self, y):
        """Compute the prediction stored in a leaf.

        TODO (students): return the majority class for classification or the
        mean of ``y`` for regression.

        Args:
            y (ndarray, shape (n_samples,)): Targets that reached the leaf.

        Returns:
            The single predicted value for the leaf.
        """
        pass

    @abstractmethod
    def _score(self, y_true, y_pred):
        """Compute the metric reported by ``eval``.

        TODO (students): e.g. accuracy for classification, R^2 for regression.

        Args:
            y_true (ndarray, shape (n_samples,)): Ground-truth targets.
            y_pred (ndarray, shape (n_samples,)): Predicted targets.

        Returns:
            float: Score (higher is better).
        """
        pass
