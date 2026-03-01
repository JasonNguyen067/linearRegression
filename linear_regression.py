"""
Linear Regression from Scratch
================================
Implement a linear regression model using gradient descent.
You will build the core math: cost function, gradients, and parameter updates.

Concepts covered:
  - Hypothesis function: h(x) = X @ theta
  - Mean Squared Error cost
  - Batch gradient descent
  - Prediction
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression model trained via batch gradient descent.

    Attributes:
        learning_rate (float): Step size for gradient descent.
        n_iterations (int): Number of gradient descent iterations.
        theta (np.ndarray): Model parameters (weights), shape (n_features,).
        cost_history (list): MSE cost recorded after each iteration.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.theta = None
        self.cost_history = []

    # -------------------------------------------------------------------------
    # Part 1: Cost Function
    # -------------------------------------------------------------------------

    def compute_cost(self, X, y):
        """
        Compute the Mean Squared Error cost.

            J(theta) = (1 / 2m) * sum((X @ theta - y) ** 2)

        Args:
            X (np.ndarray): Feature matrix with bias column, shape (m, n).
            y (np.ndarray): Target values, shape (m,).

        Returns:
            float: The MSE cost J(theta).
        """
        m = len(y)

        # TODO 1: Compute predictions using X and self.theta
        predictions = None  # shape (m,)

        # TODO 2: Compute the error (predictions - y)
        errors = None  # shape (m,)

        # TODO 3: Compute and return the cost J = (1/2m) * sum(errors^2)
        cost = None

        return cost

    # -------------------------------------------------------------------------
    # Part 2: Gradient Descent
    # -------------------------------------------------------------------------

    def compute_gradients(self, X, y):
        """
        Compute gradients of the cost with respect to theta.

            grad = (1/m) * X.T @ (X @ theta - y)

        Args:
            X (np.ndarray): Feature matrix with bias column, shape (m, n).
            y (np.ndarray): Target values, shape (m,).

        Returns:
            np.ndarray: Gradient vector, shape (n,).
        """
        m = len(y)

        # TODO 4: Compute predictions using X and self.theta
        predictions = None  # shape (m,)

        # TODO 5: Compute the error (predictions - y)
        errors = None

        # TODO 6: Compute and return the gradient: (1/m) * X.T @ errors
        gradients = None  # shape (n,)

        return gradients

    def fit(self, X, y):
        """
        Train the model by running batch gradient descent.

        Initializes theta to zeros, then iteratively updates:
            theta = theta - learning_rate * gradients

        Records the cost after every iteration in self.cost_history.

        Args:
            X (np.ndarray): Feature matrix with bias column, shape (m, n).
            y (np.ndarray): Target values, shape (m,).

        Returns:
            self
        """
        n_features = X.shape[1]

        # TODO 7: Initialize self.theta as a zero vector of shape (n_features,)
        self.theta = None

        for i in range(self.n_iterations):

            # TODO 8: Compute gradients using self.compute_gradients(X, y)
            gradients = None

            # TODO 9: Update self.theta using the gradient descent rule
            #         theta = theta - learning_rate * gradients
            pass

            # TODO 10: Compute current cost and append to self.cost_history
            cost = None
            self.cost_history.append(cost)

        return self

    # -------------------------------------------------------------------------
    # Part 3: Prediction
    # -------------------------------------------------------------------------

    def predict(self, X):
        """
        Generate predictions for input X.

            y_hat = X @ theta

        Args:
            X (np.ndarray): Feature matrix with bias column, shape (m, n).

        Returns:
            np.ndarray: Predicted values, shape (m,).
        """
        # TODO 11: Return predictions as X @ self.theta
        pass

    # -------------------------------------------------------------------------
    # Part 4 (Bonus): Normal Equation
    # -------------------------------------------------------------------------

    def fit_normal_equation(self, X, y):
        """
        Fit the model analytically using the Normal Equation (no iteration).

            theta = (X.T @ X)^{-1} @ X.T @ y

        This gives the exact optimal theta but is O(n^3) — slow for large n.

        Args:
            X (np.ndarray): Feature matrix with bias column, shape (m, n).
            y (np.ndarray): Target values, shape (m,).

        Returns:
            self
        """
        # TODO 12 (Bonus): Implement the normal equation using np.linalg.inv
        #                  or np.linalg.pinv (more numerically stable)
        pass

        return self
