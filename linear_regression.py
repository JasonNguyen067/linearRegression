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

─────────────────────────────────────────────
QUICK REFERENCE — what is X, y, and theta?
─────────────────────────────────────────────

X  — the INPUT data. A 2D table where:
       - each ROW    is one training example  (e.g. one house)
       - each COLUMN is one feature           (e.g. size, bedrooms)
     The first column is always all 1s (the bias column).

     Example with 3 houses and 2 real features:
         bias  size  bedrooms
         1     1500  3          ← house 1
         1     2200  4          ← house 2
         1      900  2          ← house 3
     shape: (m, n) = (3 houses, 3 columns)

y  — the TARGET / answer you are trying to predict. A 1D array.
     One value per training example (one price per house).

         450000   ← house 1 actual price
         620000   ← house 2 actual price
         280000   ← house 3 actual price
     shape: (m,) = (3,)

theta — the WEIGHTS the model is learning. A 1D array.
     One weight per column of X.
     Starts as all zeros. Gradient descent adjusts it every iteration.

         [bias_weight, w1, w2]  =  [5.0, 3.0, -1.5]
     shape: (n,) = (3,)

m  — number of training examples (number of rows in X)
n  — number of features including the bias column (number of columns in X)
─────────────────────────────────────────────
"""

import numpy as np


class LinearRegression:
    """
    Linear Regression model trained via batch gradient descent.

    Attributes:
        learning_rate (float): How big each gradient descent step is.
                               Too high = bounces around, too low = slow.
        n_iterations  (int):   How many times to run the gradient descent loop.
        theta (np.ndarray):    The model weights [bias, w1, w2, ...], shape (n,).
                               Starts as zeros, gets updated every iteration.
        cost_history  (list):  The cost (error) after each iteration.
                               Should decrease over time as the model learns.
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

        In plain English:
            1. Make a prediction for every house  (X @ theta)
            2. Subtract the real price            (- y)
            3. Square each error                  (** 2)
            4. Add them all up and average        (sum / 2m)
            → one number that says "how wrong is the model overall"

        Args:
            X (np.ndarray): The full dataset table, shape (m, n).
                            Rows = houses, Columns = features (bias, size, bedrooms...).
            y (np.ndarray): The real prices for every house, shape (m,).

        Returns:
            float: The cost — one number. Lower is better.
        """
        m = len(y)

        # TODO 1: Compute predictions using X and self.theta
        predictions = X @ self.theta  # shape (m,)

        # TODO 2: Compute the error (predictions - y)
        errors = predictions - y  # shape (m,)

        # TODO 3: Compute and return the cost J = (1/2m) * sum(errors^2)
        cost = (1 / (2 * m)) * sum(errors * errors)

        return cost

    # -------------------------------------------------------------------------
    # Part 2: Gradient Descent
    # -------------------------------------------------------------------------

    def compute_gradients(self, X, y):
        """
        Compute gradients of the cost with respect to theta.

            grad = (1/m) * X.T @ (X @ theta - y)

        In plain English:
            The gradient is the slope of the cost curve at the current weights.
            It tells gradient descent which direction to nudge each weight.
            - Positive gradient → weight is too high → decrease it
            - Negative gradient → weight is too low  → increase it
            - Near zero         → weight is about right → barely move

        Args:
            X (np.ndarray): The full dataset table, shape (m, n).
                            Rows = houses, Columns = features (bias, size, bedrooms...).
            y (np.ndarray): The real prices for every house, shape (m,).

        Returns:
            np.ndarray: One gradient value per weight in theta, shape (n,).
        """
        m = len(y)

        # TODO 4: Compute predictions using X and self.theta
        predictions = X @ self.theta  # shape (m,)

        # TODO 5: Compute the error (predictions - y)
        errors = predictions - y

        # TODO 6: Compute and return the gradient: (1/m) * X.T @ errors
        gradients = (1 / m) * X.T @ errors  # shape (n,)

        return gradients

    def fit(self, X, y):
        """
        Train the model by running batch gradient descent.

        Initializes theta to zeros, then iteratively updates:
            theta = theta - learning_rate * gradients

        In plain English:
            1. Start with all weights = 0 (model knows nothing)
            2. Loop n_iterations times:
               a. Compute the gradient (which way is downhill?)
               b. Nudge every weight slightly downhill
               c. Record the current cost (should get smaller each time)
            3. After the loop, theta holds the best weights found

        Args:
            X (np.ndarray): The full dataset table, shape (m, n).
                            Rows = houses, Columns = features (bias, size, bedrooms...).
            y (np.ndarray): The real prices for every house, shape (m,).

        Returns:
            self
        """
        n_features = X.shape[1]

        # TODO 7: Initialize self.theta as a zero vector of shape (n_features,)
        self.theta = np.zeros(n_features)

        for i in range(self.n_iterations):

            # TODO 8: Compute gradients using self.compute_gradients(X, y)
            gradients = self.compute_gradients(X, y)

            # TODO 9: Update self.theta using the gradient descent rule
            #         theta = theta - learning_rate * gradients
            self.theta = self.theta - self.learning_rate * gradients

            # TODO 10: Compute current cost and append to self.cost_history
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

        return self

    # -------------------------------------------------------------------------
    # Part 3: Prediction
    # -------------------------------------------------------------------------

    def predict(self, X):
        """
        Generate predictions for input X using the learned weights.

            y_hat = X @ theta

        In plain English:
            For each house, multiply every feature by its weight and sum it up.
            That sum is the predicted price for that house.

            Example for one house:
                prediction = (1 * bias) + (1500 * w1) + (3 * w2)
                           = (1 * 5)   + (1500 * 3)  + (3 * -1.5)
                           = 4500.5

            X @ theta does this for ALL houses at once.

        Args:
            X (np.ndarray): The dataset you want predictions for, shape (m, n).
                            Must have the same columns as the training data.

        Returns:
            np.ndarray: Predicted price for every house, shape (m,).
        """
        # TODO 11: Return predictions as X @ self.theta
        predictions = X @ self.theta
        return predictions

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
