"""
Data Preprocessing Utilities
==============================
Before feeding data into a model you need to:
  1. Add a bias (intercept) column to X
  2. Normalize features so gradient descent converges faster
  3. Split data into training and test sets

Concepts covered:
  - Bias / intercept term
  - Z-score (standardization) normalization
  - Train/test splitting with optional shuffling
"""

import numpy as np


# -------------------------------------------------------------------------
# Part 1: Bias Column
# -------------------------------------------------------------------------

def add_bias(X):
    """
    Prepend a column of ones to X so the model can learn an intercept.

    Example:
        X = [[2, 3],        result = [[1, 2, 3],
             [4, 5]]                  [1, 4, 5]]

    Args:
        X (np.ndarray): Feature matrix, shape (m, n).

    Returns:
        np.ndarray: Augmented matrix, shape (m, n+1).
    """
    m = X.shape[0]

    # TODO 1: Create a column of ones with shape (m, 1)
    ones = None

    # TODO 2: Horizontally stack ones with X using np.hstack
    #         and return the result
    pass


# -------------------------------------------------------------------------
# Part 2: Feature Normalization
# -------------------------------------------------------------------------

def normalize(X):
    """
    Standardize each feature column to zero mean and unit variance (z-score).

        X_norm = (X - mean) / std

    Store and return mean and std so you can apply the same transform to
    test data (never fit statistics on test data!).

    Args:
        X (np.ndarray): Feature matrix, shape (m, n). Do NOT include bias col.

    Returns:
        tuple:
            X_norm (np.ndarray): Normalized features, shape (m, n).
            mean   (np.ndarray): Per-feature mean,   shape (n,).
            std    (np.ndarray): Per-feature std,     shape (n,).
    """
    # TODO 3: Compute the mean of each column (axis=0)
    mean = None

    # TODO 4: Compute the standard deviation of each column (axis=0)
    #         Use ddof=0 (population std, same as numpy default)
    std = None

    # TODO 5: Normalize X: subtract mean, divide by std.
    #         Handle std == 0 to avoid division by zero (use np.where or add eps)
    X_norm = None

    return X_norm, mean, std


def apply_normalization(X, mean, std):
    """
    Apply previously computed mean/std to a new set of features.
    Use this to normalize test data with *training* statistics.

    Args:
        X    (np.ndarray): Feature matrix, shape (m, n).
        mean (np.ndarray): Per-feature mean from training set, shape (n,).
        std  (np.ndarray): Per-feature std  from training set, shape (n,).

    Returns:
        np.ndarray: Normalized feature matrix, shape (m, n).
    """
    # TODO 6: Apply the same formula as normalize() but using the
    #         provided mean and std (do not recompute them)
    pass


# -------------------------------------------------------------------------
# Part 3: Train / Test Split
# -------------------------------------------------------------------------

def train_test_split(X, y, test_size=0.2, seed=42):
    """
    Randomly split (X, y) into training and test sets.

    Args:
        X         (np.ndarray): Feature matrix, shape (m, n).
        y         (np.ndarray): Target vector,  shape (m,).
        test_size (float):      Fraction of data to use for testing (0 < test_size < 1).
        seed      (int):        Random seed for reproducibility.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
            X_train (np.ndarray): shape (m_train, n)
            X_test  (np.ndarray): shape (m_test,  n)
            y_train (np.ndarray): shape (m_train,)
            y_test  (np.ndarray): shape (m_test,)
    """
    np.random.seed(seed)
    m = X.shape[0]

    # TODO 7: Create a shuffled array of indices using np.random.permutation(m)
    indices = None

    # TODO 8: Compute n_test = number of test samples = int(m * test_size)
    n_test = None

    # TODO 9: Split indices into test_indices (first n_test) and
    #         train_indices (the rest)
    test_indices  = None
    train_indices = None

    # TODO 10: Index into X and y with train_indices and test_indices
    #          to produce X_train, X_test, y_train, y_test and return them
    pass
