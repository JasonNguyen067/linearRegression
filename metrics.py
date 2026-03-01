"""
Evaluation Metrics
===================
After training, you need numbers that tell you *how good* the model is.

Concepts covered:
  - Mean Absolute Error  (MAE)  — average absolute residual
  - Mean Squared Error   (MSE)  — penalizes large errors more
  - Root Mean Squared Error (RMSE) — same units as y
  - R-squared (R²)      — proportion of variance explained (1 = perfect)
"""

import numpy as np


# -------------------------------------------------------------------------
# Part 1: Error Metrics
# -------------------------------------------------------------------------

def mean_absolute_error(y_true, y_pred):
    """
    Compute Mean Absolute Error.

        MAE = (1/m) * sum(|y_pred - y_true|)

    Args:
        y_true (np.ndarray): Ground-truth targets,  shape (m,).
        y_pred (np.ndarray): Model predictions,     shape (m,).

    Returns:
        float: MAE value.
    """
    # TODO 1: Compute and return the MAE
    pass


def mean_squared_error(y_true, y_pred):
    """
    Compute Mean Squared Error.

        MSE = (1/m) * sum((y_pred - y_true)^2)

    Args:
        y_true (np.ndarray): Ground-truth targets,  shape (m,).
        y_pred (np.ndarray): Model predictions,     shape (m,).

    Returns:
        float: MSE value.
    """
    # TODO 2: Compute and return the MSE
    pass


def root_mean_squared_error(y_true, y_pred):
    """
    Compute Root Mean Squared Error.

        RMSE = sqrt(MSE)

    Args:
        y_true (np.ndarray): Ground-truth targets,  shape (m,).
        y_pred (np.ndarray): Model predictions,     shape (m,).

    Returns:
        float: RMSE value.
    """
    # TODO 3: Reuse mean_squared_error() and return its square root
    pass


# -------------------------------------------------------------------------
# Part 2: R-Squared (Coefficient of Determination)
# -------------------------------------------------------------------------

def r_squared(y_true, y_pred):
    """
    Compute R², the proportion of variance in y explained by the model.

        SS_res = sum((y_true - y_pred)^2)
        SS_tot = sum((y_true - mean(y_true))^2)
        R²     = 1 - SS_res / SS_tot

    Interpretation:
        R² = 1.0  → perfect fit
        R² = 0.0  → model is no better than predicting the mean
        R² < 0    → model is worse than predicting the mean

    Args:
        y_true (np.ndarray): Ground-truth targets,  shape (m,).
        y_pred (np.ndarray): Model predictions,     shape (m,).

    Returns:
        float: R² value.
    """
    # TODO 4: Compute SS_res = sum of squared residuals
    SS_res = None

    # TODO 5: Compute SS_tot = sum of squared deviations from the mean of y_true
    SS_tot = None

    # TODO 6: Return R² = 1 - SS_res / SS_tot
    pass


# -------------------------------------------------------------------------
# Part 3: Summary Report
# -------------------------------------------------------------------------

def evaluation_report(y_true, y_pred):
    """
    Print a formatted table of all metrics.

    Expected output (values will differ):
        ╔══════════════════════════════╗
        ║     Model Evaluation         ║
        ╠══════════════════════════════╣
        ║  MAE  :   2.3412             ║
        ║  MSE  :   9.1042             ║
        ║  RMSE :   3.0173             ║
        ║  R²   :   0.9312             ║
        ╚══════════════════════════════╝

    Args:
        y_true (np.ndarray): Ground-truth targets, shape (m,).
        y_pred (np.ndarray): Model predictions,    shape (m,).
    """
    # TODO 7: Call each metric function above and print a formatted report.
    #         You may format the table however you like — make it readable.
    pass
