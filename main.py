"""
Main Driver — Linear Regression Project
=========================================
Run this file after completing all TODOs in:
    linear_regression.py
    preprocessing.py
    metrics.py

You do NOT need to modify this file, but reading it will help you
understand how all the pieces fit together.
"""

import numpy as np
import matplotlib.pyplot as plt

from linear_regression import LinearRegression
from preprocessing import add_bias, normalize, apply_normalization, train_test_split
from metrics import evaluation_report


# ─────────────────────────────────────────────
# 1. Generate synthetic dataset
# ─────────────────────────────────────────────
np.random.seed(0)
m = 200                              # number of samples
X_raw = np.random.randn(m, 2)        # 2 features
true_theta = np.array([3.0, -1.5])  # true weights (no bias for data gen)
y = X_raw @ true_theta + 5 + np.random.randn(m) * 0.8   # y = Xw + bias + noise


# ─────────────────────────────────────────────
# 2. Preprocess
# ─────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.2)

X_train_norm, mean, std = normalize(X_train_raw)
X_test_norm              = apply_normalization(X_test_raw, mean, std)

X_train = add_bias(X_train_norm)
X_test  = add_bias(X_test_norm)


# ─────────────────────────────────────────────
# 3. Train
# ─────────────────────────────────────────────
model = LinearRegression(learning_rate=0.1, n_iterations=500)
model.fit(X_train, y_train)

print(f"Learned theta: {model.theta}")
print(f"True weights (with bias ~5): [5, 3, -1.5]\n")


# ─────────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────────
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)

print("── Train Set ──")
evaluation_report(y_train, y_pred_train)

print("\n── Test Set ──")
evaluation_report(y_test, y_pred_test)


# ─────────────────────────────────────────────
# 5. Bonus: Normal Equation comparison
# ─────────────────────────────────────────────
model_ne = LinearRegression()
model_ne.fit_normal_equation(X_train, y_train)
if model_ne.theta is not None:
    y_pred_ne = model_ne.predict(X_test)
    print("\n── Normal Equation (Test Set) ──")
    evaluation_report(y_test, y_pred_ne)


# ─────────────────────────────────────────────
# 6. Plot: Cost curve
# ─────────────────────────────────────────────
if model.cost_history:
    plt.figure(figsize=(8, 4))
    plt.plot(model.cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Cost")
    plt.title("Gradient Descent — Cost vs. Iteration")
    plt.tight_layout()
    plt.savefig("cost_curve.png", dpi=120)
    print("\nCost curve saved to cost_curve.png")
    plt.show()
