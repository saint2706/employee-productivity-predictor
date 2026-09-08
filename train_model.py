"""
Trains a linear regression model to predict Monthly_Productivity_Score
and prints a report on how good it is.

The actual "load the data" and "train the model" steps live in
model_core.py, shared with app.py (the Streamlit demo), so both always
train the exact same way. This script is the "terminal report" on top of
that shared logic — see model_core.py if you want the detailed, step-by-
step explanation of HOW the model is built.
"""

# --- Imports -----------------------------------------------------------------

import numpy as np
import pandas as pd

# Pulls in the shared loading/training logic. `load_data` reads and
# validates the CSV; `train_model` does the one-hot encoding, train/test
# split, fitting, and scoring, and hands back everything we need below.
from model_core import load_data, train_model

# --- Step 1: Load the data and train the model -----------------------------------

df = load_data()
model, model_columns, coefficients, X_test, y_test, y_pred, metrics = train_model(df)

# --- Step 2: Print the headline performance numbers -------------------------------

# r2 ("R-squared") is a number from roughly 0 to 1 (it CAN go negative for
# a very bad model) representing the fraction of the target's variation
# that the model successfully explains. 1.0 would mean perfect
# predictions; 0.0 would mean the model is no better than just always
# guessing the average productivity score.
#
# mae ("Mean Absolute Error") is, on average, how many points off each
# prediction was from the true value, ignoring whether it guessed too high
# or too low.
#
# rmse ("Root Mean Squared Error") is similar to MAE, but it squares each
# error before averaging (then un-squares the result), which makes a few
# big mistakes count for more than lots of tiny ones. If RMSE is
# noticeably larger than MAE, that's a sign the model has a handful of
# predictions that are way off, alongside many good ones.
print("Model performance on the test data (20% of rows the model never trained on):")
print(f"  R-squared (R2):            {round(metrics['r2'], 3)}")
print(f"  Mean Absolute Error (MAE): {round(metrics['mae'], 3)} productivity points")
print(f"  Root Mean Squared Error:   {round(metrics['rmse'], 3)} productivity points")

# --- Step 3: Look at what the model learned ---------------------------------------

# model.intercept_ is the model's learned "baseline" score — roughly, what
# it would predict for an employee if every single feature were 0. On its
# own this number isn't very meaningful (nobody has 0 for every feature),
# but it's part of the formula the model built.
print(f"\nLearned baseline (intercept): {round(model.intercept_, 3)}")

# `coefficients` (built by train_model in model_core.py) pairs each
# feature's NAME with its learned weight: "holding every other feature
# fixed, increasing this one feature by 1 unit changes the predicted
# productivity score by this many points." A positive weight means that
# feature tends to push productivity UP; negative means it pushes
# productivity DOWN. We sort from most positive to most negative so the
# strongest effects are easiest to spot.
print(
    "\nLearned weight (coefficient) for each feature, sorted strongest-positive to strongest-negative:"
)
print(coefficients.sort_values(ascending=False).round(3))

# --- Step 4: Sanity-check a few individual predictions --------------------------

# Grab the first 5 rows of the test set and show the model's guess next to
# the true answer, just so we can visually see how close it gets on actual
# examples (rather than only looking at the summary scores above).
comparison = pd.DataFrame(
    {
        "Actual": y_test.values[:5],
        "Predicted": np.round(y_pred[:5], 2),
    }
)
print("\nA few real vs. predicted examples from the test set:")
print(comparison)
