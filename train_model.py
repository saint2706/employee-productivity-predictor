"""
Trains a linear regression model to predict Monthly_Productivity_Score
from the other columns in employee_productivity_dataset.csv.

"Linear regression" is a way of finding a straight-line formula (just like
the one used to GENERATE Monthly_Productivity_Score in generate_dataset.py)
that best predicts one column (the "target") from the others (the
"features"). Here we're doing the reverse job: instead of building the
formula ourselves, we let the model FIGURE OUT the formula from the data.
"""

# --- Imports -----------------------------------------------------------------

# pandas ("pd") lets us load the CSV file into a table (DataFrame) and work
# with its rows/columns easily.
import pandas as pd

# train_test_split splits our table into two parts: one part the model
# learns from ("training data"), and one part we hide from it so we can
# fairly check how well it does on data it has never seen ("test data").
from sklearn.model_selection import train_test_split

# LinearRegression is the actual model: the algorithm that finds the best
# straight-line formula relating our input columns to the target column.
from sklearn.linear_model import LinearRegression

# These are "scoring" functions: after the model makes predictions, we use
# them to measure how close those predictions were to the real answers.
#   r2_score            -> how much of the variation in the target the model explains (0 to 1, higher is better)
#   mean_absolute_error -> on average, how many points off each prediction was (lower is better)
#   root_mean_squared_error -> similar to mean_absolute_error, but penalizes big misses more (lower is better)
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

import numpy as np

# --- Step 1: Load the data -----------------------------------------------------

# pd.read_csv reads the CSV file we generated earlier and turns it into a
# DataFrame (a table) called df, just like the one we built in
# generate_dataset.py — except this time we're reading it back FROM the
# file instead of building it in memory.
df = pd.read_csv("employee_productivity_dataset.csv")

# --- Step 2: Separate the target column from the feature columns --------------

# "Target" = the column we want to PREDICT. We pull it out into its own
# variable, y, which is the standard letter statisticians/ML people use for
# "the answer we're trying to predict".
y = df["Monthly_Productivity_Score"]

# "Features" = the columns we're ALLOWED to use to make that prediction.
# We build this by dropping (removing) two columns from df:
#   - Monthly_Productivity_Score, because that's the answer itself — the
#     model isn't allowed to see the answer while it's trying to predict it.
#   - Employee_ID, because it's just a made-up label ("EMP00001", "EMP00002",
#     ...) with no real relationship to productivity. Including it would
#     only confuse the model or make it "memorize" individuals instead of
#     learning a general pattern.
# X (capital letter, by convention) holds all the remaining feature columns.
X = df.drop(columns=["Monthly_Productivity_Score", "Employee_ID"])

# --- Step 3: Turn text columns into numbers ("one-hot encoding") --------------

# Linear regression is pure math — it can only multiply and add NUMBERS, it
# cannot do arithmetic on text like "Engineering" or "Senior". But our table
# has two text (categorical) columns: Department and Job_Level.
#
# pd.get_dummies() solves this with a trick called "one-hot encoding": for
# a column like Department with 7 possible values, it creates 7 new
# True/False (0/1) columns, one per department — e.g. "Department_Sales",
# "Department_Engineering", etc. A row gets a 1 in the column matching its
# real department, and 0 in all the others. This lets the model treat each
# department as a separate, purely numeric on/off switch.
#
# drop_first=True removes the FIRST category's column (e.g. it might drop
# "Department_Customer_Support"). This avoids giving the model redundant
# information: if a row is 0 in every OTHER department column, that already
# implies it must be Customer_Support, so keeping that column too would
# just be duplicate information. This is standard practice and avoids a
# statistical problem called "perfect multicollinearity".
X = pd.get_dummies(X, columns=["Department", "Job_Level"], drop_first=True)

# --- Step 4: Split into training data and test data ----------------------------

# We split our rows into two groups:
#   - 80% ("X_train", "y_train") for the model to learn from
#   - 20% ("X_test", "y_test") that we hide from the model during training,
#     so we can later check its predictions against reality on examples it
#     has genuinely never seen. This is how we detect "overfitting" (a
#     model that just memorized the training data instead of learning a
#     general pattern would do great on training data but poorly here).
# test_size=0.2 means "use 20% of rows for testing, 80% for training".
# random_state=1 is a seed (same idea as the rng seed in generate_dataset.py)
# so the split is the same every time we run this script.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# --- Step 5: Create and train the model -----------------------------------------

# LinearRegression() creates a fresh, "untrained" model object — at this
# point it doesn't know anything about our data yet.
model = LinearRegression()

# .fit(X_train, y_train) is where the actual "learning" happens: the model
# looks at every training row's features (X_train) alongside the real
# answer for that row (y_train), and works out the combination of weights
# (one number per feature, plus one overall baseline number) that best
# reproduces y_train from X_train. After this line, `model` "knows" a
# formula it can apply to NEW, unseen feature rows.
model.fit(X_train, y_train)

# --- Step 6: Use the trained model to make predictions on the test data ---------

# .predict(X_test) applies the formula the model just learned to the test
# rows (which it never saw during training) and returns its best-guess
# productivity score for each one. y_pred is an array of N_test numbers,
# lining up row-for-row with y_test (the real answers).
y_pred = model.predict(X_test)

# --- Step 7: Measure how good the model is --------------------------------------

# r2_score compares y_pred to the real y_test values and returns a number
# from roughly 0 to 1 (it CAN go negative for a very bad model) representing
# the fraction of the target's variation that the model successfully
# explains. 1.0 would mean perfect predictions; 0.0 would mean the model is
# no better than just always guessing the average productivity score.
r2 = r2_score(y_test, y_pred)

# mean_absolute_error (MAE) is, on average, how many points off each
# prediction was from the true value, ignoring whether it guessed too high
# or too low. E.g. an MAE of 3.8 means predictions are typically about 3.8
# productivity points away from the real score.
mae = mean_absolute_error(y_test, y_pred)

# root_mean_squared_error (RMSE) is similar to MAE, but it squares each
# error before averaging (then un-squares the result at the end), which
# makes a few big mistakes count for more than lots of tiny ones. If RMSE
# is noticeably larger than MAE, that's a sign the model has a handful of
# predictions that are way off, alongside many good ones.
rmse = root_mean_squared_error(y_test, y_pred)

# Print all three scores so we can see how the model performed. round(x, 3)
# trims each number to 3 decimal places just so the printout is tidy.
print("Model performance on the test data (20% of rows the model never trained on):")
print(f"  R-squared (R2):            {round(r2, 3)}")
print(f"  Mean Absolute Error (MAE): {round(mae, 3)} productivity points")
print(f"  Root Mean Squared Error:   {round(rmse, 3)} productivity points")

# --- Step 8: Look at what the model learned ---------------------------------------

# model.intercept_ is the model's learned "baseline" score — roughly, what
# it would predict for an employee if every single feature were 0. On its
# own this number isn't very meaningful (nobody has 0 for every feature),
# but it's part of the formula the model built.
print(f"\nLearned baseline (intercept): {round(model.intercept_, 3)}")

# model.coef_ is an array holding one learned "weight" per feature column,
# in the same order as the columns in X. Each weight says: "holding every
# other feature fixed, increasing this one feature by 1 unit changes the
# predicted productivity score by this many points." A positive weight
# means that feature tends to push productivity UP; negative means it
# pushes productivity DOWN.
#
# We build a small pandas Series (a labeled list of numbers) pairing each
# feature's NAME (X.columns) with its learned weight (model.coef_), then
# sort it from most positive to most negative so the strongest effects
# are easiest to spot, and print it.
coefficients = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("\nLearned weight (coefficient) for each feature, sorted strongest-positive to strongest-negative:")
print(coefficients.round(3))

# --- Step 9: Sanity-check a few individual predictions --------------------------

# Grab the first 5 rows of the test set and show the model's guess next to
# the true answer, just so we can visually see how close it gets on actual
# examples (rather than only looking at the summary scores above).
comparison = pd.DataFrame({
    "Actual": y_test.values[:5],
    "Predicted": np.round(y_pred[:5], 2),
})
print("\nA few real vs. predicted examples from the test set:")
print(comparison)
