"""
Shared "load the data and train the model" logic, used by BOTH
train_model.py (a plain script you run from a terminal) and app.py (the
Streamlit web app).

Why this file exists: train_model.py and app.py used to each contain their
own separate copy of the exact same steps (load the CSV, turn text columns
into numbers, split into train/test, fit a LinearRegression, score it).
Two copies of the same logic are a trap — if you ever tweak one (e.g. add
a feature, change the train/test split), it's easy to forget to update the
other, and the two would silently start disagreeing about what the "real"
model is. Putting the logic here ONCE means both files call the same code
and can never drift apart.

This file deliberately does NOT import streamlit. It's plain Python/pandas/
scikit-learn, so either a terminal script or a web app (or anything else)
can import and use it.
"""

# --- Imports -----------------------------------------------------------------

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

# --- Shared constants ----------------------------------------------------------

# The path to the dataset file, relative to wherever this script/app is run
# from. Kept as a named constant (rather than typed inline in load_data)
# so there's exactly one place to change it if the file ever moves.
DATA_PATH = "employee_productivity_dataset.csv"

# The numeric (already-a-number) feature columns the model uses.
NUMERIC_FEATURES = [
    "Years_of_Experience",
    "Training_Hours",
    "Monthly_Working_Hours",
    "Projects_Completed",
    "Average_Task_Completion_Time",
    "Absence_Days",
    "Engagement_Score",
]

# The text (categorical) feature columns — these need to be turned into
# numbers (see train_model below) before a linear regression model can use
# them, since the model can only do arithmetic, not read words.
CATEGORICAL_FEATURES = ["Department", "Job_Level"]

# The column we're trying to predict.
TARGET_COLUMN = "Monthly_Productivity_Score"

# The five job levels IN SENIORITY ORDER (not alphabetical — alphabetical
# would incorrectly put "Lead" before "Junior"). This must match the
# `job_levels` list in generate_dataset.py. load_data() below checks that
# match every time the CSV is loaded, so a mismatch fails loudly instead of
# quietly producing wrong predictions.
JOB_LEVEL_ORDER = ["Junior", "Mid", "Senior", "Lead", "Manager"]


# --- Step 1: Load the dataset ---------------------------------------------------


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Reads the CSV file into a pandas table (DataFrame) and sanity-checks
    that its Job_Level column only contains the values we expect.
    """
    df = pd.read_csv(path)

    # Guard against the CSV's Job_Level values silently drifting away from
    # JOB_LEVEL_ORDER (e.g. if someone edits generate_dataset.py's
    # `job_levels` list later and forgets to update JOB_LEVEL_ORDER here).
    # Without this check, a level missing from JOB_LEVEL_ORDER just
    # couldn't be selected in the app's dropdown, and a level in
    # JOB_LEVEL_ORDER but not actually in the data would silently encode
    # as all-zero dummy columns during prediction — either way, wrong
    # results with no error message. Better to fail loudly here, the
    # moment the data is loaded.
    actual_levels = set(df["Job_Level"].unique())
    if actual_levels != set(JOB_LEVEL_ORDER):
        raise ValueError(
            f"Job_Level values in {path} ({sorted(actual_levels)}) don't "
            f"match JOB_LEVEL_ORDER in model_core.py ({JOB_LEVEL_ORDER}). "
            "Update JOB_LEVEL_ORDER to match."
        )

    return df


# --- Step 2-6: Train a linear regression model on that data ---------------------


def train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 1):
    """
    Trains a LinearRegression model to predict TARGET_COLUMN from the
    other columns in df, and returns everything a caller (a script or a
    web app) would need to report on or use that model:

        model          -- the trained scikit-learn model object itself
        model_columns  -- the exact list/order of encoded feature column
                           names the model expects at prediction time
        coefficients   -- a pandas Series pairing each encoded column name
                           with its learned weight
        X_test, y_test -- the held-out test rows and their true answers
        y_pred         -- the model's predictions for those test rows
        metrics        -- a dict of r2 / mae / rmse / n_train / n_test
    """
    # Target column: what we're predicting.
    y = df[TARGET_COLUMN]

    # Feature columns: everything the model is allowed to look at.
    # Employee_ID is dropped because it's just a made-up label, not a
    # real predictor.
    X = df.drop(columns=[TARGET_COLUMN, "Employee_ID"])

    # Turn Department/Job_Level text into 0/1 dummy columns so the model
    # (pure math) can use them. This trick is called "one-hot encoding":
    # a column like Department with 7 possible values becomes 7 new
    # True/False (0/1) columns, one per department. drop_first=True drops
    # one category per original column to avoid redundant/duplicate
    # information (if a row is 0 in every OTHER department column, that
    # already implies it must be the dropped one).
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=True)

    # Remember the exact list and order of encoded column names. Anyone
    # building a single new row to predict on later (see build_input_row
    # below) needs to line their row up against this exact list, since a
    # trained scikit-learn model always expects the same columns, in the
    # same order, it was trained on.
    model_columns = X.columns.tolist()

    # Split into training data (the model learns from this) and test data
    # (hidden from the model during training, used afterward to fairly
    # check its predictions against reality).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Create a fresh, untrained model, then .fit() it — this is where the
    # model looks at every training row's features alongside the real
    # answer for that row, and works out the combination of weights that
    # best reproduces the real answers.
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Apply the learned formula to the test rows (which the model never
    # saw during training) to get its best-guess predictions for them.
    y_pred = model.predict(X_test)

    # Score those predictions against the real test answers.
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # A pandas Series pairing each encoded column name with its learned
    # weight, so a caller can look up "how much does this feature push
    # the predicted score up or down per unit?"
    coefficients = pd.Series(model.coef_, index=model_columns)

    return model, model_columns, coefficients, X_test, y_test, y_pred, metrics


# --- Helpers for turning a single set of user-picked inputs into a prediction ---


def pretty_label(column_name: str) -> str:
    """
    Turns a raw encoded column name like "Department_Engineering" or
    "Years_of_Experience" into a friendlier label for display, like
    "Department: Engineering" or "Years Of Experience". Purely cosmetic —
    it doesn't change any numbers, just how they're labeled in output.
    """
    for cat in CATEGORICAL_FEATURES:
        prefix = cat + "_"
        if column_name.startswith(prefix):
            # e.g. "Department_Engineering" -> "Department: Engineering"
            return f"{cat.replace('_', ' ')}: {column_name[len(prefix) :].replace('_', ' ')}"
    # Plain numeric feature, e.g. "Years_of_Experience" -> "Years Of Experience"
    return column_name.replace("_", " ")


def build_input_row(user_inputs: dict, model_columns: list) -> pd.DataFrame:
    """
    Takes ONE set of raw feature values (a plain dictionary, e.g.
    {"Department": "Engineering", "Years_of_Experience": 5.0, ...}) and
    turns it into a ONE-ROW table that has exactly the same columns, in
    exactly the same order, as the data train_model() trained on
    (model_columns). This is required because a trained scikit-learn
    model always expects the same columns it saw during training.
    """
    # Wrap the single dictionary of inputs in a list so pandas builds a
    # DataFrame with exactly one row.
    row = pd.DataFrame([user_inputs])

    # One-hot encode the same two text columns the same way training did.
    row = pd.get_dummies(row, columns=CATEGORICAL_FEATURES)

    # .reindex(columns=model_columns, fill_value=0) does two jobs at once:
    #   1. Adds back any dummy columns that this particular row doesn't
    #      need (e.g. if the row's Department is "Engineering", there's no
    #      "Department_Sales" column yet — reindex adds it and fills it
    #      with 0, exactly like a real Sales=0/Engineering=1 row).
    #   2. Puts every column in the exact order model_columns expects.
    # Without this step, a single row's columns could end up in a
    # different order than what the model was trained on, which would
    # silently produce wrong predictions.
    row = row.reindex(columns=model_columns, fill_value=0)

    return row
