"""
Flask web app that demos the linear regression model from train_model.py
using the data from generate_dataset.py.

Unlike Streamlit (which re-runs the whole script on every interaction),
Flask is a plain request/response web server: it renders an HTML page once
(the "/" route below), and the browser's own JavaScript (see
static/js/main.js) takes over from there, calling the JSON API route
"/api/predict" every time the form changes and updating the page in place.

The model itself is trained exactly once, when this module is first
imported (see "Load data and train the model once" below), and then reused
for every request the server handles — training on every click would be
wasteful and would make predictions for the SAME inputs change between
requests as the random train/test split moved around.

To run this app locally:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000 in a browser.
"""

# --- Imports -----------------------------------------------------------------

from flask import Flask, jsonify, render_template, request

# The actual "load the data" and "train the model" logic lives in
# model_core.py, shared with train_model.py (the plain terminal script), so
# both always compute things exactly the same way.
from model_core import (
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    JOB_LEVEL_ORDER,
    load_data,
    train_model,
    pretty_label,
    build_input_row,
)

app = Flask(__name__)

# --- Per-feature display metadata -----------------------------------------------
#
# Everything the Predict form needs to know about each numeric feature
# besides its min/max/default (which come from the dataset itself, see
# build_feature_meta() below): a friendly label, the step size for its
# input box, and whether it's a whole-number field. This mirrors the
# widget choices the original Streamlit app made for each field.
FEATURE_LABELS = {
    "Years_of_Experience": "Years of Experience",
    "Training_Hours": "Training Hours (this year)",
    "Monthly_Working_Hours": "Monthly Working Hours",
    "Projects_Completed": "Projects Completed (this month)",
    "Average_Task_Completion_Time": "Average Task Completion Time (hours)",
    "Absence_Days": "Absence Days (this month)",
    "Engagement_Score": "Engagement Score",
}
FEATURE_STEPS = {
    "Years_of_Experience": 0.5,
    "Training_Hours": 1,
    "Monthly_Working_Hours": 1,
    "Projects_Completed": 1,
    "Average_Task_Completion_Time": 0.1,
    "Absence_Days": 1,
    "Engagement_Score": 1,
}
INTEGER_FEATURES = {"Projects_Completed", "Absence_Days"}


# --- Load data and train the model once, when this module is imported -----------

df = load_data()
model, model_columns, coefficients, X_test, y_test, y_pred, metrics = train_model(df)
DEPARTMENTS = sorted(df["Department"].unique().tolist())


def build_feature_meta() -> list[dict]:
    """
    One entry per numeric feature, in form order, with everything the
    Predict form's HTML + JavaScript need to draw it and validate typed
    values against the range actually observed in the dataset (matching
    the original app's "still used, but a warning is shown" behavior).
    """
    fields = []
    for col in NUMERIC_FEATURES:
        integer = col in INTEGER_FEATURES
        lo, hi, default = df[col].min(), df[col].max(), df[col].median()
        if integer:
            lo, hi, default = int(lo), int(hi), int(round(default))
        else:
            lo, hi, default = float(lo), float(hi), float(default)
        fields.append(
            {
                "name": col,
                "label": FEATURE_LABELS[col],
                "step": FEATURE_STEPS[col],
                "integer": integer,
                "min": lo,
                "max": hi,
                "default": default,
            }
        )
    return fields


def sorted_coefficients_payload() -> dict:
    """Learned coefficients, sorted and pretty-labeled, for the bar chart."""
    sorted_coefficients = coefficients.sort_values()
    return {
        "labels": [pretty_label(c) for c in sorted_coefficients.index],
        "values": [float(v) for v in sorted_coefficients.values],
    }


def parse_user_inputs(payload: dict) -> dict:
    """
    Validates and converts the raw JSON body of a POST /api/predict request
    into the same kind of plain dict build_input_row() (model_core.py)
    expects: real department/job-level text plus numbers for every feature.
    Raises ValueError with a human-readable message on anything malformed
    -- this is now a public API endpoint, not a trusted in-process widget,
    so every field has to be checked rather than assumed valid.
    """
    department = payload.get("Department")
    if department not in DEPARTMENTS:
        raise ValueError(f"Unknown Department: {department!r}")

    job_level = payload.get("Job_Level")
    if job_level not in JOB_LEVEL_ORDER:
        raise ValueError(f"Unknown Job_Level: {job_level!r}")

    user_inputs = {"Department": department, "Job_Level": job_level}
    for col in NUMERIC_FEATURES:
        raw_value = payload.get(col)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{col} must be a number, got {raw_value!r}")
        if col in INTEGER_FEATURES:
            value = int(round(value))
        user_inputs[col] = value

    return user_inputs


# --- Routes ----------------------------------------------------------------------


@app.route("/")
def index():
    """
    Renders the whole page in one shot: the Predict form (departments, job
    levels, per-feature ranges) plus everything the Model Performance tab
    needs (headline metrics, actual-vs-predicted, residuals, coefficients).
    None of that depends on user input, so it's computed once above and
    just handed to the template -- only the Predict tab's live prediction
    goes through /api/predict afterward.
    """
    residuals = (y_test - y_pred).tolist()
    return render_template(
        "index.html",
        departments=DEPARTMENTS,
        job_levels=JOB_LEVEL_ORDER,
        feature_meta=build_feature_meta(),
        employee_count=len(df),
        metrics=metrics,
        target_scores=df[TARGET_COLUMN].tolist(),
        scatter={"actual": y_test.tolist(), "predicted": y_pred.tolist()},
        residuals=residuals,
        coefficients=sorted_coefficients_payload(),
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    The Predict tab's live-update endpoint: takes one employee's profile as
    JSON, returns the predicted productivity score plus everything needed
    to redraw the contribution (waterfall) chart. Called by main.js
    whenever a form field changes.
    """
    payload = request.get_json(silent=True) or {}
    try:
        user_inputs = parse_user_inputs(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    input_row = build_input_row(user_inputs, model_columns)
    raw_prediction = float(model.predict(input_row)[0])

    # The training target was clipped to a 0-100 scale (see
    # generate_dataset.py), but a raw linear regression formula has no idea
    # a "score" is supposed to stay in that range. Clip the DISPLAYED
    # prediction the same way, so the headline number always stays on the
    # scale the app advertises ("... / 100").
    prediction = max(0.0, min(100.0, raw_prediction))
    percentile = float((df[TARGET_COLUMN] <= prediction).mean() * 100)

    # A linear regression prediction is always:
    #     prediction = intercept + (coef_1 * feature_1) + (coef_2 * feature_2) + ...
    # so we can show exactly how each input pushed the score up or down
    # from the model's baseline -- the most "explainable" part of a linear
    # model, compared to more complex models where this isn't so simple.
    labels = ["Baseline"]
    values = [float(model.intercept_)]
    measures = ["absolute"]

    for col in NUMERIC_FEATURES:
        contribution = float(coefficients[col] * input_row.iloc[0][col])
        labels.append(pretty_label(col))
        values.append(contribution)
        measures.append("relative")

    # For the one-hot (Department_*, Job_Level_*) columns, only show a bar
    # when that column is actually "on" (equal to 1) for this employee.
    for col in model_columns:
        if col not in NUMERIC_FEATURES and input_row.iloc[0][col] == 1:
            labels.append(pretty_label(col))
            values.append(float(coefficients[col]))
            measures.append("relative")

    predicted_total = sum(values)
    labels.append("Predicted score")
    values.append(0.0)
    measures.append("total")

    return jsonify(
        {
            "prediction": prediction,
            "raw_prediction": raw_prediction,
            "percentile": percentile,
            "mae": metrics["mae"],
            "waterfall": {
                "labels": labels,
                "values": values,
                "measures": measures,
                "predicted_total": predicted_total,
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
