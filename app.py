"""
Streamlit web app that demos the linear regression model from
train_model.py using the data from generate_dataset.py.

Streamlit is a Python library that turns a plain script like this one into
an interactive website: every widget (slider, dropdown, etc.) you create
becomes a real control in the browser, and whenever the user changes one,
Streamlit re-runs this whole script top-to-bottom and redraws the page with
the new values. That "just re-run everything" model is why you'll see
`@st.cache_data` / `@st.cache_resource` below — they tell Streamlit "don't
redo this expensive step (loading the CSV, training the model) on every
single re-run, just remember the result from last time."

To run this app locally: open a terminal in this folder and run
    streamlit run app.py
It will open a browser tab automatically.
"""

# --- Imports -----------------------------------------------------------------

import streamlit as st                       # the web app framework itself
import pandas as pd                            # tables (DataFrames)
import plotly.graph_objects as go               # interactive charts

# The actual "load the data" and "train the model" logic lives in
# model_core.py, shared with train_model.py (the plain terminal script), so
# both always compute things exactly the same way. We import:
#   - the column-name constants, so this file doesn't need its own copies
#   - load_data / train_model, the functions that do the real work (renamed
#     with a leading underscore here since we wrap each in a Streamlit
#     cache decorator below, under the SAME names, for the rest of this
#     file to call)
#   - pretty_label / build_input_row, small helpers used by the Predict tab
from model_core import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
    JOB_LEVEL_ORDER,
    load_data as _load_data,
    train_model as _train_model,
    pretty_label,
    build_input_row,
)

# --- Page setup ----------------------------------------------------------------

# st.set_page_config must be the first Streamlit command in the script. It
# sets the browser tab's title/icon and makes the page use the full screen
# width instead of a narrow centered column (better for side-by-side charts).
st.set_page_config(
    page_title="Employee Productivity Predictor",
    page_icon="📈",
    layout="wide",
)


# --- Step 1: Load the dataset (cached) -----------------------------------------

# The @st.cache_data decorator above a function tells Streamlit: "run this
# function once, remember what it returned, and just hand back that saved
# result on future calls instead of re-running the whole function." Without
# this, Streamlit would re-read the CSV from disk on every single click.
# The actual reading + validation happens in model_core.load_data(); this
# wrapper just adds Streamlit's caching on top of it.
@st.cache_data
def load_data() -> pd.DataFrame:
    return _load_data()


# --- Step 2: Train the model (cached) -------------------------------------------

# @st.cache_resource is the same idea as @st.cache_data, but meant for
# objects that aren't plain data (like a trained model). The actual
# encoding/splitting/fitting/scoring happens in model_core.train_model() —
# the exact same function train_model.py calls — so this app's model can
# never quietly diverge from what that script reports. This wrapper just
# adds Streamlit's caching so we only ever train once per app session.
@st.cache_resource
def train_model(df: pd.DataFrame):
    return _train_model(df)


# --- Load data and model once, then build the page -----------------------------

df = load_data()
model, model_columns, coefficients, X_test, y_test, y_pred, metrics = train_model(df)

st.title("📈 Employee Productivity Predictor")
st.caption(
    "A linear regression demo trained on a synthetic HR dataset "
    f"({len(df):,} employees). Predicts Monthly Productivity Score (0-100)."
)

# st.tabs() creates clickable tabs at the top of the page. It returns one
# object per tab name we pass in; anything we draw "inside" that tab's
# `with` block only shows up when that tab is selected.
predict_tab, performance_tab = st.tabs(["🔮 Predict", "📊 Model Performance"])


# =================================================================================
# TAB 1: Predict — input form, live prediction, population context, contribution
# breakdown
# =================================================================================
with predict_tab:
    left, right = st.columns([1, 1.4])

    # --- Left column: the input form -------------------------------------------
    with left:
        st.subheader("Employee profile")

        department = st.selectbox(
            "Department", sorted(df["Department"].unique()), index=0
        )
        job_level = st.selectbox("Job Level", JOB_LEVEL_ORDER)

        # For every numeric slider below, we pull the real minimum/maximum
        # seen in the dataset (df[col].min() / .max()) so users can't drag
        # a slider to a value that never occurs in real data, and we
        # default to the column's median (the "middle" value) as a
        # reasonable starting point.
        years_experience = st.slider(
            "Years of Experience",
            min_value=float(df["Years_of_Experience"].min()),
            max_value=float(df["Years_of_Experience"].max()),
            value=float(df["Years_of_Experience"].median()),
            step=0.5,
        )
        training_hours = st.slider(
            "Training Hours (this year)",
            min_value=float(df["Training_Hours"].min()),
            max_value=float(df["Training_Hours"].max()),
            value=float(df["Training_Hours"].median()),
            step=1.0,
        )
        working_hours = st.slider(
            "Monthly Working Hours",
            min_value=float(df["Monthly_Working_Hours"].min()),
            max_value=float(df["Monthly_Working_Hours"].max()),
            value=float(df["Monthly_Working_Hours"].median()),
            step=1.0,
        )
        projects_completed = st.slider(
            "Projects Completed (this month)",
            min_value=int(df["Projects_Completed"].min()),
            max_value=int(df["Projects_Completed"].max()),
            value=int(df["Projects_Completed"].median()),
            step=1,
        )
        avg_task_time = st.slider(
            "Average Task Completion Time (hours)",
            min_value=float(df["Average_Task_Completion_Time"].min()),
            max_value=float(df["Average_Task_Completion_Time"].max()),
            value=float(df["Average_Task_Completion_Time"].median()),
            step=0.1,
        )
        absence_days = st.slider(
            "Absence Days (this month)",
            min_value=int(df["Absence_Days"].min()),
            max_value=int(df["Absence_Days"].max()),
            value=int(df["Absence_Days"].median()),
            step=1,
        )
        engagement_score = st.slider(
            "Engagement Score",
            min_value=float(df["Engagement_Score"].min()),
            max_value=float(df["Engagement_Score"].max()),
            value=float(df["Engagement_Score"].median()),
            step=1.0,
        )

        # Bundle every widget's current value into one plain dictionary,
        # using the SAME column names the model was trained on. This is
        # what build_input_row() (defined above) will turn into a model-
        # ready row.
        user_inputs = {
            "Department": department,
            "Job_Level": job_level,
            "Years_of_Experience": years_experience,
            "Training_Hours": training_hours,
            "Monthly_Working_Hours": working_hours,
            "Projects_Completed": projects_completed,
            "Average_Task_Completion_Time": avg_task_time,
            "Absence_Days": absence_days,
            "Engagement_Score": engagement_score,
        }

    # Every time ANY widget above changes, Streamlit re-runs the whole
    # script, so this next block always reflects the form's current state
    # — there's no separate "Predict" button to click.
    input_row = build_input_row(user_inputs, model_columns)
    raw_prediction = float(model.predict(input_row)[0])

    # The training target was clipped to a 0-100 scale (see
    # generate_dataset.py), but a raw linear regression formula has no idea
    # a "score" is supposed to stay in that range — with a favorable-enough
    # combination of slider values it can predict above 100 or below 0. We
    # clip the DISPLAYED prediction the same way the training data was
    # clipped, so the headline number always stays on the scale the app
    # advertises ("... / 100").
    prediction = max(0.0, min(100.0, raw_prediction))

    # What fraction of all employees in the dataset score at or below this
    # (clipped) prediction? This turns a raw number ("72.4") into relatable
    # context ("better than 65% of employees").
    percentile = float((df[TARGET_COLUMN] <= prediction).mean() * 100)

    # --- Right column: predicted score, population context, contribution ------
    with right:
        st.subheader("Predicted productivity")

        # st.metric draws a big, dashboard-style number with an optional
        # smaller caption/delta beneath it — good for a headline result.
        st.metric(
            "Monthly Productivity Score",
            f"{prediction:.1f} / 100",
            help=f"Model's typical error on unseen data is about ±{metrics['mae']:.1f} points (MAE).",
        )
        st.write(
            f"This is higher than **{percentile:.0f}%** of employees in the dataset."
        )
        if raw_prediction != prediction:
            st.caption(
                f"Note: the model's raw formula output {raw_prediction:.1f} for this "
                "combination of inputs; it's clipped to 0-100 here since that's the "
                "scale the training data uses."
            )

        # --- Population context histogram ---------------------------------
        # A histogram of every employee's real productivity score, with a
        # vertical dashed line marking where THIS prediction falls.
        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=df[TARGET_COLUMN],
                nbinsx=40,
                name="All employees",
                marker_color="#6C8EBF",
            )
        )
        # add_vline draws a vertical line straight across the chart at a
        # given x position — here, the predicted score.
        hist_fig.add_vline(
            x=prediction,
            line_width=3,
            line_dash="dash",
            line_color="#D9534F",
            annotation_text="Your prediction",
            annotation_position="top",
        )
        hist_fig.update_layout(
            title="Where this prediction falls among all employees",
            xaxis_title="Monthly Productivity Score",
            yaxis_title="Number of employees",
            height=320,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(hist_fig, width='stretch')

        # --- Contribution breakdown (waterfall chart) ----------------------
        # A linear regression prediction is always:
        #     prediction = intercept + (coef_1 * feature_1) + (coef_2 * feature_2) + ...
        # so we can show EXACTLY how each of this employee's inputs pushed
        # the score up or down from the model's baseline. This is the most
        # "explainable" part of a linear model, compared to more complex
        # ML models where this kind of breakdown isn't so straightforward.
        labels = ["Baseline"]
        values = [model.intercept_]
        measures = ["absolute"]  # "absolute" = draw this bar from 0, not stacked on the previous one

        # Always show every numeric feature's contribution, even if it's
        # small, since every numeric feature always has some value.
        for col in NUMERIC_FEATURES:
            contribution = coefficients[col] * input_row.iloc[0][col]
            labels.append(pretty_label(col))
            values.append(contribution)
            measures.append("relative")  # "relative" = stack on top of the running total so far

        # For the one-hot (Department_*, Job_Level_*) columns, only show a
        # bar when that column is actually "on" (equal to 1) for this
        # employee — the columns that are 0 contribute nothing and would
        # just clutter the chart.
        for col in model_columns:
            if col not in NUMERIC_FEATURES and input_row.iloc[0][col] == 1:
                labels.append(pretty_label(col))
                values.append(coefficients[col])
                measures.append("relative")

        labels.append("Predicted score")
        values.append(0)  # ignored by Plotly for "total" bars — it sums everything automatically
        measures.append("total")

        waterfall_fig = go.Figure(
            go.Waterfall(
                x=labels,
                y=values,
                measure=measures,
                connector={"line": {"color": "rgba(120,120,120,0.4)"}},
                increasing={"marker": {"color": "#5CB85C"}},
                decreasing={"marker": {"color": "#D9534F"}},
                totals={"marker": {"color": "#6C8EBF"}},
                text=[f"{v:+.1f}" if lbl not in ("Baseline", "Predicted score") else f"{v:.1f}"
                      for lbl, v in zip(labels, values)],
                textposition="outside",
            )
        )
        waterfall_fig.update_layout(
            title="How each input contributed to this prediction",
            yaxis_title="Productivity points",
            height=420,
            margin=dict(t=50, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(waterfall_fig, width='stretch')


# =================================================================================
# TAB 2: Model Performance — headline metrics, actual vs predicted, residuals,
# learned coefficients
# =================================================================================
with performance_tab:
    st.subheader("How good is this model?")
    st.write(
        "All numbers on this tab are computed on the **test set**: "
        f"{metrics['n_test']:,} employees the model never saw while learning "
        f"(it trained on the other {metrics['n_train']:,})."
    )

    # st.columns(4) creates 4 side-by-side areas; we put one stat card
    # (st.metric) in each, giving a dashboard-style row of headline numbers.
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R² (variance explained)", f"{metrics['r2']:.3f}")
    c2.metric("Mean Absolute Error", f"{metrics['mae']:.2f} pts")
    c3.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f} pts")
    c4.metric("Test set size", f"{metrics['n_test']:,} rows")

    chart_left, chart_right = st.columns(2)

    # --- Actual vs Predicted scatter plot ---------------------------------------
    with chart_left:
        scatter_fig = go.Figure()
        scatter_fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                marker=dict(color="#6C8EBF", size=6, opacity=0.5),
                name="Test employees",
            )
        )
        # A perfect model would put every point exactly on this diagonal
        # line (predicted == actual). The closer the dots hug the line,
        # the better the model.
        axis_min = float(min(y_test.min(), y_pred.min()))
        axis_max = float(max(y_test.max(), y_pred.max()))
        scatter_fig.add_trace(
            go.Scatter(
                x=[axis_min, axis_max],
                y=[axis_min, axis_max],
                mode="lines",
                line=dict(color="#D9534F", dash="dash"),
                name="Perfect prediction",
            )
        )
        scatter_fig.update_layout(
            title="Actual vs. Predicted productivity score",
            xaxis_title="Actual score",
            yaxis_title="Predicted score",
            height=400,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(scatter_fig, width='stretch')

    # --- Residuals plot -----------------------------------------------------------
    with chart_right:
        # A "residual" is (actual - predicted) — how far off, and in which
        # direction, each prediction was. A healthy model's residuals
        # should scatter randomly around 0 with no obvious pattern; a
        # clear slope or curve would mean the model is systematically
        # wrong in some situations.
        residuals = y_test - y_pred
        resid_fig = go.Figure()
        resid_fig.add_trace(
            go.Scatter(
                x=y_pred,
                y=residuals,
                mode="markers",
                marker=dict(color="#F0AD4E", size=6, opacity=0.5),
                name="Residual",
            )
        )
        resid_fig.add_hline(y=0, line_color="#D9534F", line_dash="dash")
        resid_fig.update_layout(
            title="Residuals (Actual - Predicted) vs. Predicted score",
            xaxis_title="Predicted score",
            yaxis_title="Residual",
            height=400,
            margin=dict(t=50, b=10, l=10, r=10),
        )
        st.plotly_chart(resid_fig, width='stretch')

    # --- Learned coefficients bar chart --------------------------------------------
    st.subheader("What the model learned")
    st.write(
        "Each bar is the model's learned weight for one input: holding every "
        "other input fixed, this is how many productivity points that feature "
        "adds (positive) or removes (negative) per unit increase."
    )

    sorted_coefficients = coefficients.sort_values()
    coef_fig = go.Figure(
        go.Bar(
            x=sorted_coefficients.values,
            y=[pretty_label(c) for c in sorted_coefficients.index],
            orientation="h",
            marker_color=[
                "#5CB85C" if v >= 0 else "#D9534F" for v in sorted_coefficients.values
            ],
        )
    )
    coef_fig.update_layout(
        xaxis_title="Coefficient (productivity points per unit)",
        height=500,
        margin=dict(t=10, b=10, l=10, r=10),
    )
    st.plotly_chart(coef_fig, width='stretch')
