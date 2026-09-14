"""
Employee Productivity Predictor — Streamlit web app.

Loads the trained pipeline saved by model_training.ipynb (model.pkl) and
lets a user enter an employee's profile to get a predicted Monthly
Productivity Score (0-100), along with model performance context.

To run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

DATA_PATH = "data.csv"
MODEL_PATH = "model.pkl"
TARGET_COLUMN = "Monthly_Productivity_Score"
NUMERIC_FEATURES = [
    "Years_of_Experience",
    "Training_Hours",
    "Monthly_Working_Hours",
    "Projects_Completed",
    "Average_Task_Completion_Time",
    "Absence_Days",
    "Engagement_Score",
]
CATEGORICAL_FEATURES = ["Department", "Job_Level"]
JOB_LEVEL_ORDER = ["Junior", "Mid", "Senior", "Lead", "Manager"]

st.set_page_config(
    page_title="Employee Productivity Predictor", page_icon="📈", layout="wide"
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def evaluate_model(_model, df: pd.DataFrame):
    """
    Recreates the same train/test split used in model_training.ipynb
    (same random_state) purely to report honest, held-out performance
    numbers and charts here in the app — the model itself is loaded
    pre-trained from model.pkl, not retrained.
    """
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    y_pred = _model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "n_test": len(X_test),
    }
    return X_test, y_test, y_pred, metrics


df = load_data()
model = load_model()
X_test, y_test, y_pred, metrics = evaluate_model(model, df)

st.title("📈 Employee Productivity Predictor")
st.caption(
    "Predicts Monthly Productivity Score (0-100) from an employee's profile, "
    "using the pipeline trained in model_training.ipynb."
)

predict_tab, performance_tab = st.tabs(["🔮 Predict", "📊 Model Performance"])

# =============================================================================
# TAB 1: Predict
# =============================================================================
with predict_tab:
    left, right = st.columns([1, 1.3])

    with left:
        st.subheader("Employee profile")
        department = st.selectbox("Department", sorted(df["Department"].unique()))
        job_level = st.selectbox("Job Level", JOB_LEVEL_ORDER)
        years_experience = st.number_input(
            "Years of Experience", value=float(df["Years_of_Experience"].median()), step=0.5
        )
        training_hours = st.number_input(
            "Training Hours (this year)", value=float(df["Training_Hours"].median()), step=1.0
        )
        working_hours = st.number_input(
            "Monthly Working Hours", value=float(df["Monthly_Working_Hours"].median()), step=1.0
        )
        projects_completed = st.number_input(
            "Projects Completed (this month)",
            value=int(df["Projects_Completed"].median()),
            step=1,
        )
        avg_task_time = st.number_input(
            "Average Task Completion Time (hours)",
            value=float(df["Average_Task_Completion_Time"].median()),
            step=0.1,
        )
        absence_days = st.number_input(
            "Absence Days (this month)", value=int(df["Absence_Days"].median()), step=1
        )
        engagement_score = st.number_input(
            "Engagement Score", value=float(df["Engagement_Score"].median()), step=1.0
        )

        input_row = pd.DataFrame(
            [
                {
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
            ]
        )

    raw_prediction = float(model.predict(input_row)[0])
    prediction = max(0.0, min(100.0, raw_prediction))
    percentile = float((df[TARGET_COLUMN] <= prediction).mean() * 100)

    with right:
        st.subheader("Predicted productivity")
        st.metric(
            "Monthly Productivity Score",
            f"{prediction:.1f} / 100",
            help=f"Model's typical error on unseen data is about ±{metrics['mae']:.1f} points (MAE).",
        )
        st.write(f"This is higher than **{percentile:.0f}%** of employees in the dataset.")
        if raw_prediction != prediction:
            st.caption(
                f"Note: the model's raw output was {raw_prediction:.1f}; it's clipped "
                "to 0-100 here since that's the scale the training data uses."
            )

        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=df[TARGET_COLUMN],
                xbins=dict(start=0, end=100, size=2.5),
                name="All employees",
                marker_color="#2a78d6",
            )
        )
        hist_fig.add_vline(
            x=prediction,
            line_width=2,
            line_dash="dash",
            line_color="#e34948",
            annotation_text="Your prediction",
        )
        hist_fig.update_layout(
            title="Where this prediction falls among all employees",
            xaxis_title="Monthly Productivity Score",
            yaxis_title="Number of employees",
            height=350,
            bargap=0.04,
        )
        st.plotly_chart(hist_fig, width="stretch")

# =============================================================================
# TAB 2: Model Performance
# =============================================================================
with performance_tab:
    st.subheader("How good is this model?")
    st.write(
        "Numbers below are computed on a held-out test set "
        f"({metrics['n_test']:,} employees the model never trained on)."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("R² (variance explained)", f"{metrics['r2']:.3f}")
    c2.metric("Mean Absolute Error", f"{metrics['mae']:.2f} pts")
    c3.metric("Root Mean Squared Error", f"{metrics['rmse']:.2f} pts")

    chart_left, chart_right = st.columns(2)

    with chart_left:
        scatter_fig = go.Figure()
        scatter_fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                marker=dict(color="#2a78d6", size=7, opacity=0.5),
                name="Test employees",
            )
        )
        lo, hi = float(min(y_test.min(), y_pred.min())), float(max(y_test.max(), y_pred.max()))
        scatter_fig.add_trace(
            go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines",
                line=dict(color="#52514e", dash="dash"), name="Perfect prediction",
            )
        )
        scatter_fig.update_layout(
            title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted", height=380
        )
        st.plotly_chart(scatter_fig, width="stretch")

    with chart_right:
        residuals = y_test - y_pred
        resid_fig = go.Figure()
        resid_fig.add_trace(
            go.Scatter(
                x=y_pred, y=residuals, mode="markers",
                marker=dict(color="#2a78d6", size=7, opacity=0.5), name="Residual",
            )
        )
        resid_fig.add_hline(y=0, line_color="#52514e", line_dash="dash")
        resid_fig.update_layout(
            title="Residuals vs Predicted", xaxis_title="Predicted", yaxis_title="Residual", height=380
        )
        st.plotly_chart(resid_fig, width="stretch")

    st.subheader("What the model learned")
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pd.Series(
        model.named_steps["regressor"].coef_, index=feature_names
    ).sort_values()
    coef_fig = go.Figure(
        go.Bar(
            x=coefficients.values,
            y=[c.split("__")[-1] for c in coefficients.index],
            orientation="h",
            marker_color=np.where(coefficients.values >= 0, "#2a78d6", "#e34948"),
        )
    )
    coef_fig.update_layout(
        title="Learned coefficients (scaled numeric features + one-hot categories)",
        xaxis_title="Coefficient",
        height=500,
    )
    st.plotly_chart(coef_fig, width="stretch")
