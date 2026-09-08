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

# --- Chart color palette ---------------------------------------------------------
#
# One small, fixed set of roles is reused across every chart below instead of
# each chart picking its own colors ad hoc. That matters for two reasons:
#   - a "positive contribution" is always the same blue and a "negative
#     contribution" is always the same red, whether you're looking at the
#     waterfall chart or the coefficients chart — so the two never disagree
#   - the hues are colorblind-safe (checked with a Delta-E-in-OKLab palette
#     validator, not just eyeballed) and keep enough contrast against the
#     surface to read clearly
# Both a light and a dark version are defined so charts still look
# intentional — not a jarring white box — if someone views this app in
# Streamlit's dark theme.
PALETTE_LIGHT = {
    "surface": "#fcfcfb",
    "text_primary": "#0b0b0b",
    "text_secondary": "#52514e",
    "text_muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "series": "#2a78d6",    # a single data series (histogram bars, scatter dots)
    "positive": "#2a78d6",  # a value that pushes the score up
    "negative": "#e34948",  # a value that pushes the score down
    "neutral": "#52514e",   # a running total — neither positive nor negative
    "guide": "#52514e",     # a passive reference line to compare data against
    "highlight": "#e34948", # an active marker calling out one specific value
}
PALETTE_DARK = {
    "surface": "#1a1a19",
    "text_primary": "#ffffff",
    "text_secondary": "#c3c2b7",
    "text_muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "series": "#3987e5",
    "positive": "#3987e5",
    "negative": "#e66767",
    "neutral": "#c3c2b7",
    "guide": "#c3c2b7",
    "highlight": "#e66767",
}


def active_palette() -> dict:
    """
    Picks the light or dark chart palette to match Streamlit's current
    theme, so charts don't render as a jarring white box in dark mode.
    st.context.theme.type is None when the theme can't be determined
    (falls back to light).
    """
    theme_type = getattr(st.context.theme, "type", None)
    return PALETTE_DARK if theme_type == "dark" else PALETTE_LIGHT


def style_chart(
    fig: go.Figure,
    palette: dict,
    *,
    title: str | None,
    height: int,
    margin_t: int = 50,
    margin_b: int = 10,
    legend_y: float = -0.22,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    showlegend: bool = False,
) -> go.Figure:
    """
    Applies the same chrome to every chart on the page — background, fonts,
    gridlines, legend placement, and hover box colors, all pulled from
    `palette` — so every chart reads as part of one system and moves
    together when Streamlit's theme changes. Chart-specific bits (traces,
    reference lines, axis ranges) are still set by each caller beforehand.
    """

    # Plotly's frontend renders a literal "undefined" title if `title` (or
    # an axis `title`) is explicitly set to None rather than left out of the
    # update entirely — so build the layout kwargs and only include a title
    # key at all when there's real text for it.
    layout_kwargs = dict(
        height=height,
        margin=dict(t=margin_t, b=margin_b, l=10, r=10),
        paper_bgcolor=palette["surface"],
        plot_bgcolor=palette["surface"],
        font=dict(color=palette["text_secondary"], size=13),
        showlegend=showlegend,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=legend_y,
            xanchor="center",
            x=0.5,
            font=dict(color=palette["text_secondary"], size=12),
        ),
        hoverlabel=dict(
            bgcolor=palette["surface"],
            bordercolor=palette["axis"],
            font=dict(color=palette["text_primary"]),
        ),
    )
    if title:
        layout_kwargs["title"] = dict(text=title, font=dict(size=16, color=palette["text_primary"]))
    fig.update_layout(**layout_kwargs)

    def axis_kwargs(text: str | None) -> dict:
        kwargs = dict(
            tickfont=dict(color=palette["text_muted"]),
            gridcolor=palette["grid"],
            gridwidth=1,
            griddash="solid",
            zeroline=False,
            showline=True,
            linecolor=palette["axis"],
        )
        if text:
            kwargs["title"] = dict(text=text, font=dict(color=palette["text_secondary"]))
        return kwargs

    fig.update_xaxes(**axis_kwargs(xaxis_title))
    fig.update_yaxes(**axis_kwargs(yaxis_title))
    return fig


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


def numeric_input_with_range_hint(label: str, column: str, step, integer: bool = False):
    """
    Draws a plain numeric entry box (st.number_input) for one feature.
    Unlike a slider, this has NO min/max enforced by Streamlit — the user
    can type any number, including ones nobody in the dataset ever had.
    Instead, after they type something, we compare it to the range
    actually observed in the dataset (column min/max) and — if it falls
    outside that range — show a light, non-blocking warning caption right
    underneath. The value the user typed is still used for the
    prediction either way; we're only informing, never refusing input.
    """
    lo = df[column].min()
    hi = df[column].max()
    default = df[column].median()

    if integer:
        # For whole-number features (like a count of projects or absence
        # days), pass plain Python ints so Streamlit draws an integer
        # spinner (no decimal point) instead of a float box.
        lo, hi, default = int(lo), int(hi), int(round(default))
        value = st.number_input(label, value=default, step=int(step))
    else:
        lo, hi, default = float(lo), float(hi), float(default)
        value = st.number_input(label, value=default, step=float(step))

    # :orange[...] is Streamlit's markdown syntax for colored text. Using
    # st.caption (small, muted text) with an orange warning icon keeps
    # this feeling like a gentle heads-up rather than a blocking error.
    if value < lo or value > hi:
        st.caption(
            f":orange[⚠️ Typical range in the data is {lo:g}–{hi:g}. "
            "This value is outside that range, but it'll still be used.]"
        )

    return value


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

        # Every field below is a plain number box (st.number_input), not a
        # slider — you can type any value, including unrealistic ones. If
        # what you type falls outside the range actually seen in the
        # dataset, numeric_input_with_range_hint() (defined above) shows a
        # light warning caption underneath, but never blocks the input.
        years_experience = numeric_input_with_range_hint(
            "Years of Experience", "Years_of_Experience", step=0.5
        )
        training_hours = numeric_input_with_range_hint(
            "Training Hours (this year)", "Training_Hours", step=1.0
        )
        working_hours = numeric_input_with_range_hint(
            "Monthly Working Hours", "Monthly_Working_Hours", step=1.0
        )
        projects_completed = numeric_input_with_range_hint(
            "Projects Completed (this month)", "Projects_Completed", step=1, integer=True
        )
        avg_task_time = numeric_input_with_range_hint(
            "Average Task Completion Time (hours)", "Average_Task_Completion_Time", step=0.1
        )
        absence_days = numeric_input_with_range_hint(
            "Absence Days (this month)", "Absence_Days", step=1, integer=True
        )
        engagement_score = numeric_input_with_range_hint(
            "Engagement Score", "Engagement_Score", step=1.0
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
    # (or now, with free-typed numbers, even unrealistic-enough) combination
    # of input values it can predict above 100 or below 0. We
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
        palette = active_palette()
        hist_fig = go.Figure()
        hist_fig.add_trace(
            go.Histogram(
                x=df[TARGET_COLUMN],
                # Pin the bins to the score's actual 0-100 scale instead of
                # leaving Plotly to pick "nice" bin edges on its own — with
                # nbinsx alone, Plotly can round to a bin width/offset (e.g.
                # 97.5-102.49) that overshoots past 100, implying scores
                # that can't actually occur (the target is clipped to
                # 0-100 in generate_dataset.py).
                xbins=dict(start=0, end=100, size=2.5),
                name="All employees",
                marker_color=palette["series"],
                hovertemplate="%{x} score<br>%{y} employees<extra></extra>",
            )
        )
        # A thin gap between touching bars (rather than an outlined border)
        # keeps each bin visually separate without adding ink that isn't data.
        hist_fig.update_layout(bargap=0.04)
        # add_vline draws a vertical line straight across the chart at a
        # given x position — here, the predicted score.
        hist_fig.add_vline(
            x=prediction,
            line_width=2,
            line_dash="dash",
            line_color=palette["highlight"],
            annotation_text="<b>Your prediction</b>",
            annotation_position="top",
            annotation_font_color=palette["highlight"],
            annotation_font_size=12,
        )
        style_chart(
            hist_fig,
            palette,
            title="Where this prediction falls among all employees",
            xaxis_title="Monthly Productivity Score",
            yaxis_title="Number of employees",
            height=320,
        )
        # The modebar (zoom/pan/export icons) floats over the top-right of
        # the chart; in this narrow column the title wraps right into it,
        # so it's hidden rather than fixed with fragile margin tweaks.
        st.plotly_chart(hist_fig, width='stretch', config={"displayModeBar": False})

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

        # The bar itself is drawn correctly without our help: Plotly computes
        # a "total" measure's height as the running sum of everything before
        # it (baseline + every contribution above), ignoring whatever y-value
        # we give it here — 0 is just a required placeholder. But our own
        # on-bar text labels below are plain string formatting, not magic,
        # so if we labeled this bar from that same placeholder it would
        # print "0.0" while the bar visibly reaches the real total. Compute
        # the true total (== raw_prediction, since it's baseline + every
        # contribution the model actually applied) so the label matches
        # what the bar shows.
        predicted_total = sum(values)
        labels.append("Predicted score")
        values.append(0)
        measures.append("total")

        waterfall_fig = go.Figure()
        waterfall_fig.add_trace(
            go.Waterfall(
                x=labels,
                y=values,
                measure=measures,
                connector={"line": {"color": palette["grid"], "width": 1}},
                increasing={"marker": {"color": palette["positive"]}},
                decreasing={"marker": {"color": palette["negative"]}},
                totals={"marker": {"color": palette["neutral"]}},
                text=[
                    f"{v:.1f}" if lbl == "Baseline"
                    else f"{predicted_total:.1f}" if lbl == "Predicted score"
                    else f"{v:+.1f}"
                    for lbl, v in zip(labels, values)
                ],
                textposition="outside",
                textfont=dict(color=palette["text_primary"]),
                showlegend=False,
            )
        )
        # go.Waterfall draws its three bar colors (increasing / decreasing /
        # total) from one trace, so Plotly can't generate a legend entry per
        # color on its own. Add three invisible marker-only traces purely so
        # the reader gets a legend explaining what each bar color means —
        # without one, "why is this bar red and that one blue?" has no
        # answer on the chart itself.
        for legend_name, color in (
            ("Increases score", palette["positive"]),
            ("Decreases score", palette["negative"]),
            ("Running total", palette["neutral"]),
        ):
            waterfall_fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=10, color=color, symbol="square"),
                    name=legend_name,
                    hoverinfo="skip",
                )
            )
        style_chart(
            waterfall_fig,
            palette,
            title="How each input contributed to this prediction",
            yaxis_title="Productivity points",
            height=560,
            margin_b=160,
            legend_y=-0.62,
            showlegend=True,
        )
        st.plotly_chart(waterfall_fig, width='stretch', config={"displayModeBar": False})


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
        palette = active_palette()
        scatter_fig = go.Figure()
        scatter_fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                marker=dict(
                    color=palette["series"],
                    size=8,
                    opacity=0.55,
                    line=dict(color=palette["surface"], width=1),
                ),
                name="Test employees",
                hovertemplate="Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
            )
        )
        # A perfect model would put every point exactly on this diagonal
        # line (predicted == actual). The closer the dots hug the line,
        # the better the model. It's a reference to compare the data
        # against, not data itself, so it gets a direct label instead of a
        # legend entry.
        axis_min = float(min(y_test.min(), y_pred.min()))
        axis_max = float(max(y_test.max(), y_pred.max()))
        scatter_fig.add_trace(
            go.Scatter(
                x=[axis_min, axis_max],
                y=[axis_min, axis_max],
                mode="lines",
                line=dict(color=palette["guide"], dash="dash", width=2),
                name="Perfect prediction",
                hoverinfo="skip",
            )
        )
        scatter_fig.add_annotation(
            x=axis_max,
            y=axis_max,
            text="Perfect prediction",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(color=palette["guide"], size=11),
        )
        style_chart(
            scatter_fig,
            palette,
            title="Actual vs. Predicted productivity score",
            xaxis_title="Actual score",
            yaxis_title="Predicted score",
            height=400,
        )
        st.plotly_chart(scatter_fig, width='stretch', config={"displayModeBar": False})

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
                marker=dict(
                    color=palette["series"],
                    size=8,
                    opacity=0.55,
                    line=dict(color=palette["surface"], width=1),
                ),
                name="Residual",
                hovertemplate="Predicted: %{x:.1f}<br>Residual: %{y:+.1f}<extra></extra>",
            )
        )
        resid_fig.add_hline(
            y=0,
            line_color=palette["guide"],
            line_dash="dash",
            line_width=2,
            annotation_text="Perfect prediction",
            annotation_position="top left",
            annotation_font_color=palette["guide"],
            annotation_font_size=11,
        )
        style_chart(
            resid_fig,
            palette,
            title="Residuals (Actual - Predicted) vs. Predicted score",
            xaxis_title="Predicted score",
            yaxis_title="Residual",
            height=400,
        )
        st.plotly_chart(resid_fig, width='stretch', config={"displayModeBar": False})

    # --- Learned coefficients bar chart --------------------------------------------
    st.subheader("What the model learned")
    st.write(
        "Each bar is the model's learned weight for one input: holding every "
        "other input fixed, this is how many productivity points that feature "
        "adds (positive) or removes (negative) per unit increase."
    )

    sorted_coefficients = coefficients.sort_values()
    coef_categories = [pretty_label(c) for c in sorted_coefficients.index]
    # Split into two traces (rather than one trace with a per-bar color
    # list) so each color gets its own legend entry — otherwise "blue means
    # increases, red means decreases" only lives in the prose above the
    # chart, not on the chart itself. Each row gets a real value in exactly
    # one trace and None in the other, so the two traces never overlap.
    positive_values = [v if v >= 0 else None for v in sorted_coefficients.values]
    negative_values = [v if v < 0 else None for v in sorted_coefficients.values]
    coef_fig = go.Figure()
    coef_fig.add_trace(
        go.Bar(
            x=positive_values,
            y=coef_categories,
            orientation="h",
            marker_color=palette["positive"],
            name="Increases score",
            hovertemplate="%{y}: %{x:+.2f} pts<extra></extra>",
        )
    )
    coef_fig.add_trace(
        go.Bar(
            x=negative_values,
            y=coef_categories,
            orientation="h",
            marker_color=palette["negative"],
            name="Decreases score",
            hovertemplate="%{y}: %{x:+.2f} pts<extra></extra>",
        )
    )
    coef_fig.update_layout(barmode="overlay", bargap=0.3)
    style_chart(
        coef_fig,
        palette,
        title=None,
        xaxis_title="Coefficient (productivity points per unit)",
        height=500,
        margin_t=10,
        margin_b=50,
        showlegend=True,
    )
    st.plotly_chart(coef_fig, width='stretch', config={"displayModeBar": False})
