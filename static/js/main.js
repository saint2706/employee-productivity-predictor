/* ============================================================================
   Employee Productivity Predictor — front-end behavior
   ----------------------------------------------------------------------------
   This page is server-rendered once (see templates/index.html / app.py),
   then everything below takes over: switching tabs, live-validating typed
   numbers against the dataset's observed range, calling POST /api/predict
   whenever the form changes, and drawing every chart with Plotly.js.

   Two colour palettes (light/dark) mirror the ones the model_core-powered
   backend reports metrics for, so a chart's "this pushes the score up"
   blue and "this pushes it down" red never disagree with the rest of the
   page, in either theme.
============================================================================ */

(() => {
  "use strict";

  // --- Chart color palettes --------------------------------------------------
  const PALETTE_LIGHT = {
    surface: "#fcfcfb",
    text_primary: "#0b0b0b",
    text_secondary: "#52514e",
    text_muted: "#898781",
    grid: "#e1e0d9",
    axis: "#c3c2b7",
    series: "#2a78d6",
    positive: "#2a78d6",
    negative: "#e34948",
    neutral: "#52514e",
    guide: "#52514e",
    highlight: "#e34948",
  };
  const PALETTE_DARK = {
    surface: "#1a1a19",
    text_primary: "#ffffff",
    text_secondary: "#c3c2b7",
    text_muted: "#898781",
    grid: "#2c2c2a",
    axis: "#383835",
    series: "#3987e5",
    positive: "#3987e5",
    negative: "#e66767",
    neutral: "#c3c2b7",
    guide: "#c3c2b7",
    highlight: "#e66767",
  };

  function activePalette() {
    const explicit = document.documentElement.getAttribute("data-theme");
    if (explicit === "dark") return PALETTE_DARK;
    if (explicit === "light") return PALETTE_LIGHT;
    return window.matchMedia("(prefers-color-scheme: dark)").matches
      ? PALETTE_DARK
      : PALETTE_LIGHT;
  }

  // --- Shared chart chrome, mirroring model_core-adjacent app.py's style ----
  function axisKwargs(palette, text) {
    const kwargs = {
      tickfont: { color: palette.text_muted },
      gridcolor: palette.grid,
      gridwidth: 1,
      griddash: "solid",
      zeroline: false,
      showline: true,
      linecolor: palette.axis,
      automargin: true,
    };
    if (text) kwargs.title = { text, font: { color: palette.text_secondary } };
    return kwargs;
  }

  function buildLayout(palette, opts) {
    opts = opts || {};
    const layout = {
      height: opts.height,
      margin: { t: opts.marginT ?? 50, b: opts.marginB ?? 10, l: 10, r: 10 },
      paper_bgcolor: palette.surface,
      plot_bgcolor: palette.surface,
      font: { color: palette.text_secondary, size: 13 },
      showlegend: !!opts.showlegend,
      legend: {
        orientation: "h",
        yanchor: "top",
        y: opts.legendY ?? -0.22,
        xanchor: "center",
        x: 0.5,
        font: { color: palette.text_secondary, size: 12 },
      },
      hoverlabel: {
        bgcolor: palette.surface,
        bordercolor: palette.axis,
        font: { color: palette.text_primary },
      },
      xaxis: axisKwargs(palette, opts.xaxisTitle),
      yaxis: axisKwargs(palette, opts.yaxisTitle),
    };
    if (opts.title) {
      layout.title = { text: opts.title, font: { size: 16, color: palette.text_primary } };
    }
    return layout;
  }

  const PLOT_CONFIG = { displayModeBar: false, responsive: true };

  // --- Chart builders ---------------------------------------------------------

  function renderHistogram(palette, targetScores, prediction) {
    const trace = {
      type: "histogram",
      x: targetScores,
      xbins: { start: 0, end: 100, size: 2.5 },
      marker: { color: palette.series },
      name: "All employees",
      hovertemplate: "%{x} score<br>%{y} employees<extra></extra>",
    };
    const layout = buildLayout(palette, {
      title: "Where this prediction falls among all employees",
      xaxisTitle: "Monthly Productivity Score",
      yaxisTitle: "Number of employees",
      height: 320,
    });
    layout.bargap = 0.04;
    if (prediction !== null && prediction !== undefined) {
      Object.assign(layout, predictionLineProps(palette, prediction));
    }
    Plotly.newPlot("chart-histogram", [trace], layout, PLOT_CONFIG);
  }

  function predictionLineProps(palette, prediction) {
    return {
      shapes: [
        {
          type: "line",
          x0: prediction,
          x1: prediction,
          y0: 0,
          y1: 1,
          yref: "paper",
          line: { color: palette.highlight, width: 2, dash: "dash" },
        },
      ],
      annotations: [
        {
          x: prediction,
          y: 1,
          yref: "paper",
          yanchor: "bottom",
          showarrow: false,
          text: "<b>Your prediction</b>",
          font: { color: palette.highlight, size: 12 },
        },
      ],
    };
  }

  function updateHistogramLine(palette, prediction) {
    Plotly.relayout("chart-histogram", predictionLineProps(palette, prediction));
  }

  function renderWaterfall(palette, data) {
    const text = data.labels.map((lbl, i) => {
      if (lbl === "Baseline") return data.values[i].toFixed(1);
      if (lbl === "Predicted score") return data.predicted_total.toFixed(1);
      return (data.values[i] >= 0 ? "+" : "") + data.values[i].toFixed(1);
    });
    const waterfallTrace = {
      type: "waterfall",
      x: data.labels,
      y: data.values,
      measure: data.measures,
      connector: { line: { color: palette.grid, width: 1 } },
      increasing: { marker: { color: palette.positive } },
      decreasing: { marker: { color: palette.negative } },
      totals: { marker: { color: palette.neutral } },
      text,
      textposition: "outside",
      textfont: { color: palette.text_primary },
      showlegend: false,
    };
    const legendTraces = [
      ["Increases score", palette.positive],
      ["Decreases score", palette.negative],
      ["Running total", palette.neutral],
    ].map(([name, color]) => ({
      x: [null],
      y: [null],
      mode: "markers",
      marker: { size: 10, color, symbol: "square" },
      name,
      hoverinfo: "skip",
    }));
    const layout = buildLayout(palette, {
      title: "How each input contributed to this prediction",
      yaxisTitle: "Productivity points",
      height: 560,
      marginB: 160,
      legendY: -0.35,
      showlegend: true,
    });
    Plotly.newPlot("chart-waterfall", [waterfallTrace, ...legendTraces], layout, PLOT_CONFIG);
  }

  function renderScatter(palette, scatter) {
    const { actual, predicted } = scatter;
    const axisMin = Math.min(...actual, ...predicted);
    const axisMax = Math.max(...actual, ...predicted);
    const pointsTrace = {
      type: "scatter",
      mode: "markers",
      x: actual,
      y: predicted,
      marker: { color: palette.series, size: 8, opacity: 0.55, line: { color: palette.surface, width: 1 } },
      name: "Test employees",
      hovertemplate: "Actual: %{x:.1f}<br>Predicted: %{y:.1f}<extra></extra>",
    };
    const guideTrace = {
      type: "scatter",
      mode: "lines",
      x: [axisMin, axisMax],
      y: [axisMin, axisMax],
      line: { color: palette.guide, dash: "dash", width: 2 },
      name: "Perfect prediction",
      hoverinfo: "skip",
    };
    const layout = buildLayout(palette, {
      title: "Actual vs. Predicted productivity score",
      xaxisTitle: "Actual score",
      yaxisTitle: "Predicted score",
      height: 400,
    });
    layout.annotations = [
      {
        x: axisMax,
        y: axisMax,
        text: "Perfect prediction",
        showarrow: false,
        xanchor: "right",
        yanchor: "bottom",
        font: { color: palette.guide, size: 11 },
      },
    ];
    Plotly.newPlot("chart-scatter", [pointsTrace, guideTrace], layout, PLOT_CONFIG);
  }

  function renderResiduals(palette, scatter, residuals) {
    const trace = {
      type: "scatter",
      mode: "markers",
      x: scatter.predicted,
      y: residuals,
      marker: { color: palette.series, size: 8, opacity: 0.55, line: { color: palette.surface, width: 1 } },
      name: "Residual",
      hovertemplate: "Predicted: %{x:.1f}<br>Residual: %{y:+.1f}<extra></extra>",
    };
    const layout = buildLayout(palette, {
      title: "Residuals (Actual - Predicted) vs. Predicted score",
      xaxisTitle: "Predicted score",
      yaxisTitle: "Residual",
      height: 400,
    });
    layout.shapes = [
      {
        type: "line",
        x0: 0,
        x1: 1,
        xref: "paper",
        y0: 0,
        y1: 0,
        line: { color: palette.guide, dash: "dash", width: 2 },
      },
    ];
    layout.annotations = [
      {
        x: 0,
        y: 0,
        xref: "paper",
        xanchor: "left",
        yanchor: "bottom",
        text: "Perfect prediction",
        showarrow: false,
        font: { color: palette.guide, size: 11 },
      },
    ];
    Plotly.newPlot("chart-residuals", [trace], layout, PLOT_CONFIG);
  }

  function renderCoefficients(palette, coefficients) {
    const { labels, values } = coefficients;
    const positive = values.map((v) => (v >= 0 ? v : null));
    const negative = values.map((v) => (v < 0 ? v : null));
    const posTrace = {
      type: "bar",
      x: positive,
      y: labels,
      orientation: "h",
      marker: { color: palette.positive },
      name: "Increases score",
      hovertemplate: "%{y}: %{x:+.2f} pts<extra></extra>",
    };
    const negTrace = {
      type: "bar",
      x: negative,
      y: labels,
      orientation: "h",
      marker: { color: palette.negative },
      name: "Decreases score",
      hovertemplate: "%{y}: %{x:+.2f} pts<extra></extra>",
    };
    const layout = buildLayout(palette, {
      xaxisTitle: "Coefficient (productivity points per unit)",
      height: 500,
      marginT: 10,
      marginB: 50,
      showlegend: true,
    });
    layout.barmode = "overlay";
    layout.bargap = 0.3;
    Plotly.newPlot("chart-coefficients", [posTrace, negTrace], layout, PLOT_CONFIG);
  }

  // --- App state ---------------------------------------------------------------
  const appData = JSON.parse(document.getElementById("app-data").textContent);
  let lastResult = null;
  let performanceRendered = false;

  // --- Theme toggle --------------------------------------------------------------
  const root = document.documentElement;
  const storedTheme = localStorage.getItem("theme");
  if (storedTheme) root.setAttribute("data-theme", storedTheme);

  document.getElementById("theme-toggle").addEventListener("click", () => {
    const next = activePalette() === PALETTE_DARK ? "light" : "dark";
    localStorage.setItem("theme", next);
    root.setAttribute("data-theme", next);
    rerenderAllCharts();
  });

  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (!localStorage.getItem("theme")) rerenderAllCharts();
  });

  function rerenderAllCharts() {
    const palette = activePalette();
    renderHistogram(palette, appData.targetScores, lastResult ? lastResult.prediction : null);
    if (lastResult) renderWaterfall(palette, lastResult.waterfall);
    if (performanceRendered) {
      renderScatter(palette, appData.scatter);
      renderResiduals(palette, appData.scatter, appData.residuals);
      renderCoefficients(palette, appData.coefficients);
    }
  }

  // --- Tabs ------------------------------------------------------------------
  const tabButtons = document.querySelectorAll(".tab");
  tabButtons.forEach((btn) => btn.addEventListener("click", () => activateTab(btn.dataset.tab)));

  function activateTab(name) {
    tabButtons.forEach((btn) =>
      btn.setAttribute("aria-selected", btn.dataset.tab === name ? "true" : "false")
    );
    document.getElementById("tab-predict").hidden = name !== "predict";
    document.getElementById("tab-performance").hidden = name !== "performance";

    if (name === "performance") {
      if (!performanceRendered) {
        const palette = activePalette();
        renderScatter(palette, appData.scatter);
        renderResiduals(palette, appData.scatter, appData.residuals);
        renderCoefficients(palette, appData.coefficients);
        performanceRendered = true;
      } else {
        ["chart-scatter", "chart-residuals", "chart-coefficients"].forEach((id) =>
          Plotly.Plots.resize(document.getElementById(id))
        );
      }
    }
  }

  // --- Predict form: range hints + live prediction ---------------------------
  const form = document.getElementById("predict-form");
  const numberInputs = form.querySelectorAll('input[type="number"]');
  const selects = form.querySelectorAll("select");

  function formatNum(n) {
    return Number.isInteger(n) ? String(n) : String(parseFloat(n.toFixed(4)));
  }

  function updateHint(input) {
    const hintEl = document.querySelector(`[data-hint-for="${input.name}"]`);
    if (!hintEl) return;
    const min = parseFloat(input.dataset.min);
    const max = parseFloat(input.dataset.max);
    const val = parseFloat(input.value);
    if (!Number.isNaN(val) && (val < min || val > max)) {
      hintEl.hidden = false;
      hintEl.textContent = `⚠️ Typical range in the data is ${formatNum(min)}–${formatNum(
        max
      )}. This value is outside that range, but it'll still be used.`;
    } else {
      hintEl.hidden = true;
    }
  }

  let debounceTimer;
  function scheduleUpdate() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(updatePrediction, 250);
  }

  function gatherFormData() {
    const data = {
      Department: form.elements["Department"].value,
      Job_Level: form.elements["Job_Level"].value,
    };
    let valid = true;
    numberInputs.forEach((el) => {
      const val = parseFloat(el.value);
      if (Number.isNaN(val)) valid = false;
      data[el.name] = val;
    });
    return valid ? data : null;
  }

  async function updatePrediction() {
    const data = gatherFormData();
    if (!data) return;
    let result;
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      if (!res.ok) throw new Error(`predict failed: ${res.status}`);
      result = await res.json();
    } catch (err) {
      console.error(err);
      return;
    }
    lastResult = result;
    renderPredictionResult(result);
  }

  function renderPredictionResult(result) {
    document.getElementById("prediction-value").textContent = result.prediction.toFixed(1);
    document.getElementById(
      "prediction-help"
    ).textContent = `Model's typical error on unseen data is about ±${result.mae.toFixed(1)} points (MAE).`;
    document.getElementById(
      "prediction-percentile"
    ).innerHTML = `This is higher than <strong>${result.percentile.toFixed(
      0
    )}%</strong> of employees in the dataset.`;

    const noteEl = document.getElementById("prediction-note");
    if (result.raw_prediction !== result.prediction) {
      noteEl.hidden = false;
      noteEl.textContent = `Note: the model's raw formula output ${result.raw_prediction.toFixed(
        1
      )} for this combination of inputs; it's clipped to 0–100 here since that's the scale the training data uses.`;
    } else {
      noteEl.hidden = true;
    }

    const palette = activePalette();
    updateHistogramLine(palette, result.prediction);
    renderWaterfall(palette, result.waterfall);
  }

  numberInputs.forEach((el) => {
    updateHint(el);
    el.addEventListener("input", () => {
      updateHint(el);
      scheduleUpdate();
    });
  });
  selects.forEach((el) => el.addEventListener("change", scheduleUpdate));

  // --- Initial render ----------------------------------------------------------
  renderHistogram(activePalette(), appData.targetScores, null);
  updatePrediction();
})();
