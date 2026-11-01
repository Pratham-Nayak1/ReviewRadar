"""Premium Streamlit interface for ReviewRadar.

All scoring remains delegated to ``ReviewRadarPredictor``. The app never
re-trains, reads labels for normal uploaded predictions, or invents metrics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.demo import load_demo_cases
from src.predict import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DISPLAY_IDENTIFIER,
    InferenceInputError,
    ReviewRadarPredictor,
)


ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = ROOT / "data" / "raw" / "apachejit_total.csv"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

st.set_page_config(page_title="ReviewRadar", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root { --rr-ink: #e8edf7; --rr-muted: #94a3b8; --rr-border: #263449; --rr-accent: #5eead4; }
        .stApp { background: radial-gradient(circle at 90% 0%, #172554 0, #0b1120 38%, #070b14 100%); color: var(--rr-ink); }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid var(--rr-border); }
        [data-testid="stMetric"] { background: rgba(17,24,39,.88); border: 1px solid var(--rr-border); border-radius: 14px; padding: 16px 18px; }
        [data-testid="stMetricLabel"] { color: var(--rr-muted); }
        [data-testid="stMetricValue"] { color: var(--rr-ink); }
        .rr-eyebrow { color: var(--rr-accent); font-size: .78rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 700; }
        .rr-subtitle { color: var(--rr-muted); font-size: 1.05rem; margin-top: -.65rem; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--rr-border); border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_predictor() -> ReviewRadarPredictor:
    return ReviewRadarPredictor(DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH)


@st.cache_data
def load_sample_data(feature_columns: tuple[str, ...]) -> pd.DataFrame:
    if not RAW_DATA_PATH.is_file():
        raise FileNotFoundError(f"Sample dataset not found: {RAW_DATA_PATH}")
    return pd.read_csv(RAW_DATA_PATH, usecols=[DISPLAY_IDENTIFIER, *feature_columns], nrows=50)


@st.cache_data
def load_report_csv(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


@st.cache_data
def load_config() -> dict:
    return json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) if DEFAULT_CONFIG_PATH.is_file() else {}


@st.cache_data(show_spinner="Preparing held-out Demo Lab examples...")
def cached_demo_cases() -> list[dict]:
    return load_demo_cases()


def render_brand() -> None:
    st.markdown('<div class="rr-eyebrow">Developer risk intelligence</div>', unsafe_allow_html=True)
    st.title("ReviewRadar")
    st.markdown('<div class="rr-subtitle">Prioritize software changes for review with an honest, leakage-aware risk score.</div>', unsafe_allow_html=True)


def render_sidebar() -> str:
    st.sidebar.markdown("## ReviewRadar")
    st.sidebar.caption("ApacheJIT change-risk workspace")
    page = st.sidebar.selectbox("Workspace", ["Overview", "Analyze Changes", "Model Performance", "Demo Lab", "About / Methodology"], index=0)
    st.sidebar.divider()
    st.sidebar.caption("Score = ranking signal, not a calibrated probability or a definitive bug diagnosis.")
    return page


def render_overview(config: dict, predictor: ReviewRadarPredictor) -> None:
    st.markdown('<div class="rr-eyebrow">Workspace overview</div>', unsafe_allow_html=True)
    st.header("A clearer queue for engineering review")
    st.write("ReviewRadar scores historical software changes and surfaces the changes that deserve earlier human attention. It supports review prioritization; it does not replace testing, ownership, or engineering judgment.")
    try:
        sample = load_sample_data(tuple(predictor.feature_columns))
        predictions = predictor.predict(sample)
        flagged = int(predictions["predicted_buggy"].sum()) if "predicted_buggy" in predictions else 0
        median_score = float(predictions["risk_score"].median())
    except Exception as exc:
        st.warning(f"The demo batch is unavailable: {exc}")
        sample, predictions, flagged, median_score = pd.DataFrame(), pd.DataFrame(), 0, 0.0
    classification = load_report_csv("evaluation_classification_metrics.csv")
    test_f1 = "Pending"
    if not classification.empty:
        test_rows = classification[classification["model"].astype(str).str.endswith("_final_test")]
        if not test_rows.empty:
            test_f1 = f"{float(test_rows.iloc[0]['f1']):.3f}"
    metrics = st.columns(4)
    metrics[0].metric("Demo changes analyzed", f"{len(sample):,}")
    metrics[1].metric("Threshold flagged", f"{flagged:,}")
    metrics[2].metric("Median score", f"{median_score:.3f}")
    metrics[3].metric("Held-out test F1", test_f1, help="Frozen historical benchmark result, not a live estimate.")
    if not predictions.empty:
        left, right = st.columns([1.25, 1])
        with left:
            st.subheader("Risk-score distribution")
            chart = px.histogram(predictions, x="risk_score", nbins=16, color="predicted_buggy", color_discrete_sequence=["#64748b", "#5eead4"], labels={"risk_score": "Model score", "predicted_buggy": "Threshold flag"})
            chart.update_layout(height=330, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1", legend_title_text="")
            st.plotly_chart(chart, width="stretch")
        with right:
            st.subheader("Priority queue preview")
            preview = sample[[DISPLAY_IDENTIFIER]].copy()
            preview["risk_score"] = predictions["risk_score"]
            preview["predicted_buggy"] = predictions.get("predicted_buggy", False)
            preview = preview.sort_values("risk_score", ascending=False).head(5)
            st.dataframe(preview.style.format({"risk_score": "{:.3f}"}), width="stretch", hide_index=True)
    st.info("The overview uses 50 deterministic ApacheJIT rows as a demo batch. Upload your own schema-compatible CSV from Analyze Changes to score a different batch.")


def render_analyze_sidebar(predictor: ReviewRadarPredictor) -> tuple[str, float, bool, float, object]:
    st.sidebar.subheader("Analysis controls")
    mode = st.sidebar.radio("Data source", ["Sample data", "Upload CSV"], index=0)
    uploaded = st.sidebar.file_uploader("Upload a schema-compatible CSV", type=["csv"]) if mode == "Upload CSV" else None
    review_budget = st.sidebar.slider("Review budget", 1, 100, 10, 1, format="%d%%") / 100
    only_flagged = st.sidebar.checkbox("Show only threshold-flagged changes", value=False)
    minimum_score = st.sidebar.slider("Minimum displayed score", 0.0, 1.0, 0.0, 0.01)
    st.sidebar.caption("Required: the 12 trained numeric metrics. Optional: commit_id. Targets and metadata are rejected.")
    return mode, review_budget, only_flagged, minimum_score, uploaded


def get_input_data(predictor: ReviewRadarPredictor, mode: str, uploaded) -> tuple[pd.DataFrame | None, str]:
    if mode == "Sample data":
        try:
            return load_sample_data(tuple(predictor.feature_columns)), "deterministic ApacheJIT sample rows"
        except Exception as exc:
            st.error(f"Could not load sample data: {exc}")
            return None, ""
    if uploaded is None:
        st.info("Upload a CSV to begin. The file should contain only trained features and an optional commit_id column.")
        return None, ""
    try:
        return pd.read_csv(uploaded), uploaded.name
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        return None, ""


def rank_results(input_data: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    result = predictions.copy()
    result.insert(0, "source_index", range(len(result)))
    result = result.sort_values("risk_score", ascending=False, kind="mergesort").reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def render_queue(predictor: ReviewRadarPredictor, input_data: pd.DataFrame, predictions: pd.DataFrame, review_budget: float, only_flagged: bool, minimum_score: float) -> None:
    ranked = rank_results(input_data, predictions)
    budget_k = max(1, math.ceil(len(ranked) * review_budget)) if len(ranked) else 0
    queue = ranked.head(budget_k).copy()
    queue = queue[queue["risk_score"] >= minimum_score]
    if only_flagged and "predicted_buggy" in queue:
        queue = queue[queue["predicted_buggy"]]
    st.subheader("Prioritized review queue")
    st.caption(f"Top {budget_k} of {len(ranked)} changes are in the selected budget; filters may hide rows from display.")
    metrics = st.columns(3)
    metrics[0].metric("Changes scored", f"{len(ranked):,}")
    metrics[1].metric("Budget size", f"{budget_k:,}")
    metrics[2].metric("Displayed", f"{len(queue):,}")
    if queue.empty:
        st.warning("No changes match the current filters. Clear a filter or increase the review budget.")
        return
    display_columns = [c for c in ["rank", DISPLAY_IDENTIFIER, "risk_score", "predicted_buggy", "risk_category"] if c in queue]
    st.dataframe(queue[display_columns].style.format({"risk_score": "{:.3f}"}), width="stretch", hide_index=True)
    st.download_button("Download ranked queue", queue[display_columns].to_csv(index=False), "reviewradar_ranked_queue.csv", "text/csv")
    options = queue["source_index"].tolist()
    labels = {int(source_index): (f"#{int(row['rank'])} — {row[DISPLAY_IDENTIFIER]}" if DISPLAY_IDENTIFIER in queue else f"Row {int(row['rank'])}") for source_index, row in queue.set_index("source_index").iterrows()}
    selected_source = st.selectbox("Select a change to inspect", options=options, format_func=lambda value: labels[value])
    selected_result = ranked[ranked["source_index"] == selected_source].iloc[0]
    selected_input = input_data.iloc[int(selected_source)]
    detail = st.columns(3)
    detail[0].metric("Risk score", f"{float(selected_result['risk_score']):.3f}")
    detail[1].metric("Threshold flag", "Review priority" if bool(selected_result.get("predicted_buggy", False)) else "Below threshold")
    detail[2].metric("Risk category", str(selected_result.get("risk_category", "Score-based")))
    st.caption("The score is a prioritization signal. It is not calibrated and does not prove that this change contains a defect.")
    st.markdown("**Available change metrics**")
    st.dataframe(selected_input[predictor.feature_columns].to_frame("value"), width="stretch")
    estimator = predictor.pipeline.named_steps.get("estimator")
    if hasattr(estimator, "feature_importances_"):
        st.markdown("**Global model context**")
        importance = pd.DataFrame({"feature": predictor.feature_columns, "global_importance": estimator.feature_importances_}).sort_values("global_importance", ascending=False).head(8)
        st.dataframe(importance.style.format({"global_importance": "{:.3f}"}), width="stretch", hide_index=True)
        st.caption("These are global Random Forest importances, not a causal explanation of this individual change. Use Demo Lab for local behavior-based examples.")


def render_analyze_page(predictor: ReviewRadarPredictor) -> None:
    st.markdown('<div class="rr-eyebrow">Analyze changes</div>', unsafe_allow_html=True)
    st.header("Turn a batch into a review queue")
    st.write("Upload only the metrics the saved pipeline understands, or start with the deterministic sample batch.")
    mode, budget, only_flagged, minimum_score, uploaded = render_analyze_sidebar(predictor)
    input_data, source_name = get_input_data(predictor, mode, uploaded)
    if input_data is None:
        return
    st.caption(f"Source: {source_name} · {len(input_data):,} rows loaded")
    with st.expander("Input preview", expanded=True):
        st.dataframe(input_data.head(8), width="stretch", hide_index=True)
    try:
        predictions = predictor.predict(input_data)
    except InferenceInputError as exc:
        st.error(f"Input schema validation failed: {exc}")
        st.info("Provide the 12 trained numeric features and optional commit_id. Remove target and metadata columns.")
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return
    render_queue(predictor, input_data.reset_index(drop=True), predictions, budget, only_flagged, minimum_score)


def render_performance_page() -> None:
    st.markdown('<div class="rr-eyebrow">Evidence, not promises</div>', unsafe_allow_html=True)
    st.header("Model performance")
    st.warning("These are saved historical ApacheJIT benchmark results. They are not guarantees of future performance and do not mean the model prevents bugs.")
    config = load_config()
    classification = load_report_csv("evaluation_classification_metrics.csv")
    budget = load_report_csv("evaluation_review_budget.csv")
    calibration = load_report_csv("evaluation_calibration_metrics.csv")
    if classification.empty or budget.empty:
        st.error("Saved evaluation reports are missing. Run `python -m src.evaluate` before opening this page.")
        return
    st.write(f"Positive class: **{config.get('positive_class', 'not configured')}** · Split: **chronological author_date split**")
    candidates = classification[classification["threshold_source"] == "fixed_default"]
    st.subheader("Validation candidate comparison")
    st.dataframe(candidates[["model", "precision", "recall", "f1", "pr_auc_average_precision", "threshold"]].style.format({c: "{:.3f}" for c in ["precision", "recall", "f1", "pr_auc_average_precision", "threshold"]}), width="stretch", hide_index=True)
    st.subheader("Frozen selected-model results")
    selected_names = [f"{config.get('selected_model')}_tuned", f"{config.get('selected_model')}_final_test"]
    selected = classification[classification["model"].isin(selected_names)]
    st.dataframe(selected[["model", "rows", "threshold", "precision", "recall", "f1", "pr_auc_average_precision", "brier_score"]].style.format({c: "{:.3f}" for c in ["threshold", "precision", "recall", "f1", "pr_auc_average_precision", "brier_score"]}), width="stretch", hide_index=True)
    left, right = st.columns(2)
    with left:
        pr_path = FIGURES_DIR / "pr_curve_test.png"
        st.subheader("Held-out precision-recall curve")
        if pr_path.is_file():
            st.image(str(pr_path), caption="Selected model on the frozen test split")
        else:
            st.info("PR curve figure is unavailable. Run evaluation to regenerate it.")
    with right:
        st.subheader("Review-budget trade-off")
        options = sorted(float(v) for v in budget["review_fraction"].unique())
        chosen = st.selectbox("Budget", options, format_func=lambda v: f"Top {v:.0%} of changes")
        view = budget[budget["review_fraction"] == chosen].copy()
        chart = px.bar(view, x="approach", y=["precision_at_k", "recall_at_k"], facet_col="split", barmode="group", range_y=[0, 1], labels={"value": "rate", "variable": "metric"})
        chart.update_layout(height=360, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
        st.plotly_chart(chart, width="stretch")
    st.caption("Threshold selection happened on validation only. Changing a displayed budget changes ranking coverage; it does not retrain or retune the model. Random-selection values are simulations, not measured operational savings.")
    st.subheader("Calibration diagnostics")
    st.dataframe(calibration.style.format({"brier_score": "{:.3f}", "expected_calibration_error": "{:.3f}"}), width="stretch", hide_index=True)
    matrices = load_report_csv("evaluation_confusion_matrices.csv")
    st.subheader("Confusion matrices")
    st.dataframe(matrices.rename(columns={"true_negative": "TN", "false_positive": "FP", "false_negative": "FN", "true_positive": "TP"}), width="stretch", hide_index=True)
    test_matrix = matrices[matrices["split"] == "test"] if not matrices.empty and "split" in matrices.columns else pd.DataFrame()
    if not test_matrix.empty:
        row = test_matrix.iloc[0]
        matrix_frame = pd.DataFrame(
            [[int(row["true_negative"]), int(row["false_positive"])], [int(row["false_negative"]), int(row["true_positive"])]],
            index=["Actual non-buggy", "Actual buggy"],
            columns=["Predicted non-buggy", "Predicted buggy"],
        )
        matrix_plot = px.imshow(matrix_frame, text_auto=True, color_continuous_scale="Blues", aspect="auto", labels={"x": "Prediction", "y": "Ground truth", "color": "Count"})
        matrix_plot.update_layout(height=320, margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#cbd5e1")
        st.plotly_chart(matrix_plot, width="stretch")


def render_demo_lab() -> None:
    st.markdown('<div class="rr-eyebrow">Interactive Demo Lab</div>', unsafe_allow_html=True)
    st.header("See where the model helps — and where it misses")
    st.write("These are real, inspectable records from the held-out chronological test split. They were selected after the model and threshold were frozen and were not used for fitting or tuning.")
    st.info("Local explanations use one-feature-at-a-time replacement with the training median. A positive delta means that observed value raised the model score relative to that counterfactual. It is behavior evidence, not causation.")
    cases = cached_demo_cases()
    if not cases:
        st.warning("No demo cases are available in the held-out split.")
        return
    labels = {case["case_id"]: f"{case['outcome']} · {case['case_id']}" for case in cases}
    selected_id = st.selectbox("Choose a held-out case", list(labels), format_func=lambda value: labels[value])
    case = next(case for case in cases if case["case_id"] == selected_id)
    top = st.columns(4)
    top[0].metric("Outcome", case["outcome"])
    top[1].metric("Risk score", f"{case['risk_score']:.3f}")
    top[2].metric("Threshold", f"{case['threshold']:.3f}")
    top[3].metric("Project", case["project"])
    st.caption(f"Source: {case['source']} · stable identifier: `{case['commit_id']}`")
    st.write(f"Ground truth: **buggy={case['actual_buggy']}** · Prediction: **buggy={case['predicted_buggy']}**")
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Features used by the model")
        feature_frame = pd.DataFrame({"feature": list(case["features"]), "value": list(case["features"].values())})
        st.dataframe(feature_frame, width="stretch", hide_index=True)
    with right:
        st.subheader("Local behavior explanation")
        explanation = pd.DataFrame(case["explanation"])
        st.dataframe(explanation.style.format({"score_delta_vs_median": "{:+.4f}"}), width="stretch", hide_index=True)
    st.caption("The model can be wrong because these summary metrics do not capture all code context, project conventions, defect semantics, or label noise. A false alarm is not evidence of a bad change; a missed defect is not evidence that the change was safe.")


def render_about() -> None:
    st.markdown('<div class="rr-eyebrow">Methodology</div>', unsafe_allow_html=True)
    st.header("About ReviewRadar")
    st.write("ReviewRadar is an educational classical-ML project for prioritizing historical ApacheJIT software changes for human review.")
    st.subheader("How to interpret the score")
    st.write("The Random Forest score ranks changes. It is not calibrated as a probability. The positive class is `buggy=True`, an SZZ-derived historical label meaning the commit introduced a bug according to the dataset documentation. Neither a high score nor a threshold flag proves an individual change is buggy.")
    st.subheader("Leakage-safe protocol")
    st.write("The project uses author_date for a chronological train/validation/test split, excludes identifiers and unresolved post-outcome metadata, fits preprocessing inside scikit-learn pipelines, tunes the threshold on validation only, and evaluates the held-out test after decisions are frozen.")
    st.subheader("Scope and limitations")
    st.write("ApacheJIT represents historical Apache projects and SZZ-derived labels. Performance may shift across projects and time. ReviewRadar does not predict severity, repair time, review effort, cost savings, or bugs prevented, and it should not be used as the sole basis for blocking releases or assigning blame.")
    st.markdown("Dataset documentation: `data/README.md` · Evaluation protocol: `reports/evaluation_protocol.md` · Baseline: `reports/baseline.md`")


def main() -> None:
    inject_styles()
    render_brand()
    try:
        predictor = load_predictor()
    except Exception as exc:
        st.error(f"ReviewRadar model could not be loaded: {exc}")
        st.info("Run `python -m src.train` and `python -m src.evaluate` before launching it.")
        st.stop()
    page = render_sidebar()
    config = load_config()
    if page == "Overview":
        render_overview(config, predictor)
    elif page == "Analyze Changes":
        render_analyze_page(predictor)
    elif page == "Model Performance":
        render_performance_page()
    elif page == "Demo Lab":
        render_demo_lab()
    else:
        render_about()


if __name__ == "__main__":
    main()
