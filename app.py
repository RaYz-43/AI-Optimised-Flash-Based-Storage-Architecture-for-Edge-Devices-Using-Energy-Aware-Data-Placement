from __future__ import annotations

import pandas as pd
import streamlit as st

from edge_ai_flash_project import (
    PlacementResult,
    capture_live_process_workloads,
    get_last_simulation_diagnostics,
    run_simulation,
)
from ml.inference import get_model_evaluation, load_trained_model


st.set_page_config(
    page_title="Edge AI Flash Storage Demo",
    page_icon="AI",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --bg: #f3efe6;
        --panel: rgba(255, 252, 247, 0.88);
        --panel-strong: #fffaf1;
        --ink: #1f2937;
        --muted: #6b7280;
        --accent: #c2410c;
        --accent-soft: #fb923c;
        --edge: #1d4ed8;
        --success: #0f766e;
        --border: rgba(148, 163, 184, 0.35);
        --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(251, 146, 60, 0.22), transparent 28%),
            radial-gradient(circle at top right, rgba(37, 99, 235, 0.16), transparent 24%),
            linear-gradient(180deg, #f8f5ef 0%, #efe7da 100%);
        color: var(--ink);
        font-family: 'Space Grotesk', sans-serif;
    }

    .stApp h1, .stApp h2, .stApp h3 {
        font-family: 'Space Grotesk', sans-serif;
        letter-spacing: -0.02em;
        color: #111827;
    }

    .stApp p, .stApp li, .stApp label, .stApp .stMarkdown, .stApp .stCaption {
        font-family: 'Space Grotesk', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: rgba(255, 248, 240, 0.88);
        border-right: 1px solid rgba(194, 65, 12, 0.12);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: var(--shadow);
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 0.35rem 0.55rem;
    }

    .hero-shell {
        background: linear-gradient(135deg, rgba(255, 250, 241, 0.96), rgba(255, 244, 230, 0.92));
        border: 1px solid rgba(194, 65, 12, 0.15);
        border-radius: 28px;
        padding: 2rem 2.2rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        color: var(--accent);
        text-transform: uppercase;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        margin-bottom: 0.8rem;
    }

    .hero-title {
        font-size: 3rem;
        line-height: 1.02;
        margin: 0;
        max-width: 11ch;
    }

    .hero-copy {
        font-size: 1.06rem;
        line-height: 1.7;
        color: var(--muted);
        max-width: 64ch;
        margin-top: 1rem;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: 1.2fr 0.8fr;
        gap: 1rem;
        align-items: end;
    }

    .hero-panel {
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(29, 78, 216, 0.12);
        border-radius: 22px;
        padding: 1.15rem 1.2rem;
    }

    .hero-panel h4 {
        margin: 0 0 0.6rem 0;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--edge);
    }

    .hero-panel p {
        margin: 0.35rem 0;
        color: var(--ink);
        font-size: 0.98rem;
    }

    .impact-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.85rem;
        margin: 1rem 0 1.25rem 0;
    }

    .impact-card {
        background: rgba(255, 252, 247, 0.88);
        border: 1px solid rgba(15, 118, 110, 0.14);
        border-radius: 22px;
        padding: 1rem 1rem 0.9rem 1rem;
        box-shadow: var(--shadow);
    }

    .impact-label {
        color: var(--muted);
        font-size: 0.84rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .impact-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0 0.2rem 0;
        color: #111827;
    }

    .impact-detail {
        color: var(--success);
        font-size: 0.92rem;
        font-family: 'IBM Plex Mono', monospace;
    }

    .mono-text {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--muted);
        font-size: 0.92rem;
    }

    @media (max-width: 960px) {
        .hero-grid,
        .impact-strip {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 2.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def improvement(old: float, new: float, inverse: bool = True) -> float:
    if inverse:
        return ((old - new) / old) * 100.0
    return ((new - old) / old) * 100.0


def to_rows(baseline: PlacementResult, optimized: PlacementResult) -> list[dict[str, float | str]]:
    return [
        {
            "Metric": "Average latency (ms)",
            "Baseline": round(baseline.avg_latency_ms, 4),
            "AI-Optimized": round(optimized.avg_latency_ms, 4),
        },
        {
            "Metric": "Average energy (mJ/op)",
            "Baseline": round(baseline.avg_energy_mj, 4),
            "AI-Optimized": round(optimized.avg_energy_mj, 4),
        },
        {
            "Metric": "Average wear cost",
            "Baseline": round(baseline.avg_wear_cost, 4),
            "AI-Optimized": round(optimized.avg_wear_cost, 4),
        },
        {
            "Metric": "Throughput (ops/s)",
            "Baseline": round(baseline.throughput_ops_per_s, 2),
            "AI-Optimized": round(optimized.throughput_ops_per_s, 2),
        },
    ]


def to_improvement_dataframe(
    latency_gain: float,
    energy_gain: float,
    wear_gain: float,
    throughput_gain: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Impact": [latency_gain, energy_gain, wear_gain, throughput_gain],
        },
        index=["Latency", "Energy", "Wear", "Throughput"],
    )


def describe_workload_source(workload_mode: str, seed: int, live_capture_seconds: int) -> tuple[str, str]:
    if workload_mode == "Synthetic":
        return (
            f"Synthetic benchmark • seed {seed}",
            "Controlled synthetic workload tuned to represent mixed edge AI access patterns for repeatable benchmarking.",
        )

    return (
        f"Live telemetry • process I/O capture ({live_capture_seconds}s)",
        "Short real-time capture of process-level disk I/O converted into workload profiles for an ML-guided placement run.",
    )


model_artifact = load_trained_model()
selected_model_name = "Heuristic fallback"
if model_artifact is not None:
    selected_model_name = str(model_artifact.get("model_name", "Gradient Boosting"))


with st.sidebar:
    st.header("Simulation controls")
    workload_mode = st.radio("Workload source", options=["Synthetic", "Live telemetry"], horizontal=False)
    workload_count = 220
    seed = 123
    live_capture_seconds = 8

    if workload_mode == "Synthetic":
        workload_count = st.slider("Workload blocks", min_value=60, max_value=500, value=220, step=20)
        seed = st.number_input("Random seed", min_value=1, max_value=9999, value=123, step=1)
    else:
        live_capture_seconds = st.slider("Capture duration (seconds)", min_value=4, max_value=20, value=8, step=2)
        st.caption("Samples real-time process disk I/O counters and converts them into workload profiles.")

    run_clicked = st.button("Run simulation", type="primary", width="stretch")
    st.markdown(
        f"""
        <div class="mono-text">
        Placement engine: {selected_model_name} plus an energy-aware scheduler. Synthetic mode provides repeatable benchmarking, while Live telemetry samples current process I/O in real time.
        </div>
        """,
        unsafe_allow_html=True,
    )

if run_clicked or "simulation_result" not in st.session_state:
    if workload_mode == "Synthetic":
        source_label, source_note = describe_workload_source(workload_mode, seed, live_capture_seconds)
        st.session_state.simulation_result = {
            "source_label": source_label,
            "source_note": source_note,
            "results": run_simulation(count=workload_count, seed=seed, policy_mode="ml"),
        }
    else:
        with st.spinner(f"Capturing live process disk activity for {live_capture_seconds} seconds..."):
            live_workloads = capture_live_process_workloads(duration_sec=live_capture_seconds, sample_interval_sec=0.5)
        if not live_workloads:
            st.warning("No live disk I/O activity was captured. Try opening files, copying data, or refreshing a heavy app while capture runs.")
            st.session_state.simulation_result = {
                "source_label": f"Live telemetry • idle capture ({live_capture_seconds}s)",
                "source_note": "No meaningful live activity was detected, so the app fell back to a synthetic sample to keep the demo usable.",
                "results": run_simulation(count=120, seed=123, policy_mode="ml"),
            }
        else:
            source_label, source_note = describe_workload_source(workload_mode, seed, live_capture_seconds)
            st.session_state.simulation_result = {
                "source_label": source_label,
                "source_note": source_note,
                "results": run_simulation(workloads=live_workloads, policy_mode="ml"),
            }

result_bundle = st.session_state.simulation_result
baseline, optimized, total_blocks = result_bundle["results"]
source_label = result_bundle["source_label"]
source_note = result_bundle["source_note"]
model_evaluation = get_model_evaluation(model_artifact)
diagnostics = get_last_simulation_diagnostics()
total_capacity = int(diagnostics.get("total_capacity", 0)) if diagnostics else 0
optimized_overflow = int(diagnostics.get("optimized", {}).get("overflow_count", 0)) if diagnostics else 0
baseline_overflow = int(diagnostics.get("baseline", {}).get("overflow_count", 0)) if diagnostics else 0

if total_capacity > 0:
    st.caption(f"Flash capacity model: {total_capacity} blocks across HOT_CACHE, BALANCED, and COLD_DENSE zones.")
if optimized_overflow > 0 or baseline_overflow > 0:
    st.info(
        "Capacity pressure detected: workload demand exceeded physical capacity. "
        f"Overflow handling applied to {optimized_overflow} AI-optimized blocks and {baseline_overflow} baseline blocks."
    )

latency_gain = improvement(baseline.avg_latency_ms, optimized.avg_latency_ms)
energy_gain = improvement(baseline.avg_energy_mj, optimized.avg_energy_mj)
wear_gain = improvement(baseline.avg_wear_cost, optimized.avg_wear_cost)
throughput_gain = improvement(baseline.throughput_ops_per_s, optimized.throughput_ops_per_s, inverse=False)
wear_metric_label = "Wear Reduction" if wear_gain >= 0 else "Wear Tradeoff"
wear_summary = (
    f"{wear_gain:.2f}% lower wear under the ML-guided placement engine."
    if wear_gain >= 0
    else f"{abs(wear_gain):.2f}% higher wear in this run, indicating that the current policy prioritizes latency and throughput over wear in this scenario."
)
wear_caption = (
    "Positive percentages indicate lower average wear cost."
    if wear_gain >= 0
    else "A negative wear value means this run traded a small amount of wear efficiency for faster access and higher throughput."
)

st.markdown(
    f"""
    <section class="hero-shell">
        <div class="eyebrow">Edge AI Storage Demo</div>
        <div class="hero-grid">
            <div>
                <h1 class="hero-title">AI-Optimised Flash Placement for Edge Devices</h1>
                <p class="hero-copy">
                    This presentation demo compares a random placement baseline with an ML-guided placement engine that predicts the most suitable flash zone for each workload and then applies energy-aware scheduling.
                </p>
            </div>
            <div class="hero-panel">
                <h4>Presenter takeaway</h4>
                <p><strong>{total_blocks}</strong> workload profiles evaluated under the same conditions.</p>
                <p><strong>{latency_gain:.2f}%</strong> lower latency and <strong>{energy_gain:.2f}%</strong> lower energy under the ML-guided placement engine.</p>
                <p><strong>{throughput_gain:.2f}%</strong> higher throughput because the model prioritizes high-value data for faster flash tiers.</p>
                <p><strong>Wear outcome:</strong> {wear_summary}</p>
                <p><strong>Placement engine:</strong> {selected_model_name} + energy-aware scheduler</p>
                <p><strong>Input source:</strong> {source_label}</p>
                <p>{source_note}</p>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="impact-strip">
        <div class="impact-card">
            <div class="impact-label">Latency Reduction</div>
            <div class="impact-value">{latency_gain:.2f}%</div>
            <div class="impact-detail">AI: {optimized.avg_latency_ms:.4f} ms</div>
        </div>
        <div class="impact-card">
            <div class="impact-label">Energy Reduction</div>
            <div class="impact-value">{energy_gain:.2f}%</div>
            <div class="impact-detail">AI: {optimized.avg_energy_mj:.4f} mJ/op</div>
        </div>
        <div class="impact-card">
            <div class="impact-label">{wear_metric_label}</div>
            <div class="impact-value">{wear_gain:.2f}%</div>
            <div class="impact-detail">AI: {optimized.avg_wear_cost:.4f}</div>
        </div>
        <div class="impact-card">
            <div class="impact-label">Throughput Increase</div>
            <div class="impact-value">{throughput_gain:.2f}%</div>
            <div class="impact-detail">AI: {optimized.throughput_ops_per_s:.2f} ops/s</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

comparison_col, impact_col = st.columns([1.05, 0.95])

with comparison_col:
    with st.container(border=True):
        st.subheader("Metric Comparison")
        st.table(to_rows(baseline, optimized))
        st.caption("Raw metrics for the same workload source under baseline and ML-guided placement.")

with impact_col:
    with st.container(border=True):
        st.subheader("Improvement Overview")
        st.bar_chart(to_improvement_dataframe(latency_gain, energy_gain, wear_gain, throughput_gain), color="#c2410c")
        st.caption(f"Positive percentages show the relative benefit of the ML-guided placement engine. {wear_caption}")

st.caption(f"Current input: {source_label}. Placement model: {selected_model_name}.")

if diagnostics:
    optimized_zone_mix = diagnostics.get("optimized", {}).get("assignment_counts", {})
    baseline_zone_mix = diagnostics.get("baseline", {}).get("assignment_counts", {})
    mix_frame = pd.DataFrame(
        {
            "Baseline": baseline_zone_mix,
            "AI-Optimized": optimized_zone_mix,
        }
    ).fillna(0)
    with st.container(border=True):
        st.subheader("Zone Assignment Mix")
        st.dataframe(mix_frame.astype(int), width="stretch")
        st.caption("Shows how many blocks were assigned to each zone during this run.")

if model_evaluation is not None:
    metric_col, class_col = st.columns([0.9, 1.1])

    with metric_col:
        with st.container(border=True):
            st.subheader("ML Evaluation")
            metric_frame = pd.DataFrame(
                {
                    "Score": [
                        model_evaluation["accuracy"],
                        model_evaluation["macro_precision"],
                        model_evaluation["macro_recall"],
                        model_evaluation["macro_f1"],
                        model_evaluation["weighted_f1"],
                    ]
                },
                index=["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"],
            )
            st.table(metric_frame.style.format("{:.4f}"))
            st.caption(
                f"Hold-out evaluation on {model_evaluation['test_size']} test samples after training on {model_evaluation['train_size']} samples."
            )
            st.info(
                "Why F1 is emphasized: zone classes are not perfectly balanced, so Macro F1 gives equal importance to each class and avoids inflated performance claims from majority-class accuracy."
            )

    with class_col:
        with st.container(border=True):
            st.subheader("Per-Class Performance")
            per_class_frame = pd.DataFrame.from_dict(model_evaluation["per_class"], orient="index")
            st.table(per_class_frame.style.format({"precision": "{:.4f}", "recall": "{:.4f}", "f1": "{:.4f}", "support": "{:.0f}"}))
            confusion = model_evaluation["confusion_matrix"]
            confusion_frame = pd.DataFrame(confusion["values"], index=confusion["labels"], columns=confusion["labels"])
            st.caption("Confusion matrix on the hold-out set. Rows represent true labels and columns represent predicted labels.")
            st.dataframe(confusion_frame, width="stretch")

comparison = model_artifact.get("comparison", {}) if model_artifact is not None else {}
if comparison:
    with st.container(border=True):
        st.subheader("Model Benchmark (Same Dataset & Same Hold-out Split)")
        rows = []
        for name, metrics in comparison.items():
            rows.append(
                {
                    "Model": name,
                    "Accuracy": metrics["accuracy"],
                    "Macro F1": metrics["macro_f1"],
                    "Weighted F1": metrics["weighted_f1"],
                    "Macro Precision": metrics["macro_precision"],
                    "Macro Recall": metrics["macro_recall"],
                }
            )
        benchmark_frame = pd.DataFrame(rows).sort_values(by="Macro F1", ascending=False)
        st.dataframe(
            benchmark_frame.style.format(
                {
                    "Accuracy": "{:.4f}",
                    "Macro F1": "{:.4f}",
                    "Weighted F1": "{:.4f}",
                    "Macro Precision": "{:.4f}",
                    "Macro Recall": "{:.4f}",
                }
            ),
            width="stretch",
        )

        macro_f1_chart = benchmark_frame[["Model", "Macro F1"]].set_index("Model")
        st.bar_chart(macro_f1_chart, color="#0f766e")
        st.caption("Macro F1 chart across the three candidate models. Higher is better.")

        st.success(f"Selected model: {selected_model_name} (highest Macro F1)")
        top_model_name = str(benchmark_frame.iloc[0]["Model"])
        st.info(
            f"Model selection rationale: {top_model_name} achieved the best class-balanced performance on the same train/test split, so it is selected for deployment."
        )
        st.caption(
            "Fairness check: all three models are trained and tested on the same dataset, with the same feature set and the same hold-out split."
        )

feature_importance = model_evaluation.get("feature_importance") if model_evaluation is not None else None
if feature_importance:
    with st.container(border=True):
        st.subheader("Feature Importance (Selected Model)")
        fi_frame = pd.DataFrame(
            {
                "Feature": list(feature_importance.keys()),
                "Importance": list(feature_importance.values()),
            }
        )
        st.bar_chart(fi_frame.set_index("Feature"), color="#1d4ed8")
        st.caption("Relative importance from the selected model. Higher values indicate stronger influence on zone prediction.")

with st.container(border=True):
    st.subheader("Demo Explanation Mode")
    st.markdown(
        """
- Baseline assigns blocks randomly, then suffers from capacity and behavior mismatch penalties.
- AI-Optimized predicts a suitable zone first, then applies energy-aware scheduling bonuses.
- This creates a fair baseline-vs-optimized comparison under identical workload input.
        """
    )
