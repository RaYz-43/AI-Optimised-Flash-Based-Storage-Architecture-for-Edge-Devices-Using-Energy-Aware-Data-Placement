from __future__ import annotations

import pandas as pd
import streamlit as st

from edge_ai_flash_project import (
    PlacementResult,
    capture_live_process_workloads,
    run_simulation,
)


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

    [data-testid="stSidebar"] * {
        font-family: 'Space Grotesk', sans-serif;
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

    .section-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.25rem 1.35rem;
        box-shadow: var(--shadow);
    }

    .note-card {
        background: linear-gradient(180deg, rgba(255, 253, 248, 0.96), rgba(255, 247, 234, 0.94));
        border: 1px solid rgba(194, 65, 12, 0.14);
        border-radius: 24px;
        padding: 1.25rem 1.35rem;
        box-shadow: var(--shadow);
    }

    .note-card h4,
    .section-card h4 {
        margin: 0 0 0.7rem 0;
        font-size: 1.02rem;
    }

    .note-card ol {
        margin: 0;
        padding-left: 1.15rem;
    }

    .note-card li {
        margin-bottom: 0.65rem;
        line-height: 1.55;
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
            f"Synthetic generator • seed {seed}",
            "Controlled synthetic workload tuned to represent mixed edge AI access patterns.",
        )

    return (
        f"Live telemetry • process I/O capture ({live_capture_seconds}s)",
        "Short real-time capture of process-level disk I/O converted into workload profiles for the simulator.",
    )


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

    run_clicked = st.button("Run simulation", type="primary", use_container_width=True)
    st.markdown(
        """
        <div class="mono-text">
        Synthetic mode is reproducible by seed. Live telemetry samples current process I/O activity in real time.
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
            "results": run_simulation(count=workload_count, seed=seed),
        }
    else:
        with st.spinner(f"Capturing live process disk activity for {live_capture_seconds} seconds..."):
            live_workloads = capture_live_process_workloads(duration_sec=live_capture_seconds, sample_interval_sec=0.5)
        if not live_workloads:
            st.warning("No live disk I/O activity was captured. Try opening files, copying data, or refreshing a heavy app while capture runs.")
            st.session_state.simulation_result = {
                "source_label": f"Live telemetry • idle capture ({live_capture_seconds}s)",
                "source_note": "No meaningful live activity was detected, so the app fell back to a synthetic sample to keep the demo usable.",
                "results": run_simulation(count=120, seed=123),
            }
        else:
            source_label, source_note = describe_workload_source(workload_mode, seed, live_capture_seconds)
            st.session_state.simulation_result = {
                "source_label": source_label,
                "source_note": source_note,
                "results": run_simulation(workloads=live_workloads),
            }

result_bundle = st.session_state.simulation_result
baseline, optimized, total_blocks = result_bundle["results"]
source_label = result_bundle["source_label"]
source_note = result_bundle["source_note"]

latency_gain = improvement(baseline.avg_latency_ms, optimized.avg_latency_ms)
energy_gain = improvement(baseline.avg_energy_mj, optimized.avg_energy_mj)
wear_gain = improvement(baseline.avg_wear_cost, optimized.avg_wear_cost)
throughput_gain = improvement(baseline.throughput_ops_per_s, optimized.throughput_ops_per_s, inverse=False)

st.markdown(
    f"""
    <section class="hero-shell">
        <div class="eyebrow">Edge AI Storage Demo</div>
        <div class="hero-grid">
            <div>
                <h1 class="hero-title">AI-Optimised Flash Placement for Edge Devices</h1>
                <p class="hero-copy">
                    This presentation demo compares a random flash-allocation baseline against a workload-aware policy that places hot, reusable AI data in faster storage zones and colder blocks in denser tiers.
                </p>
            </div>
            <div class="hero-panel">
                <h4>Presenter takeaway</h4>
                <p><strong>{total_blocks}</strong> workload profiles evaluated under the same conditions.</p>
                <p><strong>{latency_gain:.2f}%</strong> lower latency and <strong>{energy_gain:.2f}%</strong> lower energy under the AI-guided policy.</p>
                <p><strong>{throughput_gain:.2f}%</strong> higher throughput because the policy keeps high-value data closer to fast flash.</p>
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
            <div class="impact-label">Wear Reduction</div>
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

overview_col, note_col = st.columns([1.3, 0.9])

with overview_col:
    with st.container(border=True):
        st.subheader("Executive Summary")
        st.write(
            "Edge devices run AI inference with tight power and storage limits. This model shows that smarter data placement can cut access cost by prioritizing hot blocks for fast flash and relegating colder data to denser tiers."
        )
        st.write(
            "Both strategies use the same workload source, so any performance gap comes from placement quality rather than differences in the input data."
        )
        st.caption(f"Current input: {source_label}")

with note_col:
    with st.container(border=True):
        st.subheader("Talk Track")
        st.markdown(
            """
1. Start with the problem: edge AI has limited flash storage, power budget, and memory bandwidth.
2. Explain the baseline: random placement ignores how often a block is reused or how write-heavy it is.
3. Explain the AI-inspired policy: it scores blocks by access frequency, write behavior, and temporal reuse.
4. Close on impact: smarter placement lowers latency and energy while raising throughput under the same workload.
"""
        )

comparison_col, impact_col = st.columns([1.05, 0.95])

with comparison_col:
    with st.container(border=True):
        st.subheader("Metric Comparison")
        st.table(to_rows(baseline, optimized))
        st.caption("Raw metrics for the same workload source under both placement strategies.")

with impact_col:
    with st.container(border=True):
        st.subheader("Improvement Overview")
        st.bar_chart(to_improvement_dataframe(latency_gain, energy_gain, wear_gain, throughput_gain), color="#c2410c")
        st.caption("Positive percentages show the relative benefit of the AI-aware placement policy.")

closing_col, methodology_col = st.columns(2)

with closing_col:
    with st.container(border=True):
        st.subheader("Project Story")
        st.write(
            "This demo models an energy-aware flash architecture for edge devices. The main idea is simple: when storage decisions understand workload behavior, the system spends less time and energy fetching important data."
        )
        st.write(
            "That matters for video analytics, smart sensors, drones, and other edge AI systems where small efficiency gains can directly improve responsiveness and battery life."
        )

with methodology_col:
    with st.container(border=True):
        st.subheader("Methodology")
        st.write(
            "The simulator evaluates a random baseline and then applies an AI-inspired placement policy using the same inputs. This keeps the comparison controlled and easy to explain in a live demo."
        )
        st.write(
            "Hot, reusable blocks tend to move toward faster flash zones, while colder data is assigned to denser zones that trade a little speed for capacity efficiency."
        )
        st.caption("Live telemetry mode uses process-level disk I/O counters in real time. It is more realistic than synthetic input, but it is still not raw flash-controller logging.")