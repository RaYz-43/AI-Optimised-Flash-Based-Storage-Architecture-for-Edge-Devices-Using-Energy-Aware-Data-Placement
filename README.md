# AI-Optimised Flash Storage for Edge AI

A simulation-based proof of concept showing how a trained machine learning model combined with energy-aware flash placement can improve latency, energy use, and throughput for edge AI devices.

![Project preview](assets/project-preview.png)

## Demo options

- Terminal mode for quick verification
- Streamlit web app for presentation and live demo
- Live telemetry mode using short real-time process disk I/O capture

## What This Project Is

This project is a lightweight storage-systems simulator rather than a hardware benchmark. It models flash zones with different performance and energy characteristics, then compares:

- **Baseline random placement**
- **ML-guided workload-aware placement**

The optimisation path benchmarks multiple tabular classifiers and selects the best performer to predict the most suitable flash zone from workload features such as access frequency, write ratio, temporal reuse, and block size. A heuristic fallback remains in the backend so the simulator stays robust even if the saved model artifact is unavailable.

## Why this project stands out

This project moves beyond a generic storage simulation by framing the problem around **edge AI workloads** such as video analytics, sensor fusion, and model caching.

The goal is to show how workload-aware storage decisions can reduce latency, lower energy use, and improve throughput.

## Project idea upgrade

For lecturers, recruiters, or GitHub visitors, position the project as a:

**Lightweight Edge-AI Storage Optimisation Simulator**

That tells a stronger story than just “flash architecture.” It shows:

- systems thinking
- applied machine learning
- performance engineering
- energy-efficiency awareness
- practical experimentation

## Current features

- Synthetic benchmark workload generation for repeatable evaluation
- Live telemetry mode using process-level disk I/O activity captured from the local machine
- Zone-based flash model with different latency, energy, and wear profiles
- Offline-trained placement model selected by benchmark comparison (currently Gradient Boosting)
- Hidden heuristic fallback for robustness if the saved model is unavailable
- Capacity-aware block placement
- Capacity-aware overflow handling when workload demand exceeds physical flash capacity
- Placement-aware scheduling bonuses for matched data-zone behavior
- Baseline vs optimised comparison report
- Saved training dataset and model artifact for reproducibility
- Unit tests and GitHub Actions CI
- Browser-based Streamlit demo for presentation

## Verified sample results

Using the default synthetic benchmark setup, a representative run produced:

- **Latency reduction:** 12.21%
- **Energy reduction:** 10.88%
- **Wear-cost reduction:** 0.12%
- **Throughput increase:** 13.90%

## Capacity model and workload size

The simulator models a fixed physical flash capacity of 250 blocks across three zones:

- HOT_CACHE: 75 blocks
- BALANCED: 105 blocks
- COLD_DENSE: 70 blocks

In the web UI, you can run synthetic workloads up to 500 blocks. For runs above 250 blocks, the simulator now applies an explicit overflow pressure model instead of silently forcing all overflow into one zone. This keeps high-load experiments interpretable and avoids an artificial quality cliff immediately after the physical-capacity threshold.

## Repository structure

- [edge_ai_flash_project.py](edge_ai_flash_project.py) — main simulation script
- [app.py](app.py) — Streamlit demo application
- [ml/training.py](ml/training.py) — model training pipeline and artifact export
- [ml/inference.py](ml/inference.py) — runtime model loading and zone prediction
- [tests/test_edge_ai_flash_project.py](tests/test_edge_ai_flash_project.py) — unit tests
- [.github/workflows/python-ci.yml](.github/workflows/python-ci.yml) — GitHub Actions CI
- [.gitignore](.gitignore) — ignores Python cache and local environment files
- [requirements.txt](requirements.txt) — Python dependencies for the demo app and live telemetry mode
- [LICENSE](LICENSE) — MIT license

## How to run

1. Install Python 3.10+
2. Open the project folder
3. Create or select a virtual environment
4. Install dependencies
5. Run the main script

```bash
python -m pip install -r requirements.txt
python edge_ai_flash_project.py
```

Expected result: a report comparing the baseline strategy with ML-guided placement.

## ML pipeline

The ML-backed placement flow is:

1. Generate or capture workloads
2. Extract workload features
3. Predict the best flash zone with the highest-scoring benchmarked classifier
4. Apply capacity-aware placement in the simulator
5. Compare results against baseline random placement

The trained model artifact is stored in `data/model.joblib`, and the generated labeled training dataset is stored in `data/training_workloads.csv`.

The training artifact also stores hold-out evaluation metrics, including accuracy, macro precision, macro recall, macro F1, weighted F1, and a confusion matrix.

### Model comparison (same hold-out split)

The training pipeline benchmarks three candidate models on the same 80/20 hold-out split and selects the model with the highest **macro F1** (best balance across all classes, including minority classes).

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.8833 | 0.8596 | 0.8878 |
| Random Forest | 0.9125 | 0.8722 | 0.9114 |
| **Gradient Boosting (selected)** | **0.9292** | **0.8939** | **0.9272** |

Why this winner: Gradient Boosting achieved the highest macro F1, so it provided the best class-balanced performance for this dataset.

To regenerate the model and print the evaluation metrics:

```bash
python -m ml.training
```

## How to run the web demo

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser demo lets you switch between a controlled synthetic benchmark workload and a short live telemetry capture from the current machine. The presentation flow always uses the ML-backed placement engine.

## Live telemetry mode

The web demo supports a real-time capture mode. During a short capture window, it samples current process disk I/O counters and converts that activity into workload profiles for the simulator.

This is more realistic than synthetic generation, but it is still **process-level telemetry**, not raw flash-controller logs or true block-level NAND tracing.

Synthetic mode remains useful as a controlled benchmark source for repeatable demos, testing, and model training.

Good activities for testing live mode:

- opening large folders in File Explorer
- copying files
- launching applications
- opening several browser tabs with heavy sites

## How to test

```bash
python -m unittest discover -s tests -v
```

All tests should pass before presenting or pushing changes.

## Presentation Summary

If you need one sentence for a demo:

> This project is a simulation-based proof of concept for ML-guided flash placement on edge devices, showing that workload-aware storage decisions can reduce latency and energy use while improving throughput.

## Example project pitch

> Edge devices increasingly run AI workloads, but their flash storage is often managed with generic strategies that ignore workload behavior. This project simulates an ML-optimised flash placement layer that extracts workload features and uses the best-performing benchmarked classifier to guide data placement across flash zones. The result is lower latency, reduced energy consumption, and better throughput for next-generation edge systems.

## GitHub checklist

A strong beginner-friendly GitHub repo should include:

- clear project title
- problem statement
- setup steps
- test coverage
- license
- CI workflow
- future roadmap

This repository includes the core pieces expected in a strong beginner-friendly research or portfolio project.

## Suggested future upgrades

If you want to push this further, add one or more of these:

1. **Captured-process table**: show the top live workloads during telemetry mode
2. **Scenario presets**: healthcare edge AI, autonomous drones, smart city cameras
3. **CLI arguments**: choose workload size, random seed, or capture duration
4. **Results export**: save JSON or CSV reports for analysis
5. **Charts**: add richer visual comparisons and trend history
6. **Model selection expansion**: add XGBoost/LightGBM and cross-validation-based benchmarking

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
