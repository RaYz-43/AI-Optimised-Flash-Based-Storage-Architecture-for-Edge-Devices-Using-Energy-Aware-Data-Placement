# AI-Optimised Flash Storage for Edge AI

A simulation-based project showing how machine-learning-guided data placement can improve flash storage behavior for edge workloads.

It compares:
- **Baseline random placement**
- **ML-guided energy-aware placement**

with measurable impact on:
- latency
- energy per operation
- throughput
- wear cost

---

## Project scope (important)

This is a **storage simulation proof-of-concept**, not a physical NAND controller benchmark.

- Uses synthetic workload generation and optional live process I/O telemetry
- Models three flash zones with different cost profiles
- Evaluates policy quality through baseline vs optimized simulation outcomes

---

## Current ML setup

The training pipeline benchmarks **three required models** on the **same dataset** and **same hold-out split**:

1. Decision Tree
2. Linear Regression (converted to classifier)
3. Gradient Boosting

Model selection rule:
- Select the model with highest **Macro F1**

Why Macro F1:
- The zone classes are not perfectly balanced.
- Macro F1 treats each class equally, so model quality is not dominated by a majority class.

### Current selected model

With the latest trained artifact in this repo, the selected model is:
- **Linear Regression (as classifier)**

(Selection may change if you retrain with different data/seed.)

---

## Key features

- Repeatable synthetic workload generation
- Live telemetry mode (process-level disk I/O capture)
- Capacity-aware zone placement with overflow handling
- Baseline vs optimized comparison under identical workload input
- Model benchmark table and selection by Macro F1
- Confusion matrix and per-class metrics
- Feature importance visualization (when supported by model)
- Streamlit presentation UI

---

## Repository structure

- `edge_ai_flash_project.py` — simulation core
- `app.py` — Streamlit demo app
- `ml/training.py` — model training/benchmark pipeline
- `ml/inference.py` — model loading + runtime inference
- `ml/features.py` — feature engineering helpers
- `tests/test_edge_ai_flash_project.py` — unit tests
- `data/model.joblib` — trained model artifact
- `data/training_workloads.csv` — generated training dataset

---

## Run guide (Windows)

### 1) Install dependencies

```bash
py -3.13 -m pip install -r requirements.txt
```

### 2) Train / refresh model artifact

```bash
py -3.13 -m ml.training
```

### 3) Run terminal simulation

```bash
py -3.13 edge_ai_flash_project.py
```

### 4) Run Streamlit demo app

```bash
py -3.13 -m streamlit run app.py --server.port 8502
```

Open: `http://localhost:8502`

> If 8502 is busy, use any free port.

---

## What to show in poster/demo

Keep your presentation focused on these 3 visuals:

1. **Baseline vs AI metrics** (latency, energy, throughput, wear)
2. **Model Macro F1 comparison chart** (3 models)
3. **Confusion matrix** of selected model

And these 3 talking points:

- “Macro F1 is used because class balance matters.”
- “All models are compared on the same split for fairness.”
- “The selected model drives zone prediction in optimized placement.”

---

## Evaluation outputs stored in model artifact

`data/model.joblib` stores:
- selected model name
- full comparison results for all three models
- hold-out metrics (accuracy, macro precision/recall/F1, weighted F1)
- per-class performance
- confusion matrix
- classification report
- feature importance (if available)

---

## Testing

```bash
py -3.13 -m unittest discover -s tests -v
```

---

## Notes on realism and limits

- Live telemetry is process-level I/O activity, not hardware flash trace logs.
- Simulation cost equations are simplified but consistent and useful for comparative evaluation.
- Results should be presented as **relative improvement in simulation**, not absolute hardware guarantee.

---

## License

MIT License. See `LICENSE`.
