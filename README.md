# AI-Optimised Flash Storage for Edge AI

A simulation-based proof of concept that demonstrates how workload-aware flash placement can improve latency, energy use, and throughput for edge AI devices.

![Project preview](assets/project-preview.png)

## Demo options

- Terminal mode for quick verification
- Streamlit web app for presentation and live demo
- Live telemetry mode using short real-time process disk I/O capture

## What This Project Is

This project is a lightweight storage-systems simulator, not a hardware benchmark. It models flash zones with different performance and energy characteristics, then compares:

- **Baseline random placement**
- **AI-inspired workload-aware placement**

The current "AI" is an explainable heuristic policy rather than a trained neural network or classifier. That keeps the project transparent, lightweight, and appropriate for a proof-of-concept demo.

## Why this project stands out

This project moves beyond a generic storage simulation by framing the problem around **edge AI workloads** such as video analytics, sensor fusion, and model caching.

The goal is to show how intelligent workload-aware storage decisions can reduce latency, lower energy use, and improve throughput.

## Project idea upgrade

To make the project more impressive for lecturers, recruiters, or GitHub visitors, position it as a:

**Lightweight Edge-AI Storage Optimisation Simulator**

That tells a stronger story than just “flash architecture.” It shows:

- systems thinking
- AI-inspired decision logic
- performance engineering
- energy-efficiency awareness
- practical experimentation

## Current features

- Synthetic edge workload generation
- Live telemetry mode using process-level disk I/O activity captured from the local machine
- Zone-based flash model with different latency, energy, and wear profiles
- AI-inspired hotness scoring policy
- Capacity-aware block placement
- Placement-aware scheduling bonuses for matched data-zone behavior
- Baseline vs optimised comparison report
- Unit tests and GitHub Actions CI
- Browser-based Streamlit demo for presentation

## Verified sample results

Using the default synthetic simulation setup:

- **Latency reduction:** 12.46%
- **Energy reduction:** 11.62%
- **Wear-cost reduction:** 0.57%
- **Throughput increase:** 14.23%

## Repository structure

- [edge_ai_flash_project.py](edge_ai_flash_project.py) — main simulation script
- [app.py](app.py) — Streamlit demo application
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

Expected result: a report comparing baseline and AI-optimised placement.

## How to run the web demo

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

The browser demo lets you switch between a controlled synthetic workload and a short live telemetry capture from the current machine.

## Live telemetry mode

The web demo supports a real-time capture mode. During a short capture window, it samples current process disk I/O counters and converts that activity into workload profiles for the simulator.

This is more realistic than synthetic generation, but it is still **process-level telemetry**, not raw flash-controller logs or true block-level NAND tracing.

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

> This project is a simulation-based proof of concept for AI-aware flash placement on edge devices, showing that workload-aware storage decisions can reduce latency and energy use while improving throughput.

## Example project pitch

> Edge devices increasingly run AI workloads, but their flash storage is usually managed with generic strategies that ignore workload behavior. This project simulates an AI-optimised flash placement layer that analyses access frequency, write intensity, and data reuse to place blocks more efficiently. The result is lower latency, reduced energy consumption, and better throughput for next-generation edge systems.

## GitHub checklist

A strong beginner-friendly GitHub repo should include:

- clear project title
- problem statement
- setup steps
- test coverage
- license
- CI workflow
- future roadmap

This repository now includes the core pieces.

## Suggested future upgrades

If you want to push this further, add one or more of these:

1. **Captured-process table**: show the top live workloads during telemetry mode
2. **Scenario presets**: healthcare edge AI, autonomous drones, smart city cameras
3. **CLI arguments**: choose workload size, random seed, or capture duration
4. **Results export**: save JSON or CSV reports for analysis
5. **Charts**: add richer visual comparisons and trend history
6. **Real ML model**: replace the heuristic scoring policy with a trained classifier or regressor

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
