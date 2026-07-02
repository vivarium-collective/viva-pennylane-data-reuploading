# pbg-pennylane-data-reuploading

Process-bigraph wrapper for the **PennyLane data-reuploading quantum classifier** (Pérez-Salinas et al. 2019). A single-qubit variational quantum circuit that achieves universal classification via sequential data re-uploading.

## Architecture

The wrapper bridges PennyLane's quantum simulator as a process-bigraph `Process`. Each `update()` call advances one training epoch (or transitions through the pipeline phases: `init → training → done`).

```
┌─────────────────────────────────────────┐
│  PennyLaneDataReuploadingProcess        │
│                                         │
│  Input ports:  train_inputs,            │
│                 train_labels,           │
│                 test_inputs,            │
│                 test_labels             │
│                                         │
│  Output ports: phase, epoch, loss,      │
│                 train_accuracy,         │
│                 test_accuracy,          │
│                 best_test_accuracy      │
│                                         │
│  Config:       num_layers, learning_rate│
│                 epochs, batch_size,     │
│                 num_train, num_test,    │
│                 seed, device, radius    │
└─────────────────────────────────────────┘
```

The classifier trains on 2D points from a circle dataset (inside/outside a radius, binary classification) using fidelity-based cost and Adam optimization.

## Installation

```bash
# From PyPI (recommended):
pip install pbg-pennylane-data-reuploading

# For development (editable):
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Once installed, processes register automatically via `bigraph_schema.package.discover` — no manual `register_link()` calls are needed.

## Quick Start

```python
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter

core = allocate_core()
core.register_link("ram-emitter", RAMEmitter)

doc = {
    "classifier": {
        "_type": "process",
        "address": "local:PennyLaneDataReuploadingProcess",
        "config": {"num_layers": 3, "epochs": 10},
        "outputs": {
            "phase": ["stores", "phase"],
            "epoch": ["stores", "epoch"],
            "test_accuracy": ["stores", "test_accuracy"],
        },
    },
    "stores": {
        "phase": "init",
        "epoch": 0,
        "test_accuracy": 0.0,
    },
    "emitter": {
        "_type": "step",
        "address": "local:RAMEmitter",
        "config": {"emit": {"phase": "string", "epoch": "integer", "test_accuracy": "float"}},
        "inputs": {"phase": ["stores", "phase"], "epoch": ["stores", "epoch"],
                    "test_accuracy": ["stores", "test_accuracy"], "time": ["global_time"]},
    },
}

sim = Composite({"state": doc}, core=core)
sim.run(10.0)
```

## Composite Generators

| Generator | Description |
|---|---|
| `data_reuploading_baseline` | 3 layers, 10 epochs, standard circle dataset |
| `data_reuploading_deep` | 6 layers, 15 epochs, 500 training samples |
| `data_reuploading_lightweight` | 2 layers, 3 epochs, small dataset for smoke testing |

## Demo

```bash
source .venv/bin/activate
python demo/demo_report.py
```

Opens an interactive HTML report with time-series charts and bigraph architecture diagrams.

## API Reference

| Port | Type | Description |
|---|---|---|
| `train_inputs` | `list[list[float]]` | 2D training points |
| `train_labels` | `list[integer]` | Training labels (0 or 1) |
| `test_inputs` | `list[list[float]]` | 2D test points |
| `test_labels` | `list[integer]` | Test labels (0 or 1) |
| `phase` | `string` | Pipeline phase: init/training/done |
| `epoch` | `integer` | Current training epoch |
| `loss` | `float` | Current fidelity loss |
| `train_accuracy` | `float` | Training set accuracy |
| `test_accuracy` | `float` | Test set accuracy |
| `best_test_accuracy` | `float` | Best test accuracy seen so far |

## Use Cases

- **Minimal Genome Evaluation**: Screen simulated gene knockouts (KOs) from v2ecoli by training the classifier on metabolic trajectory embeddings.
- **Spatiotemporal State Sorting**: Categorize 4D structural cellular imaging trajectories from wriggler/live-cell data into healthy or diseased metabolic modes.
- **Quantum-Classical Hybrid Pipeline**: Insert the classifier as a `Process` in a larger composite alongside v2ecoli, FBA wrappers, and spatial simulators.

## References

- Pérez-Salinas, Adrián, et al. "Data re-uploading for a universal quantum classifier." arXiv:1907.02085 (2019).
- [PennyLane data-reuploading demo](https://pennylane.ai/demos/tutorial_data_reuploading_classifier)
