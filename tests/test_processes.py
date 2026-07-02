import numpy as np
import pytest
from process_bigraph import Composite, allocate_core

from pbg_pennylane_data_reuploading.processes import PennyLaneDataReuploadingProcess


@pytest.fixture
def core():
    c = allocate_core()
    from process_bigraph.emitter import RAMEmitter
    c.register_link("ram-emitter", RAMEmitter)
    from pbg_pennylane_data_reuploading import PennyLaneDataReuploadingProcess
    c.register_link("PennyLaneDataReuploadingProcess", PennyLaneDataReuploadingProcess)
    return c


@pytest.fixture
def fast_config():
    return {
        "num_layers": 2,
        "learning_rate": 0.6,
        "epochs": 2,
        "batch_size": 16,
        "num_train": 20,
        "num_test": 20,
        "seed": 42,
    }


def test_instantiation(core, fast_config):
    proc = PennyLaneDataReuploadingProcess(config=fast_config, core=core)
    assert proc is not None


def test_inputs_outputs_schema(core, fast_config):
    proc = PennyLaneDataReuploadingProcess(config=fast_config, core=core)
    inputs = proc.inputs()
    outputs = proc.outputs()
    assert "train_inputs" in inputs
    assert "train_labels" in inputs
    assert "test_inputs" in inputs
    assert "test_labels" in inputs
    assert "phase" in outputs
    assert "epoch" in outputs
    assert "loss" in outputs
    assert "train_accuracy" in outputs
    assert "test_accuracy" in outputs


def test_initial_state(core, fast_config):
    proc = PennyLaneDataReuploadingProcess(config=fast_config, core=core)
    state = proc.initial_state()
    assert state["phase"] == "init"
    assert state["epoch"] == 0
    assert state["loss"] == 0.0


def test_full_pipeline(core, fast_config):
    proc = PennyLaneDataReuploadingProcess(config=fast_config, core=core)
    state = proc.initial_state()

    n_max = 10
    for _ in range(n_max):
        result = proc.update(state, interval=1.0)
        state = {**state, **result}
        if result["phase"] == "done":
            break

    assert state["phase"] == "done", f"Did not reach done after {n_max} updates"
    assert state["epoch"] >= 1
    assert 0.0 <= state["train_accuracy"] <= 1.0
    assert 0.0 <= state["test_accuracy"] <= 1.0


def test_synthetic_data_via_ports(core, fast_config):
    rng = np.random.RandomState(42)
    n = 20
    train_x = (2 * rng.rand(n, 2) - 1).tolist()
    train_y = [int(rng.rand() > 0.5) for _ in range(n)]
    test_x = (2 * rng.rand(10, 2) - 1).tolist()
    test_y = [int(rng.rand() > 0.5) for _ in range(10)]

    doc = {
        "classifier": {
            "_type": "process",
            "address": "local:PennyLaneDataReuploadingProcess",
            "config": fast_config,
            "inputs": {
                "train_inputs": ["stores", "train_inputs"],
                "train_labels": ["stores", "train_labels"],
                "test_inputs": ["stores", "test_inputs"],
                "test_labels": ["stores", "test_labels"],
            },
            "outputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
            },
        },
        "stores": {
            "train_inputs": train_x,
            "train_labels": train_y,
            "test_inputs": test_x,
            "test_labels": test_y,
            "phase": "init",
            "epoch": 0,
            "loss": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "phase": "string",
                    "epoch": "integer",
                    "loss": "float",
                    "train_accuracy": "float",
                    "test_accuracy": "float",
                }
            },
            "inputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "time": ["global_time"],
            },
        },
    }

    sim = Composite({"state": doc}, core=core)
    sim.run(10.0)
    assert sim.state["stores"]["phase"] == "done"


def test_generator_is_registered(fast_config):
    from pbg_superpowers.composite_generator import _REGISTRY
    matches = [eid for eid in _REGISTRY if "pennylane_data_reuploading" in eid]
    assert matches, f"no pennylane_data_reuploading generators in registry; have {list(_REGISTRY)[:10]}"
