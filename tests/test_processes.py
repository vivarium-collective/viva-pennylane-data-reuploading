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


# ---------------------------------------------------------------------------
# Tests for GenotypeToPhenotypeProcess
# ---------------------------------------------------------------------------

@pytest.fixture
def gt2p_core():
    c = allocate_core()
    from process_bigraph.emitter import RAMEmitter
    c.register_link("ram-emitter", RAMEmitter)
    from pbg_pennylane_data_reuploading import GenotypeToPhenotypeProcess
    c.register_link("GenotypeToPhenotypeProcess", GenotypeToPhenotypeProcess)
    return c


@pytest.fixture
def gt2p_config():
    return {
        "num_omic_modalities": 3,
        "features_per_modality": 3,
        "learning_rate": 0.4,
        "epochs": 2,
        "batch_size": 8,
        "num_train": 12,
        "num_test": 12,
        "seed": 42,
        "noise_level": 0.05,
    }


def test_gt2p_instantiation(gt2p_core, gt2p_config):
    from pbg_pennylane_data_reuploading import GenotypeToPhenotypeProcess
    proc = GenotypeToPhenotypeProcess(config=gt2p_config, core=gt2p_core)
    assert proc is not None


def test_gt2p_inputs_outputs_schema(gt2p_core, gt2p_config):
    from pbg_pennylane_data_reuploading import GenotypeToPhenotypeProcess
    proc = GenotypeToPhenotypeProcess(config=gt2p_config, core=gt2p_core)
    inputs = proc.inputs()
    outputs = proc.outputs()
    assert "train_inputs" in inputs
    assert "train_labels" in inputs
    assert "test_inputs" in inputs
    assert "test_labels" in inputs
    for key in ("phase", "epoch", "loss", "train_accuracy", "test_accuracy",
                "best_test_accuracy", "auc_roc", "f1_score",
                "precision_viable", "recall_viable", "viable_ratio"):
        assert key in outputs, f"missing output key: {key}"


def test_gt2p_initial_state(gt2p_core, gt2p_config):
    from pbg_pennylane_data_reuploading import GenotypeToPhenotypeProcess
    proc = GenotypeToPhenotypeProcess(config=gt2p_config, core=gt2p_core)
    state = proc.initial_state()
    assert state["phase"] == "init"
    assert state["epoch"] == 0
    assert state["loss"] == 0.0
    assert state["auc_roc"] == 0.0
    assert state["f1_score"] == 0.0


def test_gt2p_full_pipeline(gt2p_core, gt2p_config):
    from pbg_pennylane_data_reuploading import GenotypeToPhenotypeProcess
    proc = GenotypeToPhenotypeProcess(config=gt2p_config, core=gt2p_core)
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
    assert 0.0 <= state["auc_roc"] <= 1.0
    assert 0.0 <= state["f1_score"] <= 1.0
    assert 0.0 <= state["precision_viable"] <= 1.0
    assert 0.0 <= state["recall_viable"] <= 1.0
    assert 0.0 <= state["viable_ratio"] <= 1.0


def test_gt2p_synthetic_data_via_ports(gt2p_core, gt2p_config):
    from pbg_pennylane_data_reuploading import GenotypeToPhenotypeProcess
    rng = np.random.RandomState(42)
    nfeat = gt2p_config["num_omic_modalities"] * gt2p_config["features_per_modality"]
    n_train = gt2p_config["num_train"]
    n_test = gt2p_config["num_test"]
    train_x = rng.randn(n_train, nfeat).tolist()
    train_y = [int(rng.rand() > 0.5) for _ in range(n_train)]
    test_x = rng.randn(n_test, nfeat).tolist()
    test_y = [int(rng.rand() > 0.5) for _ in range(n_test)]

    doc = {
        "classifier": {
            "_type": "process",
            "address": "local:GenotypeToPhenotypeProcess",
            "config": gt2p_config,
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
                "auc_roc": ["stores", "auc_roc"],
                "f1_score": ["stores", "f1_score"],
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
            "auc_roc": 0.0,
            "f1_score": 0.0,
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
                    "auc_roc": "float",
                    "f1_score": "float",
                }
            },
            "inputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "auc_roc": ["stores", "auc_roc"],
                "f1_score": ["stores", "f1_score"],
                "time": ["global_time"],
            },
        },
    }

    sim = Composite({"state": doc}, core=gt2p_core)
    sim.run(10.0)
    assert sim.state["stores"]["phase"] == "done"


# ---------------------------------------------------------------------------
# Tests for ZarrMultiOmicLoader
# ---------------------------------------------------------------------------

def _build_synthetic_zarr_store(tmp_path):
    """Build a minimal v2ecoli-style zarr store for testing.

    Structure (mirrors the XArrayEmitter output):
        experiment_id=test/variant=0/lineage_seed=0/
            emitstep_gen=<N>/   (array)
            time_gen=<N>/       (array)
            <observable>/generation=<N>/   (array)
    """
    import xarray as xr
    import numpy as np

    store_dir = tmp_path / "test_store.zarr"
    n_steps_per_gen = {1: 5, 2: 7}
    n_genes = 4

    tree = xr.DataTree()

    for gen, n_steps in n_steps_per_gen.items():
        base = f"experiment_id=test/variant=0/lineage_seed=0"
        emit_dim = f"emitstep_gen={gen}"

        tree[f"{base}/emitstep_gen={gen}"] = xr.DataArray(
            np.arange(n_steps, dtype=np.uint32),
            dims=[emit_dim],
            name=f"emitstep_gen={gen}",
        )
        tree[f"{base}/time_gen={gen}"] = xr.DataArray(
            np.linspace(0, gen * 4.0, n_steps, dtype=np.float32),
            dims=[emit_dim],
            name=f"time_gen={gen}",
        )

        # scalar observable
        tree[f"{base}/listeners/mass/dry_mass/generation={gen}"] = xr.DataArray(
            np.random.randn(n_steps).astype(np.float32),
            dims=[emit_dim],
            name=f"generation={gen}",
        )
        tree[f"{base}/listeners/mass/cell_mass/generation={gen}"] = xr.DataArray(
            np.random.randn(n_steps).astype(np.float32),
            dims=[emit_dim],
            name=f"generation={gen}",
        )

        # vector observable (mRNA counts — one per gene)
        tree[f"{base}/listeners/mRna_counts/generation={gen}"] = xr.DataArray(
            np.random.randn(n_steps, n_genes).astype(np.float32),
            dims=[emit_dim, "gene_id"],
            name=f"generation={gen}",
        )

    tree.to_zarr(store_dir)
    return store_dir


def test_loader_instantiation(tmp_path):
    from pbg_pennylane_data_reuploading.data_loading import ZarrMultiOmicLoader
    store = _build_synthetic_zarr_store(tmp_path)
    loader = ZarrMultiOmicLoader(store_path=store)
    assert loader is not None
    assert loader.store_path == store


def test_loader_list_generations(tmp_path):
    from pbg_pennylane_data_reuploading.data_loading import ZarrMultiOmicLoader
    store = _build_synthetic_zarr_store(tmp_path)
    loader = ZarrMultiOmicLoader(store_path=store)
    loader._open_store()
    gens = loader._list_generations()
    assert gens == [1, 2], f"expected [1, 2], got {gens}"


def test_loader_read_variable(tmp_path):
    from pbg_pennylane_data_reuploading.data_loading import ZarrMultiOmicLoader
    store = _build_synthetic_zarr_store(tmp_path)
    loader = ZarrMultiOmicLoader(store_path=store)
    loader._open_store()

    arr = loader._read_variable("listeners/mass/dry_mass", gen=1)
    assert arr is not None, "dry_mass gen=1 returned None"
    assert arr.ndim == 1, f"expected 1-D, got {arr.ndim}"
    assert arr.shape[0] == 5, f"expected 5 steps, got {arr.shape[0]}"

    arr2 = loader._read_variable("listeners/mass/dry_mass", gen=2)
    assert arr2 is not None, "dry_mass gen=2 returned None"
    assert arr2.shape[0] == 7, f"expected 7 steps, got {arr2.shape[0]}"

    # vector observable
    vec = loader._read_variable("listeners/mRna_counts", gen=1)
    assert vec is not None
    assert vec.ndim == 2
    assert vec.shape[1] == 4, f"expected 4 genes, got {vec.shape[1]}"


def test_loader_extract_observable(tmp_path):
    from pbg_pennylane_data_reuploading.data_loading import ZarrMultiOmicLoader
    store = _build_synthetic_zarr_store(tmp_path)
    loader = ZarrMultiOmicLoader(store_path=store)
    loader._open_store()

    obs = loader._extract_observable("listeners/mass/dry_mass", agg="mean")
    assert obs is not None
    assert obs.shape == (2, 1), f"expected (2, 1), got {obs.shape}"

    obs_slope = loader._extract_observable("listeners/mass/dry_mass", agg="slope")
    assert obs_slope is not None
    assert obs_slope.shape == (2, 1), f"expected (2, 1), got {obs_slope.shape}"

    obs_end = loader._extract_observable("listeners/mass/dry_mass", agg="end")
    assert obs_end is not None
    assert obs_end.shape == (2, 1), f"expected (2, 1), got {obs_end.shape}"


def test_loader_extract_vector_observable(tmp_path):
    from pbg_pennylane_data_reuploading.data_loading import ZarrMultiOmicLoader
    store = _build_synthetic_zarr_store(tmp_path)
    loader = ZarrMultiOmicLoader(store_path=store)
    loader._open_store()

    obs = loader._extract_observable("listeners/mRna_counts", agg="mean")
    assert obs is not None
    assert obs.ndim == 2, f"expected 2-D (n_gen, n_genes), got {obs.ndim}"
    assert obs.shape[1] == 4, f"expected 4 genes, got {obs.shape[1]}"


def test_loader_load_end_to_end(tmp_path):
    from pbg_pennylane_data_reuploading.data_loading import ZarrMultiOmicLoader

    n_genes = 4
    store_dir = tmp_path / "test_store.zarr"
    _build_synthetic_zarr_store(tmp_path)

    from pbg_pennylane_data_reuploading.data_loading import ModalitySpec
    modality_map = [
        ModalitySpec(name="genomics", store_paths=["listeners/mass/dry_mass", "listeners/mass/cell_mass"], aggregation="mean"),
        ModalitySpec(name="transcriptomics", store_paths=["listeners/mRna_counts"], aggregation="mean"),
    ]

    loader = ZarrMultiOmicLoader(
        store_path=store_dir,
        modality_map=modality_map,
    )
    data = loader.load(train_ratio=0.5, shuffle=False)

    assert "train_inputs" in data
    assert "train_labels" in data
    assert "test_inputs" in data
    assert "test_labels" in data
    assert data["_generations"] == [1, 2]
    assert data["_total_samples"] == 2

    # With 2 samples and 0.5 ratio: 1 train, 1 test
    assert len(data["train_inputs"]) == 1
    assert len(data["test_inputs"]) == 1

    # Features: genomics (mean of dry_mass + mean of cell_mass = 2) + transcriptomics (4 genes)
    # = 6 + padding to F (6) = 6
    # Actually, genomics has 2 paths → 2 features, transcriptomics has 1 path → 4 features
    # feature_counts = [2, 4], F = 4
    # genomics padded: 2 → 4, transcriptomics: 4
    # total features: 4 + 4 = 8
    train_x = data["train_inputs"][0]
    assert len(train_x) == 8, f"expected 8 features, got {len(train_x)}"


def test_loader_nonexistent_path_returns_placeholder(tmp_path):
    from pbg_pennylane_data_reuploading.data_loading import ZarrMultiOmicLoader, ModalitySpec
    store = _build_synthetic_zarr_store(tmp_path)
    modality_map = [
        ModalitySpec(name="nonexistent", store_paths=["listeners/nonexistent/obs"], aggregation="mean"),
    ]
    loader = ZarrMultiOmicLoader(store_path=store, modality_map=modality_map)
    data = loader.load(shuffle=False)
    assert data["_total_samples"] == 2
    # Placeholder should have correct shape
    assert len(data["train_inputs"][0]) == 3  # default placeholder size


def test_loader_generate_labels(tmp_path):
    from pbg_pennylane_data_reuploading.data_loading import (
        ZarrMultiOmicLoader,
        PhenotypeLabelRule,
    )
    store = _build_synthetic_zarr_store(tmp_path)

    label_rule = PhenotypeLabelRule(
        source_path="listeners/mass/dry_mass",
        operator=">",
        threshold=0.0,
        use_derived=True,
    )
    loader = ZarrMultiOmicLoader(store_path=store, label_rule=label_rule)
    loader._open_store()
    labels = loader._generate_labels()
    assert labels is not None
    assert labels.shape == (2,), f"expected (2,) labels, got {labels.shape}"
    assert set(labels.tolist()).issubset({0, 1})


def test_loader_missing_store_raises(tmp_path):
    from pbg_pennylane_data_reuploading.data_loading import ZarrMultiOmicLoader
    loader = ZarrMultiOmicLoader(store_path=tmp_path / "does_not_exist.zarr")
    import pytest
    with pytest.raises(Exception):
        loader.load()
