import pytest
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter, gather_emitter_results


def _core():
    c = allocate_core()
    c.register_link("ram-emitter", RAMEmitter)
    from pbg_pennylane_data_reuploading import PennyLaneDataReuploadingProcess
    c.register_link("PennyLaneDataReuploadingProcess", PennyLaneDataReuploadingProcess)
    return c


@pytest.mark.timeout(120)
def test_baseline_generator():
    from pbg_pennylane_data_reuploading.composites.classifier import (
        data_reuploading_baseline,
    )
    doc = data_reuploading_baseline(
        core=None,
        num_layers=2,
        epochs=2,
        batch_size=16,
        num_train=20,
        num_test=20,
    )
    core = _core()
    sim = Composite({"state": doc}, core=core)
    sim.run(10.0)
    from process_bigraph.emitter import gather_emitter_results
    results = gather_emitter_results(sim)
    assert results, "No emitter results"


@pytest.mark.timeout(120)
def test_lightweight_generator():
    from pbg_pennylane_data_reuploading.composites.classifier import (
        data_reuploading_lightweight,
    )
    doc = data_reuploading_lightweight(
        core=None,
        num_layers=2,
        epochs=2,
        batch_size=10,
        num_train=10,
        num_test=10,
    )
    core = _core()
    sim = Composite({"state": doc}, core=core)
    sim.run(5.0)
    results = gather_emitter_results(sim)
    assert results, "No emitter results"
