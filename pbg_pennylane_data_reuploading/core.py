from process_bigraph import allocate_core

from .processes import (
    PennyLaneDataReuploadingProcess,
    GenotypeToPhenotypeProcess,
    SurrogateReactionProcess,
    MinimalGenomeProcess,
    SpatiotemporalSortingProcess,
)


def register_processes(core):
    core.register_link("PennyLaneDataReuploadingProcess", PennyLaneDataReuploadingProcess)
    core.register_link("GenotypeToPhenotypeProcess", GenotypeToPhenotypeProcess)
    core.register_link("SurrogateReactionProcess", SurrogateReactionProcess)
    core.register_link("MinimalGenomeProcess", MinimalGenomeProcess)
    core.register_link("SpatiotemporalSortingProcess", SpatiotemporalSortingProcess)
    return core


def build_core():
    core = allocate_core()
    core = register_processes(core)
    return core
