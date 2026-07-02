from process_bigraph import allocate_core

from .processes import PennyLaneDataReuploadingProcess


def register_processes(core):
    core.register_link("PennyLaneDataReuploadingProcess", PennyLaneDataReuploadingProcess)
    return core


def build_core():
    core = allocate_core()
    core = register_processes(core)
    return core
