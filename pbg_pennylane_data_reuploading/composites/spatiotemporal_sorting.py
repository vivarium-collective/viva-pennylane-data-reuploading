from pbg_superpowers.composite_generator import composite_generator


def _build_document(config):
    return {
        "spatiotemporal_classifier": {
            "_type": "process",
            "address": "local:SpatiotemporalSortingProcess",
            "config": config,
            "outputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "best_test_accuracy": ["stores", "best_test_accuracy"],
                "healthy_mode_ratio": ["stores", "healthy_mode_ratio"],
                "confusion_matrix": ["stores", "confusion_matrix"],
            },
        },
        "stores": {
            "phase": "init",
            "epoch": 0,
            "loss": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
            "healthy_mode_ratio": 0.0,
            "confusion_matrix": "",
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
                    "best_test_accuracy": "float",
                    "healthy_mode_ratio": "float",
                    "confusion_matrix": "string",
                }
            },
            "inputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "best_test_accuracy": ["stores", "best_test_accuracy"],
                "healthy_mode_ratio": ["stores", "healthy_mode_ratio"],
                "confusion_matrix": ["stores", "confusion_matrix"],
                "time": ["global_time"],
            },
        },
    }


@composite_generator(
    name="spatiotemporal_state_sorting",
    description=(
        "Categorise 4D structural cellular imaging trajectories (time × 3D space) "
        "into healthy or diseased metabolic modes. One re-uploading layer per "
        "time frame enables the circuit to track spatial motion through time."
    ),
    parameters={
        "num_timestep_frames": {
            "type": "integer", "default": 5,
            "description": "Number of time frames per trajectory"},
        "spatial_dims": {
            "type": "integer", "default": 3,
            "description": "Spatial dimensions (X, Y, Z)"},
        "learning_rate": {
            "type": "float", "default": 0.3,
            "description": "Adam learning rate"},
        "epochs": {
            "type": "integer", "default": 8,
            "description": "Training epochs"},
        "batch_size": {
            "type": "integer", "default": 16,
            "description": "Mini-batch size"},
    },
)
def spatiotemporal_state_sorting(core=None, *, num_timestep_frames=5,
                                  spatial_dims=3, learning_rate=0.3,
                                  epochs=8, batch_size=16):
    return _build_document({
        "num_timestep_frames": num_timestep_frames,
        "spatial_dims": spatial_dims,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    })
