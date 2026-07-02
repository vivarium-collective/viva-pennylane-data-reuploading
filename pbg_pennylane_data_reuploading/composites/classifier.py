from pbg_superpowers.composite_generator import composite_generator


def _build_document(config):
    return {
        "classifier": {
            "_type": "process",
            "address": "local:PennyLaneDataReuploadingProcess",
            "config": config,
            "outputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "best_test_accuracy": ["stores", "best_test_accuracy"],
            },
        },
        "stores": {
            "phase": "init",
            "epoch": 0,
            "loss": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
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
                }
            },
            "inputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "best_test_accuracy": ["stores", "best_test_accuracy"],
                "time": ["global_time"],
            },
        },
    }


@composite_generator(
    name="pennylane_data_reuploading_baseline",
    description="Standard data-reuploading quantum classifier: 3 layers, 10 training epochs, circle dataset.",
    parameters={
        "num_layers": {"type": "integer", "default": 3,
                        "description": "Number of re-uploading layers"},
        "learning_rate": {"type": "float", "default": 0.6,
                          "description": "Adam optimizer learning rate"},
        "epochs": {"type": "integer", "default": 10,
                   "description": "Number of training epochs"},
        "batch_size": {"type": "integer", "default": 32,
                       "description": "Mini-batch size"},
        "num_train": {"type": "integer", "default": 200,
                      "description": "Number of training samples"},
        "num_test": {"type": "integer", "default": 2000,
                     "description": "Number of test samples"},
    },
)
def data_reuploading_baseline(core=None, *, num_layers=3, learning_rate=0.6,
                               epochs=10, batch_size=32, num_train=200,
                               num_test=2000):
    return _build_document({
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_train": num_train,
        "num_test": num_test,
    })


@composite_generator(
    name="pennylane_data_reuploading_deep",
    description="Deep data-reuploading classifier: 6 layers, 15 epochs for higher accuracy.",
    parameters={
        "num_layers": {"type": "integer", "default": 6,
                        "description": "Number of re-uploading layers"},
        "learning_rate": {"type": "float", "default": 0.4,
                          "description": "Adam optimizer learning rate"},
        "epochs": {"type": "integer", "default": 15,
                   "description": "Number of training epochs"},
        "batch_size": {"type": "integer", "default": 32,
                       "description": "Mini-batch size"},
        "num_train": {"type": "integer", "default": 500,
                      "description": "Number of training samples"},
        "num_test": {"type": "integer", "default": 2000,
                     "description": "Number of test samples"},
    },
)
def data_reuploading_deep(core=None, *, num_layers=6, learning_rate=0.4,
                           epochs=15, batch_size=32, num_train=500,
                           num_test=2000):
    return _build_document({
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_train": num_train,
        "num_test": num_test,
    })


@composite_generator(
    name="pennylane_data_reuploading_lightweight",
    description="Fast data-reuploading classifier: 2 layers, 3 epochs, small dataset for quick smoke testing.",
    parameters={
        "num_layers": {"type": "integer", "default": 2,
                        "description": "Number of re-uploading layers"},
        "learning_rate": {"type": "float", "default": 0.6,
                          "description": "Adam optimizer learning rate"},
        "epochs": {"type": "integer", "default": 3,
                   "description": "Number of training epochs"},
        "batch_size": {"type": "integer", "default": 16,
                       "description": "Mini-batch size"},
        "num_train": {"type": "integer", "default": 50,
                      "description": "Number of training samples"},
        "num_test": {"type": "integer", "default": 100,
                     "description": "Number of test samples"},
    },
)
def data_reuploading_lightweight(core=None, *, num_layers=2, learning_rate=0.6,
                                  epochs=3, batch_size=16, num_train=50,
                                  num_test=100):
    return _build_document({
        "num_layers": num_layers,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_train": num_train,
        "num_test": num_test,
    })
