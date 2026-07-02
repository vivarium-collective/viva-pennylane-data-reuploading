from pbg_superpowers.composite_generator import composite_generator


def _build_document(config):
    return {
        "genome_evaluator": {
            "_type": "process",
            "address": "local:MinimalGenomeProcess",
            "config": config,
            "outputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "best_test_accuracy": ["stores", "best_test_accuracy"],
                "failure_ratio": ["stores", "failure_ratio"],
                "critical_ko_count": ["stores", "critical_ko_count"],
            },
        },
        "stores": {
            "phase": "init",
            "epoch": 0,
            "loss": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
            "failure_ratio": 0.0,
            "critical_ko_count": 0,
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
                    "failure_ratio": "float",
                    "critical_ko_count": "integer",
                }
            },
            "inputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "best_test_accuracy": ["stores", "best_test_accuracy"],
                "failure_ratio": ["stores", "failure_ratio"],
                "critical_ko_count": ["stores", "critical_ko_count"],
                "time": ["global_time"],
            },
        },
    }


@composite_generator(
    name="minimal_genome_evaluation",
    description=(
        "Screen simulated gene knockouts in silico to predict whether a reduced "
        "cell subsystem fails. The quantum circuit learns the nonlinear mapping "
        "from knockout mask to subsystem-viability status."
    ),
    parameters={
        "num_genes": {
            "type": "integer", "default": 20,
            "description": "Number of genes in the model"},
        "num_subsystems": {
            "type": "integer", "default": 4,
            "description": "Number of cellular subsystems"},
        "learning_rate": {
            "type": "float", "default": 0.5,
            "description": "Adam learning rate"},
        "epochs": {
            "type": "integer", "default": 8,
            "description": "Training epochs"},
        "batch_size": {
            "type": "integer", "default": 32,
            "description": "Mini-batch size"},
    },
)
def minimal_genome_evaluation(core=None, *, num_genes=20, num_subsystems=4,
                               learning_rate=0.5, epochs=8, batch_size=32):
    return _build_document({
        "num_genes": num_genes,
        "num_subsystems": num_subsystems,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    })
