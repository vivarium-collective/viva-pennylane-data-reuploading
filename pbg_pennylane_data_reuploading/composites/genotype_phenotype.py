from pbg_superpowers.composite_generator import composite_generator


def _build_document(config):
    return {
        "genotype_classifier": {
            "_type": "process",
            "address": "local:GenotypeToPhenotypeProcess",
            "config": config,
            "outputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "best_test_accuracy": ["stores", "best_test_accuracy"],
                "viable_ratio": ["stores", "viable_ratio"],
                "precision_viable": ["stores", "precision_viable"],
                "recall_viable": ["stores", "recall_viable"],
            },
        },
        "stores": {
            "phase": "init",
            "epoch": 0,
            "loss": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
            "viable_ratio": 0.0,
            "precision_viable": 0.0,
            "recall_viable": 0.0,
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
                    "viable_ratio": "float",
                    "precision_viable": "float",
                    "recall_viable": "float",
                }
            },
            "inputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_accuracy": ["stores", "train_accuracy"],
                "test_accuracy": ["stores", "test_accuracy"],
                "best_test_accuracy": ["stores", "best_test_accuracy"],
                "viable_ratio": ["stores", "viable_ratio"],
                "precision_viable": ["stores", "precision_viable"],
                "recall_viable": ["stores", "recall_viable"],
                "time": ["global_time"],
            },
        },
    }


@composite_generator(
    name="genotype_to_phenotype_mapping",
    description=(
        "Classify multi-omic expression profiles (transcriptomics, proteomics, "
        "metabolomics, epigenomics) into viable vs non-viable cell replication using "
        "quantum data re-uploading. Each omic modality occupies one re-uploading layer."
    ),
    parameters={
        "num_omic_modalities": {
            "type": "integer", "default": 4,
            "description": "Number of omic input modalities"},
        "features_per_modality": {
            "type": "integer", "default": 8,
            "description": "Features per modality"},
        "learning_rate": {
            "type": "float", "default": 0.4,
            "description": "Adam learning rate"},
        "epochs": {
            "type": "integer", "default": 8,
            "description": "Training epochs"},
        "batch_size": {
            "type": "integer", "default": 16,
            "description": "Mini-batch size"},
    },
)
def genotype_to_phenotype_mapping(core=None, *, num_omic_modalities=4,
                                   features_per_modality=8, learning_rate=0.4,
                                   epochs=8, batch_size=16):
    return _build_document({
        "num_omic_modalities": num_omic_modalities,
        "features_per_modality": features_per_modality,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    })
