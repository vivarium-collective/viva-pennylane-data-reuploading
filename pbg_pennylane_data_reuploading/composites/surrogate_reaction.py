from pbg_superpowers.composite_generator import composite_generator


def _build_document(config):
    return {
        "reaction_surrogate": {
            "_type": "process",
            "address": "local:SurrogateReactionProcess",
            "config": config,
            "outputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_mse": ["stores", "train_mse"],
                "test_mse": ["stores", "test_mse"],
                "best_mse": ["stores", "best_mse"],
                "r2_score": ["stores", "r2_score"],
            },
        },
        "stores": {
            "phase": "init",
            "epoch": 0,
            "loss": 0.0,
            "train_mse": 0.0,
            "test_mse": 0.0,
            "best_mse": 1e9,
            "r2_score": 0.0,
        },
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {
                "emit": {
                    "phase": "string",
                    "epoch": "integer",
                    "loss": "float",
                    "train_mse": "float",
                    "test_mse": "float",
                    "best_mse": "float",
                    "r2_score": "float",
                }
            },
            "inputs": {
                "phase": ["stores", "phase"],
                "epoch": ["stores", "epoch"],
                "loss": ["stores", "loss"],
                "train_mse": ["stores", "train_mse"],
                "test_mse": ["stores", "test_mse"],
                "best_mse": ["stores", "best_mse"],
                "r2_score": ["stores", "r2_score"],
                "time": ["global_time"],
            },
        },
    }


@composite_generator(
    name="surrogate_reaction_modeling",
    description=(
        "Replace slow reaction-diffusion master-equation (RDME / SSA) solvers "
        "with a lightweight quantum function approximator. Encodes current species "
        "concentrations and reaction rates, predicts next-step concentrations."
    ),
    parameters={
        "num_species": {
            "type": "integer", "default": 5,
            "description": "Number of chemical species"},
        "num_reactions": {
            "type": "integer", "default": 3,
            "description": "Number of reactions"},
        "learning_rate": {
            "type": "float", "default": 0.3,
            "description": "Adam learning rate"},
        "epochs": {
            "type": "integer", "default": 8,
            "description": "Training epochs"},
        "batch_size": {
            "type": "integer", "default": 32,
            "description": "Mini-batch size"},
    },
)
def surrogate_reaction_modeling(core=None, *, num_species=5, num_reactions=3,
                                 learning_rate=0.3, epochs=8, batch_size=32):
    return _build_document({
        "num_species": num_species,
        "num_reactions": num_reactions,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
    })
