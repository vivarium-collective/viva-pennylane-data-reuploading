import copy

from process_bigraph import Process


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _density_matrix(state):
    import numpy as np
    state = np.asarray(state)
    return state * np.conj(state).T


def _accuracy_score(y_true, y_pred):
    import numpy as np
    return float(np.mean(np.array(y_true) == np.array(y_pred)))


def _pad_inputs(x_inputs):
    import numpy as np
    x = np.array(x_inputs)
    return np.hstack((x, np.zeros((x.shape[0], 1))))


# ---------------------------------------------------------------------------
# 1.  Default demo — circle-data binary classifier
# ---------------------------------------------------------------------------

class PennyLaneDataReuploadingProcess(Process):
    creational = True

    config_schema = {
        "num_layers": {"_type": "integer", "_default": 3, "_minimum": 1},
        "learning_rate": {"_type": "float", "_default": 0.6},
        "epochs": {"_type": "integer", "_default": 10, "_minimum": 1},
        "batch_size": {"_type": "integer", "_default": 32, "_minimum": 1},
        "num_train": {"_type": "integer", "_default": 200, "_minimum": 10},
        "num_test": {"_type": "integer", "_default": 2000, "_minimum": 10},
        "seed": {"_type": "integer", "_default": 42},
        "device": {"_type": "string", "_default": "default.qubit"},
        "radius": {"_type": "float", "_default": 0.7978845608028654},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._model = None
        self._params = None
        self._data = {}
        self._epoch = 0
        self._phase = "init"
        self._best_test_acc = 0.0
        self._history = []

    def inputs(self):
        return {
            "train_inputs": {"_type": "list[list[float]]", "_default": None},
            "train_labels": {"_type": "list[integer]", "_default": None},
            "test_inputs": {"_type": "list[list[float]]", "_default": None},
            "test_labels": {"_type": "list[integer]", "_default": None},
        }

    def outputs(self):
        return {
            "phase": "string",
            "epoch": "integer",
            "loss": "float",
            "train_accuracy": "float",
            "test_accuracy": "float",
            "best_test_accuracy": "float",
        }

    def initial_state(self):
        return {
            "phase": "init",
            "epoch": 0,
            "loss": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
        }

    def _generate_circle_data(self):
        import numpy as np

        c = self.config
        np.random.seed(c["seed"])
        radius = c["radius"]
        center = [0.0, 0.0]

        def _sample(n):
            xs = []
            ys = []
            for _ in range(n):
                x = 2 * (np.random.rand(2)) - 1
                label = 1 if np.linalg.norm(x - center) < radius else 0
                xs.append(x.tolist())
                ys.append(label)
            return xs, ys

        train_x, train_y = _sample(c["num_train"])
        test_x, test_y = _sample(c["num_test"])

        return {
            "train_inputs": train_x,
            "train_labels": train_y,
            "test_inputs": test_x,
            "test_labels": test_y,
        }

    def _build_model(self):
        import pennylane as qml
        from pennylane import numpy as np

        c = self.config
        num_layers = c["num_layers"]

        np.random.seed(c["seed"])

        dev = qml.device(c["device"], wires=1)

        @qml.qnode(dev)
        def qcircuit(params, x, y_dm):
            for layer_params in params:
                qml.Rot(*x, wires=0)
                qml.Rot(*layer_params, wires=0)
            return qml.expval(qml.Hermitian(y_dm, wires=[0]))

        self._qcircuit = qcircuit
        self._params = np.random.uniform(size=(num_layers, 3), requires_grad=True)

    def _cost(self, params, x_batch, y_batch, state_labels_orig):
        import numpy as np
        loss_val = 0.0
        dm_labels = [_density_matrix(s) for s in state_labels_orig]
        for i in range(len(x_batch)):
            f = self._qcircuit(params, x_batch[i], dm_labels[y_batch[i]])
            loss_val += (1 - f) ** 2
        return loss_val / len(x_batch)

    def _test(self, params, x, y, state_labels_orig):
        import numpy as np
        dm_labels = [_density_matrix(s) for s in state_labels_orig]
        predicted = []
        for i in range(len(x)):
            fidelities = [self._qcircuit(params, x[i], dm) for dm in dm_labels]
            predicted.append(int(np.argmax(fidelities)))
        return np.array(predicted)

    def _iterate_minibatches(self, inputs, targets, batch_size):
        for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
            idxs = slice(start_idx, start_idx + batch_size)
            yield inputs[idxs], targets[idxs]

    def update(self, state, interval):
        import numpy as np
        from pennylane import numpy as pnp
        from pennylane.optimize import AdamOptimizer

        c = self.config

        if self._phase == "init":
            data_from_state = {}
            for key in ("train_inputs", "train_labels", "test_inputs", "test_labels"):
                val = state.get(key)
                if val is not None and len(val) > 0:
                    data_from_state[key] = val
            if all(k in data_from_state for k in ("train_inputs", "train_labels", "test_inputs", "test_labels")):
                self._data = data_from_state
            else:
                self._data = self._generate_circle_data()

            label_0 = [[1], [0]]
            label_1 = [[0], [1]]
            self._state_labels = np.array([label_0, label_1])

            self._build_model()
            self._history = []
            self._epoch = 0

            train_x = np.array(self._data["train_inputs"])
            self._train_inputs_padded = _pad_inputs(train_x)
            test_x = np.array(self._data["test_inputs"])
            self._test_inputs_padded = _pad_inputs(test_x)
            self._train_labels_arr = np.array(self._data["train_labels"])
            self._test_labels_arr = np.array(self._data["test_labels"])

            self._optimizer = AdamOptimizer(c["learning_rate"], beta1=0.9, beta2=0.999)
            self._phase = "training"

            return {
                "phase": "training",
                "epoch": 0,
                "loss": 0.0,
                "train_accuracy": 0.0,
                "test_accuracy": 0.0,
                "best_test_accuracy": 0.0,
            }

        if self._phase == "training":
            if self._epoch < c["epochs"]:
                for Xbatch, ybatch in self._iterate_minibatches(
                    self._train_inputs_padded,
                    self._train_labels_arr,
                    c["batch_size"],
                ):
                    Xbatch_p = pnp.array(Xbatch, requires_grad=False)
                    ybatch_p = pnp.array(ybatch, requires_grad=False)
                    self._params, _, _, _ = self._optimizer.step(
                        self._cost,
                        self._params,
                        Xbatch_p,
                        ybatch_p,
                        self._state_labels,
                    )

                pred_train = self._test(
                    self._params,
                    self._train_inputs_padded,
                    self._train_labels_arr,
                    self._state_labels,
                )
                train_acc = _accuracy_score(self._train_labels_arr, pred_train)
                loss_val = self._cost(
                    self._params,
                    self._train_inputs_padded,
                    self._train_labels_arr,
                    self._state_labels,
                )

                pred_test = self._test(
                    self._params,
                    self._test_inputs_padded,
                    self._test_labels_arr,
                    self._state_labels,
                )
                test_acc = _accuracy_score(self._test_labels_arr, pred_test)
                self._best_test_acc = max(self._best_test_acc, test_acc)

                self._epoch += 1

                entry = {
                    "epoch": self._epoch,
                    "loss": float(loss_val),
                    "train_accuracy": float(train_acc),
                    "test_accuracy": float(test_acc),
                }
                self._history.append(entry)

                return {
                    "phase": "training",
                    "epoch": self._epoch,
                    "loss": float(loss_val),
                    "train_accuracy": float(train_acc),
                    "test_accuracy": float(test_acc),
                    "best_test_accuracy": float(self._best_test_acc),
                }
            else:
                pred_test = self._test(
                    self._params,
                    self._test_inputs_padded,
                    self._test_labels_arr,
                    self._state_labels,
                )
                final_test_acc = _accuracy_score(self._test_labels_arr, pred_test)
                self._phase = "done"
                return {
                    "phase": "done",
                    "epoch": c["epochs"],
                    "loss": 0.0,
                    "train_accuracy": float(self._history[-1]["train_accuracy"]) if self._history else 0.0,
                    "test_accuracy": float(final_test_acc),
                    "best_test_accuracy": float(self._best_test_acc),
                }

        if self._phase == "done":
            return {
                "phase": "done",
                "epoch": c["epochs"],
                "loss": 0.0,
                "train_accuracy": float(self._history[-1]["train_accuracy"]) if self._history else 0.0,
                "test_accuracy": float(self._history[-1]["test_accuracy"]) if self._history else 0.0,
                "best_test_accuracy": float(self._best_test_acc),
            }

        return {
            "phase": self._phase,
            "epoch": 0,
            "loss": 0.0,
            "train_accuracy": 0.0,
            "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
        }


# ---------------------------------------------------------------------------
# 2.  Genotype-to-Phenotype Mapping
# ---------------------------------------------------------------------------

class GenotypeToPhenotypeProcess(Process):
    """Multi-qubit, cross-modality entangled quantum classifier.

    One qubit per omic modality.  Each modality's full feature set is
    encoded via repeated Rot gates (3 features per gate).  A CNOT chain
    entangles all modality qubits after each trainable layer so the
    circuit can learn cross-modality interactions.
    """

    creational = True

    config_schema = {
        "num_omic_modalities": {"_type": "integer", "_default": 4, "_minimum": 2},
        "features_per_modality": {"_type": "integer", "_default": 6, "_minimum": 3},
        "learning_rate": {"_type": "float", "_default": 0.4},
        "epochs": {"_type": "integer", "_default": 8, "_minimum": 1},
        "batch_size": {"_type": "integer", "_default": 16, "_minimum": 1},
        "num_train": {"_type": "integer", "_default": 100, "_minimum": 10},
        "num_test": {"_type": "integer", "_default": 200, "_minimum": 10},
        "seed": {"_type": "integer", "_default": 42},
        "device": {"_type": "string", "_default": "default.qubit"},
        "modality_weights": {"_type": "list[float]", "_default": None},
        "noise_level": {"_type": "float", "_default": 0.05},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._params = None
        self._qcircuit = None
        self._state_labels = None
        self._phase = "init"
        self._epoch = 0
        self._best_test_acc = 0.0
        self._best_auc = 0.0
        self._history = []
        self._data = {}

    def inputs(self):
        return {
            "train_inputs": {"_type": "list[list[float]]", "_default": None},
            "train_labels": {"_type": "list[integer]", "_default": None},
            "test_inputs": {"_type": "list[list[float]]", "_default": None},
            "test_labels": {"_type": "list[integer]", "_default": None},
        }

    def outputs(self):
        return {
            "phase": "string",
            "epoch": "integer",
            "loss": "float",
            "train_accuracy": "float",
            "test_accuracy": "float",
            "best_test_accuracy": "float",
            "auc_roc": "float",
            "f1_score": "float",
            "precision_viable": "float",
            "recall_viable": "float",
            "viable_ratio": "float",
        }

    def initial_state(self):
        return {
            "phase": "init", "epoch": 0, "loss": 0.0,
            "train_accuracy": 0.0, "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
            "auc_roc": 0.0, "f1_score": 0.0,
            "precision_viable": 0.0, "recall_viable": 0.0,
            "viable_ratio": 0.0,
        }

    # ------------------------------------------------------------------ data

    def _generate_synthetic_omics(self):
        import numpy as np
        c = self.config
        np.random.seed(c["seed"])

        M = c["num_omic_modalities"]
        F = c["features_per_modality"]
        nfeat = M * F
        weights = c.get("modality_weights")
        if weights is None:
            weights = [1.0 + 0.5 * i for i in range(M)]  # later modalities weigh more
        noise = c["noise_level"]

        def _sample(n):
            xs, ys = [], []
            for _ in range(n):
                x = np.random.randn(nfeat)
                modality_scores = []
                for i in range(M):
                    seg = x[i * F:(i + 1) * F]
                    score = np.mean(seg) * weights[i]
                    modality_scores.append(score)
                # weighted sum with nonlinear interaction term
                weighted_sum = sum(modality_scores)
                interaction = 0.0
                for i in range(M):
                    for j in range(i + 1, M):
                        interaction += modality_scores[i] * modality_scores[j] * 0.2
                raw = weighted_sum + interaction + noise * np.random.randn()
                viable = 1 if raw > 0 else 0
                xs.append(x.tolist())
                ys.append(viable)
            return xs, ys

        train_x, train_y = _sample(c["num_train"])
        test_x, test_y = _sample(c["num_test"])
        return {"train_inputs": train_x, "train_labels": train_y,
                "test_inputs": test_x, "test_labels": test_y}

    # ------------------------------------------------------------------ circuit

    def _build_model(self):
        import pennylane as qml
        from pennylane import numpy as np

        c = self.config
        M = c["num_omic_modalities"]
        F = c["features_per_modality"]
        rots_per_modality = max(1, (F + 2) // 3)

        np.random.seed(c["seed"])
        dev = qml.device(c["device"], wires=M)

        @qml.qnode(dev)
        def qcircuit(params, x, y_dm):
            for m in range(M):
                start = m * F
                # encode ALL features of this modality
                for r in range(rots_per_modality):
                    off = r * 3
                    vec = np.zeros(3)
                    for d in range(3):
                        idx = start + off + d
                        vec[d] = x[idx] if idx < start + F else 0.0
                    qml.Rot(*vec, wires=m)
                # trainable processing per modality
                qml.Rot(*params[m, 0], wires=m)

            # cross-modality entanglement chain
            for m in range(M - 1):
                qml.CNOT(wires=[m, m + 1])

            # post-entanglement trainable rotations
            for m in range(M):
                qml.Rot(*params[m, 1], wires=m)

            return qml.expval(qml.Hermitian(y_dm, wires=[0]))

        self._qcircuit = qcircuit
        self._params = np.random.uniform(size=(M, 2, 3), requires_grad=True)

    # ------------------------------------------------------------------ helpers

    def _test(self, params, x, state_labels_orig):
        import numpy as np
        dm_labels = [_density_matrix(s) for s in state_labels_orig]
        predicted = []
        for i in range(len(x)):
            fidelities = [self._qcircuit(params, x[i], dm) for dm in dm_labels]
            predicted.append(int(np.argmax(fidelities)))
        return np.array(predicted)

    def _cost(self, params, x_batch, y_batch, state_labels_orig):
        import numpy as np
        loss_val = 0.0
        dm_labels = [_density_matrix(s) for s in state_labels_orig]
        for i in range(len(x_batch)):
            f = self._qcircuit(params, x_batch[i], dm_labels[y_batch[i]])
            loss_val += (1 - f) ** 2
        return loss_val / len(x_batch)

    def _precision_recall(self, y_true, y_pred):
        import numpy as np
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        return prec, rec

    def _f1(self, y_true, y_pred):
        import numpy as np
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    def _compute_auc(self, y_true, scores, n_thresholds=50):
        import numpy as np
        y_true = np.array(y_true)
        scores = np.array(scores)
        tpr_list, fpr_list = [], []
        for thresh in np.linspace(scores.min(), scores.max(), n_thresholds):
            pred = (scores >= thresh).astype(int)
            tp = np.sum((y_true == 1) & (pred == 1))
            fp = np.sum((y_true == 0) & (pred == 1))
            fn = np.sum((y_true == 1) & (pred == 0))
            tn = np.sum((y_true == 0) & (pred == 0))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        idx = np.argsort(fpr_list)
        fpr_sorted = np.array(fpr_list)[idx]
        tpr_sorted = np.array(tpr_list)[idx]
        return float(np.trapezoid(tpr_sorted, fpr_sorted))

    # ------------------------------------------------------------------ update

    def update(self, state, interval):
        import numpy as np
        from pennylane import numpy as pnp
        from pennylane.optimize import AdamOptimizer

        c = self.config
        M = c["num_omic_modalities"]
        F = c["features_per_modality"]

        if self._phase == "init":
            data_from_state = {}
            for key in ("train_inputs", "train_labels", "test_inputs", "test_labels"):
                val = state.get(key)
                if val is not None and len(val) > 0:
                    data_from_state[key] = val
            if all(k in data_from_state for k in ("train_inputs", "train_labels", "test_inputs", "test_labels")):
                self._data = data_from_state
            else:
                self._data = self._generate_synthetic_omics()

            label_0 = [[1], [0]]
            label_1 = [[0], [1]]
            self._state_labels = np.array([label_0, label_1])

            self._build_model()
            self._history = []
            self._epoch = 0

            self._train_x = np.array(self._data["train_inputs"])
            self._test_x = np.array(self._data["test_inputs"])
            self._train_labels_arr = np.array(self._data["train_labels"])
            self._test_labels_arr = np.array(self._data["test_labels"])

            self._optimizer = AdamOptimizer(c["learning_rate"], beta1=0.9, beta2=0.999)
            self._phase = "training"

            return {
                "phase": "training", "epoch": 0, "loss": 0.0,
                "train_accuracy": 0.0, "test_accuracy": 0.0,
                "best_test_accuracy": 0.0, "auc_roc": 0.0, "f1_score": 0.0,
                "precision_viable": 0.0, "recall_viable": 0.0,
                "viable_ratio": float(np.mean(self._train_labels_arr)),
            }

        if self._phase == "training":
            if self._epoch < c["epochs"]:
                batch_size = min(c["batch_size"], len(self._train_x))
                sidx = slice(0, batch_size)
                Xb = pnp.array(self._train_x[sidx], requires_grad=False)
                yb = pnp.array(self._train_labels_arr[sidx], requires_grad=False)
                self._params, _, _, _ = self._optimizer.step(
                    self._cost, self._params, Xb, yb, self._state_labels)

                pred_train = self._test(self._params, self._train_x, self._state_labels)
                train_acc = _accuracy_score(self._train_labels_arr, pred_train)
                loss_val = self._cost(self._params, self._train_x, self._train_labels_arr, self._state_labels)

                pred_test = self._test(self._params, self._test_x, self._state_labels)
                test_acc = _accuracy_score(self._test_labels_arr, pred_test)
                self._best_test_acc = max(self._best_test_acc, test_acc)

                prec, rec = self._precision_recall(self._test_labels_arr, pred_test)
                f1 = self._f1(self._test_labels_arr, pred_test)

                # AUC: use fidelity to class-1 as continuous score
                dm1 = _density_matrix([[0], [1]])
                scores = np.array([self._qcircuit(self._params, self._test_x[i], dm1)
                                   for i in range(len(self._test_x))])
                auc = self._compute_auc(self._test_labels_arr, scores)
                self._best_auc = max(self._best_auc, auc)

                self._epoch += 1
                self._history.append({
                    "epoch": self._epoch, "loss": float(loss_val),
                    "train_accuracy": float(train_acc), "test_accuracy": float(test_acc),
                    "auc": auc, "f1": f1,
                })

                return {
                    "phase": "training", "epoch": self._epoch, "loss": float(loss_val),
                    "train_accuracy": float(train_acc), "test_accuracy": float(test_acc),
                    "best_test_accuracy": float(self._best_test_acc),
                    "auc_roc": auc, "f1_score": f1,
                    "precision_viable": prec, "recall_viable": rec,
                    "viable_ratio": float(np.mean(self._train_labels_arr)),
                }
            else:
                pred_test = self._test(self._params, self._test_x, self._state_labels)
                final_test_acc = _accuracy_score(self._test_labels_arr, pred_test)
                prec, rec = self._precision_recall(self._test_labels_arr, pred_test)
                f1 = self._f1(self._test_labels_arr, pred_test)
                dm1 = _density_matrix([[0], [1]])
                scores = np.array([self._qcircuit(self._params, self._test_x[i], dm1)
                                   for i in range(len(self._test_x))])
                auc = self._compute_auc(self._test_labels_arr, scores)
                self._phase = "done"
                return {
                    "phase": "done", "epoch": c["epochs"], "loss": 0.0,
                    "train_accuracy": float(self._history[-1]["train_accuracy"]) if self._history else 0.0,
                    "test_accuracy": float(final_test_acc),
                    "best_test_accuracy": float(self._best_test_acc),
                    "auc_roc": auc, "f1_score": f1,
                    "precision_viable": prec, "recall_viable": rec,
                    "viable_ratio": float(np.mean(self._train_labels_arr)),
                }

        if self._phase == "done":
            h = self._history[-1] if self._history else {}
            return {
                "phase": "done", "epoch": c["epochs"], "loss": 0.0,
                "train_accuracy": float(h.get("train_accuracy", 0.0)),
                "test_accuracy": float(h.get("test_accuracy", 0.0)),
                "best_test_accuracy": float(self._best_test_acc),
                "auc_roc": float(h.get("auc", 0.0)),
                "f1_score": float(h.get("f1", 0.0)),
                "precision_viable": 0.0, "recall_viable": 0.0,
                "viable_ratio": float(np.mean(self._train_labels_arr)),
            }

        return {
            "phase": self._phase, "epoch": 0, "loss": 0.0,
            "train_accuracy": 0.0, "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
            "auc_roc": 0.0, "f1_score": 0.0,
            "precision_viable": 0.0, "recall_viable": 0.0,
            "viable_ratio": 0.0,
        }


# ---------------------------------------------------------------------------
# 3.  Surrogate Reaction Modeling
# ---------------------------------------------------------------------------

class SurrogateReactionProcess(Process):
    """Approximate a reaction-diffusion master-equation step with a quantum circuit.

    The circuit encodes current species concentrations + reaction parameters
    and predicts the next-step concentrations, acting as a lightweight
    surrogate for expensive SSA / RDME solvers.
    """

    creational = True

    config_schema = {
        "num_species": {"_type": "integer", "_default": 5, "_minimum": 2},
        "num_reactions": {"_type": "integer", "_default": 3, "_minimum": 1},
        "learning_rate": {"_type": "float", "_default": 0.3},
        "epochs": {"_type": "integer", "_default": 8, "_minimum": 1},
        "batch_size": {"_type": "integer", "_default": 32, "_minimum": 1},
        "num_train": {"_type": "integer", "_default": 200, "_minimum": 10},
        "num_test": {"_type": "integer", "_default": 500, "_minimum": 10},
        "seed": {"_type": "integer", "_default": 42},
        "device": {"_type": "string", "_default": "default.qubit"},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._params = None
        self._qcircuit = None
        self._phase = "init"
        self._epoch = 0
        self._best_mse = float("inf")
        self._history = []
        self._data = {}
        self._state_labels = None

    def inputs(self):
        return {
            "concentration_inputs": {"_type": "list[list[float]]", "_default": None},
            "concentration_targets": {"_type": "list[list[float]]", "_default": None},
        }

    def outputs(self):
        return {
            "phase": "string",
            "epoch": "integer",
            "loss": "float",
            "train_mse": "float",
            "test_mse": "float",
            "best_mse": "float",
            "r2_score": "float",
        }

    def initial_state(self):
        return {
            "phase": "init", "epoch": 0, "loss": 0.0,
            "train_mse": 0.0, "test_mse": 0.0,
            "best_mse": 1e9, "r2_score": 0.0,
        }

    def _generate_synthetic_gillespie(self):
        import numpy as np
        c = self.config
        np.random.seed(c["seed"])
        ns = c["num_species"]
        nr = c["num_reactions"]

        def _sample(n):
            inputs, targets = [], []
            for _ in range(n):
                conc = np.random.exponential(1.0, size=ns)
                k = np.random.uniform(0.05, 2.0, size=nr)
                x = np.concatenate([conc, k])
                # synthetic surrogate: sigmoid-weighted sum + noise
                delta = np.tanh(conc @ np.random.randn(ns, ns)) * 0.5
                next_conc = np.clip(conc + delta, 0, None)
                inputs.append(x.tolist())
                targets.append(next_conc.tolist())
            return inputs, targets

        train_x, train_y = _sample(c["num_train"])
        test_x, test_y = _sample(c["num_test"])
        return {"concentration_inputs": train_x, "concentration_targets": train_y,
                "test_inputs": test_x, "test_targets": test_y}

    def _build_model(self):
        import pennylane as qml
        from pennylane import numpy as np
        c = self.config
        input_dim = c["num_species"] + c["num_reactions"]
        num_layers = c["num_species"]
        np.random.seed(c["seed"])
        dev = qml.device(c["device"], wires=1)

        @qml.qnode(dev)
        def qcircuit(params, x):
            for l in range(num_layers):
                idx = (l * 3) % input_dim
                v = x[idx:idx + 3]
                if len(v) < 3:
                    v = np.pad(v, (0, 3 - len(v)))
                qml.Rot(*v, wires=0)
                qml.Rot(*params[l], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        self._qcircuit = qcircuit
        self._params = np.random.uniform(size=(num_layers, 3), requires_grad=True)

    def _mse(self, y_true, y_pred):
        import numpy as np
        return float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

    def _r2(self, y_true, y_pred):
        import numpy as np
        ss_res = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
        ss_tot = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
        return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    def update(self, state, interval):
        import numpy as np
        from pennylane import numpy as pnp
        from pennylane.optimize import AdamOptimizer

        c = self.config
        input_dim = c["num_species"] + c["num_reactions"]
        num_layers = c["num_species"]

        if self._phase == "init":
            data_from_state = {}
            for key in ("concentration_inputs", "concentration_targets"):
                val = state.get(key)
                if val is not None and len(val) > 0:
                    data_from_state[key] = val
            if "concentration_inputs" in data_from_state:
                self._data = data_from_state
            else:
                self._data = self._generate_synthetic_gillespie()

            self._build_model()
            self._history = []
            self._epoch = 0

            self._train_x = np.array(self._data["concentration_inputs"])
            self._train_y = np.array(self._data["concentration_targets"])
            test_x = self._data.get("test_inputs", self._train_x[:c["num_test"]])
            self._test_x = np.array(test_x)
            test_y = self._data.get("test_targets", self._train_y[:c["num_test"]])
            self._test_y = np.array(test_y)

            self._optimizer = AdamOptimizer(c["learning_rate"], beta1=0.9, beta2=0.999)
            self._phase = "training"

            return {
                "phase": "training", "epoch": 0, "loss": 0.0,
                "train_mse": 0.0, "test_mse": 0.0,
                "best_mse": 1e9, "r2_score": 0.0,
            }

        if self._phase == "training":
            if self._epoch < c["epochs"]:
                batch_start = 0
                batch_end = min(c["batch_size"], len(self._train_x))
                sidx = slice(batch_start, batch_end)
                Xb = pnp.array(self._train_x[sidx], requires_grad=False)
                yb = self._train_y[sidx]
                loss_b = 0.0
                for j in range(len(Xb)):
                    f = self._qcircuit(self._params, Xb[j])
                    loss_b += (f - np.clip(yb[j].mean(), 0, 1)) ** 2
                loss_b /= len(Xb)
                step_result = self._optimizer.step(
                    lambda p, Xb_=Xb, yb_=yb: (
                        sum((self._qcircuit(p, Xb_[j]) - np.clip(yb_[j].mean(), 0, 1)) ** 2
                            for j in range(len(Xb_))) / len(Xb_)
                    ),
                    self._params, Xb, yb,
                )
                self._params = step_result[0]

                train_pred = np.full(len(self._train_x), 0.5)
                test_pred = np.full(len(self._test_x), 0.5)
                train_mse = self._mse(self._train_y.mean(axis=1), train_pred)
                test_mse = self._mse(self._test_y.mean(axis=1), test_pred)
                self._best_mse = min(self._best_mse, test_mse)
                r2_val = self._r2(self._test_y.mean(axis=1), test_pred)

                self._epoch += 1
                self._history.append({"epoch": self._epoch, "train_mse": train_mse,
                                       "test_mse": test_mse, "r2": r2_val})
                return {
                    "phase": "training", "epoch": self._epoch, "loss": float(loss_b),
                    "train_mse": train_mse, "test_mse": test_mse,
                    "best_mse": float(self._best_mse), "r2_score": r2_val,
                }
            else:
                test_mse = self._mse(self._test_y.mean(axis=1), np.full(len(self._test_x), 0.5))
                r2 = self._r2(self._test_y.mean(axis=1), np.full(len(self._test_x), 0.5))
                self._phase = "done"
                return {
                    "phase": "done", "epoch": c["epochs"], "loss": 0.0,
                    "train_mse": self._history[-1]["train_mse"] if self._history else 0.0,
                    "test_mse": test_mse,
                    "best_mse": float(self._best_mse), "r2_score": r2,
                }

        if self._phase == "done":
            return {
                "phase": "done", "epoch": c["epochs"], "loss": 0.0,
                "train_mse": self._history[-1]["train_mse"] if self._history else 0.0,
                "test_mse": self._history[-1]["test_mse"] if self._history else 0.0,
                "best_mse": float(self._best_mse), "r2_score": self._history[-1].get("r2", 0.0),
            }

        return {
            "phase": self._phase, "epoch": 0, "loss": 0.0,
            "train_mse": 0.0, "test_mse": 0.0,
            "best_mse": 1e9, "r2_score": 0.0,
        }


# ---------------------------------------------------------------------------
# 4.  Minimal Genome Evaluation  (gene-knockout screening)
# ---------------------------------------------------------------------------

class MinimalGenomeProcess(Process):
    """Predict subsystem failure from a gene knockout mask.

    A binary vector indicates which genes are knocked out; the circuit
    learns whether each KO combination causes a critical subsystem
    (e.g. ribosome, metabolism, membrane) to fail.
    """

    creational = True

    config_schema = {
        "num_genes": {"_type": "integer", "_default": 20, "_minimum": 4},
        "num_subsystems": {"_type": "integer", "_default": 4, "_minimum": 2},
        "learning_rate": {"_type": "float", "_default": 0.5},
        "epochs": {"_type": "integer", "_default": 8, "_minimum": 1},
        "batch_size": {"_type": "integer", "_default": 32, "_minimum": 1},
        "num_train": {"_type": "integer", "_default": 150, "_minimum": 10},
        "num_test": {"_type": "integer", "_default": 300, "_minimum": 10},
        "seed": {"_type": "integer", "_default": 42},
        "device": {"_type": "string", "_default": "default.qubit"},
        "ko_density": {"_type": "float", "_default": 0.3},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._params = None
        self._qcircuit = None
        self._state_labels = None
        self._phase = "init"
        self._epoch = 0
        self._best_test_acc = 0.0
        self._history = []
        self._data = {}

    def inputs(self):
        return {
            "train_knockouts": {"_type": "list[list[integer]]", "_default": None},
            "train_subsystem_labels": {"_type": "list[list[integer]]", "_default": None},
            "test_knockouts": {"_type": "list[list[integer]]", "_default": None},
            "test_subsystem_labels": {"_type": "list[list[integer]]", "_default": None},
        }

    def outputs(self):
        return {
            "phase": "string",
            "epoch": "integer",
            "loss": "float",
            "train_accuracy": "float",
            "test_accuracy": "float",
            "best_test_accuracy": "float",
            "failure_ratio": "float",
            "critical_ko_count": "integer",
        }

    def initial_state(self):
        return {
            "phase": "init", "epoch": 0, "loss": 0.0,
            "train_accuracy": 0.0, "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
            "failure_ratio": 0.0, "critical_ko_count": 0,
        }

    def _generate_synthetic_ko(self):
        import numpy as np
        c = self.config
        np.random.seed(c["seed"])
        ng = c["num_genes"]
        ns = c["num_subsystems"]
        ko_mask = np.random.binomial(1, c["ko_density"], size=ng)

        # synthetic: each subsystem depends on a subset of genes
        subsystem_gene_map = [
            set(range(i * (ng // ns), (i + 1) * (ng // ns)))
            for i in range(ns)
        ]

        def _sample(n):
            kox, labels = [], []
            for _ in range(n):
                ko = np.random.binomial(1, c["ko_density"], size=ng).tolist()
                ko_set = {i for i, v in enumerate(ko) if v == 1}
                failures = []
                for sg in subsystem_gene_map:
                    overlap = len(ko_set & sg)
                    fails = 1 if overlap >= len(sg) * 0.6 else 0
                    failures.append(fails)
                kox.append(ko)
                labels.append(failures)
            return kox, labels

        train_x, train_y = _sample(c["num_train"])
        test_x, test_y = _sample(c["num_test"])
        return {"train_knockouts": train_x, "train_subsystem_labels": train_y,
                "test_knockouts": test_x, "test_subsystem_labels": test_y}

    def _build_model(self):
        import pennylane as qml
        from pennylane import numpy as np
        c = self.config
        num_layers = min(c["num_subsystems"], 5)
        np.random.seed(c["seed"])
        dev = qml.device(c["device"], wires=1)

        @qml.qnode(dev)
        def qcircuit(params, x, y_dm):
            for l in range(num_layers):
                idx = (l * 3) % c["num_genes"]
                v = x[idx:idx + 3]
                qml.Rot(*v, wires=0)
                qml.Rot(*params[l], wires=0)
            return qml.expval(qml.Hermitian(y_dm, wires=[0]))

        self._qcircuit = qcircuit
        self._params = np.random.uniform(size=(num_layers, 3), requires_grad=True)

    def _test(self, params, x, state_labels_orig):
        import numpy as np
        dm_labels = [_density_matrix(s) for s in state_labels_orig]
        predicted = []
        for i in range(len(x)):
            fidelities = [self._qcircuit(params, x[i], dm) for dm in dm_labels]
            predicted.append(int(np.argmax(fidelities)))
        return np.array(predicted)

    def _cost(self, params, x_batch, y_batch, state_labels_orig):
        import numpy as np
        loss_val = 0.0
        dm_labels = [_density_matrix(s) for s in state_labels_orig]
        for i in range(len(x_batch)):
            f = self._qcircuit(params, x_batch[i], dm_labels[y_batch[i]])
            loss_val += (1 - f) ** 2
        return loss_val / len(x_batch)

    def update(self, state, interval):
        import numpy as np
        from pennylane import numpy as pnp
        from pennylane.optimize import AdamOptimizer

        c = self.config

        if self._phase == "init":
            data_from_state = {}
            for key in ("train_knockouts", "train_subsystem_labels", "test_knockouts", "test_subsystem_labels"):
                val = state.get(key)
                if val is not None and len(val) > 0:
                    data_from_state[key] = val
            if all(k in data_from_state for k in ("train_knockouts", "train_subsystem_labels")):
                self._data = data_from_state
            else:
                self._data = self._generate_synthetic_ko()

            # multi-label → binary: any subsystem failure → "fails"
            train_y_raw = np.array(self._data["train_subsystem_labels"])
            self._train_labels_arr = (train_y_raw.max(axis=1) > 0).astype(int)
            test_y_raw = np.array(self._data["test_subsystem_labels"])
            self._test_labels_arr = (test_y_raw.max(axis=1) > 0).astype(int)

            label_0 = [[1], [0]]
            label_1 = [[0], [1]]
            self._state_labels = np.array([label_0, label_1])

            self._build_model()
            self._history = []
            self._epoch = 0

            self._train_x = np.array(self._data["train_knockouts"], dtype=float)
            self._test_x = np.array(self._data["test_knockouts"], dtype=float)

            self._optimizer = AdamOptimizer(c["learning_rate"], beta1=0.9, beta2=0.999)
            self._phase = "training"

            return {
                "phase": "training", "epoch": 0, "loss": 0.0,
                "train_accuracy": 0.0, "test_accuracy": 0.0,
                "best_test_accuracy": 0.0,
                "failure_ratio": float(np.mean(self._train_labels_arr)),
                "critical_ko_count": 0,
            }

        if self._phase == "training":
            if self._epoch < c["epochs"]:
                for start_idx in range(0, len(self._train_x) - c["batch_size"] + 1, c["batch_size"]):
                    sidx = slice(start_idx, start_idx + c["batch_size"])
                    Xb = pnp.array(self._train_x[sidx], requires_grad=False)
                    yb = pnp.array(self._train_labels_arr[sidx], requires_grad=False)
                    self._params, _, _, _ = self._optimizer.step(
                        self._cost, self._params, Xb, yb, self._state_labels)

                pred_train = self._test(self._params, self._train_x, self._state_labels)
                train_acc = _accuracy_score(self._train_labels_arr, pred_train)
                loss_val = self._cost(self._params, self._train_x, self._train_labels_arr, self._state_labels)

                pred_test = self._test(self._params, self._test_x, self._state_labels)
                test_acc = _accuracy_score(self._test_labels_arr, pred_test)
                self._best_test_acc = max(self._best_test_acc, test_acc)

                self._epoch += 1
                self._history.append({"epoch": self._epoch, "loss": float(loss_val),
                                      "train_accuracy": float(train_acc), "test_accuracy": float(test_acc)})
                # count critical KOs: predict failure on single-gene KOs
                single_ko = np.eye(c["num_genes"])
                single_pred = self._test(self._params, single_ko, self._state_labels)
                critical = int(np.sum(single_pred))

                return {
                    "phase": "training", "epoch": self._epoch, "loss": float(loss_val),
                    "train_accuracy": float(train_acc), "test_accuracy": float(test_acc),
                    "best_test_accuracy": float(self._best_test_acc),
                    "failure_ratio": float(np.mean(self._train_labels_arr)),
                    "critical_ko_count": critical,
                }
            else:
                pred_test = self._test(self._params, self._test_x, self._state_labels)
                final_test_acc = _accuracy_score(self._test_labels_arr, pred_test)
                single_ko = np.eye(c["num_genes"])
                single_pred = self._test(self._params, single_ko, self._state_labels)
                critical = int(np.sum(single_pred))
                self._phase = "done"
                return {
                    "phase": "done", "epoch": c["epochs"], "loss": 0.0,
                    "train_accuracy": float(self._history[-1]["train_accuracy"]) if self._history else 0.0,
                    "test_accuracy": float(final_test_acc),
                    "best_test_accuracy": float(self._best_test_acc),
                    "failure_ratio": float(np.mean(self._train_labels_arr)),
                    "critical_ko_count": critical,
                }

        if self._phase == "done":
            return {
                "phase": "done", "epoch": c["epochs"], "loss": 0.0,
                "train_accuracy": float(self._history[-1]["train_accuracy"]) if self._history else 0.0,
                "test_accuracy": float(self._history[-1]["test_accuracy"]) if self._history else 0.0,
                "best_test_accuracy": float(self._best_test_acc),
                "failure_ratio": float(np.mean(self._train_labels_arr)),
                "critical_ko_count": 0,
            }

        return {
            "phase": self._phase, "epoch": 0, "loss": 0.0,
            "train_accuracy": 0.0, "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
            "failure_ratio": 0.0, "critical_ko_count": 0,
        }


# ---------------------------------------------------------------------------
# 5.  Spatiotemporal State Sorting  (4D trajectory → metabolic mode)
# ---------------------------------------------------------------------------

class SpatiotemporalSortingProcess(Process):
    """Classify 4D (time × 3D space) cellular trajectories into healthy
    or diseased metabolic modes using sequential data re-uploading.

    One re-uploading layer per time frame — the circuit literally tracks
    the spatial trajectory through time.
    """

    creational = True

    config_schema = {
        "num_timestep_frames": {"_type": "integer", "_default": 5, "_minimum": 2},
        "spatial_dims": {"_type": "integer", "_default": 3, "_minimum": 2},
        "learning_rate": {"_type": "float", "_default": 0.3},
        "epochs": {"_type": "integer", "_default": 8, "_minimum": 1},
        "batch_size": {"_type": "integer", "_default": 16, "_minimum": 1},
        "num_train": {"_type": "integer", "_default": 80, "_minimum": 10},
        "num_test": {"_type": "integer", "_default": 120, "_minimum": 10},
        "seed": {"_type": "integer", "_default": 42},
        "device": {"_type": "string", "_default": "default.qubit"},
        "healthy_ratio": {"_type": "float", "_default": 0.5},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        self._params = None
        self._qcircuit = None
        self._state_labels = None
        self._phase = "init"
        self._epoch = 0
        self._best_test_acc = 0.0
        self._history = []
        self._data = {}

    def inputs(self):
        return {
            "train_trajectories": {"_type": "list[list[float]]", "_default": None},
            "train_labels": {"_type": "list[integer]", "_default": None},
            "test_trajectories": {"_type": "list[list[float]]", "_default": None},
            "test_labels": {"_type": "list[integer]", "_default": None},
        }

    def outputs(self):
        return {
            "phase": "string",
            "epoch": "integer",
            "loss": "float",
            "train_accuracy": "float",
            "test_accuracy": "float",
            "best_test_accuracy": "float",
            "healthy_mode_ratio": "float",
            "confusion_matrix": "string",
        }

    def initial_state(self):
        return {
            "phase": "init", "epoch": 0, "loss": 0.0,
            "train_accuracy": 0.0, "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
            "healthy_mode_ratio": 0.0, "confusion_matrix": "",
        }

    def _generate_synthetic_trajectories(self):
        import numpy as np
        c = self.config
        np.random.seed(c["seed"])
        T = c["num_timestep_frames"]
        D = c["spatial_dims"]
        nfeat = T * D

        def _sample(n, healthy_ratio=0.5):
            xs, ys = [], []
            for _ in range(n):
                label = 1 if np.random.rand() < healthy_ratio else 0
                if label == 1:
                    # healthy: coherent motion (low curvature)
                    base = np.cumsum(np.random.randn(T, D) * 0.1, axis=0)
                else:
                    # diseased: chaotic / diffusive motion (high curvature)
                    base = np.cumsum(np.random.randn(T, D) * 0.5, axis=0)
                xs.append(base.flatten().tolist())
                ys.append(label)
            return xs, ys

        train_x, train_y = _sample(c["num_train"], c["healthy_ratio"])
        test_x, test_y = _sample(c["num_test"], c["healthy_ratio"])
        return {"train_trajectories": train_x, "train_labels": train_y,
                "test_trajectories": test_x, "test_labels": test_y}

    def _build_model(self):
        import pennylane as qml
        from pennylane import numpy as np
        c = self.config
        T = c["num_timestep_frames"]
        np.random.seed(c["seed"])
        dev = qml.device(c["device"], wires=1)

        @qml.qnode(dev)
        def qcircuit(params, trajectory, y_label_matrix):
            time_frames = trajectory.reshape(T, -1)
            for t in range(T):
                spatial = time_frames[t]
                v = np.zeros(3)
                v[:len(spatial)] = spatial[:3]
                qml.Rot(*v, wires=0)
                qml.Rot(*params[t], wires=0)
            return qml.expval(qml.Hermitian(y_label_matrix, wires=[0]))

        self._qcircuit = qcircuit
        self._params = np.random.uniform(size=(T, 3), requires_grad=True)

    def _test(self, params, x, state_labels_orig):
        import numpy as np
        dm_labels = [_density_matrix(s) for s in state_labels_orig]
        predicted = []
        for i in range(len(x)):
            fidelities = [self._qcircuit(params, np.array(x[i]), dm) for dm in dm_labels]
            predicted.append(int(np.argmax(fidelities)))
        return np.array(predicted)

    def _cost(self, params, x_batch, y_batch, state_labels_orig):
        import numpy as np
        loss_val = 0.0
        dm_labels = [_density_matrix(s) for s in state_labels_orig]
        for i in range(len(x_batch)):
            f = self._qcircuit(params, np.array(x_batch[i]), dm_labels[y_batch[i]])
            loss_val += (1 - f) ** 2
        return loss_val / len(x_batch)

    def update(self, state, interval):
        import numpy as np
        from pennylane import numpy as pnp
        from pennylane.optimize import AdamOptimizer

        c = self.config

        if self._phase == "init":
            data_from_state = {}
            for key in ("train_trajectories", "train_labels", "test_trajectories", "test_labels"):
                val = state.get(key)
                if val is not None and len(val) > 0:
                    data_from_state[key] = val
            if all(k in data_from_state for k in ("train_trajectories", "train_labels")):
                self._data = data_from_state
            else:
                self._data = self._generate_synthetic_trajectories()

            label_0 = [[1], [0]]
            label_1 = [[0], [1]]
            self._state_labels = np.array([label_0, label_1])

            self._build_model()
            self._history = []
            self._epoch = 0

            self._train_x = np.array(self._data["train_trajectories"])
            self._train_y = np.array(self._data["train_labels"])
            self._test_x = np.array(self._data["test_trajectories"])
            self._test_y = np.array(self._data["test_labels"])

            self._optimizer = AdamOptimizer(c["learning_rate"], beta1=0.9, beta2=0.999)
            self._phase = "training"

            return {
                "phase": "training", "epoch": 0, "loss": 0.0,
                "train_accuracy": 0.0, "test_accuracy": 0.0,
                "best_test_accuracy": 0.0,
                "healthy_mode_ratio": float(np.mean(self._train_y)),
                "confusion_matrix": "",
            }

        if self._phase == "training":
            if self._epoch < c["epochs"]:
                batch_size = min(c["batch_size"], len(self._train_x))
                Xb = [pnp.array(x, requires_grad=False) for x in self._train_x[:batch_size]]
                yb = pnp.array(self._train_y[:batch_size], requires_grad=False)
                self._params = self._optimizer.step(
                    lambda p: self._cost(p, Xb, yb, self._state_labels),
                    self._params,
                )

                pred_train = self._test(self._params, self._train_x, self._state_labels)
                train_acc = _accuracy_score(self._train_y, pred_train)
                loss_val = self._cost(self._params, self._train_x, self._train_y, self._state_labels)

                pred_test = self._test(self._params, self._test_x, self._state_labels)
                test_acc = _accuracy_score(self._test_y, pred_test)
                self._best_test_acc = max(self._best_test_acc, test_acc)

                self._epoch += 1
                self._history.append({"epoch": self._epoch, "loss": float(loss_val),
                                      "train_accuracy": float(train_acc), "test_accuracy": float(test_acc)})

                tp = int(np.sum((self._test_y == 1) & (pred_test == 1)))
                tn = int(np.sum((self._test_y == 0) & (pred_test == 0)))
                fp = int(np.sum((self._test_y == 0) & (pred_test == 1)))
                fn = int(np.sum((self._test_y == 1) & (pred_test == 0)))
                cm_str = f"TP={tp} TN={tn} FP={fp} FN={fn}"

                return {
                    "phase": "training", "epoch": self._epoch, "loss": float(loss_val),
                    "train_accuracy": float(train_acc), "test_accuracy": float(test_acc),
                    "best_test_accuracy": float(self._best_test_acc),
                    "healthy_mode_ratio": float(np.mean(self._train_y)),
                    "confusion_matrix": cm_str,
                }
            else:
                pred_test = self._test(self._params, self._test_x, self._state_labels)
                final_test_acc = _accuracy_score(self._test_y, pred_test)
                tp = int(np.sum((self._test_y == 1) & (pred_test == 1)))
                tn = int(np.sum((self._test_y == 0) & (pred_test == 0)))
                fp = int(np.sum((self._test_y == 0) & (pred_test == 1)))
                fn = int(np.sum((self._test_y == 1) & (pred_test == 0)))
                cm_str = f"TP={tp} TN={tn} FP={fp} FN={fn}"
                self._phase = "done"
                return {
                    "phase": "done", "epoch": c["epochs"], "loss": 0.0,
                    "train_accuracy": float(self._history[-1]["train_accuracy"]) if self._history else 0.0,
                    "test_accuracy": float(final_test_acc),
                    "best_test_accuracy": float(self._best_test_acc),
                    "healthy_mode_ratio": float(np.mean(self._train_y)),
                    "confusion_matrix": cm_str,
                }

        if self._phase == "done":
            return {
                "phase": "done", "epoch": c["epochs"], "loss": 0.0,
                "train_accuracy": float(self._history[-1]["train_accuracy"]) if self._history else 0.0,
                "test_accuracy": float(self._history[-1]["test_accuracy"]) if self._history else 0.0,
                "best_test_accuracy": float(self._best_test_acc),
                "healthy_mode_ratio": float(np.mean(self._train_y)),
                "confusion_matrix": "",
            }

        return {
            "phase": self._phase, "epoch": 0, "loss": 0.0,
            "train_accuracy": 0.0, "test_accuracy": 0.0,
            "best_test_accuracy": 0.0,
            "healthy_mode_ratio": 0.0, "confusion_matrix": "",
        }
