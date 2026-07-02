import copy

from process_bigraph import Process


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

    def _density_matrix(self, state):
        import numpy as np
        state = np.asarray(state)
        return state * np.conj(state).T

    def _cost(self, params, x_batch, y_batch, state_labels_orig):
        import numpy as np
        loss_val = 0.0
        dm_labels = [self._density_matrix(s) for s in state_labels_orig]
        for i in range(len(x_batch)):
            f = self._qcircuit(params, x_batch[i], dm_labels[y_batch[i]])
            loss_val += (1 - f) ** 2
        return loss_val / len(x_batch)

    def _test(self, params, x, y, state_labels_orig):
        import numpy as np
        dm_labels = [self._density_matrix(s) for s in state_labels_orig]
        predicted = []
        for i in range(len(x)):
            fidelities = [self._qcircuit(params, x[i], dm) for dm in dm_labels]
            predicted.append(int(np.argmax(fidelities)))
        return np.array(predicted)

    def _accuracy_score(self, y_true, y_pred):
        import numpy as np
        return float(np.mean(np.array(y_true) == np.array(y_pred)))

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

            # pad inputs with zero third dimension for qml.Rot
            train_x = np.array(self._data["train_inputs"])
            self._train_inputs_padded = np.hstack((
                train_x, np.zeros((train_x.shape[0], 1))
            ))
            test_x = np.array(self._data["test_inputs"])
            self._test_inputs_padded = np.hstack((
                test_x, np.zeros((test_x.shape[0], 1))
            ))
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
                train_acc = self._accuracy_score(self._train_labels_arr, pred_train)
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
                test_acc = self._accuracy_score(self._test_labels_arr, pred_test)
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
                final_test_acc = self._accuracy_score(self._test_labels_arr, pred_test)
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
