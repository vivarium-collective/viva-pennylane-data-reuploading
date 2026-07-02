"""Data loaders that bridge real (or synthetic) multi-omic data sources
to the GenotypeToPhenotypeProcess input format.

Primary loader: :class:`ZarrMultiOmicLoader` reads v2ecoli xarray/zarr
stores produced by the XArrayEmitter.  A factory function
:func:`build_modality_config` helps biologists map observable paths to
modality slots without editing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

_HIVE_KEYS = ["experiment_id", "variant", "lineage_seed"]


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class ModalitySpec:
    """One omic modality: which observables feed into it and how.

    Attributes
    ----------
    name:
        Human-readable modality label (e.g. ``"genomics"``).
    store_paths:
        Observable paths *relative to the view root* in the zarr store
        (e.g. ``["listeners/mass/dry_mass", "listeners/mass/cell_mass"]``).
    aggregation:
        Per-observable time-series reduction.
        ``"mean"`` — take the mean over all time steps.
        ``"end"`` — take the last value.
        ``"slope"`` — linear-fit slope.
        ``"mean+slope"`` — concatenate mean and slope (doubles feature count).
    flatten:
        Whether to flatten vector coordinates (e.g. monomer names) into
        individual feature columns.  ``True`` by default.
    """
    name: str
    store_paths: list[str] = field(default_factory=list)
    aggregation: str = "mean"
    flatten: bool = True


@dataclass
class PhenotypeLabelRule:
    """Rule for assigning binary phenotype labels from simulation data.

    Attributes
    ----------
    source_path:
        Observable path used for the rule (e.g. ``"listeners/mass/dry_mass"``).
    operator:
        Comparison: ``">"``, ``"<"``, ``">="``, ``"<="``.
    threshold:
        Value to compare against.
    use_derived:
        If True, labels are derived from the rule.  If False, labels must
        be provided externally.
    """
    source_path: str = "listeners/mass/dry_mass"
    operator: str = ">"
    threshold: float = 0.0
    use_derived: bool = True


# ---------------------------------------------------------------------------
# Default v2ecoli modality map
# ---------------------------------------------------------------------------

def v2ecoli_default_modality_map() -> list[ModalitySpec]:
    """Return a default modality-to-observable mapping for v2ecoli.

    Maps four omic layers:
        genomics    → growth-related scalars  (dry_mass, cell_mass, volume)
        transcriptomics → vector: mRNA counts  (``mRna_counts``)
        proteomics  → vector: monomer counts  (``monomer_counts``)
        metabolomics → scalar: FBA exchange fluxes + internal concs

    Biologists should adjust paths to match their composite's declared
    ``listeners`` view.
    """
    return [
        ModalitySpec(
            name="genomics",
            store_paths=[
                "listeners/mass/dry_mass",
                "listeners/mass/cell_mass",
            ],
            aggregation="mean+slope",
        ),
        ModalitySpec(
            name="transcriptomics",
            store_paths=[
                "listeners/mRna_counts",
            ],
            aggregation="mean",
        ),
        ModalitySpec(
            name="proteomics",
            store_paths=[
                "listeners/monomer_counts",
            ],
            aggregation="mean",
        ),
        ModalitySpec(
            name="metabolomics",
            store_paths=[
                "listeners/fba_output/exchange_fluxes",
                "listeners/fba_output/internal_concentrations",
            ],
            aggregation="end",
        ),
    ]


def _default_labels(loader: ZarrMultiOmicLoader) -> np.ndarray | None:
    """Default phenotype-label derivation: viable if final dry_mass > initial."""
    try:
        dry_mass = loader._extract_observable("listeners/mass/dry_mass", agg="mean")
        if dry_mass is None or dry_mass.size == 0:
            return None
        median = float(np.median(dry_mass))
        return (dry_mass > median).astype(int)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class ZarrMultiOmicLoader:
    """Load a v2ecoli xarray/zarr store into classifier-ready arrays.

    Usage::

        loader = ZarrMultiOmicLoader(
            store_path="out/experiment.zarr",
            modality_map=v2ecoli_default_modality_map(),
            label_rule=PhenotypeLabelRule(),
        )
        data = loader.load(
            variant=0, seed=42,
            train_ratio=0.6,
        )
        # data == {"train_inputs": ..., "train_labels": ...,
        #           "test_inputs": ..., "test_labels": ...}
        proc = GenotypeToPhenotypeProcess(config={...}, core=core)
        state = proc.initial_state()
        state.update(data)
    """

    def __init__(
        self,
        store_path: str | Path,
        modality_map: list[ModalitySpec] | None = None,
        label_rule: PhenotypeLabelRule | None = None,
        label_fn: Callable[["ZarrMultiOmicLoader"], np.ndarray | None] | None = None,
    ):
        self.store_path = Path(store_path)
        self.modality_map = modality_map or v2ecoli_default_modality_map()
        self.label_rule = label_rule or PhenotypeLabelRule()
        self.label_fn = label_fn or _default_labels
        self._tree: Any = None
        self._lineage_root: Any = None

    def _find_lineage_root(self):
        """Navigate hive-partition levels to reach the lineage root node.

        v2ecoli stores use zarr groups as hive partitions:
            experiment_id=<id>/variant=<n>/lineage_seed=<n>/

        Returns the ``lineage_seed=<n>`` group node, or the root if no
        hive partitions are found.
        """
        node = self._tree
        for key in _HIVE_KEYS:
            if not hasattr(node, "children"):
                break
            match = next(
                (c for c in node.children if c.startswith(key + "=")),
                None,
            )
            if match is not None:
                node = node[match]
            else:
                break
        return node

    def _open_store(self):
        import xarray as xr
        self._tree = xr.open_datatree(self.store_path, engine="zarr")
        self._lineage_root = self._find_lineage_root()

    def _walk_gens(self, node: Any, gens: set[int]) -> None:
        """Recursively collect ``generation=N`` integers from data variables."""
        for var_name in node.data_vars:
            if var_name.startswith("generation="):
                try:
                    gens.add(int(var_name.split("=", 1)[1]))
                except ValueError:
                    pass
        for _child_name, child_node in node.children.items():
            self._walk_gens(child_node, gens)

    def _list_generations(self) -> list[int]:
        """Discover generation numbers from data-variable ``generation=N`` arrays."""
        if self._lineage_root is None:
            return []
        gens: set[int] = set()
        self._walk_gens(self._lineage_root, gens)
        return sorted(gens)

    def _resolve_path(self, path: str) -> Any:
        """Walk the DataTree from the lineage root following a dotted/slash path.

        Paths are relative to the observable root
        (e.g. ``"listeners/mass/dry_mass"``).
        """
        if self._lineage_root is None:
            return None
        parts = path.replace(".", "/").split("/")
        node: Any = self._lineage_root
        for p in parts:
            try:
                node = node[p]
            except (KeyError, ValueError, TypeError):
                return None
        return node

    def _read_variable(self, path: str, gen: int) -> np.ndarray | None:
        """Read a data variable at a given generation.

        In v2ecoli's zarr layout the observable path is the group and
        ``generation=<N>`` is the data variable inside it:

            <lineage_root>/<path>/  has data-var  ``generation=<N>``

        Returns a numpy array, or None if the path doesn't exist.
        """
        if self._lineage_root is None:
            return None
        leaf = self._resolve_path(path)
        if leaf is None:
            return None
        gen_key = f"generation={gen}"
        try:
            arr_node = leaf[gen_key]
        except (KeyError, ValueError, TypeError):
            return None
        if hasattr(arr_node, "values"):
            return np.asarray(arr_node.values)
        if isinstance(arr_node, np.ndarray):
            return arr_node
        return None

    def _extract_observable(
        self,
        path: str,
        agg: str = "mean",
        generations: list[int] | None = None,
    ) -> np.ndarray | None:
        """Extract one observable across generations and aggregate over time.

        Returns a 1-D array of shape ``(n_generations,)`` for scalars,
        or ``(n_generations, n_features)`` for vectors.
        """
        if generations is None:
            generations = self._list_generations()
        if not generations:
            return None

        samples: list[np.ndarray] = []
        for gen in generations:
            arr = self._read_variable(path, gen)
            if arr is None:
                continue

            # arr shape: (emit_steps,) for scalars, (emit_steps, N) for vectors
            if arr.ndim == 0:
                arr = arr.reshape(1)
            if arr.ndim == 1:
                # scalar observable — aggregate over time
                if agg == "mean":
                    val = np.mean(arr)
                elif agg == "end":
                    val = arr[-1] if len(arr) > 0 else 0.0
                elif agg == "slope":
                    if len(arr) > 1:
                        val = np.polyfit(np.arange(len(arr)), arr, 1)[0]
                    else:
                        val = 0.0
                elif agg == "mean+slope":
                    m = np.mean(arr)
                    s = np.polyfit(np.arange(len(arr)), arr, 1)[0] if len(arr) > 1 else 0.0
                    val = np.array([m, s])
                else:
                    val = np.mean(arr)
                samples.append(np.atleast_1d(val))
            elif arr.ndim == 2:
                # vector observable — aggregate each column over time
                if agg == "mean":
                    val = np.mean(arr, axis=0)
                elif agg == "end":
                    val = arr[-1, :] if arr.shape[0] > 0 else np.zeros(arr.shape[1])
                elif agg == "slope":
                    if arr.shape[0] > 1:
                        val = np.polyfit(np.arange(arr.shape[0]), arr, 1)[0]
                    else:
                        val = np.zeros(arr.shape[1])
                elif agg == "mean+slope":
                    m = np.mean(arr, axis=0)
                    s = np.polyfit(np.arange(arr.shape[0]), arr, 1)[0] if arr.shape[0] > 1 else np.zeros(arr.shape[1])
                    val = np.concatenate([m, s])
                else:
                    val = np.mean(arr, axis=0)
                samples.append(np.atleast_1d(val))
            else:
                samples.append(np.atleast_1d(np.mean(arr)))

        if not samples:
            return None
        return np.array(samples)

    def _generate_labels(
        self,
        generations: list[int] | None = None,
    ) -> np.ndarray | None:
        """Generate binary labels per generation using the configured rule."""
        if not self.label_rule.use_derived:
            return None

        if generations is None:
            generations = self._list_generations()
        if not generations:
            return None

        src = self.label_rule.source_path
        agg = "mean"
        values = self._extract_observable(src, agg=agg, generations=generations)
        if values is None:
            return None
        values = values.ravel()

        op = self.label_rule.operator
        th = self.label_rule.threshold
        if op == ">":
            return (values > th).astype(int)
        elif op == "<":
            return (values < th).astype(int)
        elif op == ">=":
            return (values >= th).astype(int)
        elif op == "<=":
            return (values <= th).astype(int)
        else:
            return (values > th).astype(int)

    def load(
        self,
        *,
        variant: int = 0,
        seed: int = 42,
        train_ratio: float = 0.6,
        generations: list[int] | None = None,
        shuffle: bool = True,
        rng_seed: int = 42,
    ) -> dict[str, Any]:
        """Load data from the store and format for GenotypeToPhenotypeProcess.

        Parameters
        ----------
        variant:
            Hive-partition variant index (used to resolve store path if
            the store uses hive partitioning).
        seed:
            Lineage seed index.
        train_ratio:
            Fraction of samples to use for training (remainder = test).
        generations:
            Specific generations to load (default: all discovered).
        shuffle:
            Whether to shuffle samples before train/test split.
        rng_seed:
            Seed for shuffling.

        Returns
        -------
        dict
            With keys ``train_inputs``, ``train_labels``, ``test_inputs``,
            ``test_labels``, and metadata ``_generations``, ``_modalities``,
            ``_features_per_modality``.
        """
        # Open store if not already
        if self._tree is None:
            self._open_store()

        if generations is None:
            generations = self._list_generations()
        if not generations:
            raise ValueError(
                f"No ``generation=N`` arrays found under {self.store_path}. "
                "The store must contain observable leaf groups with "
                "``generation=<int>`` data variables — the v2ecoli "
                "XArrayEmitter output format."
            )

        # Extract features for each modality
        modality_feature_arrays: list[np.ndarray] = []
        feature_counts: list[int] = []
        for spec in self.modality_map:
            modality_vectors: list[np.ndarray] = []
            for path in spec.store_paths:
                arr = self._extract_observable(
                    path, agg=spec.aggregation, generations=generations
                )
                if arr is not None:
                    modality_vectors.append(arr)

            if modality_vectors:
                combined = np.concatenate(modality_vectors, axis=1) if len(modality_vectors) > 1 else modality_vectors[0]
                modality_feature_arrays.append(combined)
                feature_counts.append(combined.shape[1])
            else:
                # Fallback: zero-filled placeholder so modality count stays consistent
                n_samp = len(generations)
                placeholder = np.zeros((n_samp, 3))
                modality_feature_arrays.append(placeholder)
                feature_counts.append(3)

        if not modality_feature_arrays:
            raise ValueError(
                "No observables could be loaded from the store. "
                "Check that ``modality_map`` paths exist in the zarr store."
            )

        # Concatenate all modalities → (n_samples, total_features)
        X = np.concatenate(modality_feature_arrays, axis=1)
        M = len(self.modality_map)
        F = max(feature_counts) if feature_counts else 3

        # Pad so all modalities have the same feature count
        X_padded_list = []
        offset = 0
        for m_idx, n_feat in enumerate(feature_counts):
            seg = X[:, offset : offset + n_feat]
            offset += n_feat
            if n_feat < F:
                pad_width = F - n_feat
                seg = np.pad(seg, ((0, 0), (0, pad_width)), mode="constant")
            X_padded_list.append(seg)
        if X_padded_list:
            X_padded = np.concatenate(X_padded_list, axis=1)
        else:
            X_padded = X

        # Generate labels
        labels = self._generate_labels(generations=generations)
        if labels is None:
            rng = np.random.RandomState(rng_seed)
            labels = rng.randint(0, 2, size=len(generations)).astype(int)

        # Shuffle
        n = len(X_padded)
        idx = np.arange(n)
        if shuffle:
            rng = np.random.RandomState(rng_seed)
            rng.shuffle(idx)
            X_padded = X_padded[idx]
            labels = labels[idx]
            generations_shuffled = [generations[i] for i in idx]
        else:
            generations_shuffled = list(generations)

        # Train/test split
        split = int(n * train_ratio)
        if split < 1:
            split = max(1, n // 2)
        if split >= n:
            split = n - 1 if n > 1 else 1

        result = {
            "train_inputs": X_padded[:split].tolist(),
            "train_labels": labels[:split].tolist(),
            "test_inputs": X_padded[split:].tolist(),
            "test_labels": labels[split:].tolist(),
            "_generations": generations_shuffled,
            "_modalities": [s.name for s in self.modality_map],
            "_features_per_modality": F,
            "_total_samples": n,
        }
        return result


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_modality_config(
    store_path: str | Path,
    modality_assignments: dict[str, list[str]],
    default_agg: str = "mean",
) -> ZarrMultiOmicLoader:
    """Quickly build a loader with a dict-based modality map.

    Example::

        loader = build_modality_config(
            "out/my_sim.zarr",
            {
                "genomics":    ["listeners/mass/dry_mass", "listeners/mass/cell_mass"],
                "proteomics":  ["listeners/monomer_counts"],
            },
        )
        data = loader.load()
    """
    specs = [
        ModalitySpec(name=name, store_paths=paths, aggregation=default_agg)
        for name, paths in modality_assignments.items()
    ]
    return ZarrMultiOmicLoader(store_path=store_path, modality_map=specs)


# ---------------------------------------------------------------------------
# Synthetic fallback (mirrors GenotypeToPhenotypeProcess._generate_synthetic_omics)
# ---------------------------------------------------------------------------

def generate_synthetic_omics(
    num_modalities: int = 4,
    features_per_modality: int = 6,
    num_train: int = 60,
    num_test: int = 100,
    noise_level: float = 0.15,
    seed: int = 42,
    modality_weights: list[float] | None = None,
) -> dict[str, Any]:
    """Generate biologically plausible synthetic multi-omic data.

    The generative model mimics statistical properties of real multi-omic
    data: modality-specific weights, pairwise cross-modality interaction
    terms, and configurable noise.

    Returns the same dict shape as :meth:`ZarrMultiOmicLoader.load`.
    """
    rng = np.random.RandomState(seed)
    M = num_modalities
    F = features_per_modality
    nfeat = M * F

    if modality_weights is None:
        weights = [1.0 + 0.5 * i for i in range(M)]
    else:
        weights = modality_weights

    def _sample(n: int):
        xs, ys = [], []
        for _ in range(n):
            x = rng.randn(nfeat)
            scores = []
            for i in range(M):
                seg = x[i * F : (i + 1) * F]
                scores.append(float(np.mean(seg)) * weights[i])
            weighted = sum(scores)
            interaction = 0.0
            for i in range(M):
                for j in range(i + 1, M):
                    interaction += scores[i] * scores[j] * 0.2
            raw = weighted + interaction + noise_level * rng.randn()
            ys.append(1 if raw > 0 else 0)
            xs.append(x.tolist())
        return xs, ys

    train_x, train_y = _sample(num_train)
    test_x, test_y = _sample(num_test)

    return {
        "train_inputs": train_x,
        "train_labels": train_y,
        "test_inputs": test_x,
        "test_labels": test_y,
        "_modalities": [f"modality_{i}" for i in range(M)],
        "_features_per_modality": F,
        "_total_samples": num_train + num_test,
    }
