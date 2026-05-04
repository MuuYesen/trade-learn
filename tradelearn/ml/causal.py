"""Causal feature selection helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from tradelearn.utils.console import smart_tqdm as tqdm

CausalBackend = Callable[[pd.DataFrame, pd.Series], dict[str, float] | pd.Series]
CausalLearnRunner = Callable[..., Any]


@dataclass
class CausalSelector:
    """Select candidate causal features from a single dataset."""

    target: str
    features: tuple[str, ...] | list[str] | None = None
    max_features: int | None = None
    min_score: float = 0.0
    method: str = "correlation"
    alpha: float = 0.05
    indep_test: str = "fisherz"
    backend: CausalBackend | None = None
    pc_runner: CausalLearnRunner | None = None
    fci_runner: CausalLearnRunner | None = None

    def fit(self, data: pd.DataFrame) -> CausalSelector:
        """Fit selector scores and selected feature names."""
        method = self.method.lower()
        if method not in {"correlation", "pc", "fci"}:
            raise ValueError("method must be one of 'correlation', 'pc', or 'fci'")
        
        with tqdm(total=1, desc=f"CausalSelector.fit({method.upper()})", leave=True) as pbar:
            frame, target = self._split_dataset(data)
            scores = self._score(frame, target)
            ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            threshold = self.min_score
            if method in {"pc", "fci"} and threshold <= 0.0:
                threshold = 1e-12
            selected = [name for name, score in ranked if score >= threshold]
            if self.max_features is not None:
                selected = selected[: self.max_features]
            self.scores_ = scores
            self.selected_features_ = selected
            pbar.update(1)
            pbar.set_postfix(selected=len(selected))
        
        return self

    def plot(self, filename: str | None = None) -> str:
        """Render the fitted causal graph as a PNG image.

        Requires ``method='pc'`` or ``method='fci'`` and must be called after
        :meth:`fit`. Uses ``causallearn.utils.GraphUtils.to_pydot`` to convert
        the discovered graph structure into a PNG file, mirroring the 1.x
        ``Graph.fit_causal()`` API.
        """
        if not hasattr(self, "graph_"):
            raise RuntimeError("No fitted graph found. Call fit() first.")
        try:
            from causallearn.utils.GraphUtils import GraphUtils
            import pydot
        except ImportError as exc:
            raise ImportError("plot() requires causal-learn and pydot.") from exc
        
        if filename is None:
            filename = f"{self.method}_graph.png"
        
        pdy = GraphUtils.to_pydot(self.graph_)
        try:
            pdy.write_png(filename)
        except OSError as exc:
            if "dot" in str(exc):
                raise OSError("Graphviz 'dot' binary not found.") from exc
            raise
        return filename

    def to_mermaid(self) -> str:
        """Return a Mermaid.js graph representation of the causal structure."""
        if not hasattr(self, "graph_"):
            return "graph LR\n    No_Data"
        edges = []
        for edge in self.graph_.get_graph_edges():
            n1 = edge.get_node1().get_name()
            n2 = edge.get_node2().get_name()
            edges.append(f"    {n1} --> {n2}")
        return "graph LR\n" + ("\n".join(edges) if edges else "    No_Relationships")

    def to_section(self, title: str = "Causal Diagram") -> dict[str, str]:
        """Return a report section dictionary containing the causal diagram."""
        graph_code = self.to_mermaid()
        return {
            "html": f"""
            <section class="compact-section">
                <h2>{title}</h2>
                <div style="background:white; padding:20px; text-align:center; border:1px solid #eee; border-radius:8px;">
                    <pre class="mermaid">{graph_code}</pre>
                </div>
                <script type="module">
                    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
                    mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
                </script>
            </section>
            """
        }

    def select(self, data: pd.DataFrame) -> list[str]:
        """Return selected feature names after fitting."""
        return self.fit(data).selected_features_

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame containing only selected features."""
        if not hasattr(self, "selected_features_"):
            raise ValueError("CausalSelector must be fitted before transform().")
        return pd.DataFrame(data).loc[:, self.selected_features_]

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit selector and return selected feature columns."""
        return self.fit(data).transform(data)

    def _split_dataset(self, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        frame = pd.DataFrame(data).copy()
        if self.target not in frame.columns:
            raise ValueError(f"target column {self.target!r} not found")
        if self.features is None:
            feature_names = [
                str(column)
                for column in frame.select_dtypes(include="number").columns
                if str(column) != self.target
            ]
        else:
            feature_names = [str(column) for column in self.features]
        missing = [name for name in feature_names if name not in frame.columns]
        if missing:
            raise ValueError(f"feature column(s) not found: {missing}")
        self.feature_names_ = feature_names
        return frame.loc[:, feature_names], frame[self.target].astype(float)

    def _score(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        if self.backend is not None:
            raw_scores = self.backend(X, y)
            if isinstance(raw_scores, pd.Series):
                return {str(name): float(score) for name, score in raw_scores.items()}
            return {str(name): float(score) for name, score in raw_scores.items()}
        method = self.method.lower()
        if method in {"pc", "fci"}:
            return self._causal_learn_scores(X, y, method=method)
        aligned = pd.concat([X, y.rename("__target__")], axis=1).dropna()
        if aligned.empty:
            return {str(column): 0.0 for column in X.columns}
        target = aligned["__target__"].astype(float)
        scores: dict[str, float] = {}
        for column in X.columns:
            feature = aligned[column].astype(float)
            if feature.nunique(dropna=True) <= 1 or target.nunique(dropna=True) <= 1:
                scores[str(column)] = 0.0
                continue
            score = feature.corr(target)
            scores[str(column)] = 0.0 if pd.isna(score) else abs(float(score))
        return scores

    def _causal_learn_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        method: str,
    ) -> dict[str, float]:
        frame = pd.concat([X, y.rename("__target__")], axis=1).dropna()
        feature_names = [str(column) for column in X.columns]
        if frame.empty:
            return {name: 0.0 for name in feature_names}
        data = frame.to_numpy(dtype=float)
        node_names = [*feature_names, "__target__"]
        target_idx = len(node_names) - 1
        if method == "pc":
            runner = self.pc_runner or _load_pc_runner()
            result = runner(
                data,
                alpha=float(self.alpha),
                indep_test=self.indep_test,
                verbose=False,
                show_progress=False,
                node_names=node_names,
            )
            graph = getattr(result, "G", result)
        else:
            runner = self.fci_runner or _load_fci_runner()
            result = runner(
                data,
                alpha=float(self.alpha),
                independence_test_method=self.indep_test,
                verbose=False,
                show_progress=False,
                node_names=node_names,
            )
            graph = result[0] if isinstance(result, tuple) else result
        self.graph_ = graph
        adjacent = _target_adjacent_feature_indices(graph, target_idx, len(feature_names))
        return {name: (1.0 if idx in adjacent else 0.0) for idx, name in enumerate(feature_names)}


def _load_pc_runner() -> CausalLearnRunner:
    try:
        from causallearn.search.ConstraintBased.PC import pc
    except ImportError as exc:
        raise ImportError("CausalSelector(method='pc') requires causal-learn.") from exc
    return pc


def _load_fci_runner() -> CausalLearnRunner:
    try:
        from causallearn.search.ConstraintBased.FCI import fci
    except ImportError as exc:
        raise ImportError("CausalSelector(method='fci') requires causal-learn.") from exc
    return fci


def _target_adjacent_feature_indices(graph: Any, target_idx: int, feature_count: int) -> set[int]:
    matrix = getattr(graph, "graph", None)
    if matrix is not None:
        values = np.asarray(matrix)
        return {
            idx
            for idx in range(feature_count)
            if values[idx, target_idx] != 0 or values[target_idx, idx] != 0
        }
    get_graph_edges = getattr(graph, "get_graph_edges", None)
    get_nodes = getattr(graph, "get_nodes", None)
    if not callable(get_graph_edges) or not callable(get_nodes):
        return set()
    nodes = list(get_nodes())
    target = nodes[target_idx]
    adjacent = set()
    for edge in get_graph_edges():
        node1 = getattr(edge, "node1", None) or getattr(edge, "get_node1", lambda: None)()
        node2 = getattr(edge, "node2", None) or getattr(edge, "get_node2", lambda: None)()
        if node1 is target and node2 in nodes[:feature_count]:
            adjacent.add(nodes.index(node2))
        elif node2 is target and node1 in nodes[:feature_count]:
            adjacent.add(nodes.index(node1))
    return adjacent
