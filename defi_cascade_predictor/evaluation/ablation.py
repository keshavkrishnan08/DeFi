"""
Ablation study framework.

Systematically removes components to quantify their contribution:
  1. Feature group ablation (remove each feature group)
  2. Edge type ablation (remove each edge type from graph)
  3. Temporal window ablation (vary lookback period)
  4. Model component ablation (remove memory, attention, etc.)
  5. Prediction horizon analysis
"""

import copy
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from loguru import logger

from evaluation.metrics import MetricsCalculator


class AblationStudy:
    """Systematic ablation study framework for the TGN model."""

    def __init__(
        self,
        base_model: torch.nn.Module,
        config: dict,
        metrics_calculator: MetricsCalculator,
        output_dir: str = "outputs/results",
    ):
        self.base_model = base_model
        self.config = config
        self.metrics = metrics_calculator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_feature_group_ablation(
        self,
        train_fn: callable,
        evaluate_fn: callable,
        feature_groups: list[str],
        base_metrics: dict,
    ) -> dict:
        """Ablate each feature group by zeroing it out.

        For each group, retrain the model with that feature group zeroed
        and measure the performance drop.

        Args:
            train_fn: Function(exclude_groups) -> trained_model.
            evaluate_fn: Function(model) -> metrics_dict.
            feature_groups: List of feature group names.
            base_metrics: Full model metrics for comparison.

        Returns:
            Dict of group_name -> {metrics, delta_from_full}.
        """
        logger.info("=" * 60)
        logger.info("ABLATION: Feature Groups")
        logger.info("=" * 60)

        results = {"full_model": base_metrics}

        for group in feature_groups:
            logger.info(f"Ablating feature group: {group}")
            try:
                model = train_fn(exclude_groups=[group])
                group_metrics = evaluate_fn(model)

                # Compute delta
                delta = {}
                for horizon_key in base_metrics:
                    if horizon_key in group_metrics:
                        delta[horizon_key] = {}
                        for metric_name in base_metrics[horizon_key]:
                            base_val = base_metrics[horizon_key][metric_name]
                            ablated_val = group_metrics[horizon_key][metric_name]
                            if isinstance(base_val, (int, float)):
                                delta[horizon_key][metric_name] = (
                                    base_val - ablated_val
                                )

                results[f"without_{group}"] = {
                    "metrics": group_metrics,
                    "delta": delta,
                }

                # Log impact
                for hk in delta:
                    if "auroc" in delta[hk]:
                        logger.info(
                            f"  {group} -> {hk} AUROC drop: "
                            f"{delta[hk]['auroc']:.4f}"
                        )

            except Exception as e:
                logger.error(f"Feature ablation failed for {group}: {e}")
                results[f"without_{group}"] = {"error": str(e)}

        self.results["feature_group_ablation"] = results
        return results

    def run_edge_type_ablation(
        self,
        train_fn: callable,
        evaluate_fn: callable,
        edge_types: list[str],
        base_metrics: dict,
    ) -> dict:
        """Ablate each edge type from the composability graph.

        Args:
            train_fn: Function(exclude_edges) -> trained_model.
            evaluate_fn: Function(model) -> metrics_dict.
            edge_types: List of edge type names.
            base_metrics: Full model metrics.
        """
        logger.info("=" * 60)
        logger.info("ABLATION: Edge Types")
        logger.info("=" * 60)

        results = {"full_model": base_metrics}

        for etype in edge_types:
            logger.info(f"Ablating edge type: {etype}")
            try:
                model = train_fn(exclude_edges=[etype])
                etype_metrics = evaluate_fn(model)

                delta = {}
                for hk in base_metrics:
                    if hk in etype_metrics:
                        delta[hk] = {}
                        for mk in base_metrics[hk]:
                            bv = base_metrics[hk][mk]
                            av = etype_metrics[hk][mk]
                            if isinstance(bv, (int, float)):
                                delta[hk][mk] = bv - av

                results[f"without_{etype}"] = {
                    "metrics": etype_metrics,
                    "delta": delta,
                }

            except Exception as e:
                logger.error(f"Edge ablation failed for {etype}: {e}")
                results[f"without_{etype}"] = {"error": str(e)}

        self.results["edge_type_ablation"] = results
        return results

    def run_temporal_window_ablation(
        self,
        train_fn: callable,
        evaluate_fn: callable,
        windows: list[int],
        base_window: int = 30,
        base_metrics: dict = None,
    ) -> dict:
        """Test different temporal lookback windows.

        Args:
            train_fn: Function(window) -> trained_model.
            evaluate_fn: Function(model) -> metrics_dict.
            windows: List of window sizes to test.
            base_window: Default window size.
            base_metrics: Metrics for the base window.
        """
        logger.info("=" * 60)
        logger.info("ABLATION: Temporal Windows")
        logger.info("=" * 60)

        results = {}

        for window in windows:
            logger.info(f"Testing temporal window: {window} days")
            try:
                model = train_fn(window=window)
                window_metrics = evaluate_fn(model)
                results[f"window_{window}d"] = window_metrics

            except Exception as e:
                logger.error(f"Window ablation failed for {window}: {e}")
                results[f"window_{window}d"] = {"error": str(e)}

        if base_metrics is not None:
            results[f"window_{base_window}d_base"] = base_metrics

        self.results["temporal_window_ablation"] = results
        return results

    def run_model_component_ablation(
        self,
        evaluate_fn: callable,
        base_metrics: dict,
        components: list[str] = None,
    ) -> dict:
        """Ablate model components to measure their individual contribution.

        Components:
          - memory_module: Remove per-node memory (use zero memory)
          - temporal_encoding: Remove time encoding
          - multi_head_attention: Use single-head attention
          - graph_structure: Remove all edges (isolated nodes)
        """
        logger.info("=" * 60)
        logger.info("ABLATION: Model Components")
        logger.info("=" * 60)

        if components is None:
            components = [
                "memory_module",
                "temporal_encoding",
                "multi_head_attention",
                "graph_structure",
            ]

        results = {"full_model": base_metrics}

        for component in components:
            logger.info(f"Ablating component: {component}")
            try:
                # Create modified model
                modified_model = copy.deepcopy(self.base_model)

                if component == "memory_module":
                    # Zero out memory and disable updates
                    if hasattr(modified_model, "memory"):
                        modified_model.memory.reset_memory()
                        # Replace update with no-op
                        original_update = modified_model.memory.update_memory
                        modified_model.memory.update_memory = (
                            lambda *args, **kwargs: None
                        )

                elif component == "temporal_encoding":
                    # Replace time encoder with zeros
                    if hasattr(modified_model, "time_encoder"):
                        modified_model.time_encoder = _ZeroTimeEncoder(
                            modified_model.time_encoder.dim
                        )

                elif component == "graph_structure":
                    # Will be handled by passing empty edge dicts
                    pass

                comp_metrics = evaluate_fn(
                    modified_model, ablate_component=component
                )

                delta = {}
                for hk in base_metrics:
                    if hk in comp_metrics:
                        delta[hk] = {}
                        for mk in base_metrics[hk]:
                            bv = base_metrics[hk][mk]
                            av = comp_metrics[hk][mk]
                            if isinstance(bv, (int, float)):
                                delta[hk][mk] = bv - av

                results[f"without_{component}"] = {
                    "metrics": comp_metrics,
                    "delta": delta,
                }

            except Exception as e:
                logger.error(f"Component ablation failed for {component}: {e}")
                results[f"without_{component}"] = {"error": str(e)}

        self.results["model_component_ablation"] = results
        return results

    def generate_ablation_table(self) -> str:
        """Generate a formatted LaTeX table of ablation results."""
        lines = []
        lines.append("\\begin{table}[htbp]")
        lines.append("\\centering")
        lines.append("\\caption{Ablation Study Results}")
        lines.append("\\label{tab:ablation}")

        horizons = [f"cascade_{h}h" for h in [24, 72, 168, 720]]
        horizon_labels = ["1d", "3d", "7d", "30d"]

        # Header
        cols = "l" + "c" * len(horizons)
        lines.append(f"\\begin{{tabular}}{{{cols}}}")
        lines.append("\\toprule")
        header = "Configuration & " + " & ".join(
            [f"AUROC ({h})" for h in horizon_labels]
        )
        lines.append(header + " \\\\")
        lines.append("\\midrule")

        # Full model
        lines.append("\\textbf{Full Model (TGN)} & " + " & ".join(
            ["--"] * len(horizons)
        ) + " \\\\")
        lines.append("\\midrule")

        for ablation_type, type_results in self.results.items():
            lines.append(f"\\multicolumn{{{len(horizons) + 1}}}{{l}}"
                         f"{{\\textit{{{ablation_type.replace('_', ' ').title()}}}}} \\\\")

            for config_name, config_data in type_results.items():
                if config_name == "full_model":
                    continue
                if isinstance(config_data, dict) and "delta" in config_data:
                    values = []
                    for hk in horizons:
                        d = config_data["delta"].get(hk, {})
                        auroc_drop = d.get("auroc", 0)
                        values.append(f"{auroc_drop:+.4f}")

                    clean_name = config_name.replace("without_", "w/o ").replace("_", " ")
                    lines.append(
                        f"  {clean_name} & " + " & ".join(values) + " \\\\"
                    )

            lines.append("\\midrule")

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        table = "\n".join(lines)

        # Save
        with open(self.output_dir / "ablation_table.tex", "w") as f:
            f.write(table)

        return table

    def save_results(self):
        """Save all ablation results to JSON."""

        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not serializable: {type(obj)}")

        with open(self.output_dir / "ablation_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=convert)
        logger.info(f"Ablation results saved to {self.output_dir}")


class _ZeroTimeEncoder(torch.nn.Module):
    """Dummy time encoder that always returns zeros (for ablation)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            return torch.zeros(t.size(0), self.dim, device=t.device)
        return torch.zeros(t.size(0), self.dim, device=t.device)
