"""
Publication-quality visualization for IEEE TCSS paper.

Generates all figures needed for the paper:
  1. ROC and PR curves (multi-horizon)
  2. Training loss curves
  3. Model comparison bar charts
  4. Ablation heatmap
  5. Case study timeline (cascade events)
  6. Composability graph visualization
  7. Feature importance analysis
  8. Confidence interval plots
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from loguru import logger


# IEEE-style formatting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "TGN": "#2196F3",
    "Static GNN": "#FF9800",
    "LSTM": "#4CAF50",
    "XGBoost": "#F44336",
    "SIR": "#9C27B0",
    "Centrality": "#795548",
}

HORIZON_COLORS = {
    "1d": "#E53935",
    "3d": "#FB8C00",
    "7d": "#43A047",
    "30d": "#1E88E5",
}


class PaperVisualizer:
    """Generates all publication figures for the IEEE TCSS paper."""

    def __init__(self, output_dir: str = "outputs/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        model_predictions: dict[str, np.ndarray],
        horizon_label: str = "24h",
        filename: str = "roc_curves.pdf",
    ):
        """Plot ROC curves comparing all models."""
        fig, ax = plt.subplots(figsize=(4.5, 4))

        for model_name, y_prob in model_predictions.items():
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auroc = auc(fpr, tpr)
            color = COLORS.get(model_name, "#666666")
            linewidth = 2.5 if model_name == "TGN" else 1.5
            linestyle = "-" if model_name == "TGN" else "--"
            ax.plot(
                fpr, tpr,
                label=f"{model_name} (AUC={auroc:.3f})",
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )

        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curves — {horizon_label} Horizon")
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved ROC curves to {filename}")

    def plot_pr_curves(
        self,
        y_true: np.ndarray,
        model_predictions: dict[str, np.ndarray],
        horizon_label: str = "24h",
        filename: str = "pr_curves.pdf",
    ):
        """Plot Precision-Recall curves comparing all models."""
        fig, ax = plt.subplots(figsize=(4.5, 4))

        baseline_pr = y_true.mean()

        for model_name, y_prob in model_predictions.items():
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = auc(rec, prec)
            color = COLORS.get(model_name, "#666666")
            linewidth = 2.5 if model_name == "TGN" else 1.5
            linestyle = "-" if model_name == "TGN" else "--"
            ax.plot(
                rec, prec,
                label=f"{model_name} (AP={ap:.3f})",
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
            )

        ax.axhline(baseline_pr, color="gray", linestyle=":", alpha=0.5,
                    label=f"Random ({baseline_pr:.3f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curves — {horizon_label} Horizon")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])

        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved PR curves to {filename}")

    def plot_multi_horizon_comparison(
        self,
        all_metrics: dict[str, dict[str, dict]],
        metric_name: str = "auroc",
        filename: str = "multi_horizon_comparison.pdf",
    ):
        """Bar chart comparing models across all prediction horizons.

        Args:
            all_metrics: model_name -> horizon_key -> metrics_dict.
        """
        fig, ax = plt.subplots(figsize=(7, 4))

        horizons = ["cascade_24h", "cascade_72h", "cascade_168h", "cascade_720h"]
        horizon_labels = ["1d", "3d", "7d", "30d"]
        models = list(all_metrics.keys())
        n_models = len(models)
        n_horizons = len(horizons)

        bar_width = 0.8 / n_models
        x = np.arange(n_horizons)

        for i, model in enumerate(models):
            values = []
            for h in horizons:
                m = all_metrics.get(model, {}).get(h, {})
                values.append(m.get(metric_name, 0))

            color = COLORS.get(model, "#666666")
            offset = (i - n_models / 2 + 0.5) * bar_width
            bars = ax.bar(
                x + offset, values, bar_width,
                label=model, color=color, alpha=0.85,
                edgecolor="white", linewidth=0.5,
            )

            # Add value labels on top
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(horizon_labels)
        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel(metric_name.upper())
        ax.set_title(f"Model Comparison Across Prediction Horizons")
        ax.legend(loc="upper right", ncol=2, framealpha=0.9)
        ax.set_ylim([0, 1.1])

        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved multi-horizon comparison to {filename}")

    def plot_training_curves(
        self,
        train_losses: list[float],
        val_losses: list[float],
        filename: str = "training_curves.pdf",
    ):
        """Plot training and validation loss curves."""
        fig, ax = plt.subplots(figsize=(5, 3.5))

        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Train", color=COLORS["TGN"],
                linewidth=1.5)
        ax.plot(epochs, val_losses, label="Validation", color=COLORS["XGBoost"],
                linewidth=1.5)

        best_epoch = np.argmin(val_losses) + 1
        ax.axvline(best_epoch, color="gray", linestyle="--", alpha=0.5,
                    label=f"Best epoch ({best_epoch})")

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Convergence")
        ax.legend(framealpha=0.9)

        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved training curves to {filename}")

    def plot_ablation_heatmap(
        self,
        ablation_results: dict,
        filename: str = "ablation_heatmap.pdf",
    ):
        """Heatmap showing AUROC drop for each ablation."""
        fig, ax = plt.subplots(figsize=(6, 5))

        horizons = ["cascade_24h", "cascade_72h", "cascade_168h", "cascade_720h"]
        horizon_labels = ["1d", "3d", "7d", "30d"]

        configs = []
        data = []

        for config_name, config_data in ablation_results.items():
            if config_name == "full_model":
                continue
            if isinstance(config_data, dict) and "delta" in config_data:
                row = []
                for hk in horizons:
                    d = config_data.get("delta", {}).get(hk, {})
                    row.append(d.get("auroc", 0))
                data.append(row)
                configs.append(
                    config_name.replace("without_", "w/o ")
                    .replace("_", " ").title()
                )

        if not data:
            logger.warning("No ablation data to plot")
            return

        data = np.array(data)
        df = pd.DataFrame(data, index=configs, columns=horizon_labels)

        sns.heatmap(
            df, annot=True, fmt=".4f", cmap="RdYlBu_r",
            center=0, ax=ax, linewidths=0.5,
            cbar_kws={"label": "AUROC Drop (positive = component helps)"},
        )
        ax.set_title("Feature/Component Ablation Impact")
        ax.set_xlabel("Prediction Horizon")

        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved ablation heatmap to {filename}")

    def plot_cascade_case_study(
        self,
        dates: np.ndarray,
        tvl_values: np.ndarray,
        predictions: np.ndarray,
        event_name: str,
        event_start: str,
        event_end: str,
        filename: str = "case_study.pdf",
    ):
        """Timeline plot for a specific cascade event case study."""
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(7, 5), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )

        # TVL timeline
        ax1.plot(dates, tvl_values / 1e9, color=COLORS["TGN"], linewidth=1.5)
        ax1.axvspan(
            pd.Timestamp(event_start), pd.Timestamp(event_end),
            alpha=0.15, color="red", label="Cascade Event",
        )
        ax1.set_ylabel("Total TVL ($ Billion)")
        ax1.set_title(f"Case Study: {event_name}")
        ax1.legend(loc="upper right")

        # Prediction timeline
        ax2.plot(dates, predictions, color=COLORS["XGBoost"], linewidth=1.5)
        ax2.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax2.axvspan(
            pd.Timestamp(event_start), pd.Timestamp(event_end),
            alpha=0.15, color="red",
        )
        ax2.set_ylabel("Cascade Probability")
        ax2.set_xlabel("Date")
        ax2.set_ylim([0, 1])

        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved case study to {filename}")

    def plot_confidence_intervals(
        self,
        model_names: list[str],
        means: list[float],
        ci_lowers: list[float],
        ci_uppers: list[float],
        metric_name: str = "AUROC",
        filename: str = "confidence_intervals.pdf",
    ):
        """Forest plot of model performance with confidence intervals."""
        fig, ax = plt.subplots(figsize=(5, 3.5))

        y_pos = np.arange(len(model_names))
        errors = np.array(
            [[m - l for m, l in zip(means, ci_lowers)],
             [u - m for m, u in zip(means, ci_uppers)]]
        )

        colors = [COLORS.get(m, "#666666") for m in model_names]
        ax.barh(y_pos, means, xerr=errors, color=colors, alpha=0.8,
                capsize=3, edgecolor="white", linewidth=0.5)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(model_names)
        ax.set_xlabel(metric_name)
        ax.set_title(f"{metric_name} with 95% Bootstrap CIs")
        ax.invert_yaxis()

        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved CI plot to {filename}")

    def plot_composability_graph(
        self,
        adjacency: np.ndarray,
        protocol_names: list[str],
        risk_scores: Optional[np.ndarray] = None,
        filename: str = "composability_graph.pdf",
    ):
        """Visualize the DeFi composability graph."""
        import networkx as nx

        fig, ax = plt.subplots(figsize=(7, 6))
        G = nx.from_numpy_array(adjacency)

        # Layout
        pos = nx.spring_layout(G, k=2, seed=42)

        # Node sizes and colors based on risk
        if risk_scores is not None:
            node_colors = risk_scores
            cmap = plt.cm.RdYlGn_r
        else:
            node_colors = [0.5] * len(protocol_names)
            cmap = plt.cm.Blues

        # Draw edges
        edge_widths = [adjacency[u][v] * 2 for u, v in G.edges()]
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=edge_widths, alpha=0.3,
            edge_color="gray",
        )

        # Draw nodes
        nodes = nx.draw_networkx_nodes(
            G, pos, ax=ax, node_size=800,
            node_color=node_colors, cmap=cmap,
            vmin=0, vmax=1, edgecolors="black", linewidths=0.5,
        )

        # Labels
        labels = {i: name.replace("-", "\n") for i, name in enumerate(protocol_names)}
        nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

        if risk_scores is not None:
            plt.colorbar(nodes, ax=ax, label="Risk Score", shrink=0.8)

        ax.set_title("DeFi Composability Graph")
        ax.axis("off")

        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved graph visualization to {filename}")

    def plot_early_warning(
        self,
        early_warning_data: dict,
        horizons: list[int],
        filename: str = "early_warning.pdf",
    ):
        """Bar chart of average lead times per horizon and threshold."""
        fig, ax = plt.subplots(figsize=(6, 4))
        avg_lead = early_warning_data.get("average_lead_times", {})
        thresholds = early_warning_data.get("thresholds", [0.3, 0.5, 0.7])
        horizon_labels = {24: "1d", 72: "3d", 168: "7d", 720: "30d"}

        x = np.arange(len(horizons))
        bar_w = 0.8 / len(thresholds)
        colors = ["#4CAF50", "#FF9800", "#F44336"]

        for i, thr in enumerate(thresholds):
            vals = [avg_lead.get(thr, {}).get(f"cascade_{h}h", 0) / 24 for h in horizons]
            offset = (i - len(thresholds) / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=f"Threshold={thr}",
                         color=colors[i % len(colors)], alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.0f}d", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([horizon_labels.get(h, f"{h}h") for h in horizons])
        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel("Average Lead Time (days)")
        ax.set_title("Early Warning Lead Time Analysis")
        ax.legend(framealpha=0.9)
        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved early warning plot to {filename}")

    def plot_sensitivity(
        self,
        sensitivity_data: dict,
        horizons: list[int],
        filename: str = "sensitivity_analysis.pdf",
    ):
        """Heatmap of AUROC across hyperparameter configurations."""
        fig, ax = plt.subplots(figsize=(6, 4))
        horizon_labels = {24: "1d", 72: "3d", 168: "7d", 720: "30d"}

        configs = []
        data = []
        for config_name, config_data in sensitivity_data.items():
            metrics = config_data.get("metrics", {})
            row = [metrics.get(f"cascade_{h}h", {}).get("auroc", 0) for h in horizons]
            data.append(row)
            configs.append(config_name.replace("_", " ").title())

        if not data:
            return

        df = pd.DataFrame(data, index=configs,
                          columns=[horizon_labels.get(h, f"{h}h") for h in horizons])
        sns.heatmap(df, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, vmin=0.5, vmax=1.0)
        ax.set_title("Sensitivity Analysis: AUROC by Configuration")
        ax.set_xlabel("Prediction Horizon")
        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved sensitivity plot to {filename}")

    def plot_calibration(
        self,
        calibration_data: dict,
        horizons: list[int],
        filename: str = "calibration.pdf",
    ):
        """Reliability diagram showing predicted vs actual probabilities."""
        n_horizons = len(horizons)
        fig, axes = plt.subplots(1, n_horizons, figsize=(3.5 * n_horizons, 3.5))
        if n_horizons == 1:
            axes = [axes]
        horizon_labels = {24: "1d", 72: "3d", 168: "7d", 720: "30d"}

        for idx, h in enumerate(horizons):
            ax = axes[idx]
            key = f"cascade_{h}h"
            for model_name, color in [("TGN", COLORS["TGN"]), ("XGBoost", COLORS["XGBoost"])]:
                cal = calibration_data.get(model_name, {}).get(key, {})
                if cal:
                    ax.plot(cal["bin_means"], cal["bin_true_freqs"], "o-",
                            color=color, label=f'{model_name} (ECE={cal["ece"]:.3f})',
                            markersize=4, linewidth=1.5)
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
            ax.set_xlabel("Predicted Probability")
            if idx == 0:
                ax.set_ylabel("Observed Frequency")
            ax.set_title(f"{horizon_labels.get(h, f'{h}h')}")
            ax.legend(fontsize=7, framealpha=0.9)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

        fig.suptitle("Calibration (Reliability Diagram)", fontsize=12)
        fig.tight_layout()
        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved calibration plot to {filename}")

    def plot_attention_heatmap(
        self,
        protocol_importance: dict,
        protocols: list[str],
        filename: str = "attention_importance.pdf",
    ):
        """Bar chart of protocol attention importance."""
        fig, ax = plt.subplots(figsize=(6, 4))
        sorted_items = sorted(protocol_importance.items(), key=lambda x: x[1], reverse=True)
        names = [k.replace("-", "\n") for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(names)))
        ax.barh(range(len(names)), values, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Average Attention Weight")
        ax.set_title("Protocol Importance (Attention Pooling)")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved attention importance to {filename}")

    def plot_backtest(
        self,
        backtest_data: dict,
        horizons: list[int],
        filename: str = "backtest.pdf",
    ):
        """Detection rate vs precision at different thresholds."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        horizon_labels = {24: "1d", 72: "3d", 168: "7d", 720: "30d"}
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

        for h in horizons:
            key = f"cascade_{h}h"
            det_rates = []
            precisions = []
            for thr in thresholds:
                bt_key = f"threshold_{thr}_{key}"
                bt = backtest_data.get(bt_key, {})
                det_rates.append(bt.get("detection_rate", 0))
                precisions.append(bt.get("alert_precision", 0))

            color = HORIZON_COLORS.get(horizon_labels.get(h, ""), "#666")
            label = horizon_labels.get(h, f"{h}h")
            ax1.plot(thresholds, det_rates, "o-", color=color, label=label, linewidth=1.5)
            ax2.plot(thresholds, precisions, "o-", color=color, label=label, linewidth=1.5)

        ax1.set_xlabel("Alert Threshold")
        ax1.set_ylabel("Detection Rate")
        ax1.set_title("Cascade Detection Rate")
        ax1.legend(framealpha=0.9)
        ax1.set_ylim([0, 1.05])

        ax2.set_xlabel("Alert Threshold")
        ax2.set_ylabel("Alert Precision")
        ax2.set_title("Alert Precision")
        ax2.legend(framealpha=0.9)
        ax2.set_ylim([0, 1.05])

        fig.tight_layout()
        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved backtest plot to {filename}")

    def plot_temporal_robustness(
        self,
        robustness_data: dict,
        horizons: list[int],
        filename: str = "temporal_robustness.pdf",
    ):
        """Grouped bar chart of AUROC across temporal windows."""
        fig, ax = plt.subplots(figsize=(6, 4))
        horizon_labels = {24: "1d", 72: "3d", 168: "7d", 720: "30d"}
        windows = list(robustness_data.keys())
        n_windows = len(windows)
        x = np.arange(len(horizons))
        bar_w = 0.8 / max(n_windows, 1)
        colors = ["#2196F3", "#4CAF50", "#FF9800"]

        for i, window_name in enumerate(windows):
            metrics = robustness_data[window_name]
            vals = [metrics.get(f"cascade_{h}h", {}).get("auroc", 0) for h in horizons]
            offset = (i - n_windows / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=window_name.title(),
                         color=colors[i % len(colors)], alpha=0.85)
            for bar, val in zip(bars, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([horizon_labels.get(h, f"{h}h") for h in horizons])
        ax.set_xlabel("Prediction Horizon")
        ax.set_ylabel("AUROC")
        ax.set_title("Temporal Robustness: AUROC Across Time Windows")
        ax.legend(framealpha=0.9)
        ax.set_ylim([0, 1.1])
        fig.savefig(self.output_dir / filename)
        plt.close(fig)
        logger.info(f"Saved temporal robustness plot to {filename}")

    def generate_all_paper_figures(
        self,
        results: dict,
    ):
        """Generate all figures for the paper from experiment results."""
        logger.info("Generating all paper figures")

        if "training_history" in results:
            self.plot_training_curves(
                results["training_history"]["train_losses"],
                results["training_history"]["val_losses"],
            )

        if "model_comparison" in results:
            self.plot_multi_horizon_comparison(
                results["model_comparison"],
                metric_name="auroc",
            )

        if "ablation" in results:
            for ablation_type, abl_results in results["ablation"].items():
                self.plot_ablation_heatmap(
                    abl_results,
                    filename=f"ablation_{ablation_type}.pdf",
                )

        logger.info(f"All figures saved to {self.output_dir}")
