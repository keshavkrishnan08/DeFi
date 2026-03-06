"""
Statistical tests for model comparison.

Implements:
  - Diebold-Mariano test (forecast comparison)
  - McNemar's test (classification comparison)
  - Paired t-test (metric comparison across folds)
  - Wilcoxon signed-rank test (non-parametric paired comparison)
  - Bootstrap confidence intervals
  - Bonferroni correction for multiple comparisons
"""

import numpy as np
from scipy import stats
from typing import Optional
from loguru import logger


class StatisticalTestSuite:
    """Suite of statistical tests for rigorous model comparison."""

    def __init__(
        self,
        confidence_level: float = 0.95,
        bootstrap_iterations: int = 10000,
    ):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.bootstrap_iterations = bootstrap_iterations

    def diebold_mariano_test(
        self,
        y_true: np.ndarray,
        pred_1: np.ndarray,
        pred_2: np.ndarray,
        loss_fn: str = "squared",
        h: int = 1,
    ) -> dict[str, float]:
        """Diebold-Mariano test for comparing forecast accuracy.

        Tests H0: E[d_t] = 0 where d_t = L(e1_t) - L(e2_t).
        Rejection means model 1 and model 2 have significantly different
        forecast accuracy.

        Args:
            y_true: True values [n].
            pred_1: Predictions from model 1 [n].
            pred_2: Predictions from model 2 [n].
            loss_fn: "squared" or "absolute".
            h: Forecast horizon for HAC correction.

        Returns:
            Dict with test_statistic, p_value, significant, preferred_model.
        """
        e1 = y_true - pred_1
        e2 = y_true - pred_2

        if loss_fn == "squared":
            d = e1 ** 2 - e2 ** 2
        elif loss_fn == "absolute":
            d = np.abs(e1) - np.abs(e2)
        else:
            d = e1 ** 2 - e2 ** 2

        n = len(d)
        d_mean = d.mean()

        # HAC (Heteroskedasticity and Autocorrelation Consistent) variance
        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0
        for k in range(1, h):
            gamma_k = np.cov(d[k:], d[:-k])[0, 1] if len(d) > k else 0
            gamma_sum += 2 * gamma_k

        var_d = (gamma_0 + gamma_sum) / n

        if var_d <= 0:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "preferred_model": "neither",
            }

        dm_stat = d_mean / np.sqrt(var_d)
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        preferred = "neither"
        if p_value < self.alpha:
            preferred = "model_1" if d_mean < 0 else "model_2"

        return {
            "test_statistic": float(dm_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "preferred_model": preferred,
            "mean_loss_diff": float(d_mean),
        }

    def mcnemar_test(
        self,
        y_true: np.ndarray,
        pred_1: np.ndarray,
        pred_2: np.ndarray,
        threshold: float = 0.5,
    ) -> dict[str, float]:
        """McNemar's test for comparing two classifiers.

        Tests whether two classifiers have the same error rate.

        Args:
            y_true: True binary labels.
            pred_1: Probabilities or binary predictions from model 1.
            pred_2: Probabilities or binary predictions from model 2.
            threshold: Classification threshold.
        """
        c1 = (np.array(pred_1) >= threshold).astype(int)
        c2 = (np.array(pred_2) >= threshold).astype(int)
        y = np.array(y_true).astype(int)

        # Correct/incorrect for each model
        correct_1 = (c1 == y)
        correct_2 = (c2 == y)

        # Contingency table
        # b: model 1 correct, model 2 incorrect
        # c: model 1 incorrect, model 2 correct
        b = np.sum(correct_1 & ~correct_2)
        c = np.sum(~correct_1 & correct_2)

        # McNemar's test with continuity correction
        if b + c == 0:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "b": int(b),
                "c": int(c),
            }

        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return {
            "test_statistic": float(chi2),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "b": int(b),
            "c": int(c),
            "preferred_model": (
                "model_1" if b > c else "model_2" if c > b else "neither"
            ),
        }

    def paired_ttest(
        self,
        scores_1: np.ndarray,
        scores_2: np.ndarray,
    ) -> dict[str, float]:
        """Paired t-test comparing metric scores across CV folds.

        Args:
            scores_1: Metric values from model 1 across folds.
            scores_2: Metric values from model 2 across folds.
        """
        scores_1 = np.array(scores_1)
        scores_2 = np.array(scores_2)

        if len(scores_1) < 2:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "mean_diff": 0.0,
            }

        t_stat, p_value = stats.ttest_rel(scores_1, scores_2)

        return {
            "test_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "mean_diff": float(np.mean(scores_1 - scores_2)),
            "std_diff": float(np.std(scores_1 - scores_2)),
            "preferred_model": (
                "model_1" if np.mean(scores_1) > np.mean(scores_2)
                else "model_2"
            ),
        }

    def wilcoxon_signed_rank(
        self,
        scores_1: np.ndarray,
        scores_2: np.ndarray,
    ) -> dict[str, float]:
        """Wilcoxon signed-rank test (non-parametric paired comparison).

        More robust than paired t-test when normality assumption is violated.
        """
        scores_1 = np.array(scores_1)
        scores_2 = np.array(scores_2)

        if len(scores_1) < 6:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "note": "Too few samples for Wilcoxon test",
            }

        diff = scores_1 - scores_2
        if np.all(diff == 0):
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
            }

        try:
            stat, p_value = stats.wilcoxon(scores_1, scores_2)
        except ValueError:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
            }

        return {
            "test_statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
            "median_diff": float(np.median(diff)),
        }

    def bootstrap_confidence_interval(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metric_fn: callable,
        n_bootstrap: int = None,
    ) -> dict[str, float]:
        """Bootstrap confidence interval for any metric.

        Args:
            y_true: True labels.
            y_prob: Predicted probabilities.
            metric_fn: Function(y_true, y_prob) -> float.
            n_bootstrap: Number of bootstrap samples.
        """
        if n_bootstrap is None:
            n_bootstrap = self.bootstrap_iterations

        rng = np.random.RandomState(42)
        n = len(y_true)
        scores = []

        for _ in range(n_bootstrap):
            idx = rng.randint(0, n, size=n)
            boot_true = y_true[idx]
            boot_prob = y_prob[idx]
            try:
                score = metric_fn(boot_true, boot_prob)
                if np.isfinite(score):
                    scores.append(score)
            except Exception:
                continue

        if not scores:
            return {
                "mean": 0.0, "std": 0.0,
                "ci_lower": 0.0, "ci_upper": 0.0,
            }

        scores = np.array(scores)
        ci_lower = np.percentile(scores, (1 - self.confidence_level) / 2 * 100)
        ci_upper = np.percentile(scores, (1 + self.confidence_level) / 2 * 100)

        return {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
        }

    def bonferroni_correction(
        self, p_values: list[float], alpha: float = None
    ) -> dict[str, list]:
        """Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values from multiple tests.
            alpha: Significance level.

        Returns:
            Dict with corrected_alpha, adjusted_p_values, significant flags.
        """
        if alpha is None:
            alpha = self.alpha

        m = len(p_values)
        corrected_alpha = alpha / m
        adjusted = [min(p * m, 1.0) for p in p_values]

        return {
            "corrected_alpha": corrected_alpha,
            "adjusted_p_values": adjusted,
            "significant": [p < corrected_alpha for p in p_values],
            "num_comparisons": m,
        }

    def run_full_comparison(
        self,
        y_true: np.ndarray,
        model_predictions: dict[str, np.ndarray],
        fold_scores: Optional[dict[str, list]] = None,
        reference_model: str = "TGN",
    ) -> dict:
        """Run all statistical tests comparing models against reference.

        Args:
            y_true: Ground truth.
            model_predictions: Dict of model_name -> predictions.
            fold_scores: Optional dict of model_name -> list of fold scores.
            reference_model: Name of the reference model.

        Returns:
            Comprehensive comparison results.
        """
        results = {}
        ref_preds = model_predictions.get(reference_model)

        if ref_preds is None:
            logger.error(f"Reference model '{reference_model}' not found")
            return results

        all_p_values = []

        for model_name, preds in model_predictions.items():
            if model_name == reference_model:
                continue

            comparison = {"vs": f"{reference_model} vs {model_name}"}

            # Diebold-Mariano
            comparison["diebold_mariano"] = self.diebold_mariano_test(
                y_true, ref_preds, preds
            )
            all_p_values.append(comparison["diebold_mariano"]["p_value"])

            # McNemar's
            comparison["mcnemar"] = self.mcnemar_test(
                y_true, ref_preds, preds
            )
            all_p_values.append(comparison["mcnemar"]["p_value"])

            # Paired t-test on fold scores
            if fold_scores and model_name in fold_scores:
                ref_scores = fold_scores.get(reference_model, [])
                mod_scores = fold_scores.get(model_name, [])
                if ref_scores and mod_scores:
                    comparison["paired_ttest"] = self.paired_ttest(
                        np.array(ref_scores), np.array(mod_scores)
                    )
                    comparison["wilcoxon"] = self.wilcoxon_signed_rank(
                        np.array(ref_scores), np.array(mod_scores)
                    )
                    all_p_values.append(
                        comparison["paired_ttest"]["p_value"]
                    )

            results[model_name] = comparison

        # Bonferroni correction
        if all_p_values:
            results["bonferroni"] = self.bonferroni_correction(all_p_values)

        return results
