"""Statistical significance tests, permutation testing, and baseline comparisons for financial ML models."""

import logging
from typing import Any, Dict, Optional
import numpy as np
from scipy import stats
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


class StatisticalComparator:
    """Evaluates whether a model possesses genuine statistical advantage over random guessing and baselines."""

    @staticmethod
    def compute_information_coefficient(y_proba: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """Compute the Spearman rank correlation (Information Coefficient) between forecasts and realized returns.

        Args:
            y_proba: Model predicted probability for positive class.
            returns: Actual realized price returns over horizon.

        Returns:
            Dictionary with 'ic' (correlation coefficient) and 'ic_pvalue'.
        """
        if len(y_proba) < 5 or np.all(y_proba == y_proba[0]) or np.all(returns == returns[0]):
            return {"ic": 0.0, "ic_pvalue": 1.0}

        corr, pval = stats.spearmanr(y_proba, returns)
        if np.isnan(corr):
            corr, pval = 0.0, 1.0

        return {"ic": float(corr), "ic_pvalue": float(pval)}

    @staticmethod
    def compute_brier_skill_score(y_true: np.ndarray, y_proba: np.ndarray) -> float:
        """Compute Brier Skill Score (BSS) against climatology baseline (mean event frequency).

        Formula:
            BSS = 1 - (BS_model / BS_reference)
            Where BS_ref = p_base * (1 - p_base)

        Returns:
            BSS score: > 0 means better than base rate, < 0 means worse than base rate.
        """
        if len(y_true) == 0:
            return 0.0

        bs_model = float(brier_score_loss(y_true, y_proba))
        base_rate = float(np.mean(y_true))
        bs_ref = base_rate * (1.0 - base_rate)

        if bs_ref <= 1e-7:
            return 0.0  # Perfect certainty in reference, avoid div zero

        bss = 1.0 - (bs_model / bs_ref)
        return float(bss)

    @staticmethod
    def permutation_test_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_permutations: int = 1000,
        seed: int = 42,
    ) -> Dict[str, float]:
        """Execute Monte Carlo permutation test to compute empirical p-value for accuracy against random chance.

        Null Hypothesis (H0): Model predictions have no relationship with the true labels.

        Args:
            y_true: Actual binary labels.
            y_pred: Model predicted binary labels.
            n_permutations: Number of random label shuffle iterations.
            seed: Random seed.

        Returns:
            Dictionary with 'observed_accuracy', 'null_mean_accuracy', 'null_std_accuracy', 'p_value'.
        """
        n = len(y_true)
        if n == 0:
            return {"observed_accuracy": 0.0, "p_value": 1.0}

        obs_acc = float(np.mean(y_true == y_pred))

        rng = np.random.default_rng(seed)
        null_accuracies = np.zeros(n_permutations)

        y_shuffled = y_true.copy()
        for i in range(n_permutations):
            rng.shuffle(y_shuffled)
            null_accuracies[i] = np.mean(y_shuffled == y_pred)

        # Empirical p-value with Laplace correction
        extreme_count = np.sum(null_accuracies >= obs_acc)
        p_val = float((extreme_count + 1) / (n_permutations + 1))

        return {
            "observed_accuracy": obs_acc,
            "null_mean_accuracy": float(np.mean(null_accuracies)),
            "null_std_accuracy": float(np.std(null_accuracies)),
            "p_value": p_val,
        }

    @classmethod
    def evaluate_statistical_advantage(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        returns: Optional[np.ndarray] = None,
        n_permutations: int = 1000,
    ) -> Dict[str, Any]:
        """Run all statistical tests and determine whether the model holds confirmed predictive advantage."""
        perm_results = cls.permutation_test_accuracy(y_true, y_pred, n_permutations=n_permutations)
        bss = cls.compute_brier_skill_score(y_true, y_proba)

        ic_results = {"ic": 0.0, "ic_pvalue": 1.0}
        if returns is not None and len(returns) == len(y_proba):
            ic_results = cls.compute_information_coefficient(y_proba, returns)

        # Statistical advantage criteria:
        # 1. p-value < 0.05 (significantly better than random shuffling) OR
        # 2. BSS > 0 (better calibrated than baseline rate)
        has_advantage = bool(perm_results["p_value"] < 0.05 or bss > 0.0)

        return {
            "permutation_test": perm_results,
            "brier_skill_score": bss,
            "information_coefficient": ic_results["ic"],
            "ic_pvalue": ic_results["ic_pvalue"],
            "has_statistical_advantage": has_advantage,
        }
