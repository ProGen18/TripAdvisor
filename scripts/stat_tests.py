"""Wrappers de tests statistiques — retournent des dicts prets pour l'UI.

Chaque fonction retourne un dict avec test_name, statistic, p_value, effect_size,
et interpretation en francais.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _format_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.4f}"
    elif p < 0.05:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.3f}"


def _interpret_p(p: float, alpha: float = 0.05) -> str:
    if p < 0.001:
        return "Hautement significatif"
    elif p < 0.01:
        return "Tres significatif"
    elif p < 0.05:
        return "Significatif"
    else:
        return "Non significatif"


def chi2_contingency(observed: pd.DataFrame | np.ndarray) -> dict:
    """Test du Khi² d'independance sur une table de contingence.

    Args:
        observed: Table de contingence (DataFrame ou array 2D)

    Returns:
        dict avec chi2_stat, p_value, dof, cramers_v, interpretation
    """
    if isinstance(observed, pd.DataFrame):
        index_name = observed.index.name
        columns_name = observed.columns.name
        labels = {
            "rows": observed.index.tolist(),
            "cols": observed.columns.tolist(),
            "row_name": index_name or "lignes",
            "col_name": columns_name or "colonnes",
        }
        values = observed.values
    else:
        values = observed
        labels = None

    chi2, p, dof, expected = scipy_stats.chi2_contingency(values)

    n = values.sum()
    min_dim = min(values.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0.0

    interpretation = _interpret_p(p)
    if cramers_v < 0.1:
        effect_label = "tres faible"
    elif cramers_v < 0.2:
        effect_label = "faible"
    elif cramers_v < 0.4:
        effect_label = "moderee"
    elif cramers_v < 0.6:
        effect_label = "forte"
    else:
        effect_label = "tres forte"

    result = {
        "test_name": "Khi² d'independance",
        "statistic": float(chi2),
        "p_value": float(p),
        "p_formatted": _format_p(p),
        "dof": int(dof),
        "effect_size": float(cramers_v),
        "effect_name": "Cramer's V",
        "effect_label": effect_label,
        "interpretation": interpretation,
        "n_samples": int(n),
        "expected": expected.tolist() if labels is None else None,
    }
    if labels:
        result["labels"] = labels

    return result


def anova(*groups: np.ndarray, group_names: list[str] | None = None) -> dict:
    """ANOVA a un facteur.

    Args:
        *groups: Tableaux de valeurs par groupe
        group_names: Noms des groupes (optionnel)

    Returns:
        dict avec f_stat, p_value, eta_squared, interpretation
    """
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {"test_name": "ANOVA", "error": "Moins de 2 groupes valides"}

    f_stat, p = scipy_stats.f_oneway(*groups)

    # Eta-carré (taille d'effet)
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((all_vals - grand_mean) ** 2).sum()
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    if eta_sq < 0.01:
        effect_label = "tres faible"
    elif eta_sq < 0.06:
        effect_label = "faible"
    elif eta_sq < 0.14:
        effect_label = "moderee"
    else:
        effect_label = "forte"

    result = {
        "test_name": "ANOVA a un facteur",
        "statistic": float(f_stat),
        "p_value": float(p),
        "p_formatted": _format_p(p),
        "effect_size": float(eta_sq),
        "effect_name": "Eta-carre",
        "effect_label": effect_label,
        "interpretation": _interpret_p(p),
        "n_groups": len(groups),
        "n_total": sum(len(g) for g in groups),
        "group_means": [float(g.mean()) for g in groups],
        "group_stds": [float(g.std()) for g in groups],
    }
    if group_names:
        result["group_names"] = group_names

    return result


def kruskal_wallis(*groups: np.ndarray, group_names: list[str] | None = None) -> dict:
    """Test de Kruskal-Wallis (alternative non-parametrique a l'ANOVA).

    Returns:
        dict avec h_stat, p_value, epsilon_squared, interpretation
    """
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return {"test_name": "Kruskal-Wallis", "error": "Moins de 2 groupes valides"}

    h_stat, p = scipy_stats.kruskal(*groups)

    # Epsilon-carré (taille d'effet pour KW)
    N = sum(len(g) for g in groups)
    k = len(groups)
    epsilon_sq = (h_stat - k + 1) / (N - k) if N > k else 0.0
    epsilon_sq = max(0.0, min(1.0, epsilon_sq))

    if epsilon_sq < 0.01:
        effect_label = "tres faible"
    elif epsilon_sq < 0.06:
        effect_label = "faible"
    elif epsilon_sq < 0.14:
        effect_label = "moderee"
    else:
        effect_label = "forte"

    result = {
        "test_name": "Kruskal-Wallis",
        "statistic": float(h_stat),
        "p_value": float(p),
        "p_formatted": _format_p(p),
        "effect_size": float(epsilon_sq),
        "effect_name": "Epsilon-carre",
        "effect_label": effect_label,
        "interpretation": _interpret_p(p),
        "n_groups": k,
        "n_total": int(N),
        "group_medians": [float(np.median(g)) for g in groups],
        "group_iqrs": [
            float(np.percentile(g, 75) - np.percentile(g, 25)) for g in groups
        ],
    }
    if group_names:
        result["group_names"] = group_names

    return result


def spearman_r(x: np.ndarray, y: np.ndarray) -> dict:
    """Correlation de Spearman avec p-value.

    Returns:
        dict avec rho, p_value, interpretation
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean, y_clean = x[mask], y[mask]

    if len(x_clean) < 3:
        return {"test_name": "Spearman", "error": "Moins de 3 observations valides"}

    rho, p = scipy_stats.spearmanr(x_clean, y_clean)

    abs_rho = abs(rho)
    if abs_rho < 0.1:
        effect_label = "tres faible"
    elif abs_rho < 0.3:
        effect_label = "faible"
    elif abs_rho < 0.5:
        effect_label = "moderee"
    elif abs_rho < 0.7:
        effect_label = "forte"
    else:
        effect_label = "tres forte"

    direction = "positive" if rho > 0 else "negative"

    return {
        "test_name": "Correlation de Spearman",
        "statistic": float(rho),
        "p_value": float(p),
        "p_formatted": _format_p(p),
        "effect_size": abs_rho,
        "effect_name": "|rho|",
        "effect_label": effect_label,
        "direction": direction,
        "interpretation": _interpret_p(p),
        "n_samples": len(x_clean),
    }


def welch_ttest(group_a: np.ndarray, group_b: np.ndarray,
                name_a: str = "Groupe A", name_b: str = "Groupe B") -> dict:
    """Test t de Welch (variances inegales).

    Returns:
        dict avec t_stat, p_value, cohens_d, interpretation
    """
    a = group_a[~np.isnan(group_a)]
    b = group_b[~np.isnan(group_b)]

    if len(a) < 2 or len(b) < 2:
        return {"test_name": "Welch t-test", "error": "Moins de 2 observations"}

    t_stat, p = scipy_stats.ttest_ind(a, b, equal_var=False)

    # Cohen's d
    n_a, n_b = len(a), len(b)
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    cohens_d = (a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0.0
    abs_d = abs(cohens_d)

    if abs_d < 0.2:
        effect_label = "tres faible"
    elif abs_d < 0.5:
        effect_label = "faible"
    elif abs_d < 0.8:
        effect_label = "moderee"
    elif abs_d < 1.2:
        effect_label = "forte"
    else:
        effect_label = "tres forte"

    return {
        "test_name": "Welch t-test",
        "statistic": float(t_stat),
        "p_value": float(p),
        "p_formatted": _format_p(p),
        "effect_size": abs_d,
        "effect_name": "Cohen's d",
        "effect_label": effect_label,
        "interpretation": _interpret_p(p),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "std_a": float(a.std()),
        "std_b": float(b.std()),
        "n_a": n_a,
        "n_b": n_b,
        "name_a": name_a,
        "name_b": name_b,
    }


def format_stat_for_ui(stat_result: dict) -> str:
    """Formate un resultat de test stat pour affichage Streamlit."""
    if "error" in stat_result:
        return f"*Test indisponible : {stat_result['error']}*"

    test = stat_result["test_name"]
    if "p_formatted" in stat_result:
        p_str = stat_result["p_formatted"]
    else:
        p_str = f"p = {stat_result['p_value']:.4f}"

    effect = stat_result.get("effect_name", "")
    effect_val = stat_result.get("effect_size", 0)
    effect_label = stat_result.get("effect_label", "")
    interp = stat_result.get("interpretation", "")

    lines = [
        f"**{test}** : {p_str} — {interp}",
        f"Taille d'effet ({effect}) = {effect_val:.4f} ({effect_label})",
    ]

    if "n_samples" in stat_result:
        lines.append(f"N = {stat_result['n_samples']}")
    elif "n_total" in stat_result:
        lines.append(f"N = {stat_result['n_total']}")

    return "  \n".join(lines)
