import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats


BASE = Path("result/figure8_degree_maxq")


def ols_quadratic(x, y):
    design = np.column_stack([np.ones_like(x), x, x * x])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    fitted = design @ beta
    resid = y - fitted
    n = len(y)
    p = design.shape[1]
    df = n - p
    sse = float((resid ** 2).sum())
    tss = float(((y - y.mean()) ** 2).sum())
    sigma2 = sse / df
    cov = sigma2 * np.linalg.inv(design.T @ design)
    se = np.sqrt(np.diag(cov))
    t_stat = beta / se
    p_value = 2 * stats.t.sf(np.abs(t_stat), df)
    r2 = 1 - sse / tss if tss > 0 else np.nan
    return beta, se, t_stat, p_value, r2, df


def mean_ci95(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    half_width = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n) if n > 1 else 0.0
    return mean, mean - half_width, mean + half_width, sd


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    files = sorted((BASE / "csv").glob("*_degree_maxq.csv"))
    if not files:
        raise SystemExit("No degree_maxq CSV files found")

    summary_rows = []
    regression_rows = []
    group_rows = []

    for path in files:
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        dataset = rows[0]["dataset"]
        model = rows[0]["model"]
        degree = np.array([float(row["degree"]) for row in rows], dtype=float)
        max_q = np.array([float(row["max_q"]) for row in rows], dtype=float)
        n = len(max_q)

        rho, spearman_p = stats.spearmanr(degree, max_q)
        beta, se, t_stat, p_value, r2, df_resid = ols_quadratic(degree, max_q)

        # Equal-size tertiles by degree rank. This avoids qcut collapsing because
        # graph degrees are highly discrete and tied.
        order = np.argsort(degree, kind="mergesort")
        group = np.empty(n, dtype=object)
        first = n // 3
        second = 2 * n // 3
        group[order[:first]] = "low"
        group[order[first:second]] = "medium"
        group[order[second:]] = "high"
        anova_values = [max_q[group == name] for name in ("low", "medium", "high")]
        anova_f, anova_p = stats.f_oneway(*anova_values)

        summary_rows.append({
            "dataset": dataset,
            "model": model,
            "num_nodes": int(n),
            "degree_min": int(degree.min()),
            "degree_median": float(np.median(degree)),
            "degree_max": int(degree.max()),
            "maxq_mean": float(max_q.mean()),
            "maxq_median": float(np.median(max_q)),
            "maxq_max": int(max_q.max()),
            "spearman_rho": float(rho),
            "spearman_p": float(spearman_p),
            "quad_degree2_coef": float(beta[2]),
            "quad_degree2_p": float(p_value[2]),
            "quad_r2": float(r2),
            "anova_F": float(anova_f),
            "anova_p": float(anova_p),
        })

        for term, coef, err, t_val, p_val in zip(
            ("intercept", "degree", "degree2"), beta, se, t_stat, p_value
        ):
            regression_rows.append({
                "dataset": dataset,
                "model": model,
                "term": term,
                "coef": float(coef),
                "std_err": float(err),
                "t": float(t_val),
                "p_value": float(p_val),
                "r2": float(r2),
                "df_resid": int(df_resid),
            })

        for name in ("low", "medium", "high"):
            mask = group == name
            mean, ci_low, ci_high, sd = mean_ci95(max_q[mask])
            group_rows.append({
                "dataset": dataset,
                "model": model,
                "group": name,
                "num_nodes": int(mask.sum()),
                "degree_min": int(degree[mask].min()),
                "degree_max": int(degree[mask].max()),
                "maxq_mean": mean,
                "maxq_ci95_low": ci_low,
                "maxq_ci95_high": ci_high,
                "maxq_sd": sd,
                "anova_F": float(anova_f),
                "anova_p": float(anova_p),
            })

    write_csv(BASE / "degree_maxq_full_stat_summary.csv", summary_rows)
    write_csv(BASE / "degree_maxq_quadratic_regression.csv", regression_rows)
    write_csv(BASE / "degree_maxq_group_anova_ci.csv", group_rows)

    print("Wrote:")
    print(BASE / "degree_maxq_full_stat_summary.csv")
    print(BASE / "degree_maxq_quadratic_regression.csv")
    print(BASE / "degree_maxq_group_anova_ci.csv")
    print()
    for row in summary_rows:
        print(row)


if __name__ == "__main__":
    main()
