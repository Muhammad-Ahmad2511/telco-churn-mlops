# src/monitoring/monitor.py
"""Data drift and data-quality check (spec §3.12, §3.12A).

Compares *reference* data (what the production model was trained on) with *current*
data (recent inputs) column by column:
  numeric columns     -> two-sample Kolmogorov-Smirnov test (scipy.stats.ks_2samp)
  categorical columns -> chi-square test on category counts (scipy.stats.chi2_contingency)
A column has drifted if p < p_value_threshold. The dataset has drifted (the retraining
trigger) if at least `drift_share` of the columns drifted. Thresholds: configs/monitoring.yaml.

Evidently was the first choice, but its latest release (0.7.23) fails to import with
pandas 3 (it uses a removed private pandas function). The spec allows "Evidently or an
equivalent approach"; these are the same tests Evidently uses by default on small samples.

    python -m src.monitoring.monitor --simulate none          # current = unchanged sample
    python -m src.monitoring.monitor --simulate drift         # current = SIMULATED drift
    python -m src.monitoring.monitor --current data.csv       # current = your own file
    add --exit-code to exit with 2 when dataset drift is detected (for Airflow)
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import stats

from src.features.preprocessing import load_config, load_processed, split_train_val_test
from src.serving.schema import CATEGORICAL, FEATURE_ORDER, NUMERIC

logger = logging.getLogger(__name__)

DRIFT_EXIT_CODE = 2


@dataclass
class ColumnResult:
    column: str
    kind: str          # "numeric" or "categorical"
    test: str          # "ks" or "chi2"
    statistic: float
    p_value: float
    drifted: bool
    reference_summary: str
    current_summary: str


def load_settings(path: str = "configs/monitoring.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def reference_frame(pipeline_cfg_path: str = "configs/pipeline_config.yaml") -> pd.DataFrame:
    """Train + validation rows: exactly the data the production model was fitted on."""
    cfg = load_config(pipeline_cfg_path)
    X_train, X_val, *_ = split_train_val_test(load_processed(cfg), cfg)
    return pd.concat([X_train, X_val])[list(FEATURE_ORDER)]


def holdout_frame(pipeline_cfg_path: str = "configs/pipeline_config.yaml") -> pd.DataFrame:
    """Test rows: unseen by the model, same distribution as training (a 'no drift' sample)."""
    cfg = load_config(pipeline_cfg_path)
    _, _, X_test, *_ = split_train_val_test(load_processed(cfg), cfg)
    return X_test[list(FEATURE_ORDER)]


def simulate_drift(df: pd.DataFrame, sim: dict, seed: int = 0) -> pd.DataFrame:
    """SIMULATED drift for the demo: higher prices, newer customers, shifted categories."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["MonthlyCharges"] = (out["MonthlyCharges"] * sim["monthly_charges_factor"]).round(2)
    out["tenure"] = (out["tenure"] * sim["tenure_factor"]).round().astype("int64")
    out["TotalCharges"] = (out["MonthlyCharges"] * out["tenure"]).round(2)
    for column, value in sim["categorical_shift"].items():
        moved = rng.random(len(out)) < sim["shift_share"]
        out.loc[moved, column] = value
    return out


def _summary_numeric(s: pd.Series) -> str:
    return f"mean {s.mean():.2f}, median {s.median():.2f}"


def _summary_categorical(s: pd.Series) -> str:
    top = s.value_counts(normalize=True).head(2)
    return ", ".join(f"{k} {v:.0%}" for k, v in top.items())


def column_drift(reference: pd.Series, current: pd.Series, kind: str, threshold: float) -> ColumnResult:
    if kind == "numeric":
        result = stats.ks_2samp(reference.dropna(), current.dropna())
        statistic, p_value, test = float(result.statistic), float(result.pvalue), "ks"
        ref_s, cur_s = _summary_numeric(reference), _summary_numeric(current)
    else:
        categories = sorted(set(reference.dropna()) | set(current.dropna()))
        table = np.array([
            [int((reference == c).sum()) for c in categories],
            [int((current == c).sum()) for c in categories],
        ])
        table = table[:, table.sum(axis=0) > 0]
        if table.shape[1] < 2:  # a single category in both: nothing can differ
            statistic, p_value = 0.0, 1.0
        else:
            chi2, p_value, _, _ = stats.chi2_contingency(table)
            statistic, p_value = float(chi2), float(p_value)
        test = "chi2"
        ref_s, cur_s = _summary_categorical(reference), _summary_categorical(current)
    return ColumnResult(reference.name, kind, test, statistic, p_value, p_value < threshold, ref_s, cur_s)


def data_quality(current: pd.DataFrame) -> dict:
    """Missing values, unknown categories and out-of-range numbers in the current data."""
    issues: dict[str, dict] = {}
    for column in FEATURE_ORDER:
        if column not in current:
            issues[column] = {"missing_column": True}
            continue
        col = current[column]
        found = {}
        if col.isna().any():
            found["missing_share"] = round(float(col.isna().mean()), 4)
        if column in CATEGORICAL:
            unknown = sorted(set(col.dropna()) - set(CATEGORICAL[column]))
            if unknown:
                found["unknown_categories"] = unknown
        else:
            _, low, high = NUMERIC[column]
            numeric = pd.to_numeric(col, errors="coerce")
            out_of_range = float(((numeric < low) | (numeric > high)).mean())
            if out_of_range:
                found["out_of_range_share"] = round(out_of_range, 4)
        if found:
            issues[column] = found
    return issues


def check_drift(reference: pd.DataFrame, current: pd.DataFrame, settings: dict) -> dict:
    drift_cfg = settings["drift"]
    if len(current) < drift_cfg["min_rows"]:
        raise ValueError(f"current data has {len(current)} rows; need at least {drift_cfg['min_rows']}")
    columns = [
        column_drift(reference[c], current[c], "categorical" if c in CATEGORICAL else "numeric",
                     drift_cfg["p_value_threshold"])
        for c in FEATURE_ORDER
    ]
    drifted = [c.column for c in columns if c.drifted]
    share = len(drifted) / len(columns)
    return {
        "checked_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reference_rows": len(reference),
        "current_rows": len(current),
        "p_value_threshold": drift_cfg["p_value_threshold"],
        "drift_share_threshold": drift_cfg["drift_share"],
        "drifted_columns": drifted,
        "drifted_share": round(share, 4),
        "dataset_drift": share >= drift_cfg["drift_share"],
        "data_quality_issues": data_quality(current),
        "columns": [asdict(c) for c in columns],
    }


def _plot(reference: pd.Series, current: pd.Series, kind: str) -> str:
    fig, ax = plt.subplots(figsize=(4.2, 2.4))
    if kind == "numeric":
        bins = np.histogram_bin_edges(pd.concat([reference, current]).dropna(), bins=30)
        ax.hist(reference, bins=bins, density=True, alpha=0.55, label="reference")
        ax.hist(current, bins=bins, density=True, alpha=0.55, label="current")
    else:
        ref = reference.value_counts(normalize=True)
        cur = current.value_counts(normalize=True)
        cats = sorted(set(ref.index) | set(cur.index))
        x = np.arange(len(cats))
        ax.bar(x - 0.2, [ref.get(c, 0) for c in cats], 0.4, label="reference")
        ax.bar(x + 0.2, [cur.get(c, 0) for c in cats], 0.4, label="current")
        ax.set_xticks(x, [str(c)[:14] for c in cats], rotation=25, ha="right", fontsize=7)
    ax.set_title(reference.name, fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def write_html(result: dict, reference: pd.DataFrame, current: pd.DataFrame, path: Path, label: str) -> None:
    verdict = "DATASET DRIFT DETECTED: retraining recommended" if result["dataset_drift"] else "No dataset drift"
    color = "#c62828" if result["dataset_drift"] else "#2e7d32"
    rows, plots = [], []
    for c in sorted(result["columns"], key=lambda c: c["p_value"]):
        mark = "&#10060; drift" if c["drifted"] else "ok"
        rows.append(
            f"<tr><td>{html.escape(c['column'])}</td><td>{c['kind']}</td><td>{c['test']}</td>"
            f"<td>{c['p_value']:.4g}</td><td>{mark}</td><td>{html.escape(c['reference_summary'])}</td>"
            f"<td>{html.escape(c['current_summary'])}</td></tr>"
        )
        if c["drifted"]:
            img = _plot(reference[c["column"]], current[c["column"]], c["kind"])
            plots.append(f'<img alt="{html.escape(c["column"])}" src="data:image/png;base64,{img}">')
    quality = html.escape(json.dumps(result["data_quality_issues"], indent=1)) or "{}"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Data drift report</title>
<style>body{{font-family:sans-serif;margin:24px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:4px 8px;font-size:13px}}img{{margin:4px}}</style></head><body>
<h1>Data drift report</h1>
<p><b>Current data:</b> {html.escape(label)} &middot; checked {result['checked_at']}</p>
<h2 style="color:{color}">{verdict}</h2>
<p>{len(result['drifted_columns'])} of {len(result['columns'])} columns drifted
({result['drifted_share']:.0%}; trigger at {result['drift_share_threshold']:.0%}, p &lt; {result['p_value_threshold']}).
Reference rows: {result['reference_rows']}, current rows: {result['current_rows']}.</p>
<table><tr><th>column</th><th>type</th><th>test</th><th>p-value</th><th>result</th>
<th>reference</th><th>current</th></tr>{''.join(rows)}</table>
<h2>Drifted columns</h2>{''.join(plots) or '<p>none</p>'}
<h2>Data quality issues (current data)</h2><pre>{quality}</pre>
</body></html>"""
    path.write_text(page, encoding="utf-8")


def run(simulate: str = "none", current_path: str | None = None, seed: int = 0,
        settings_path: str = "configs/monitoring.yaml") -> dict:
    settings = load_settings(settings_path)
    reference = reference_frame()
    if current_path:
        p = Path(current_path)
        current = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        label = str(p)
    else:
        sample = holdout_frame().sample(n=settings["simulation"]["sample_size"], random_state=seed)
        if simulate == "drift":
            current, label = simulate_drift(sample, settings["simulation"], seed), "SIMULATED drift"
        else:
            current, label = sample, "held-out sample (no drift expected)"
    result = check_drift(reference, current, settings)
    result["current_label"] = label

    out = Path(settings["drift"]["report_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_html(result, reference, current, out / "drift_report.html", label)
    logger.info(
        "%s: %d/%d columns drifted (%.0f%%) -> dataset_drift=%s; drifted: %s",
        label, len(result["drifted_columns"]), len(result["columns"]), 100 * result["drifted_share"],
        result["dataset_drift"], ", ".join(result["drifted_columns"]) or "none",
    )
    logger.info("Report: %s", (out / "drift_report.html").resolve())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Data drift and data-quality check.")
    parser.add_argument("--simulate", choices=["none", "drift"], default="none")
    parser.add_argument("--current", help="CSV or Parquet file with recent inputs (overrides --simulate)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exit-code", action="store_true",
                        help=f"exit {DRIFT_EXIT_CODE} when dataset drift is detected (retraining trigger)")
    args = parser.parse_args(argv)
    result = run(args.simulate, args.current, args.seed)
    return DRIFT_EXIT_CODE if (args.exit_code and result["dataset_drift"]) else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(main())
