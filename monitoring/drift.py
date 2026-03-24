import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
REPORTS_DIR = ROOT / "monitoring" / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def generate_reference_data() -> pd.DataFrame:
    """Reference data matching training distribution."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "store":                np.random.randint(1, 1115, n),
        "day_of_week":          np.random.randint(1, 8, n),
        "promo":                np.random.randint(0, 2, n),
        "school_holiday":       np.random.randint(0, 2, n),
        "competition_distance": np.random.exponential(2000, n),
        "promo2":               np.random.randint(0, 2, n),
        "month":                np.random.randint(1, 13, n),
        "year":                 np.full(n, 2015),
        "day":                  np.random.randint(1, 29, n),
    })


def generate_current_data(drift: bool = False) -> pd.DataFrame:
    """Current production data — optionally with simulated drift."""
    np.random.seed(99)
    n = 500
    if drift:
        return pd.DataFrame({
            "store":                np.random.randint(1, 1115, n),
            "day_of_week":          np.random.randint(1, 8, n),
            "promo":                np.random.randint(0, 2, n),
            "school_holiday":       np.random.randint(0, 2, n),
            "competition_distance": np.random.exponential(5000, n),  # shifted
            "promo2":               np.random.randint(0, 2, n),
            "month":                np.random.randint(1, 13, n),
            "year":                 np.full(n, 2026),  # year drift
            "day":                  np.random.randint(1, 29, n),
        })
    else:
        return pd.DataFrame({
            "store":                np.random.randint(1, 1115, n),
            "day_of_week":          np.random.randint(1, 8, n),
            "promo":                np.random.randint(0, 2, n),
            "school_holiday":       np.random.randint(0, 2, n),
            "competition_distance": np.random.exponential(2000, n),
            "promo2":               np.random.randint(0, 2, n),
            "month":                np.random.randint(1, 13, n),
            "year":                 np.full(n, 2015),
            "day":                  np.random.randint(1, 29, n),
        })


def check_drift_alert(threshold: float = 0.3) -> dict:
    """
    Run drift check using KS test on each numeric column.
    Returns alert status if too many features have drifted.
    """
    from scipy import stats

    reference = generate_reference_data()
    current   = generate_current_data(drift=True)

    columns = reference.columns.tolist()
    drifted = []

    for col in columns:
        stat, p_value = stats.ks_2samp(reference[col], current[col])
        if p_value < 0.05:  # statistically significant drift
            drifted.append(col)
            print(f"[drift] ⚠️  {col}: p={p_value:.4f} — DRIFT DETECTED")
        else:
            print(f"[drift] ✅  {col}: p={p_value:.4f} — stable")

    drift_share = len(drifted) / len(columns)
    alert = drift_share >= threshold

    summary = {
        "drift_detected":  alert,
        "drifted_columns": len(drifted),
        "drifted_names":   drifted,
        "total_columns":   len(columns),
        "drift_share":     round(drift_share, 3),
        "threshold":       threshold,
        "status":          "⚠️ DRIFT ALERT" if alert else "✅ No significant drift",
    }

    return summary


def save_drift_report(drift: bool = False) -> str:
    """Save drift results as a simple HTML report."""
    reference = generate_reference_data()
    current   = generate_current_data(drift=drift)

    from scipy import stats
    rows = []
    for col in reference.columns:
        stat, p_value = stats.ks_2samp(reference[col], current[col])
        drifted = p_value < 0.05
        rows.append({
            "Feature":        col,
            "KS Statistic":   round(stat, 4),
            "P-Value":        round(p_value, 4),
            "Drift Detected": "⚠️ YES" if drifted else "✅ NO",
        })

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "WITH_DRIFT" if drift else "NO_DRIFT"
    report_path = REPORTS_DIR / f"drift_report_{label}_{timestamp}.html"

    html = f"""
    <html><head><title>Drift Report</title>
    <style>
        body {{ font-family: Arial; padding: 20px; }}
        h1 {{ color: #1B3A5C; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th {{ background: #1B3A5C; color: white; padding: 10px; text-align: left; }}
        td {{ padding: 8px; border-bottom: 1px solid #ddd; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
    </style></head>
    <body>
    <h1>Rossmann Forecaster — Drift Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mode: {"Simulated Drift" if drift else "No Drift"}</p>
    {df.to_html(index=False)}
    </body></html>
    """

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[drift] Report saved to {report_path}")
    return str(report_path)


if __name__ == "__main__":
    print("=" * 50)
    print("Running drift check (simulated drift)...")
    print("=" * 50)
    summary = check_drift_alert()
    print(f"\n{summary['status']}")
    print(f"Drifted: {summary['drifted_columns']}/{summary['total_columns']} features")
    print(f"Drifted columns: {summary['drifted_names']}")

    print("\nSaving drift reports...")
    save_drift_report(drift=False)
    save_drift_report(drift=True)
    print("\nDone. Check monitoring/reports/ for HTML reports.")