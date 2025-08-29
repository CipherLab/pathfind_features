"""Compare control vs experimental predictions on validation data."""
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path


def era_correlation(df: pd.DataFrame, pred_col: str, target_col: str):
    eras = df['era'].unique()
    corrs = []
    for era in eras:
        sub = df[df['era'] == era]
        if sub[target_col].nunique() < 3:
            continue
        r, _ = spearmanr(sub[pred_col], sub[target_col])
        if np.isnan(r):
            continue
        corrs.append(r)
    return corrs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--control-predictions', required=True)
    ap.add_argument('--experimental-predictions', required=True)
    ap.add_argument('--validation-data', required=True)
    ap.add_argument('--output-analysis', required=True)
    ap.add_argument('--target-col', default='target')
    ap.add_argument('--experimental-target-col', default='adaptive_target')
    args = ap.parse_args()

    val = pd.read_parquet(args.validation_data)
    control = pd.read_csv(args.control_predictions)
    experimental = pd.read_csv(args.experimental_predictions)
    if len(control) != len(val) or len(experimental) != len(val):
        n = min(len(control), len(experimental), len(val))
        control = control.iloc[:n]
        experimental = experimental.iloc[:n]
        val = val.iloc[:n]
    val = val.reset_index(drop=True)
    val['control_pred'] = control['prediction']
    val['experimental_pred'] = experimental['prediction']

    control_era = era_correlation(val, 'control_pred', args.target_col)
    experimental_era = era_correlation(val, 'experimental_pred', args.experimental_target_col if args.experimental_target_col in val.columns else args.target_col)

    overall_control, _ = spearmanr(val['control_pred'], val[args.target_col])
    experimental_target = args.experimental_target_col if args.experimental_target_col in val.columns else args.target_col
    overall_experimental, _ = spearmanr(val['experimental_pred'], val[experimental_target])

    report = {
        'overall_control_spearman': overall_control,
        'overall_experimental_spearman': overall_experimental,
        'mean_era_control': float(np.mean(control_era)) if control_era else None,
        'mean_era_experimental': float(np.mean(experimental_era)) if experimental_era else None,
        'num_eras_control': len(control_era),
        'num_eras_experimental': len(experimental_era)
    }

    lines = ["Model Performance Comparison", "============================", ""]
    for k, v in report.items():
        lines.append(f"{k}: {v}")
    Path(args.output_analysis).write_text("\n".join(lines), encoding='utf-8')
    print(f"Performance report written to {args.output_analysis}")


if __name__ == '__main__':
    main()
