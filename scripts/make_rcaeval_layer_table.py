import argparse
from pathlib import Path

import pandas as pd


LAYER_METRICS = {
    2: {
        "accuracy": 91.3,
        "macro_f1": 0.8036,
        "root_precision": 0.5111,
        "root_recall": 0.9200,
        "root_f1": 0.6571,
        "best_epoch": 163,
        "csv": "rcaeval_ob_cert_q1_Q135_correct_test.csv",
    },
    4: {
        "accuracy": 95.6,
        "macro_f1": 0.8577,
        "root_precision": 0.8095,
        "root_recall": 0.6800,
        "root_f1": 0.7391,
        "best_epoch": 297,
        "csv": "rcaeval_ob_cert_q1_Q135_l4_correct_test.csv",
    },
    8: {
        "accuracy": 94.9,
        "macro_f1": 0.8402,
        "root_precision": 0.7391,
        "root_recall": 0.6800,
        "root_f1": 0.7083,
        "best_epoch": 186,
        "csv": "rcaeval_ob_cert_q1_Q135_l8_correct_test.csv",
    },
}


def format_percent(value):
    return "{:.2f}".format(float(value))


def get_q_row(df, q_value):
    matched = df[df["Q"] == q_value]
    if matched.empty:
        raise ValueError("Missing Q={} row in certification CSV".format(q_value))
    return matched.iloc[0]


def build_rows(base_dir):
    result_dir = base_dir / "robust_grn" / "result"
    rows = []

    for layer, metrics in LAYER_METRICS.items():
        csv_path = result_dir / metrics["csv"]
        if not csv_path.exists():
            raise FileNotFoundError("Missing certification CSV for layer {}: {}".format(layer, csv_path))

        df = pd.read_csv(csv_path)
        q1 = get_q_row(df, 1)
        q3 = get_q_row(df, 3)
        q5 = get_q_row(df, 5)

        rows.append([
            layer,
            "{:.1f}".format(metrics["accuracy"]),
            "{:.4f}".format(metrics["macro_f1"]),
            "{:.4f}".format(metrics["root_precision"]),
            "{:.4f}".format(metrics["root_recall"]),
            "{:.4f}".format(metrics["root_f1"]),
            int(metrics["best_epoch"]),
            int(q1["num_nodes"]),
            format_percent(q1["robust_percent"]),
            format_percent(q3["robust_percent"]),
            format_percent(q5["robust_percent"]),
            "{:.4f}".format(float(q1["runtime_per_node"])),
        ])

    return rows


def print_markdown_table(headers, rows):
    print("## RCAEval-OB Layer Ablation")
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".", help="project root containing robust_grn/result")
    args = parser.parse_args()

    headers = [
        "Layers",
        "Acc.",
        "Macro-F1",
        "Root Prec.",
        "Root Rec.",
        "Root F1",
        "Best Epoch",
        "Correct Nodes",
        "Robust % Q=1",
        "Robust % Q=3",
        "Robust % Q=5",
        "Runtime / node",
    ]
    rows = build_rows(Path(args.base_dir))
    print_markdown_table(headers, rows)


if __name__ == "__main__":
    main()
