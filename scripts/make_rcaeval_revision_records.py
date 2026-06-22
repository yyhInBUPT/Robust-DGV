import argparse
import json
from pathlib import Path

import pandas as pd


DATASETS = {
    "RCAEval-OB": {
        "system": "Online Boutique",
        "stats": "robust_grn/dataset/RCAEval_OB/stats.json",
    },
    "RCAEval-SS": {
        "system": "Sock Shop",
        "stats": "robust_grn/dataset/RCAEval_SS/stats.json",
    },
}

EXPERIMENTS = {
    ("RCAEval-OB", 2): {
        "acc": 91.3,
        "macro_f1": 0.8036,
        "root_precision": 0.5111,
        "root_recall": 0.9200,
        "root_f1": 0.6571,
        "best_epoch": 163,
        "checkpoint": "robust_grn/pretrained/rcaeval_ob_gcnii_l2.pt",
        "certification_csv": "robust_grn/result/rcaeval_ob_cert_q1_Q135_correct_test.csv",
    },
    ("RCAEval-OB", 4): {
        "acc": 95.6,
        "macro_f1": 0.8577,
        "root_precision": 0.8095,
        "root_recall": 0.6800,
        "root_f1": 0.7391,
        "best_epoch": 297,
        "checkpoint": "robust_grn/pretrained/rcaeval_ob_gcnii_l4.pt",
        "certification_csv": "robust_grn/result/rcaeval_ob_cert_q1_Q135_l4_correct_test.csv",
    },
    ("RCAEval-OB", 8): {
        "acc": 94.9,
        "macro_f1": 0.8402,
        "root_precision": 0.7391,
        "root_recall": 0.6800,
        "root_f1": 0.7083,
        "best_epoch": 186,
        "checkpoint": "robust_grn/pretrained/rcaeval_ob_gcnii_l8.pt",
        "certification_csv": "robust_grn/result/rcaeval_ob_cert_q1_Q135_l8_correct_test.csv",
    },
    ("RCAEval-SS", 4): {
        "acc": 94.9,
        "macro_f1": 0.8252,
        "root_precision": 0.5882,
        "root_recall": 0.8000,
        "root_f1": 0.6780,
        "best_epoch": 266,
        "checkpoint": "robust_grn/pretrained/rcaeval_ss_gcnii_l4.pt",
        "certification_csv": "robust_grn/result/rcaeval_ss_cert_q1_Q135_l4_correct_test.csv",
    },
}

MAIN_RESULTS = [("RCAEval-OB", 4), ("RCAEval-SS", 4)]
OB_ABLATION = [("RCAEval-OB", 2), ("RCAEval-OB", 4), ("RCAEval-OB", 8)]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="experiments/rcaeval_revision")
    return parser.parse_args()


def require_path(path):
    if not path.exists():
        raise FileNotFoundError("Missing required input file: {}".format(path))
    return path


def read_json(path):
    return json.loads(require_path(path).read_text())


def read_cert_csv(path):
    return pd.read_csv(require_path(path))


def q_row(df, q_value):
    rows = df[df["Q"] == q_value]
    if rows.empty:
        raise ValueError("Missing Q={} row in {}".format(q_value, "certification CSV"))
    return rows.iloc[0]


def stat_value(stats, *keys, default=None):
    for key in keys:
        if key in stats:
            return stats[key]
    return default


def dataset_summary_row(dataset, info, stats):
    feature_names = stat_value(stats, "feature_names", "metric_types", default=[])
    return {
        "dataset": dataset,
        "system": info["system"],
        "num_cases": stat_value(stats, "num_cases"),
        "num_services": stat_value(stats, "num_services", "num_services_per_case"),
        "num_nodes": stat_value(stats, "num_nodes"),
        "feature_dim": stat_value(stats, "feature_dim", "num_features"),
        "train_cases": stat_value(stats, "train_cases"),
        "val_cases": stat_value(stats, "val_cases"),
        "test_cases": stat_value(stats, "test_cases"),
        "train_nodes": stat_value(stats, "train_nodes", "train_size"),
        "val_nodes": stat_value(stats, "val_nodes", "val_size"),
        "test_nodes": stat_value(stats, "test_nodes", "test_size"),
        "positive_nodes": stat_value(stats, "positive_nodes", "num_positive_nodes"),
        "negative_nodes": stat_value(stats, "negative_nodes", "num_negative_nodes"),
        "feature_names": ";".join(feature_names),
    }


def experiment_result_row(dataset, layer, include_system):
    metrics = EXPERIMENTS[(dataset, layer)]
    system = DATASETS[dataset]["system"]
    cert_df = read_cert_csv(BASE_DIR / metrics["certification_csv"])
    q1 = q_row(cert_df, 1)
    q3 = q_row(cert_df, 3)
    q5 = q_row(cert_df, 5)

    row = {
        "dataset": dataset,
        "layer": layer,
        "acc": metrics["acc"],
        "macro_f1": metrics["macro_f1"],
        "root_precision": metrics["root_precision"],
        "root_recall": metrics["root_recall"],
        "root_f1": metrics["root_f1"],
        "best_epoch": metrics["best_epoch"],
        "correct_nodes": int(q1["num_nodes"]),
        "robust_q1": float(q1["robust_percent"]),
        "robust_q3": float(q3["robust_percent"]),
        "robust_q5": float(q5["robust_percent"]),
        "neither_q1": float(q1["neither_percent"]),
        "neither_q3": float(q3["neither_percent"]),
        "neither_q5": float(q5["neither_percent"]),
        "runtime_q1": float(q1["runtime_per_node"]),
        "checkpoint": metrics["checkpoint"],
        "certification_csv": metrics["certification_csv"],
    }
    if include_system:
        row = {"dataset": dataset, "system": system, **{k: v for k, v in row.items() if k != "dataset"}}
    return row


def grouped_rows():
    rows = []
    for dataset, layer in EXPERIMENTS:
        system = DATASETS[dataset]["system"]
        metrics = EXPERIMENTS[(dataset, layer)]
        cert_df = read_cert_csv(BASE_DIR / metrics["certification_csv"])
        for _, row in cert_df.iterrows():
            rows.append({
                "dataset": dataset,
                "system": system,
                "layer": layer,
                "Q": int(row["Q"]),
                "q": int(row["q"]),
                "group": "Root-cause",
                "nodes": int(row["root_num_nodes"]),
                "robust_percent": float(row["root_robust_percent"]),
                "nonrobust_percent": float(row["root_nonrobust_percent"]),
                "neither_percent": float(row["root_neither_percent"]),
            })
            rows.append({
                "dataset": dataset,
                "system": system,
                "layer": layer,
                "Q": int(row["Q"]),
                "q": int(row["q"]),
                "group": "Non-root",
                "nodes": int(row["nonroot_num_nodes"]),
                "robust_percent": float(row["nonroot_robust_percent"]),
                "nonrobust_percent": float(row["nonroot_nonrobust_percent"]),
                "neither_percent": float(row["nonroot_neither_percent"]),
            })
    return rows


def fmt(value, decimals):
    return ("{:." + str(decimals) + "f}").format(float(value))


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def dataset_summary_markdown(df):
    rows = []
    for _, row in df.iterrows():
        split = "{}/{}/{}".format(int(row["train_nodes"]), int(row["val_nodes"]), int(row["test_nodes"]))
        rows.append([
            row["dataset"],
            row["system"],
            int(row["num_cases"]),
            int(row["num_services"]),
            int(row["num_nodes"]),
            int(row["feature_dim"]),
            int(row["positive_nodes"]),
            int(row["negative_nodes"]),
            split,
        ])
    return markdown_table(
        ["Dataset", "System", "Cases", "Services", "Nodes", "Feature Dim", "Positive", "Negative", "Split"],
        rows,
    )


def main_results_markdown(df):
    rows = []
    for _, row in df.iterrows():
        rows.append([
            row["dataset"],
            row["system"],
            int(row["layer"]),
            fmt(row["acc"], 1),
            fmt(row["macro_f1"], 4),
            fmt(row["root_precision"], 4),
            fmt(row["root_recall"], 4),
            fmt(row["root_f1"], 4),
            int(row["correct_nodes"]),
            fmt(row["robust_q1"], 2),
            fmt(row["robust_q3"], 2),
            fmt(row["robust_q5"], 2),
        ])
    return markdown_table(
        [
            "Dataset",
            "System",
            "Layer",
            "Acc.",
            "Macro-F1",
            "Root Prec.",
            "Root Rec.",
            "Root F1",
            "Correct Nodes",
            "Robust % Q=1",
            "Robust % Q=3",
            "Robust % Q=5",
        ],
        rows,
    )


def ob_ablation_markdown(df):
    rows = []
    for _, row in df.iterrows():
        rows.append([
            int(row["layer"]),
            fmt(row["acc"], 1),
            fmt(row["macro_f1"], 4),
            fmt(row["root_precision"], 4),
            fmt(row["root_recall"], 4),
            fmt(row["root_f1"], 4),
            int(row["best_epoch"]),
            int(row["correct_nodes"]),
            fmt(row["robust_q1"], 2),
            fmt(row["robust_q3"], 2),
            fmt(row["robust_q5"], 2),
            fmt(row["runtime_q1"], 4),
        ])
    return markdown_table(
        [
            "Layer",
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
        ],
        rows,
    )


def grouped_observation_markdown(group_df):
    q1 = group_df[group_df["Q"] == 1]
    rows = []
    for _, row in q1.iterrows():
        rows.append([
            row["dataset"],
            int(row["layer"]),
            row["group"],
            int(row["nodes"]),
            fmt(row["robust_percent"], 2),
            fmt(row["nonrobust_percent"], 2),
            fmt(row["neither_percent"], 2),
        ])
    table = markdown_table(
        ["Dataset", "Layer", "Group", "Nodes", "Robust %", "Non-robust %", "Neither %"],
        rows,
    )
    note = (
        "\n\nRoot-cause nodes mostly fall into neither, while non-root nodes have much higher robust ratios. "
        "This pattern appears in both OB and SS, suggesting that root-cause service decisions are harder to certify "
        "under node-feature perturbations than stable non-root decisions."
    )
    return table + note


def write_readme(output_dir):
    content = """# RCAEval Revision Records

This directory stores RCAEval experiment records for the IEEE TSC major revision of
\"Towards Formal Assurance of Robust Graph Intelligence in Service Computing Systems\".

Files:

- `rcaeval_dataset_summary.csv`: dataset construction statistics for RCAEval-OB and RCAEval-SS.
- `rcaeval_system_main_results.csv`: cross-system main results for the current recommended models.
- `rcaeval_ob_layer_ablation.csv`: RCAEval-OB GCNII layer ablation.
- `rcaeval_group_certification.csv`: root-cause and non-root grouped certification results.
- `rcaeval_revision_record.md`: human-readable Markdown record for paper writing.

Classification metrics are manually recorded from training logs. Certification metrics are read from CSV files
generated by `robust_grn/certify.py` under `robust_grn/result/`.

Current records include RCAEval-OB layer=2/4/8 and RCAEval-SS layer=4.
"""
    (output_dir / "README.md").write_text(content, encoding="utf-8")


def write_revision_record(output_dir, dataset_md, main_md, ob_md, group_md):
    content = f"""# RCAEval Revision Experiment Record

## 1. Dataset Construction Summary

{dataset_md}

## 2. Cross-System Main Results

{main_md}

## 3. OB Layer Ablation

{ob_md}

## 4. Grouped Certification Observation

{group_md}

## 5. Important Notes for Paper Writing

1. 当前实验认证的是 GNN-based service intelligence 的节点特征扰动鲁棒性。
2. 输出任务是 root-cause service node classification，不是直接 QoS/SLA guarantee。
3. 特征来自 metrics，不使用 logs/traces。
4. RCAEval-SS 的 topology 是基于 Sock Shop 静态服务依赖手工构造，需要在论文中保守表述。
5. 当前结果可支持 “DGV can be adapted to real microservice service graphs”，但不要声称完整 production guarantee。
"""
    (output_dir / "rcaeval_revision_record.md").write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    global BASE_DIR
    BASE_DIR = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    for dataset, info in DATASETS.items():
        stats = read_json(BASE_DIR / info["stats"])
        dataset_rows.append(dataset_summary_row(dataset, info, stats))
    dataset_df = pd.DataFrame(dataset_rows)

    main_df = pd.DataFrame([experiment_result_row(dataset, layer, include_system=True) for dataset, layer in MAIN_RESULTS])
    ob_df = pd.DataFrame([experiment_result_row(dataset, layer, include_system=False) for dataset, layer in OB_ABLATION])
    group_df = pd.DataFrame(grouped_rows())

    dataset_df.to_csv(output_dir / "rcaeval_dataset_summary.csv", index=False)
    main_df.to_csv(output_dir / "rcaeval_system_main_results.csv", index=False)
    ob_df.to_csv(output_dir / "rcaeval_ob_layer_ablation.csv", index=False)
    group_df.to_csv(output_dir / "rcaeval_group_certification.csv", index=False)

    dataset_md = dataset_summary_markdown(dataset_df)
    main_md = main_results_markdown(main_df)
    ob_md = ob_ablation_markdown(ob_df)
    group_md = grouped_observation_markdown(group_df)

    write_readme(output_dir)
    write_revision_record(output_dir, dataset_md, main_md, ob_md, group_md)

    written = [
        output_dir / "README.md",
        output_dir / "rcaeval_dataset_summary.csv",
        output_dir / "rcaeval_system_main_results.csv",
        output_dir / "rcaeval_ob_layer_ablation.csv",
        output_dir / "rcaeval_group_certification.csv",
        output_dir / "rcaeval_revision_record.md",
    ]
    print("Written files:")
    for path in written:
        print("-", path)

    print("\nDataset Construction Summary preview:")
    print(dataset_md)
    print("\nCross-System Main Results preview:")
    print(main_md)
    print("\nOB Layer Ablation preview:")
    print(ob_md)
    print("\nGrouped Certification Observation preview:")
    print(group_md)


if __name__ == "__main__":
    main()
