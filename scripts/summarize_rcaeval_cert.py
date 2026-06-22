import argparse

import pandas as pd


def format_percent(value):
    return "{:.2f}".format(float(value))


def format_runtime(value):
    return "{:.4f}".format(float(value))


def print_markdown_table(title, headers, rows):
    print("## {}".format(title))
    print()
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print("| " + " | ".join(str(item) for item in row) + " |")
    print()


def build_overall_rows(df):
    rows = []
    for _, row in df.iterrows():
        rows.append([
            int(row["Q"]),
            int(row["q"]),
            row["split"],
            str(bool(row["only_correct"])),
            int(row["num_nodes"]),
            format_percent(row["robust_percent"]),
            format_percent(row["nonrobust_percent"]),
            format_percent(row["neither_percent"]),
            format_runtime(row["runtime_per_node"]),
        ])
    return rows


def build_grouped_rows(df):
    rows = []
    for _, row in df.iterrows():
        rows.append([
            int(row["Q"]),
            int(row["q"]),
            "Root-cause",
            int(row["root_num_nodes"]),
            format_percent(row["root_robust_percent"]),
            format_percent(row["root_nonrobust_percent"]),
            format_percent(row["root_neither_percent"]),
        ])
        rows.append([
            int(row["Q"]),
            int(row["q"]),
            "Non-root",
            int(row["nonroot_num_nodes"]),
            format_percent(row["nonroot_robust_percent"]),
            format_percent(row["nonroot_nonrobust_percent"]),
            format_percent(row["nonroot_neither_percent"]),
        ])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="path to certification result csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    print_markdown_table(
        "Overall Certification Results",
        [
            "Q",
            "q",
            "Split",
            "Only Correct",
            "Nodes",
            "Robust %",
            "Non-robust %",
            "Neither %",
            "Runtime / node",
        ],
        build_overall_rows(df),
    )

    print_markdown_table(
        "Grouped Certification Results",
        [
            "Q",
            "q",
            "Group",
            "Nodes",
            "Robust %",
            "Non-robust %",
            "Neither %",
        ],
        build_grouped_rows(df),
    )


if __name__ == "__main__":
    main()
