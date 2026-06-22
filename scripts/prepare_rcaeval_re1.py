import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import block_diag, csr_matrix, save_npz


SERVICES_SS = [
    "carts",
    "carts-db",
    "catalogue",
    "catalogue-db",
    "front-end",
    "orders",
    "orders-db",
    "payment",
    "queue-master",
    "rabbitmq",
    "rabbitmq-exporter",
    "session-db",
    "shipping",
    "user",
    "user-db",
]

EDGES_SS = [
    ("front-end", "catalogue"),
    ("front-end", "carts"),
    ("front-end", "orders"),
    ("front-end", "user"),
    ("front-end", "session-db"),
    ("carts", "carts-db"),
    ("catalogue", "catalogue-db"),
    ("orders", "orders-db"),
    ("user", "user-db"),
    ("orders", "payment"),
    ("orders", "shipping"),
    ("orders", "queue-master"),
    ("queue-master", "rabbitmq"),
    ("rabbitmq", "rabbitmq-exporter"),
]

SERVICES_TT = [
    "ts-assurance-service",
    "ts-auth-service",
    "ts-avatar-service",
    "ts-basic-service",
    "ts-cancel-service",
    "ts-config-service",
    "ts-consign-price-service",
    "ts-consign-service",
    "ts-contacts-service",
    "ts-execute-service",
    "ts-food-map-service",
    "ts-food-service",
    "ts-inside-payment-service",
    "ts-news-service",
    "ts-notification-service",
    "ts-order-other-service",
    "ts-order-service",
    "ts-payment-service",
    "ts-preserve-other-service",
    "ts-preserve-service",
    "ts-price-service",
    "ts-rebook-service",
    "ts-route-plan-service",
    "ts-route-service",
    "ts-seat-service",
    "ts-security-service",
    "ts-station-service",
    "ts-ticket-office-service",
    "ts-ticketinfo-service",
    "ts-train-service",
    "ts-travel-plan-service",
    "ts-travel-service",
    "ts-travel2-service",
    "ts-user-service",
    "ts-verification-code-service",
    "ts-voucher-service",
]

EDGES_TT = [
    ("ts-auth-service", "ts-verification-code-service"),
    ("ts-user-service", "ts-auth-service"),
    ("ts-basic-service", "ts-station-service"),
    ("ts-basic-service", "ts-train-service"),
    ("ts-basic-service", "ts-route-service"),
    ("ts-basic-service", "ts-price-service"),
    ("ts-travel-service", "ts-train-service"),
    ("ts-travel-service", "ts-order-service"),
    ("ts-travel-service", "ts-route-service"),
    ("ts-travel-service", "ts-basic-service"),
    ("ts-travel-service", "ts-seat-service"),
    ("ts-travel2-service", "ts-basic-service"),
    ("ts-travel2-service", "ts-train-service"),
    ("ts-travel2-service", "ts-route-service"),
    ("ts-travel2-service", "ts-seat-service"),
    ("ts-route-plan-service", "ts-route-service"),
    ("ts-route-plan-service", "ts-travel-service"),
    ("ts-route-plan-service", "ts-travel2-service"),
    ("ts-travel-plan-service", "ts-seat-service"),
    ("ts-travel-plan-service", "ts-route-plan-service"),
    ("ts-travel-plan-service", "ts-travel-service"),
    ("ts-travel-plan-service", "ts-travel2-service"),
    ("ts-travel-plan-service", "ts-train-service"),
    ("ts-seat-service", "ts-order-service"),
    ("ts-seat-service", "ts-order-other-service"),
    ("ts-seat-service", "ts-config-service"),
    ("ts-order-service", "ts-station-service"),
    ("ts-order-other-service", "ts-station-service"),
    ("ts-security-service", "ts-order-service"),
    ("ts-security-service", "ts-order-other-service"),
    ("ts-inside-payment-service", "ts-order-service"),
    ("ts-inside-payment-service", "ts-order-other-service"),
    ("ts-inside-payment-service", "ts-payment-service"),
    ("ts-execute-service", "ts-order-service"),
    ("ts-execute-service", "ts-order-other-service"),
    ("ts-cancel-service", "ts-notification-service"),
    ("ts-cancel-service", "ts-order-service"),
    ("ts-cancel-service", "ts-order-other-service"),
    ("ts-cancel-service", "ts-inside-payment-service"),
    ("ts-cancel-service", "ts-user-service"),
    ("ts-rebook-service", "ts-seat-service"),
    ("ts-rebook-service", "ts-travel-service"),
    ("ts-rebook-service", "ts-travel2-service"),
    ("ts-rebook-service", "ts-order-service"),
    ("ts-rebook-service", "ts-order-other-service"),
    ("ts-rebook-service", "ts-train-service"),
    ("ts-rebook-service", "ts-route-service"),
    ("ts-rebook-service", "ts-inside-payment-service"),
    ("ts-preserve-service", "ts-basic-service"),
    ("ts-preserve-service", "ts-seat-service"),
    ("ts-preserve-service", "ts-user-service"),
    ("ts-preserve-service", "ts-assurance-service"),
    ("ts-preserve-service", "ts-station-service"),
    ("ts-preserve-service", "ts-security-service"),
    ("ts-preserve-service", "ts-travel-service"),
    ("ts-preserve-service", "ts-contacts-service"),
    ("ts-preserve-service", "ts-order-service"),
    ("ts-preserve-service", "ts-food-service"),
    ("ts-preserve-service", "ts-consign-service"),
    ("ts-preserve-other-service", "ts-basic-service"),
    ("ts-preserve-other-service", "ts-seat-service"),
    ("ts-preserve-other-service", "ts-user-service"),
    ("ts-preserve-other-service", "ts-assurance-service"),
    ("ts-preserve-other-service", "ts-station-service"),
    ("ts-preserve-other-service", "ts-security-service"),
    ("ts-preserve-other-service", "ts-travel2-service"),
    ("ts-preserve-other-service", "ts-contacts-service"),
    ("ts-preserve-other-service", "ts-order-other-service"),
    ("ts-preserve-other-service", "ts-food-service"),
    ("ts-preserve-other-service", "ts-consign-service"),
    ("ts-food-service", "ts-station-service"),
    ("ts-food-service", "ts-travel-service"),
    ("ts-consign-service", "ts-consign-price-service"),
    ("ts-voucher-service", "ts-order-service"),
    ("ts-voucher-service", "ts-order-other-service"),
]

FEATURE_NAMES = [
    "cpu",
    "mem",
    "load",
    "workload",
    "latency",
    "latency-50",
    "latency-90",
    "error",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="~/RCAEval/data/RE1")
    parser.add_argument("--system", default="SS", choices=["SS", "TT", "ss", "tt"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--threshold-std", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_single_adj(services, edges):
    service_to_id = {service: idx for idx, service in enumerate(services)}
    adj = np.zeros((len(services), len(services)), dtype=np.float32)

    for src, dst in edges:
        if src not in service_to_id or dst not in service_to_id:
            raise ValueError("Unknown edge endpoint: {} -- {}".format(src, dst))
        src_id = service_to_id[src]
        dst_id = service_to_id[dst]
        adj[src_id, dst_id] = 1.0
        adj[dst_id, src_id] = 1.0

    return csr_matrix(adj)


def parse_case_dir(case_dir):
    root_service, fault_type = case_dir.name.rsplit("_", 1)
    return root_service, fault_type


def read_inject_time(path):
    return int(float(path.read_text().strip()))


def find_time_column(df):
    for column in ("time", "timestamp"):
        if column in df.columns:
            return column
    raise ValueError("simple_data.csv does not contain a time or timestamp column")


def extract_binary_features(df, inject_time, services, threshold_std):
    time_column = find_time_column(df)
    normal_df = df[df[time_column] < inject_time]
    fault_df = df[df[time_column] >= inject_time]

    if len(normal_df) == 0 or len(fault_df) == 0:
        raise ValueError("empty normal or fault window, inject_time={}".format(inject_time))

    features = np.zeros((len(services), len(FEATURE_NAMES)), dtype=np.float32)

    for service_id, service in enumerate(services):
        for metric_id, metric in enumerate(FEATURE_NAMES):
            column = "{}_{}".format(service, metric)
            if column not in df.columns:
                continue

            normal_values = pd.to_numeric(normal_df[column], errors="coerce").dropna()
            fault_values = pd.to_numeric(fault_df[column], errors="coerce").dropna()
            if len(normal_values) == 0 or len(fault_values) == 0:
                continue

            normal_mean = normal_values.mean()
            normal_std = normal_values.std()
            fault_mean = fault_values.mean()
            if np.isnan(normal_std):
                normal_std = 0.0

            threshold = normal_mean + threshold_std * (normal_std + 1e-12)
            if fault_mean > threshold:
                features[service_id, metric_id] = 1.0

    return features


def make_case_level_split(num_cases, nodes_per_case, seed):
    rng = np.random.default_rng(seed)
    case_ids = np.arange(num_cases, dtype=np.int64)
    rng.shuffle(case_ids)

    num_train = int(num_cases * 0.6)
    num_val = int(num_cases * 0.2)
    train_case_ids = case_ids[:num_train]
    val_case_ids = case_ids[num_train:num_train + num_val]
    test_case_ids = case_ids[num_train + num_val:]

    def expand_case_ids(selected_case_ids):
        node_indices = [
            np.arange(case_id * nodes_per_case, (case_id + 1) * nodes_per_case, dtype=np.int64)
            for case_id in selected_case_ids
        ]
        return np.concatenate(node_indices) if node_indices else np.array([], dtype=np.int64)

    return (
        expand_case_ids(train_case_ids),
        expand_case_ids(val_case_ids),
        expand_case_ids(test_case_ids),
        train_case_ids,
        val_case_ids,
        test_case_ids,
    )


def main():
    args = parse_args()
    system = args.system.upper()

    raw_root = Path(args.raw_root).expanduser()
    input_dir = raw_root / "RE1-{}".format(system)
    if args.output is None:
        output_dir = Path("robust_grn/dataset/RCAEval_{}".format(system))
    else:
        output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError("RCAEval RE1-{} input directory not found: {}".format(system, input_dir))

    if system == "SS":
        services = SERVICES_SS
        edges = EDGES_SS
        source_system = "Sock Shop"
        graph_type = "service-graph"
        topology_source = "manual Sock Shop static service dependency graph"
    elif system == "TT":
        services = SERVICES_TT
        edges = EDGES_TT
        source_system = "Train Ticket"
        graph_type = "application-service-only"
        topology_source = (
            "official Train Ticket architecture/deployment/source evidence; "
            "see experiments/rcaeval_revision/tt_official_topology_notes.md"
        )
    else:
        raise ValueError("Unsupported --system: {}".format(args.system))

    service_to_id = {service: idx for idx, service in enumerate(services)}
    single_adj = build_single_adj(services, edges)

    all_features = []
    all_labels = []
    all_adjs = []
    case_meta = []
    node_meta = []
    case_id = 0

    case_dirs = sorted([path for path in input_dir.iterdir() if path.is_dir()])
    for case_dir in case_dirs:
        root_service, fault_type = parse_case_dir(case_dir)
        if root_service not in service_to_id:
            raise ValueError("root_service not in service list: {}, case={}".format(root_service, case_dir))

        repeat_dirs = sorted([path for path in case_dir.iterdir() if path.is_dir()], key=lambda path: int(path.name))
        for repeat_dir in repeat_dirs:
            data_path = repeat_dir / "simple_data.csv"
            inject_path = repeat_dir / "inject_time.txt"
            if not data_path.exists() or not inject_path.exists():
                continue

            df = pd.read_csv(data_path)
            inject_time = read_inject_time(inject_path)
            features = extract_binary_features(df, inject_time, services, args.threshold_std)

            labels = np.zeros(len(services), dtype=np.int64)
            labels[service_to_id[root_service]] = 1

            all_features.append(features)
            all_labels.append(labels)
            all_adjs.append(single_adj)

            case_meta.append({
                "case_id": case_id,
                "case_group": case_dir.name,
                "root_service": root_service,
                "fault_type": fault_type,
                "repeat": repeat_dir.name,
                "path": str(repeat_dir),
            })

            for service in services:
                label = int(service == root_service)
                node_meta.append({
                    "node_id": len(node_meta),
                    "case_id": case_id,
                    "service": service,
                    "root_service": root_service,
                    "fault_type": fault_type,
                    "repeat": repeat_dir.name,
                    "label": label,
                })

            case_id += 1

    if not all_features:
        raise ValueError("No RE1-{} cases were processed from {}".format(system, input_dir))

    features_big = np.vstack(all_features).astype(np.float32)
    labels_big = np.concatenate(all_labels).astype(np.int64)
    adj_big = block_diag(all_adjs, format="csr").astype(np.float32)

    idx_train, idx_val, idx_test, train_case_ids, val_case_ids, test_case_ids = make_case_level_split(
        case_id,
        len(services),
        args.seed,
    )

    np.save(output_dir / "features.npy", features_big)
    np.save(output_dir / "labels.npy", labels_big)
    np.save(output_dir / "idx_train.npy", idx_train)
    np.save(output_dir / "idx_val.npy", idx_val)
    np.save(output_dir / "idx_test.npy", idx_test)
    save_npz(output_dir / "adj.npz", adj_big)

    with (output_dir / "service_names.json").open("w", encoding="utf-8") as f:
        json.dump(services, f, indent=2, ensure_ascii=False)
    with (output_dir / "case_meta.json").open("w", encoding="utf-8") as f:
        json.dump(case_meta, f, indent=2, ensure_ascii=False)
    with (output_dir / "node_meta.json").open("w", encoding="utf-8") as f:
        json.dump(node_meta, f, indent=2, ensure_ascii=False)

    stats = {
        "system": system,
        "source_system": source_system,
        "graph_type": graph_type,
        "topology_source": topology_source,
        "num_cases": int(case_id),
        "num_services": int(len(services)),
        "num_nodes": int(features_big.shape[0]),
        "feature_dim": int(features_big.shape[1]),
        "num_edges_per_case": int(single_adj.sum() // 2),
        "train_cases": int(len(train_case_ids)),
        "val_cases": int(len(val_case_ids)),
        "test_cases": int(len(test_case_ids)),
        "train_nodes": int(len(idx_train)),
        "val_nodes": int(len(idx_val)),
        "test_nodes": int(len(idx_test)),
        "positive_nodes": int(labels_big.sum()),
        "negative_nodes": int((labels_big == 0).sum()),
        "threshold_std": float(args.threshold_std),
        "feature_names": FEATURE_NAMES,
        "seed": int(args.seed),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }
    with (output_dir / "stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("Saved RCAEval RE1-{} DGV dataset to:".format(system), output_dir)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
