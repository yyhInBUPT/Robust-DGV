import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import block_diag, csr_matrix, save_npz


# =========================
# 1. 路径配置
# =========================

RCA_ROOT = Path.home() / "RCAEval" / "data" / "RE1" / "RE1-OB"
OUT_DIR = Path.home() / "Robust_deepGNN" / "robust_grn" / "dataset" / "RCAEval_OB"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. Online Boutique 服务列表
# =========================
# 第一版只保留核心服务。后续如果需要，可以再加入 frontend-external、main 等组件。

SERVICES = [
    "frontend",
    "cartservice",
    "checkoutservice",
    "currencyservice",
    "emailservice",
    "paymentservice",
    "productcatalogservice",
    "recommendationservice",
    "shippingservice",
    "adservice",
    "redis",
]

SERVICE_TO_ID = {s: i for i, s in enumerate(SERVICES)}


# =========================
# 3. 静态服务调用关系
# =========================
# RE1 没有 traces，所以第一版使用 Online Boutique 的静态依赖关系。
# 这里按无向图处理，便于 GCN 聚合邻居信息。

EDGES = [
    ("frontend", "cartservice"),
    ("frontend", "checkoutservice"),
    ("frontend", "currencyservice"),
    ("frontend", "productcatalogservice"),
    ("frontend", "recommendationservice"),
    ("frontend", "shippingservice"),
    ("frontend", "adservice"),

    ("checkoutservice", "cartservice"),
    ("checkoutservice", "currencyservice"),
    ("checkoutservice", "emailservice"),
    ("checkoutservice", "paymentservice"),
    ("checkoutservice", "productcatalogservice"),
    ("checkoutservice", "shippingservice"),

    ("cartservice", "redis"),
    ("recommendationservice", "productcatalogservice"),
]


# =========================
# 4. 指标解析规则
# =========================

METRIC_TYPES = [
    "cpu",
    "mem",
    "load",
    "workload",
    "latency",
    "latency-50",
    "latency-90",
    "error",
]

METRIC_TO_ID = {m: i for i, m in enumerate(METRIC_TYPES)}


def build_single_adj():
    n = len(SERVICES)
    adj = np.zeros((n, n), dtype=np.float32)

    for u, v in EDGES:
        if u not in SERVICE_TO_ID or v not in SERVICE_TO_ID:
            continue
        i, j = SERVICE_TO_ID[u], SERVICE_TO_ID[v]
        adj[i, j] = 1.0
        adj[j, i] = 1.0

    return csr_matrix(adj)


def parse_case_dir(case_dir: Path):
    """
    case_dir name example:
    adservice_cpu
    productcatalogservice_delay
    """
    name = case_dir.name

    # fault type 在最后一个下划线后面
    root_service, fault_type = name.rsplit("_", 1)
    return root_service, fault_type


def read_inject_time(path: Path):
    text = path.read_text().strip()
    return int(float(text))


def match_metric_column(col: str):
    """
    将 data.csv 中的列名解析为 service + metric type。

    例如：
    adservice_cpu -> adservice, cpu
    frontend_latency-90 -> frontend, latency-90

    如果列不属于我们保留的核心服务或指标类型，返回 None。
    """
    if col == "time":
        return None

    matched_metric = None
    for m in sorted(METRIC_TYPES, key=len, reverse=True):
        suffix = "_" + m
        if col.endswith(suffix):
            matched_metric = m
            service = col[: -len(suffix)]
            break

    if matched_metric is None:
        return None

    if service not in SERVICE_TO_ID:
        return None

    return service, matched_metric


def extract_binary_features(df: pd.DataFrame, inject_time: int):
    """
    对单个 case 生成服务级二值异常特征。
    输出 shape: [num_services, num_metric_types]

    第一版规则：
    - 正常窗口：inject_time 前的数据
    - 故障窗口：inject_time 后的数据
    - 若故障窗口均值 > 正常窗口均值 + 2 * 正常窗口标准差，则记为异常 1
    - 否则为 0
    """
    features = np.zeros((len(SERVICES), len(METRIC_TYPES)), dtype=np.float32)

    if "time" not in df.columns:
        raise ValueError("data.csv does not contain 'time' column")

    normal_df = df[df["time"] < inject_time]
    fault_df = df[df["time"] >= inject_time]

    if len(normal_df) == 0 or len(fault_df) == 0:
        raise ValueError(f"empty normal or fault window, inject_time={inject_time}")

    for col in df.columns:
        parsed = match_metric_column(col)
        if parsed is None:
            continue

        service, metric = parsed
        s_id = SERVICE_TO_ID[service]
        m_id = METRIC_TO_ID[metric]

        normal_values = pd.to_numeric(normal_df[col], errors="coerce").dropna()
        fault_values = pd.to_numeric(fault_df[col], errors="coerce").dropna()

        if len(normal_values) == 0 or len(fault_values) == 0:
            continue

        normal_mean = normal_values.mean()
        normal_std = normal_values.std()
        fault_mean = fault_values.mean()

        if np.isnan(normal_std):
            normal_std = 0.0

        threshold = normal_mean + 2.0 * normal_std

        if fault_mean > threshold:
            features[s_id, m_id] = 1.0

    return features


def make_case_level_split(num_cases, nodes_per_case, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    按故障 case 划分 train / val / test。
    每个 case 的所有服务节点作为整体进入同一个 split，避免 case leakage。
    """
    rng = np.random.default_rng(seed)

    case_ids = np.arange(num_cases, dtype=np.int64)
    rng.shuffle(case_ids)

    n_train = int(num_cases * train_ratio)
    n_val = int(num_cases * val_ratio)

    train_case_ids = case_ids[:n_train]
    val_case_ids = case_ids[n_train:n_train + n_val]
    test_case_ids = case_ids[n_train + n_val:]

    def cases_to_node_indices(selected_case_ids):
        node_indices = [
            np.arange(case_id * nodes_per_case, (case_id + 1) * nodes_per_case, dtype=np.int64)
            for case_id in selected_case_ids
        ]
        return np.concatenate(node_indices) if node_indices else np.array([], dtype=np.int64)

    idx_train = cases_to_node_indices(train_case_ids)
    idx_val = cases_to_node_indices(val_case_ids)
    idx_test = cases_to_node_indices(test_case_ids)

    return idx_train, idx_val, idx_test, train_case_ids, val_case_ids, test_case_ids


def main():
    if not RCA_ROOT.exists():
        raise FileNotFoundError(f"RCA root not found: {RCA_ROOT}")

    case_dirs = sorted([p for p in RCA_ROOT.iterdir() if p.is_dir()])
    print(f"Found case groups: {len(case_dirs)}")

    single_adj = build_single_adj()

    all_features = []
    all_labels = []
    all_adjs = []
    node_meta = []
    case_meta = []

    case_id = 0

    for case_dir in case_dirs:
        root_service, fault_type = parse_case_dir(case_dir)

        repeat_dirs = sorted([p for p in case_dir.iterdir() if p.is_dir()], key=lambda x: int(x.name))

        for rep_dir in repeat_dirs:
            data_path = rep_dir / "data.csv"
            inject_path = rep_dir / "inject_time.txt"

            if not data_path.exists() or not inject_path.exists():
                continue

            df = pd.read_csv(data_path)
            inject_time = read_inject_time(inject_path)

            features = extract_binary_features(df, inject_time)

            labels = np.zeros(len(SERVICES), dtype=np.int64)
            if root_service in SERVICE_TO_ID:
                labels[SERVICE_TO_ID[root_service]] = 1
            else:
                print(f"[WARN] root service not in SERVICES: {root_service}, case={rep_dir}")

            all_features.append(features)
            all_labels.append(labels)
            all_adjs.append(single_adj)

            for service in SERVICES:
                node_meta.append({
                    "global_node_id": len(node_meta),
                    "case_id": case_id,
                    "service": service,
                    "root_service": root_service,
                    "fault_type": fault_type,
                    "repeat": rep_dir.name,
                    "is_root_cause": int(service == root_service),
                    "case_path": str(rep_dir),
                })

            case_meta.append({
                "case_id": case_id,
                "case_group": case_dir.name,
                "root_service": root_service,
                "fault_type": fault_type,
                "repeat": rep_dir.name,
                "data_path": str(data_path),
                "inject_time": inject_time,
                "num_rows": int(df.shape[0]),
                "num_columns": int(df.shape[1]),
            })

            case_id += 1

    features_big = np.vstack(all_features).astype(np.float32)
    labels_big = np.concatenate(all_labels).astype(np.int64)

    adj_big = block_diag(all_adjs, format="csr").astype(np.float32)

    idx_train, idx_val, idx_test, train_case_ids, val_case_ids, test_case_ids = make_case_level_split(
        case_id,
        len(SERVICES),
        seed=42,
    )

    # 保存
    np.save(OUT_DIR / "features.npy", features_big)
    np.save(OUT_DIR / "labels.npy", labels_big)
    np.save(OUT_DIR / "idx_train.npy", idx_train)
    np.save(OUT_DIR / "idx_val.npy", idx_val)
    np.save(OUT_DIR / "idx_test.npy", idx_test)
    save_npz(OUT_DIR / "adj.npz", adj_big)

    with open(OUT_DIR / "service_names.json", "w", encoding="utf-8") as f:
        json.dump(SERVICES, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "node_meta.json", "w", encoding="utf-8") as f:
        json.dump(node_meta, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "case_meta.json", "w", encoding="utf-8") as f:
        json.dump(case_meta, f, indent=2, ensure_ascii=False)

    # 统计信息
    stats = {
        "dataset": "RCAEval RE1-OB",
        "num_cases": case_id,
        "num_services_per_case": len(SERVICES),
        "num_nodes": int(features_big.shape[0]),
        "num_edges_single_graph_undirected": int(single_adj.sum() // 2),
        "num_edges_block_graph_undirected": int(adj_big.sum() // 2),
        "num_features": int(features_big.shape[1]),
        "num_positive_nodes": int(labels_big.sum()),
        "num_negative_nodes": int((labels_big == 0).sum()),
        "feature_density": float(features_big.mean()),
        "train_size": int(len(idx_train)),
        "val_size": int(len(idx_val)),
        "test_size": int(len(idx_test)),
        "split_strategy": "case_level",
        "train_cases": int(len(train_case_ids)),
        "val_cases": int(len(val_case_ids)),
        "test_cases": int(len(test_case_ids)),
        "train_case_ids": train_case_ids.astype(int).tolist(),
        "val_case_ids": val_case_ids.astype(int).tolist(),
        "test_case_ids": test_case_ids.astype(int).tolist(),
        "services": SERVICES,
        "metric_types": METRIC_TYPES,
    }

    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nSaved to:", OUT_DIR)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
