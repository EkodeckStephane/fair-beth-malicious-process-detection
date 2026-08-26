#!/usr/bin/env python3
"""Major-revision data-accounting audit for BETH.

This script answers the dataset-accounting questions raised during peer review.
It never trains a model and never changes the scientific split.  It inventories
all BETH CSV files, verifies raw row counts and hashes, computes process-level
counts when the schema supports (hostName, processId), records DNS/non-DNS
schema differences, and checks exact host-name overlap with the official test
file.

Outputs are written to results/revision_audits/.  The raw dataset is not copied
into the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

GROUP_COLS = ["hostName", "processId"]
OFFICIAL_TEST = "labelled_testing_data.csv"
OFFICIAL_TRAIN = "labelled_training_data.csv"
OFFICIAL_VALIDATION = "labelled_validation_data.csv"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def schema_signature(columns: list[str]) -> str:
    payload = "\n".join(columns).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def safe_min(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.min()) if len(values) else None


def safe_max(series: pd.Series) -> float | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.max()) if len(values) else None


def inspect_csv(path: Path) -> tuple[dict[str, Any], set[str]]:
    df = pd.read_csv(path, low_memory=False)
    columns = list(df.columns)
    hosts: set[str] = set()
    if "hostName" in df.columns:
        hosts = set(df["hostName"].dropna().astype(str).unique().tolist())

    evil_events = None
    if "evil" in df.columns:
        evil = pd.to_numeric(df["evil"], errors="coerce").fillna(0)
        evil_events = int((evil > 0).sum())

    processes = None
    evil_processes = None
    if all(col in df.columns for col in GROUP_COLS):
        grouped = df.groupby(GROUP_COLS, dropna=False)
        processes = int(grouped.ngroups)
        if "evil" in df.columns:
            proc_labels = grouped["evil"].max()
            evil_processes = int((pd.to_numeric(proc_labels, errors="coerce").fillna(0) > 0).sum())

    row = {
        "file": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "rows": int(len(df)),
        "columns": int(len(columns)),
        "schema_signature": schema_signature(columns),
        "is_dns": path.name.endswith("-dns.csv"),
        "is_official_train": path.name == OFFICIAL_TRAIN,
        "is_official_validation": path.name == OFFICIAL_VALIDATION,
        "is_official_test": path.name == OFFICIAL_TEST,
        "is_2021may_supplement": path.name.startswith("labelled_2021may"),
        "evil_events": evil_events,
        "processes": processes,
        "evil_processes": evil_processes,
        "unique_hosts": int(len(hosts)) if "hostName" in df.columns else None,
        "first_timestamp": safe_min(df["timestamp"]) if "timestamp" in df.columns else None,
        "last_timestamp": safe_max(df["timestamp"]) if "timestamp" in df.columns else None,
        "column_names": "|".join(columns),
    }
    return row, hosts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        default=Path("results/revision_audits"),
        type=Path,
    )
    args = parser.parse_args()

    csv_paths = sorted(args.data_dir.rglob("*.csv"))
    if not csv_paths:
        raise SystemExit(f"No CSV files found below {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    hosts_by_file: dict[str, set[str]] = {}
    for path in csv_paths:
        print(f"[audit] {path.name}", flush=True)
        row, hosts = inspect_csv(path)
        rows.append(row)
        hosts_by_file[path.name] = hosts
        del hosts

    inventory = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    inventory.to_csv(args.output_dir / "dataset_inventory.csv", index=False)

    test_hosts = hosts_by_file.get(OFFICIAL_TEST, set())
    overlap_rows: list[dict[str, Any]] = []
    for filename, hosts in sorted(hosts_by_file.items()):
        if filename == OFFICIAL_TEST:
            continue
        overlap = sorted(hosts & test_hosts)
        overlap_rows.append(
            {
                "file": filename,
                "host_count": len(hosts),
                "official_test_host_count": len(test_hosts),
                "overlap_count": len(overlap),
                "overlap_hosts": "|".join(overlap),
            }
        )
    pd.DataFrame(overlap_rows).to_csv(args.output_dir / "host_overlap.csv", index=False)

    dns = inventory[inventory["is_dns"]].copy()
    non_dns = inventory[~inventory["is_dns"]].copy()
    schema_rows: list[dict[str, Any]] = []
    for _, drow in dns.iterrows():
        base_name = drow["file"].replace("-dns.csv", ".csv")
        peer = non_dns[non_dns["file"] == base_name]
        schema_rows.append(
            {
                "dns_file": drow["file"],
                "paired_process_file": base_name if len(peer) else None,
                "dns_schema_signature": drow["schema_signature"],
                "process_schema_signature": peer.iloc[0]["schema_signature"] if len(peer) else None,
                "same_schema": bool(len(peer) and drow["schema_signature"] == peer.iloc[0]["schema_signature"]),
                "dns_columns": drow["column_names"],
                "process_columns": peer.iloc[0]["column_names"] if len(peer) else None,
            }
        )
    pd.DataFrame(schema_rows).to_csv(args.output_dir / "dns_schema_comparison.csv", index=False)

    by_name = inventory.set_index("file")
    val_rows = int(by_name.loc[OFFICIAL_VALIDATION, "rows"]) if OFFICIAL_VALIDATION in by_name.index else None
    test_rows = int(by_name.loc[OFFICIAL_TEST, "rows"]) if OFFICIAL_TEST in by_name.index else None

    supplement = inventory[(inventory["is_2021may_supplement"]) & (~inventory["is_dns"])].copy()
    summary = {
        "csv_file_count": int(len(inventory)),
        "dns_file_count": int(inventory["is_dns"].sum()),
        "non_dns_file_count": int((~inventory["is_dns"]).sum()),
        "official_validation_rows": val_rows,
        "official_test_rows": test_rows,
        "validation_and_test_row_counts_equal": bool(val_rows is not None and test_rows is not None and val_rows == test_rows),
        "official_test_hosts": sorted(test_hosts),
        "supplement_non_dns_files": supplement["file"].tolist(),
        "supplement_evil_processes_by_file": {
            str(row["file"]): (None if pd.isna(row["evil_processes"]) else int(row["evil_processes"]))
            for _, row in supplement.iterrows()
        },
        "supplement_total_evil_processes_sum_by_file": int(
            pd.to_numeric(supplement["evil_processes"], errors="coerce").fillna(0).sum()
        ),
    }
    with (args.output_dir / "dataset_audit_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print("\n=== DATASET AUDIT SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("\n=== INVENTORY (selected columns) ===")
    print(
        inventory[
            [
                "file",
                "rows",
                "evil_events",
                "processes",
                "evil_processes",
                "unique_hosts",
                "schema_signature",
            ]
        ].to_string(index=False)
    )
    print("\n=== HOST OVERLAP WITH OFFICIAL TEST ===")
    print(pd.DataFrame(overlap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
