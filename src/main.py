import argparse
import json
import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd

from src.utils import add_normalized_columns, parse_truth
from src.blocking import CandidateGenerator
from src.features import build_pair_matrix
from src.model import PairMatcher
from src.evaluate import find_best_threshold


def read_tsv(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_train(data_dir):
    base = os.path.join(data_dir, "train")
    s1 = add_normalized_columns(read_tsv(os.path.join(base, "train_source1.tsv")))
    s2 = add_normalized_columns(read_tsv(os.path.join(base, "train_source2.tsv")))
    s3 = add_normalized_columns(read_tsv(os.path.join(base, "train_source3.tsv")))
    gt = read_tsv(os.path.join(base, "train_ground_truth.tsv"))
    return s1, s2, s3, parse_truth(gt)


def load_test(data_dir):
    base = os.path.join(data_dir, "test")
    s1 = add_normalized_columns(read_tsv(os.path.join(base, "test_source1.tsv")))
    s2 = add_normalized_columns(read_tsv(os.path.join(base, "test_source2.tsv")))
    s3 = add_normalized_columns(read_tsv(os.path.join(base, "test_source3.tsv")))
    return s1, s2, s3


def make_training_pairs(s1, targets, truth):
    gen = CandidateGenerator(max_approx=10)
    gen.fit(targets)
    pairs = gen.generate(s1)

    source_ids = s1["entity_id"].to_numpy()
    target_ids = targets["entity_id"].to_numpy()

    pair_rows = []
    y = []
    existing = set()

    # Precomputed ID arrays avoid expensive DataFrame.iloc lookups for every pair.
    for i, j in pairs:
        sid = source_ids[i]
        tid = target_ids[j]
        pair_rows.append((i, j, tid))
        y.append(int(tid in truth.get(sid, set())))
        existing.add((i, j))

    # Add any positive pair missed by blocking so training cannot lose positives.
    id_to_idx = {tid: j for j, tid in enumerate(target_ids)}
    for i, sid in enumerate(source_ids):
        for tid in truth.get(sid, set()):
            j = id_to_idx.get(tid)
            if j is not None and (i, j) not in existing:
                pair_rows.append((i, j, tid))
                y.append(1)
                existing.add((i, j))

    return pair_rows, np.array(y, dtype=int), gen


def split_source1(s1, fraction=0.20, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(s1))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * fraction))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def subset_truth(truth, s1):
    return {sid: truth.get(sid, set()) for sid in s1["entity_id"]}


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/6] Loading training data...", flush=True)
    s1, s2, s3, truth = load_train(args.data_dir)
    train_counts = (len(s1), len(s2), len(s3))
    targets = pd.concat([s2, s3], ignore_index=True)

    train_idx, val_idx = split_source1(s1, args.val_fraction)
    train_s1 = s1.iloc[train_idx].reset_index(drop=True)
    val_s1 = s1.iloc[val_idx].reset_index(drop=True)

    print("[2/6] Blocking + features for validation split...", flush=True)
    # Validation/training split.
    train_truth = subset_truth(truth, train_s1)
    train_rows, y, train_gen = make_training_pairs(train_s1, targets, train_truth)
    X_train = build_pair_matrix(
        train_s1, targets, [(i, j) for i, j, _ in train_rows]
    )

    if len(np.unique(y)) < 2:
        raise RuntimeError(
            "Training pairs contain only one class. "
            "Increase data or inspect ground truth/blocking."
        )

    print("[3/6] Training model and selecting threshold...", flush=True)
    model = PairMatcher().fit(X_train, y)

    val_gen = CandidateGenerator(max_approx=10).fit(targets)
    val_pairs = val_gen.generate(val_s1)
    val_rows = [
        (i, j, targets.iloc[j]["entity_id"]) for i, j in val_pairs
    ]
    X_val = build_pair_matrix(
        val_s1, targets, [(i, j) for i, j, _ in val_rows]
    )
    val_prob = model.predict_proba(X_val)

    val_truth = subset_truth(truth, val_s1)
    val_ids = val_s1["entity_id"].tolist()
    threshold, val_score = find_best_threshold(
        val_ids, val_rows, val_prob, val_truth
    )

    # Release validation/training matrices before building the final model.
    del X_train, X_val, train_rows, y, val_rows, val_prob

    print("[4/6] Training final model on all training records...", flush=True)
    # Final model uses all Source-1 training records.
    full_rows, y_full, _ = make_training_pairs(s1, targets, truth)
    X_full = build_pair_matrix(
        s1, targets, [(i, j) for i, j, _ in full_rows]
    )
    if len(np.unique(y_full)) < 2:
        raise RuntimeError("Full training pairs contain only one class.")
    final_training_pairs = len(y_full)
    final_positive_pairs = int(y_full.sum())
    final_model = PairMatcher().fit(X_full, y_full)
    del X_full, full_rows, y_full
    import gc
    del train_s1, val_s1, train_truth, train_idx, val_idx, train_gen, val_gen
    del s2, s3, targets, truth, model
    gc.collect()

    print("[5/6] Running inference on test data...", flush=True)
    # Test inference.
    test_s1, test_s2, test_s3 = load_test(args.data_dir)
    test_targets = pd.concat([test_s2, test_s3], ignore_index=True)

    test_gen = CandidateGenerator(max_approx=10).fit(test_targets)
    test_pairs = test_gen.generate(test_s1)
    test_rows = [
        (i, j, test_targets.iloc[j]["entity_id"])
        for i, j in test_pairs
    ]
    # Build/predict test features in chunks to keep peak memory low.
    test_prob_parts = []
    chunk_size = 5000
    for start in range(0, len(test_rows), chunk_size):
        chunk = test_rows[start:start + chunk_size]
        X_chunk = build_pair_matrix(
            test_s1, test_targets, [(i, j) for i, j, _ in chunk]
        )
        test_prob_parts.append(final_model.predict_proba(X_chunk))
        del X_chunk
    test_prob = np.concatenate(test_prob_parts) if test_prob_parts else np.array([])
    del test_prob_parts

    matches = {sid: [] for sid in test_s1["entity_id"]}
    candidates = {sid: [] for sid in test_s1["entity_id"]}

    for row, p in zip(test_rows, test_prob):
        i, _, tid = row
        sid = test_s1.iloc[i]["entity_id"]
        candidates[sid].append(tid)
        if p >= threshold:
            matches[sid].append(tid)

    for sid in matches:
        matches[sid] = list(dict.fromkeys(matches[sid]))
        candidates[sid] = list(dict.fromkeys(candidates[sid]))

    print("[6/6] Writing output files...", flush=True)
    matching_path = os.path.join(args.output_dir, "matching_results.tsv")
    candidate_path = os.path.join(args.output_dir, "candidate_pairs.tsv")

    matching_df = pd.DataFrame({
        "source1_entity_id": test_s1["entity_id"].tolist(),
        "matched_entity_ids": [
            ",".join(matches[sid]) for sid in test_s1["entity_id"]
        ]
    })
    candidate_df = pd.DataFrame({
        "source1_entity_id": test_s1["entity_id"].tolist(),
        "candidate_entity_ids": [
            ",".join(candidates[sid]) for sid in test_s1["entity_id"]
        ]
    })

    matching_df.to_csv(matching_path, sep="\t", index=False)
    candidate_df.to_csv(candidate_path, sep="\t", index=False)

    report = {
        "validation_f05": val_score,
        "selected_probability_threshold": threshold,
        "train_source1_records": train_counts[0],
        "train_source2_records": train_counts[1],
        "train_source3_records": train_counts[2],
        "test_source1_records": len(test_s1),
        "test_source2_records": len(test_s2),
        "test_source3_records": len(test_s3),
        "training_pairs": int(final_training_pairs),
        "positive_training_pairs": int(final_positive_pairs),
        "test_candidate_pairs": int(len(test_rows)),
        "test_predicted_matches": int(sum(len(v) for v in matches.values())),
    }


    with open(
        os.path.join(args.output_dir, "validation_report.json"),
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2)

    joblib.dump(
        {
            "model": final_model,
            "threshold": threshold,
            "feature_version": "v1",
        },
        os.path.join(args.output_dir, "model.joblib")
    )

    print(json.dumps(report, indent=2))
    print(f"\nCreated: {matching_path}")
    print(f"Created: {candidate_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    args = parser.parse_args()
    run(args)
