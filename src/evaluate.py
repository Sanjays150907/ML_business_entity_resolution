import numpy as np
from src.utils import entity_level_f05


def score_predictions(source1_ids, pair_rows, probabilities, threshold, truth):
    pred = {sid: set() for sid in source1_ids}

    for (i, target_idx, target_id), p in zip(pair_rows, probabilities):
        if p >= threshold:
            pred[source1_ids[i]].add(target_id)

    scores = []
    for sid in source1_ids:
        scores.append(entity_level_f05(pred[sid], truth.get(sid, set())))

    return float(np.mean(scores)), pred


def find_best_threshold(source1_ids, pair_rows, probabilities, truth):
    # Precision-heavy metric: search a practical high-confidence range.
    thresholds = np.arange(0.30, 0.951, 0.025)
    best_t, best_score = 0.70, -1

    for t in thresholds:
        score, _ = score_predictions(
            source1_ids, pair_rows, probabilities, t, truth
        )
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, best_score
