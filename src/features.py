import numpy as np
import pandas as pd
from src.utils import safe_ratio, w_ratio, jaccard


FEATURE_NAMES = [
    "name_ratio",
    "name_wratio",
    "name_jaccard",
    "address_ratio",
    "address_wratio",
    "address_jaccard",
    "country_equal",
    "name_exact",
    "address_exact",
    "name_prefix_equal",
    "address_numbers_overlap",
    "name_len_diff",
    "address_len_diff",
]


def pair_features(a, b):
    name_a, name_b = a["name_norm"], b["name_norm"]
    addr_a, addr_b = a["address_norm"], b["address_norm"]

    nums_a = a["address_numbers"]
    nums_b = b["address_numbers"]

    number_overlap = (
        len(nums_a & nums_b) / len(nums_a | nums_b)
        if nums_a or nums_b else 1.0
    )

    return [
        safe_ratio(name_a, name_b),
        w_ratio(name_a, name_b),
        jaccard(name_a, name_b),
        safe_ratio(addr_a, addr_b),
        w_ratio(addr_a, addr_b),
        jaccard(addr_a, addr_b),
        float(a["country_norm"] == b["country_norm"]),
        float(bool(name_a) and name_a == name_b),
        float(bool(addr_a) and addr_a == addr_b),
        float(bool(name_a) and bool(name_b) and
              (name_a[:5] == name_b[:5])),
        number_overlap,
        min(abs(len(name_a) - len(name_b)), 100) / 100.0,
        min(abs(len(addr_a) - len(addr_b)), 200) / 200.0,
    ]


def build_pair_matrix(left_df, right_df, pairs):
    X = np.zeros((len(pairs), len(FEATURE_NAMES)), dtype=float)
    for k, (i, j) in enumerate(pairs):
        X[k] = pair_features(left_df.iloc[i], right_df.iloc[j])
    return X
