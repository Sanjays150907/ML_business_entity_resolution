import re
import unicodedata
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio, WRatio


LEGAL_WORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "llp", "llc", "pvt", "private", "plc",
    "gmbh", "sa", "sas", "ag", "bv", "pte"
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    s = str(value).lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_name(value):
    s = normalize_text(value)
    tokens = [t for t in s.split() if t not in LEGAL_WORDS]
    return " ".join(tokens)


def normalize_address(value):
    s = normalize_text(value)
    replacements = {
        "street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
        "lane": "ln", "drive": "dr", "apartment": "apt", "floor": "fl",
        "building": "bldg", "highway": "hwy", "place": "pl", "parkway": "pkwy"
    }
    tokens = [replacements.get(t, t) for t in s.split()]
    return " ".join(tokens)


def tokens(s):
    return set(str(s).split()) if s else set()


def jaccard(a, b):
    A, B = tokens(a), tokens(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def safe_ratio(a, b):
    if not a or not b:
        return 0.0
    return ratio(a, b) / 100.0


def w_ratio(a, b):
    if not a or not b:
        return 0.0
    return WRatio(a, b) / 100.0


def extract_numbers(s):
    return set(re.findall(r"\d+", str(s)))


def add_normalized_columns(df):
    out = df.copy()
    out["name_norm"] = out["business_name"].map(normalize_name)
    out["address_norm"] = out["business_address"].map(normalize_address)
    out["country_norm"] = out["country"].map(normalize_text)
    out["name_compact"] = out["name_norm"].str.replace(" ", "", regex=False)
    out["address_compact"] = out["address_norm"].str.replace(" ", "", regex=False)
    out["name_tokens"] = out["name_norm"].map(tokens)
    out["address_tokens"] = out["address_norm"].map(tokens)
    out["address_numbers"] = out["address_norm"].map(extract_numbers)
    return out


def f05(precision, recall):
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return 1.25 * precision * recall / denom


def entity_level_f05(pred, truth):
    pred = set(pred)
    truth = set(truth)
    if not pred and not truth:
        return 1.0
    if not pred:
        return 0.0
    tp = len(pred & truth)
    precision = tp / len(pred)
    recall = tp / len(truth) if truth else 0.0
    return f05(precision, recall)


def parse_truth(df):
    mapping = {}
    for _, row in df.iterrows():
        s1 = str(row["source1_entity_id"])
        raw = "" if pd.isna(row["matched_entity_ids"]) else str(row["matched_entity_ids"])
        ids = [x.strip() for x in raw.split(",") if x.strip()]
        mapping[s1] = set(ids)
    return mapping
