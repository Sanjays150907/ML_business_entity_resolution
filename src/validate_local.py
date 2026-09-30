import argparse
import os
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matching", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--test-dir", required=True)
    args = p.parse_args()

    m = pd.read_csv(args.matching, sep="\t", dtype=str, keep_default_na=False)
    c = pd.read_csv(args.candidate, sep="\t", dtype=str, keep_default_na=False)

    s1 = pd.read_csv(
        os.path.join(args.test_dir, "test_source1.tsv"),
        sep="\t", dtype=str
    )
    s2 = pd.read_csv(
        os.path.join(args.test_dir, "test_source2.tsv"),
        sep="\t", dtype=str
    )
    s3 = pd.read_csv(
        os.path.join(args.test_dir, "test_source3.tsv"),
        sep="\t", dtype=str
    )

    valid_s1 = set(s1.entity_id)
    valid_targets = set(s2.entity_id) | set(s3.entity_id)

    assert list(m.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(c.columns) == ["source1_entity_id", "candidate_entity_ids"]

    assert len(m) == len(s1), "Wrong number of matching rows"
    assert len(c) == len(s1), "Wrong number of candidate rows"

    assert m.source1_entity_id.is_unique
    assert c.source1_entity_id.is_unique

    assert set(m.source1_entity_id) == valid_s1
    assert set(c.source1_entity_id) == valid_s1

    cand_map = dict(zip(c.source1_entity_id, c.candidate_entity_ids))

    for _, row in m.iterrows():
        sid = row.source1_entity_id
        mids = [x for x in row.matched_entity_ids.split(",") if x]
        assert len(mids) == len(set(mids)), f"Duplicate match: {sid}"
        assert set(mids) <= valid_targets, f"Invalid target ID: {sid}"
        cand = set(x for x in cand_map[sid].split(",") if x)
        assert set(mids) <= cand, f"Match is not in candidate set: {sid}"

    print("PASS")


if __name__ == "__main__":
    main()
