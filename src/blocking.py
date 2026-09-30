import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


class CandidateGenerator:
    """Generate a small, high-recall candidate set for each Source-1 row."""

    def __init__(self, max_approx=10, query_batch_size=256):
        self.max_approx = max_approx
        self.query_batch_size = query_batch_size

    def fit(self, target_df):
        self.target_df = target_df.reset_index(drop=True)
        self.exact_name = defaultdict(list)
        self.exact_address = defaultdict(list)

        for i, r in self.target_df.iterrows():
            name = r["name_norm"]
            address = r["address_norm"]
            country = r["country_norm"]
            if name:
                self.exact_name[(country, name)].append(i)
            if address:
                self.exact_address[(country, address)].append(i)

        corpus = self.target_df["name_norm"].fillna("").tolist()
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5),
            min_df=1, sublinear_tf=True
        )
        X = self.vectorizer.fit_transform(corpus)
        self.nn = NearestNeighbors(
            n_neighbors=min(self.max_approx, max(1, len(corpus))),
            metric="cosine", algorithm="brute", n_jobs=-1
        )
        self.nn.fit(X)
        self.target_countries = self.target_df["country_norm"].to_numpy()
        return self

    def _exact_candidates(self, row):
        country = row["country_norm"]
        cand = set()
        if row["name_norm"]:
            cand.update(self.exact_name.get((country, row["name_norm"]), []))
        if row["address_norm"]:
            cand.update(self.exact_address.get((country, row["address_norm"]), []))
        return cand

    def candidates_for_row(self, row):
        cand = self._exact_candidates(row)
        q = self.vectorizer.transform([row["name_norm"]])
        _, inds = self.nn.kneighbors(q, n_neighbors=self.max_approx)
        same_country = [
            int(j) for j in inds[0]
            if self.target_countries[int(j)] == row["country_norm"]
        ]
        # Keep a small global fallback only when country-filtered retrieval
        # finds nothing; this avoids huge candidate sets for common names.
        cand.update(same_country)
        if not same_country:
            cand.update(map(int, inds[0][:min(10, len(inds[0]))]))
        return sorted(cand)

    def generate(self, source1_df):
        source1_df = source1_df.reset_index(drop=True)
        rows = []
        for start in range(0, len(source1_df), self.query_batch_size):
            stop = min(start + self.query_batch_size, len(source1_df))
            batch = source1_df.iloc[start:stop]
            queries = self.vectorizer.transform(batch["name_norm"].fillna("").tolist())
            _, batch_inds = self.nn.kneighbors(
                queries, n_neighbors=self.max_approx
            )

            for local_idx, (_, row) in enumerate(batch.iterrows()):
                cand = self._exact_candidates(row)
                same_country = [
                    int(j) for j in batch_inds[local_idx]
                    if self.target_countries[int(j)] == row["country_norm"]
                ]
                cand.update(same_country)
                if not same_country:
                    cand.update(map(int, batch_inds[local_idx][:10]))
                source_idx = start + local_idx
                rows.extend((source_idx, j) for j in sorted(cand))
        return rows
