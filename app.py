"""
Browser UI for the Business Entity Resolution pipeline.

Run:
    python app.py
then open http://127.0.0.1:5000  (it opens automatically).
"""
import argparse
import contextlib
import io
import json
import os
import sys
import threading
import time
import webbrowser
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from src.utils import add_normalized_columns
from src.blocking import CandidateGenerator
from src.features import build_pair_matrix

DATA_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DOWNLOADS = {
    "matching_results.tsv", "candidate_pairs.tsv",
    "validation_report.json", "model.joblib",
}

app = Flask(__name__)


# --------------------------------------------------------------------------
# Lazy cache of test data, model and candidate generator
# --------------------------------------------------------------------------
class Cache:
    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        self.loaded = False
        self.s1 = self.targets = self.gen = None
        self.model = None
        self.threshold = 0.5
        self.id_to_idx = {}
        self.s1_to_idx = {}
        self._file_mtimes = {}
        self._file_data = {}

    def _read_tsv(self, path):
        return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)

    def ensure(self):
        with self.lock:
            if self.loaded:
                return
            base = os.path.join(DATA_DIR, "test")
            s1 = add_normalized_columns(self._read_tsv(os.path.join(base, "test_source1.tsv")))
            s2 = add_normalized_columns(self._read_tsv(os.path.join(base, "test_source2.tsv")))
            s3 = add_normalized_columns(self._read_tsv(os.path.join(base, "test_source3.tsv")))
            self.s1 = s1
            self.targets = pd.concat([s2, s3], ignore_index=True)
            self.id_to_idx = {t: i for i, t in enumerate(self.targets["entity_id"])}
            self.s1_to_idx = {t: i for i, t in enumerate(self.s1["entity_id"])}
            self.gen = CandidateGenerator(max_approx=10).fit(self.targets)

            model_path = os.path.join(OUTPUT_DIR, "model.joblib")
            if os.path.exists(model_path):
                bundle = joblib.load(model_path)
                self.model = bundle["model"]
                self.threshold = float(bundle["threshold"])
            self.loaded = True

    def output_table(self, name):
        """Read an output TSV as {source1_id: [ids]}; cached by mtime."""
        path = os.path.join(OUTPUT_DIR, name)
        if not os.path.exists(path):
            return None
        m = os.path.getmtime(path)
        if self._file_mtimes.get(name) != m:
            df = self._read_tsv(path)
            col = df.columns[1]
            self._file_data[name] = {
                r[0]: [x for x in r[1].split(",") if x]
                for r in df[[df.columns[0], col]].itertuples(index=False)
            }
            self._file_mtimes[name] = m
        return self._file_data[name]


cache = Cache()


def rec(df, idx):
    r = df.iloc[idx]
    return {
        "id": r["entity_id"], "name": r["business_name"],
        "address": r["business_address"], "country": r["country"],
    }


def score_pairs(left_df, target_indices):
    """Probabilities for one left row (index 0 of left_df) vs target rows."""
    if cache.model is None or not target_indices:
        return [None] * len(target_indices)
    X = build_pair_matrix(left_df, cache.targets, [(0, j) for j in target_indices])
    return [float(p) for p in cache.model.predict_proba(X)]


# --------------------------------------------------------------------------
# Pipeline job (runs src.main.run in a background thread)
# --------------------------------------------------------------------------
job = {"state": "idle", "log": "", "started": None, "finished": None, "error": None}
job_lock = threading.Lock()


class LogWriter(io.TextIOBase):
    def write(self, s):
        with job_lock:
            job["log"] += s
        return len(s)


def _run_job(val_fraction):
    from src.main import run
    args = SimpleNamespace(data_dir=DATA_DIR, output_dir=OUTPUT_DIR, val_fraction=val_fraction)
    try:
        with contextlib.redirect_stdout(LogWriter()):
            run(args)
        with job_lock:
            job["state"] = "done"
    except Exception as e:  # noqa: BLE001
        with job_lock:
            job["state"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
    finally:
        with job_lock:
            job["finished"] = time.time()
        cache.reset()


@app.post("/api/run")
def api_run():
    body = request.get_json(silent=True) or {}
    try:
        vf = float(body.get("val_fraction", 0.2))
    except (TypeError, ValueError):
        return jsonify(error="val_fraction must be a number"), 400
    if not 0.05 <= vf <= 0.5:
        return jsonify(error="val_fraction must be between 0.05 and 0.5"), 400
    with job_lock:
        if job["state"] == "running":
            return jsonify(error="A run is already in progress"), 409
        job.update(state="running", log="", started=time.time(), finished=None, error=None)
    threading.Thread(target=_run_job, args=(vf,), daemon=True).start()
    return jsonify(ok=True)


@app.get("/api/job")
def api_job():
    with job_lock:
        j = dict(job)
    j["now"] = time.time()
    return jsonify(j)


# --------------------------------------------------------------------------
# Read-only API
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/summary")
def api_summary():
    report = None
    rp = os.path.join(OUTPUT_DIR, "validation_report.json")
    if os.path.exists(rp):
        with open(rp, encoding="utf-8") as f:
            report = json.load(f)

    files = {}
    for name in sorted(DOWNLOADS):
        p = os.path.join(OUTPUT_DIR, name)
        if os.path.exists(p):
            files[name] = {"size": os.path.getsize(p), "modified": os.path.getmtime(p)}

    out = {"report": report, "files": files, "distribution": None}
    matches = cache.output_table("matching_results.tsv")
    if matches is not None:
        cache.ensure()
        dist, by_country, src_counts = {}, {}, {"S2": 0, "S3": 0}
        country_of = dict(zip(cache.s1["entity_id"], cache.s1["country"]))
        for sid, ids in matches.items():
            n = min(len(ids), 4)
            dist[n] = dist.get(n, 0) + 1
            c = country_of.get(sid, "?")
            d = by_country.setdefault(c, {"total": 0, "matched": 0})
            d["total"] += 1
            d["matched"] += 1 if ids else 0
            for t in ids:
                src_counts[t[:2]] = src_counts.get(t[:2], 0) + 1
        out["distribution"] = {
            "matches_per_record": [{"n": k, "count": dist[k]} for k in sorted(dist)],
            "by_country": [{"country": k, **v} for k, v in sorted(by_country.items())],
            "by_source": src_counts,
        }
    return jsonify(out)


@app.get("/api/results")
def api_results():
    matches = cache.output_table("matching_results.tsv")
    cands = cache.output_table("candidate_pairs.tsv")
    if matches is None:
        return jsonify(error="No results yet. Run the pipeline first.", rows=[], total=0), 200
    cache.ensure()

    q = (request.args.get("q") or "").strip().lower()
    flt = request.args.get("filter", "all")
    page = max(1, int(request.args.get("page", 1)))
    per = min(100, max(5, int(request.args.get("per", 25))))

    df = cache.s1
    rows = []
    for r in df.itertuples(index=False):
        sid = r.entity_id
        ids = matches.get(sid, [])
        if flt == "matched" and not ids:
            continue
        if flt == "unmatched" and ids:
            continue
        if q and q not in f"{sid} {r.business_name} {r.business_address} {r.country}".lower():
            continue
        rows.append({
            "id": sid, "name": r.business_name, "address": r.business_address,
            "country": r.country, "matches": ids,
            "n_candidates": len((cands or {}).get(sid, [])),
        })
    total = len(rows)
    start = (page - 1) * per
    return jsonify(rows=rows[start:start + per], total=total, page=page, per=per)


@app.get("/api/entity/<sid>")
def api_entity(sid):
    cache.ensure()
    if sid not in cache.s1_to_idx:
        abort(404)
    i = cache.s1_to_idx[sid]
    left = cache.s1.iloc[[i]].reset_index(drop=True)

    cands = cache.output_table("candidate_pairs.tsv")
    matches = cache.output_table("matching_results.tsv")
    if cands is not None and sid in cands:
        idxs = [cache.id_to_idx[t] for t in cands[sid] if t in cache.id_to_idx]
    else:
        idxs = cache.gen.candidates_for_row(left.iloc[0])
    matched = set((matches or {}).get(sid, []))

    probs = score_pairs(left, idxs)
    items = []
    for j, p in zip(idxs, probs):
        r = rec(cache.targets, j)
        r["prob"] = p
        r["matched"] = r["id"] in matched
        items.append(r)
    items.sort(key=lambda x: (x["prob"] is None, -(x["prob"] or 0)))
    return jsonify(source=rec(cache.s1, i), threshold=cache.threshold, candidates=items)


@app.post("/api/match")
def api_match():
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify(error="Business name is required"), 400
    if cache.output_table("matching_results.tsv") is None and not os.path.exists(
            os.path.join(OUTPUT_DIR, "model.joblib")):
        return jsonify(error="No trained model yet. Run the pipeline first."), 400
    cache.ensure()
    if cache.model is None:
        return jsonify(error="No trained model yet. Run the pipeline first."), 400

    q = pd.DataFrame([{
        "entity_id": "QUERY", "business_name": name,
        "business_address": (body.get("address") or "").strip(),
        "country": (body.get("country") or "").strip(),
    }])
    q = add_normalized_columns(q)
    idxs = cache.gen.candidates_for_row(q.iloc[0])
    probs = score_pairs(q, idxs)
    items = []
    for j, p in zip(idxs, probs):
        r = rec(cache.targets, j)
        r["prob"] = p
        r["matched"] = p is not None and p >= cache.threshold
        items.append(r)
    items.sort(key=lambda x: -(x["prob"] or 0))
    return jsonify(threshold=cache.threshold, candidates=items)


@app.get("/download/<name>")
def download(name):
    if name not in DOWNLOADS or not os.path.exists(os.path.join(OUTPUT_DIR, name)):
        abort(404)
    return send_from_directory(OUTPUT_DIR, name, as_attachment=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(f"http://{a.host}:{a.port}")).start()
    app.run(host=a.host, port=a.port, threaded=True, debug=False)
