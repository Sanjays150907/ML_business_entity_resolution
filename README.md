# Business Entity Resolution

End-to-end ML pipeline for the Business Entity Resolution Challenge.

## Run in the browser

```bash
python -m pip install -r requirements.txt
python app.py
```

Your browser opens at http://127.0.0.1:5000 (use `--port 8000` or `--no-browser` if needed). The UI has four tabs:

- **Dashboard** – validation F0.5, threshold, counts, match distribution, country match rates, output downloads.
- **Results** – searchable/filterable list of all test Source-1 records; click a row to see every candidate with its match probability and a threshold slider.
- **Try a record** – type any business name/address/country and get scored matches from the test Source-2/3 data.
- **Run pipeline** – trains the model and regenerates the `output/` files with a live log.

## Run from the command line

From the `business_entity_resolution` project folder:

```powershell
python -m src.main --data-dir dataset --output-dir output
```

You can also run it directly:

```powershell
python src\main.py --data-dir dataset --output-dir output
```

Optional validation split:

```bash
python -m src.main --data-dir dataset --output-dir output --val-fraction 0.20
```

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

The program creates `matching_results.tsv`, `candidate_pairs.tsv`, `validation_report.json`, and `model.joblib` in `output/`.
