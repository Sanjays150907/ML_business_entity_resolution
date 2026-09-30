# Business Entity Resolution — Methodology

## 1. Methodology Used

Describe the complete pipeline:
- data loading
- text normalization
- blocking
- pairwise feature engineering
- ML classifier
- threshold selection
- final inference

## 2. Candidate Generation / Blocking

The implementation uses multiple blocking keys and approximate name retrieval:
- normalized name prefixes
- name tokens
- address tokens
- country + name prefix
- country
- TF-IDF character n-gram nearest-neighbor retrieval

Explain why blocking is necessary and how it affects recall and computational cost.

## 3. Model Architecture

The pair classifier is Logistic Regression with:
- StandardScaler
- class_weight="balanced"
- deterministic random_state

Features include name/address similarity, token Jaccard similarity, country equality, exact matches, prefix agreement, address-number overlap, and length differences.

## 4. Threshold Selection

A Source-1-level validation split is used. Candidate pair probabilities are converted into entity match lists. The probability threshold is selected by maximizing the challenge's macro-average F_0.5 on the validation entities.

## 5. Precision / Recall Considerations

The challenge uses F_0.5, which gives greater importance to precision and penalizes false merges more heavily. Singleton predictions are therefore important.

## 6. Fair Play

The pipeline uses only the supplied challenge data. It does not call external business databases, APIs, geocoders, search engines, or internet sources.
