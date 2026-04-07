# FitNova - AI-Powered Fitness Planning

FitNova is a graduation project for personalized fitness planning. The backend uses a 3-layer recommendation pipeline that combines rule-based content filtering, NeuMF collaborative re-ranking, and an LLM adaptation layer for weekly plan generation.

## Architecture

```text
User Profile (8 inputs)
    ->
Layer 1: Content-Based Filtering (weighted compatibility)
    -> Returns top-50 candidate programs
Layer 2: NeuMF (Neural Matrix Factorization)
    -> Re-ranks candidates and selects top program
Layer 3: LLM Adaptation (Gemini / Claude)
    -> Personalizes the weekly plan
Final Output: JSON weekly plan
```

### Layer 1
- Matches a user profile against 2,598 programs with weighted compatibility rules.
- Uses experience level, workout-type intent, session duration, and weekly frequency.
- Solves the cold-start problem immediately for new users.

### Layer 2
- Implements He et al. (2017) Neural Collaborative Filtering.
- Combines GMF and MLP into a fused NeuMF recommender.
- Trained on 809,439 synthetic interactions across 3,048 programs (2,598 real + 450 augmented).
- Current performance: `HR@10 = 0.9270`, `NDCG@10 = 0.7469`.

### Layer 3
- Takes the top-ranked program and user context.
- Adapts it into a structured weekly plan with an LLM.

## Project Structure

```text
FitNova Application/
|-- backend/
|   |-- data/
|   |   |-- build_catalog.py
|   |   |-- generate_interactions.py
|   |   |-- workout_taxonomy.py
|   |   |-- program_catalog.pkl
|   |   |-- interactions.csv
|   |   |-- program_features.csv
|   |   `-- user_features.csv
|   |-- models/
|   |   |-- neumf_final.keras
|   |   |-- gmf_pretrained.keras
|   |   |-- mlp_pretrained.keras
|   |   |-- neumf_metadata.pkl
|   |   `-- user_pos_items.pkl
|   |-- services/
|   |   |-- content_filter.py
|   |   |-- neumf_ranker.py
|   |   `-- recommender.py
|   |-- tests/
|   |   |-- test_content_filter.py
|   |   |-- test_recommender.py
|   |   `-- test_workout_taxonomy.py
|   |-- training/
|   |   |-- train_neumf.py
|   |   `-- train_log.txt
|   `-- requirements.txt
|-- docs/
|-- lib/
|-- smoke_test_content_filter.py
`-- smoke_test_recommender.py
```

## Quick Start

### Backend setup

```bash
cd backend
pip install -r requirements.txt
```

### Rebuild derived data

```bash
python backend/data/build_catalog.py
python backend/data/generate_interactions.py
```

### Retrain NeuMF

```bash
cd backend
python training/train_neumf.py
```

Expected summary:

```text
GMF   - HR@10: 0.9075 | NDCG@10: 0.7251
MLP   - HR@10: 0.9065 | NDCG@10: 0.7208
NeuMF - HR@10: 0.9270 | NDCG@10: 0.7469
```

### Run backend tests

```bash
python -m pytest backend/tests -q
```

## Roadmap

| Phase | Status | Deliverable |
|---|---|---|
| 1. Data Pipeline | Complete | Catalog + synthetic data generation |
| 2. Layer 1 | Complete | Intent-aware content filtering |
| 3. Layer 1+2 | Complete | NeuMF re-ranking pipeline |
| 4. Layer 3 | In Progress | LLM weekly-plan adaptation |
| 5. FastAPI | Upcoming | `POST /generate-plan` endpoint |
| 6. Flutter | Upcoming | Mobile UI |
| 7. Testing | Backend Complete | `pytest` coverage for taxonomy, filtering, and recommender |

## Datasets

- Fitness Programs: 605K+ exercise rows collapsed into 2,598 programs.
- Gym Members: 973 user profiles with workout type, duration, frequency, BMI, age, and gender.

## Model Performance

The current NeuMF artifacts were evaluated with leave-one-out testing:

| Component | HR@10 | NDCG@10 |
|---|---:|---:|
| GMF | 0.9075 | 0.7251 |
| MLP | 0.9065 | 0.7208 |
| NeuMF | 0.9270 | 0.7469 |

NeuMF outperforms both component models on the committed synthetic dataset.

## Documentation

- [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- [docs/DECISIONS.md](docs/DECISIONS.md)
- [docs/REFERENCES.md](docs/REFERENCES.md)
- [CHANGELOG.md](CHANGELOG.md)

## Status

Phase 4 backend development is active. The recommender pipeline, taxonomy, synthetic interaction generation, and backend regression tests are in place.
