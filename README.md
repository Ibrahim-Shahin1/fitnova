# FitNova — AI-Powered Fitness Planning

An intelligent fitness planning application that generates personalized weekly workout plans using a **3-layer recommendation pipeline** combining content-based filtering, neural collaborative filtering (NeuMF), and large language model adaptation.

## Project Overview

FitNova is a graduation project that demonstrates how to build an AI-powered mobile fitness application. The core feature—**fitness planning**—uses a novel hybrid approach that addresses both the cold-start problem and delivers highly personalized, context-aware workout recommendations.

### Key Features

1. **Fitness Planning** — The core AI feature (in development)
2. Exercise Form Correction — Computer vision module (separate)
3. Nutrition Guidance — Dietary recommendations (separate)

## Architecture

```
User Profile (8 inputs)
    ↓
Layer 1: Content-Based Filtering (cosine similarity)
    ↓ Returns top-50 candidate programs
Layer 2: NeuMF (Neural Matrix Factorization)
    ↓ Re-ranks and selects top program
Layer 3: LLM Adaptation (Claude/Gemini)
    ↓ Personalizes weekly plan
Final Output: JSON Weekly Plan
```

### Layer Details

**Layer 1 — Content-Based Filtering**
- Matches user profile to 2,598 fitness programs using cosine similarity
- Features: experience level, goal, equipment, workout type, duration, frequency
- Solves the cold-start problem (works for new users immediately)

**Layer 2 — Neural Matrix Factorization (NeuMF)**
- Implements He et al. (2017) "Neural Collaborative Filtering"
- Combines GMF (linear relationships) + MLP (non-linear interactions)
- Trained on 1.36M synthetic user-item interactions
- Performance: **HR@10 = 0.8756** | **NDCG@10 = 0.6363** ✓

**Layer 3 — LLM Personalization**
- Takes top-ranked program and user context
- Uses LLM (Gemini Flash) to generate personalized weekly plan
- Outputs structured JSON with exercises, sets, reps, rest times

## Project Structure

```
FitNova Application/
├── backend/                    # Python FastAPI backend
│   ├── models/                # Trained TensorFlow models
│   │   ├── neumf_final.keras  # Production NeuMF model
│   │   ├── gmf_pretrained.keras
│   │   ├── mlp_pretrained.keras
│   │   ├── neumf_metadata.pkl
│   │   └── user_pos_items.pkl
│   ├── training/              # Training pipeline
│   │   ├── train_neumf.py     # NeuMF training script
│   │   └── train_log.txt      # Training log
│   ├── services/              # (Phase 2+) Service modules
│   ├── data/                  # Data processing
│   ├── main.py                # (Phase 5) FastAPI app
│   └── requirements.txt        # Python dependencies
├── lib/                        # Flutter Dart source
│   └── main.dart              # (Phase 6) Flutter app
├── docs/                       # Documentation
│   ├── DEVELOPMENT.md         # Development roadmap
│   ├── ARCHITECTURE.md        # System design details
│   ├── DECISIONS.md           # Technical decisions
│   └── REFERENCES.md          # Academic citations
├── CHANGELOG.md               # Version history
└── .gitignore                 # Git ignore rules
```

## Quick Start

### Prerequisites
- Python 3.10+
- Flutter 3.0+
- Git

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

### Run Training (optional — models already trained)

```bash
cd backend
python training/train_neumf.py
```

Expected output:
```
GMF   — HR@10: 0.8510  |  NDCG@10: 0.6275
MLP   — HR@10: 0.8674  |  NDCG@10: 0.6203
NeuMF — HR@10: 0.8756  |  NDCG@10: 0.6363
```

### Run FastAPI Server (Phase 5+)

```bash
cd backend
python main.py
# Navigate to http://localhost:8000/docs for interactive API docs
```

### Run Flutter App (Phase 6+)

```bash
flutter run
```

## Development Roadmap

The project follows a **7-phase implementation plan**:

| Phase | Status | Deliverable |
|-------|--------|-------------|
| **1. Data Pipeline** | Upcoming | Program catalog builder |
| **2. Layer 1** | Upcoming | Content-based filtering |
| **3. Layer 1+2** | Upcoming | Recommendation pipeline |
| **4. Layer 3** | Upcoming | LLM adapter |
| **5. FastAPI** | Upcoming | REST API endpoint |
| **6. Flutter** | Upcoming | Mobile UI |
| **7. Testing** | Upcoming | Unit/integration tests + docs |

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed phase breakdown.

## Technical Decisions

Key architectural choices and their rationale are documented in [docs/DECISIONS.md](docs/DECISIONS.md):
- Why NeuMF (He et al. 2017) over other collaborative filtering models
- Why synthetic interaction data grounded in Gupta et al. (2024) research
- Why a 3-layer pipeline (cold-start + collaboration + personalization)

## Academic References

- **He et al. (2017)** — *Neural Collaborative Filtering*, WWW 2017
- **Gupta et al. (2024)** — *Analyzing Gym Members' Fitness Patterns*, IEEE ICTACS 2024

Full references in [docs/REFERENCES.md](docs/REFERENCES.md).

## Datasets

**Dataset 1 — Fitness Programs** (600K+ rows, 2,598 programs)
- Boostcamp Kaggle Dataset
- Includes: exercise names, sets, reps, program level, goal, equipment, weekly schedule

**Dataset 2 — Gym Members** (973 rows)
- Kaggle dataset linked to Gupta et al. (2024) research
- Includes: age, gender, experience level, BMI, workout type, frequency, duration

## Model Performance

The trained NeuMF model was evaluated on a leave-one-out protocol using HR@10 and NDCG@10 metrics:

| Component | HR@10 | NDCG@10 |
|-----------|-------|---------|
| GMF alone | 0.8510 | 0.6275 |
| MLP alone | 0.8674 | 0.6203 |
| **NeuMF (Fused)** | **0.8756** | **0.6363** |

NeuMF outperforms both components independently, validating the hybrid architecture.

## Documentation

- **[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)** — Setup, running phases, dependencies
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — System design, data flow, model details
- **[docs/DECISIONS.md](docs/DECISIONS.md)** — Why each major choice was made
- **[docs/REFERENCES.md](docs/REFERENCES.md)** — Academic citations and dataset sources
- **[CHANGELOG.md](CHANGELOG.md)** — Version history and milestones

## License

MIT (or Apache 2.0, to be confirmed)

## Contact / Supervisor

[Your Name]
[Your Email]
[University / Institution]

---

**Status:** Phase 1 development (Program Catalog Builder)
**Last Updated:** April 2026
