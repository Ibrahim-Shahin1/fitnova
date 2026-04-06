# Changelog

All notable changes to the FitNova project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.1.0] — 2026-04-06

### ✨ Added

#### Core ML Component
- **NeuMF Model Training Pipeline** (`backend/training/train_neumf.py`)
  - Implements He et al. (2017) Neural Collaborative Filtering architecture
  - Two-pathway design: GMF (linear) + MLP (non-linear)
  - Pre-training + fine-tuning protocol with Adam and SGD optimizers
  - Binary cross-entropy loss on 1.36M synthetic user-item interactions

- **Synthetic Interaction Data Generation** (completed in prior phase)
  - 1,361,094 total interactions (305,807 positive, 1,055,287 negative)
  - Generated using Gupta et al. (2024) compatibility rules
  - Feature weights: experience (0.35), workout_type (0.30), duration (0.20), frequency (0.15)
  - 87.90% sparsity (973 users × 2,598 programs)
  - Gaussian noise (σ=0.08) + 1.35× correction for Expert-Advanced pairs

#### Trained Models
- `backend/models/neumf_final.keras` — Final NeuMF model (~760 KB)
- `backend/models/gmf_pretrained.keras` — Pre-trained GMF component (~705 KB)
- `backend/models/mlp_pretrained.keras` — Pre-trained MLP component (~1.5 MB)
- `backend/models/neumf_metadata.pkl` — Model hyperparameters and dimensions
- `backend/models/user_pos_items.pkl` — User interaction history (874 KB)

#### Project Infrastructure
- **Git Repository** — Initialized with comprehensive `.gitignore`
- **Documentation**
  - `README.md` — Project overview, architecture, quick start
  - `docs/DEVELOPMENT.md` — 7-phase implementation roadmap
  - `docs/ARCHITECTURE.md` — System design, layer details, data flow
  - `docs/DECISIONS.md` — Architectural decisions and rationale (6 major decisions)
  - `docs/REFERENCES.md` — Academic citations, datasets, evaluation metrics
  - `CHANGELOG.md` — Version history
- **Python Dependencies** — `backend/requirements.txt` with TensorFlow, FastAPI, etc.
- **Code Structure**
  - `backend/training/` — NeuMF training pipeline
  - `backend/models/` — Trained model artifacts
  - `backend/` — Ready for Phase 1 (data pipeline) and beyond
  - `lib/main.dart` — Flutter scaffold (to be updated in Phase 6)

### 📊 Model Performance

Evaluated on leave-one-out protocol with HR@10 and NDCG@10 metrics (973 test users):

| Component | HR@10 | NDCG@10 |
|-----------|-------|---------|
| GMF | 0.8510 | 0.6275 |
| MLP | 0.8674 | 0.6203 |
| **NeuMF (Fused)** | **0.8756** | **0.6363** |

**Interpretation:**
- 87.56% of test users have the held-out program in top-10 recommendations
- NDCG accounts for ranking position; NeuMF best balances precision and ranking

**Validation:** Results align with He et al. (2017) findings on Netflix/MovieLens

### 📋 Known Limitations

- **Layer 1** (Content-Based Filtering) not yet implemented
- **Layer 2** (NeuMF) trained but not integrated into recommendation pipeline
- **Layer 3** (LLM Adaptation) not yet implemented
- **FastAPI Server** not yet built
- **Flutter App** only has default template; no UI for fitness planning
- **Testing** — No unit/integration tests yet (Phase 7)

### 🔮 Next Steps (Phases 1-7)

1. **Phase 1** — Program Catalog Builder: Map program_ids to exercise data
2. **Phase 2** — Layer 1: Content-based filtering with cosine similarity
3. **Phase 3** — Layer 1+2 Integration: Recommendation pipeline with NeuMF
4. **Phase 4** — Layer 3: LLM adaptation for personalization
5. **Phase 5** — FastAPI Server: `/generate-plan` REST endpoint
6. **Phase 6** — Flutter Frontend: Onboarding form + plan display
7. **Phase 7** — Testing & Documentation: Unit tests, integration tests, academic write-up

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed phase breakdown.

### 📚 References

- **He et al. (2017)** — "Neural Collaborative Filtering," WWW 2017
- **Gupta et al. (2024)** — "Analyzing Gym Members' Fitness Patterns," IEEE ICTACS 2024
- Datasets: Boostcamp Kaggle (605K+ exercises, 2,598 programs) + Gym Members (973 users)

### 🎯 Academic Grounding

This project is grounded in peer-reviewed research:
- NeuMF architecture directly from WWW 2017 seminal paper
- Feature weights derived from Gupta et al. (2024) empirical analysis
- Synthetic data generation uses compatibility rules justified by real user behavior
- Evaluation metrics (HR@10, NDCG@10) standard in collaborative filtering literature

---

## Versioning & Release Strategy

- **0.1.x** — Pre-release (Phases 1-7 implementation)
- **0.2.0** — First stable release (all 7 phases complete, tested)
- **1.0.0** — Production ready (with API stability guarantee, full docs)

---

## Future Milestones (Planned)

### v0.2.0 (Fitness Planning Complete)
- [ ] Phase 1–7 fully implemented
- [ ] All tests passing
- [ ] Full API documentation
- [ ] Flutter app functional end-to-end

### v1.0.0 (Production Release)
- [ ] Deployed to app stores (iOS, Android)
- [ ] User authentication system
- [ ] Persistent user data (database)
- [ ] Interaction logging for model retraining

### v2.0.0 (Form Correction Integration)
- [ ] Exercise form detection via computer vision
- [ ] Real-time feedback on exercise form
- [ ] Integration with fitness planning

### v3.0.0 (Nutrition Integration)
- [ ] Nutrition guidance module
- [ ] Integration with fitness & form correction
- [ ] Full AI fitness ecosystem

---

## Contributors

- [Your Name] — NeuMF training, architecture design, project setup

---

## License

This project is licensed under the MIT License (or Apache 2.0 — to be confirmed).

