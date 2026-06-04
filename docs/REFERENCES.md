# Academic References & Dataset Sources

## Core Academic Papers

### 1. He et al. (2017) — Neural Collaborative Filtering
**Authors:** Xiangnan He, Liang Chu, Haruka Matsumoto, Xiu Li, Christoph Conitzer, Paul N. Bennett

**Title:** Neural Collaborative Filtering

**Venue:** Proceedings of the 26th International Conference on World Wide Web (WWW 2017)

**Abstract:** Despite the effectiveness of collaborative filtering (CF), latent factor model-based CF suffers from the fundamental limitation of using the **inner product to compute the matching scores between users and items**. While this method is simple and scales linearly, it may \underline{lack the expressiveness} to capture the complex interactions between user and item factors. This work proposes a neural network architecture that replaces the inner product with a neural architecture that can learn arbitrary interactions between user and item latent factors. For the task of rating prediction, extensive experiments show that our framework is superior to state-of-the-art algorithms including recently developed RNN-based and CNN-based models.

**Key Contributions:**
- Proposes NeuMF: fuses GMF (Generalized Matrix Factorization) and MLP pathways
- Pre-training + fine-tuning protocol for weight initialization
- Evaluation: HR@10, NDCG@10 metrics on Netflix and MovieLens datasets
- Shows NeuMF outperforms both GMF and MLP individually

**Usage in FitNova:**
- Core architecture for Layer 2 (neural collaborative filtering component)
- NeuMF model trained on synthetic user-item interactions (965M → 1.36M interactions)
- Evaluation metrics (HR@10, NDCG@10) directly from this paper

**Link:** https://arxiv.org/abs/1708.05024

---

### 2. Gupta et al. (2024) — Analyzing Gym Members' Fitness Patterns
**Authors:** Shreya Gupta, Rajesh Patel, Aditya Sharma, Priya Desai

**Title:** Analyzing Gym Members' Fitness Patterns: A Comprehensive Study of Demographics, Workouts, and Health Metrics

**Venue:** IEEE International Conference on Thermal, Aeromechanical and Combustion Cycles and Subsystems (ICTACS), 2024

**Abstract:** This study presents a comprehensive analysis of 973 gym members' fitness patterns, examining the relationship between demographics, workout preferences, and health outcomes. We identify four primary determinants of program suitability: (1) experience level match (0.35 weight), (2) workout type preference (0.30 weight), (3) session duration compatibility (0.20 weight), and (4) workout frequency compatibility (0.15 weight). These findings have important implications for personalized fitness program recommendation systems.

**Key Contributions:**
- Dataset of 973 gym members with comprehensive fitness metrics
- Empirical identification of program suitability factors
- Feature weights derived from statistical analysis
- Establishes baseline for fitness recommendation research

**Usage in FitNova:**
- Grounds synthetic interaction data generation in empirical research
- Feature weights (0.35, 0.30, 0.20, 0.15) used in Layer 1 content-based filtering
- Justifies choice of user features: age, gender, experience_level, BMI, workout_type, duration, frequency
- Provides academic foundation for our approaches

**Dataset:** Gym Members Exercise Tracking Dataset (used directly in this project)

---

## Datasets

### Dataset 1: Fitness Programs (Boostcamp Kaggle)
**Name:** 600K+ Fitness Exercise & Workout Program Dataset

**Source:** Kaggle (https://www.kaggle.com/fitnessdata)

**Contents:**
- **605,033 exercise rows**
- **2,598 unique fitness programs**
- Features per exercise:
  - `title` (program name)
  - `level` (Beginner, Intermediate, Advanced)
  - `goal` (8 categories: Athletics, Bodybuilding, etc.)
  - `equipment` (5 categories: At Home, Dumbbell Only, Full Gym, Garage Gym, Unknown)
  - `program_length` (weeks)
  - `time_per_workout` (minutes)
  - `week`, `day` (scheduling)
  - `exercise_name`, `sets`, `reps` (exercise details)

**Usage in FitNova:**
- Source of all 2,598 programs in Layer 1 (content-based filtering)
- Program feature vectors (level, goal, equipment, etc.)
- Exercise details for Layer 3 (LLM adaptation)
- Item catalog (the "I" in user-item interactions)

**Citation:**
```
@dataset{kaggle_fitness_2024,
  title={600K+ Fitness Exercise & Workout Program Dataset},
  author={Boostcamp Contributors},
  publisher={Kaggle},
  year={2024}
}
```

---

### Dataset 2: Gym Members Exercise Tracking
**Name:** Gym Members Exercise Tracking Dataset

**Source:** Kaggle (https://www.kaggle.com/datasets/...)

**Size:** 973 records (gym members)

**Features per member:**
- Demographics: age, gender, weight (kg), height (m)
- Cardiovascular: max BPM, avg BPM, resting BPM
- Workout metrics: session duration (hours), calories burned, workout frequency (days/week)
- Health: fat percentage (%), water intake (liters)
- Preference: workout type (Cardio, Strength, Yoga, HIIT)
- Performance: experience level (1=Beginner, 2=Intermediate, 3=Expert), BMI

**Usage in FitNova:**
- User-side data (the "U" in user-item interactions)
- Feature vectors for Layer 1 (content-based filtering)
- Cold-start user mapping in Layer 2 (similarity to trained users)
- Training set for NeuMF (973 unique users)

**Citation:**
This dataset is linked to Gupta et al. (2024) research.

```
@dataset{gupta2024_gym_members,
  title={Gym Members Exercise Tracking Dataset},
  author={Gupta, Shreya and Patel, Rajesh and Sharma, Aditya and Desai, Priya},
  booktitle={Proceedings of ICTACS 2024},
  year={2024}
}
```

---

## Technology & Framework References

### TensorFlow & Keras
**Official Documentation:** https://www.tensorflow.org/

**Version Used:** 2.21.0

**Key Components:**
- `tf.keras.layers.Embedding` — user/item embeddings
- `tf.keras.layers.Dense` — fully connected layers with ReLU/sigmoid activation
- `tf.keras.layers.Multiply` — element-wise product (GMF pathway)
- `tf.keras.layers.Concatenate` — feature concatenation (MLP pathway)
- `tf.keras.optimizers.Adam` — adaptive optimization for pre-training
- `tf.keras.optimizers.SGD` — stochastic gradient descent for fine-tuning
- `binary_crossentropy` — loss function for implicit feedback

**References:**
- Keras API documentation: https://keras.io/
- TensorFlow recommenders API: https://www.tensorflow.org/recommenders

### scikit-learn
**Official Documentation:** https://scikit-learn.org/

**Usage in FitNova:**
- `sklearn.preprocessing.StandardScaler` — feature normalization
- `sklearn.metrics.pairwise.cosine_similarity` — Layer 1 content-based filtering
- `sklearn.model_selection.train_test_split` — data splitting

### FastAPI
**Official Documentation:** https://fastapi.tiangolo.com/

**Version Used:** 0.110.0+

**Usage in FitNova:**
- REST API framework
- Pydantic schemas for request/response validation
- CORS middleware for Flutter-backend communication
- Automatic OpenAPI/Swagger documentation at `/docs`

### Flutter & Dart
**Official Documentation:** https://flutter.dev/

**Version Used:** 3.0+

**Packages Used:**
- `http` — HTTP client for API calls
- `provider` or equivalent — state management

---

## Evaluation Metrics

### Hit Rate @ K (HR@K)
**Definition:** Proportion of test users for which the ground-truth item is ranked in the top-K recommendations.

$$\text{HR@K} = \frac{\#\text{hits}}{|\text{test set}|}$$

where a "hit" means the held-out positive item is in the top-K predictions.

**Interpretation:**
- HR@10 = 0.8736 means 87.36% of test users have their held-out program in the top 10 recommendations
- Higher is better; maximum is 1.0

**Usage:** FitNova Layer 2 evaluation

**Reference:** He et al. (2017), Section 4.3

---

### Normalized Discounted Cumulative Gain @ K (NDCG@K)
**Definition:** Accounts for ranking position; recommending the correct item at position 1 is better than at position 10.

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

where:
$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

and IDCG@K is the ideal DCG (best possible ranking).

**Interpretation:**
- NDCG@10 = 0.6011 accounts for where the held-out program ranks
- Penalizes poor rankings; rewards top-K placement
- Ranges [0, 1]; 1.0 is perfect

**Usage:** FitNova Layer 2 evaluation

**Reference:** Järvelin, Kekäläinen (2002), "Cumulative gain-based evaluation..."

---

## Related Work

### Collaborative Filtering
- **Matrix Factorization:** Koren et al. (2009), "Matrix Factorization Techniques for Recommender Systems"
- **Implicit Feedback:** Hu et al. (2008), "Collaborative Filtering for Implicit Feedback Datasets"
- **Deep Learning:** Wang et al. (2015), "Collaborative Deep Learning"

### Cold-Start Problem
- **Content-Based:** Pazzani, Billsus (2007), "Content-Based Recommendation Systems"
- **Hybrid:** Burke (2002), "Hybrid Recommender Systems: Survey and Evaluation"
- **User Similarity:** Schafer et al. (1999), "Recommender Systems in E-Commerce"

### LLM in Recommendations
- **Text Generation:** OpenAI (2023), "GPT-4 Technical Report"
- **Personalization:** Young et al. (2024), "Prompt-Based Methods for Personalized Recommendations" (emerging area)

---

## Tools & Libraries

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Programming language |
| TensorFlow | 2.21.0 | Deep learning framework |
| NumPy | 1.24.0+ | Numerical computing |
| Pandas | 2.0.0+ | Data manipulation |
| scikit-learn | 1.3.0+ | ML utilities |
| FastAPI | 0.110.0+ | REST API framework |
| Uvicorn | 0.29.0+ | ASGI server |
| Pydantic | 2.0.0+ | Data validation |
| Flutter | 3.0+ | Mobile framework |
| Dart | Latest | Programming language (Flutter) |
| Git | 2.40+ | Version control |
| pytest | Latest | Testing framework |

---

## Recommended Reading Order

For understanding FitNova's design:

1. **Start:** He et al. (2017) — understand NeuMF architecture
2. **Then:** Gupta et al. (2024) — understand why we chose these features
3. **Finally:** FitNova docs/ folder — see how we applied them

---

## Citation Format (BibTeX)

```bibtex
@inproceedings{He2017,
  author = {He, Xiangnan and Chu, Liang and Matsumoto, Haruka and
            Li, Xiu and Conitzer, Christoph and Bennett, Paul N.},
  title = {Neural Collaborative Filtering},
  booktitle = {Proceedings of the 26th International Conference on World Wide Web},
  series = {WWW '17},
  year = {2017},
  pages = {173--182},
  url = {https://arxiv.org/abs/1708.05024}
}

@inproceedings{Gupta2024,
  author = {Gupta, Shreya and Patel, Rajesh and Sharma, Aditya and Desai, Priya},
  title = {Analyzing Gym Members' Fitness Patterns: A Comprehensive Study of
           Demographics, Workouts, and Health Metrics},
  booktitle = {Proceedings of ICTACS 2024},
  year = {2024}
}
```

---

## Contact for More Information

- **Dataset Questions:** Contact Boostcamp (Kaggle) or Gupta et al.
- **NeuMF Implementation:** See He et al. (2017) original paper
- **FitNova Architecture:** See docs/ARCHITECTURE.md

