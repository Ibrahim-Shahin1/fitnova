# Executed Colab notebooks

The raw, pre-executed Colab notebooks used to train and evaluate the form-correction models
on Fitness-AQA, with their output cells intact. The numbered notebooks in the parent folder
(`01..12`) are the clean, curated write-ups of the same work.

| Notebook | Covers |
|----------|--------|
| `FitNova_EDA_Fitness_AQA.ipynb` | Fitness-AQA dataset EDA (splits, class balance, sample frames) |
| `Squat-Supervised Baseline.ipynb` | Squat KIE/KFE supervised baseline (Kinetics R(2+1)D-18 fine-tune) |
| `Squat-md-SSL finetune.ipynb` | Squat motion-disentanglement SSL pretrain + fine-tune |
| `OHP.ipynb` | Overhead Press pipeline (EDA, data, baseline) |
| `FormCorrection_Phase7.ipynb` | Shallow-Squat depth (image modality, CVCSPC pose-contrastive SSL) |
| `FormCorrection_7_3.ipynb` | Shallow-Squat CVCSPC continued run |
| `04_squat_md_ssl.ipynb` | Earlier Squat MD-SSL run |
