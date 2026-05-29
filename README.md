<p align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/thumb/2/2e/University_of_portsmouth_logo.png/320px-University_of_portsmouth_logo.png" alt="University of Portsmouth" width="260"/>
</p>

# 🎓 University of Portsmouth — Erasmus Exchange
### Artificial Intelligence and Machine Learning MSc · Summer Semester 2025/26

This repository contains the coursework submissions I completed during a one-semester Erasmus exchange at the **University of Portsmouth**. As part of the MSc programme in Artificial Intelligence and Machine Learning, I enrolled in three modules and produced four graded submissions.

> 🏅 **Note on the UK grading scale:** British universities use a notoriously strict marking system. A score of 70 or above already constitutes a **Distinction** — the highest possible classification — making anything in the 70s, 80s, or 90s exceptional. All three of my modules were graded in the Distinction band.

---

## 📚 Modules & Scores

| Module | Submissions | Score (out of 100) |
|---|---|---|
| Programming for AI | 1 | **75** |
| AI Foundations in Practice | 2 | **85** |
| Cloud Computing | 1 | **88** |

---

## 📋 Submissions

### 1 · 🩺 Heart Attack Prediction
**Module:** Programming for AI

A machine learning project predicting cardiovascular risk using a stroke dataset, covering data preprocessing, feature engineering, clustering, and neural network hyperparameter tuning.

**💻 Code**
- Notebook: [`code_stroke_assignment.ipynb`](code_stroke_assignment.ipynb)
- Processed data & model outputs: [`ProgrammingForAI/output/`](ProgrammingForAI/output/)

**📄 Report**

> Click the link below

[Heart Attack Prediction.pdf](Heart%20Attack%20Prediction.pdf)

---

### 2 · 🚦 Traffic Light Image Detection
**Module:** AI Foundations in Practice — Submission 2

A single consolidated notebook (required format for this submission) implementing a CNN-based pipeline for traffic light classification from image data, including YOLO, data augmentation and Keras Tuner hyperparameter search.

**💻 Code**
- Notebook: [`traffic_classification.ipynb`](traffic_classification.ipynb)
- Dataset: [`AIFoundations/traffic/`](AIFoundations/traffic/)
- Processed data & model outputs: [`AIFoundations/output/`](AIFoundations/output/)

**📄 Report**

[Traffic Light Image Detection.pdf](Traffic%20Light%20Image%20Detection.pdf)

---

### 3 · 🧬 Stroke Prediction
**Module:** AI Foundations in Practice — Submission 1

A multi-notebook pipeline exploring classical and deep learning models for stroke prediction, including dataset preparation, model comparison, neural network architecture search, and t-SNE visualisation.

**💻 Code** — [`AIFoundations/coursework_code_1/`](AIFoundations/coursework_code_1/)

| File | Description |
|---|---|
| [`dataset_preparation.ipynb`](AIFoundations/coursework_code_1/dataset_preparation.ipynb) | Data cleaning and feature engineering |
| [`dataset_visualization.ipynb`](AIFoundations/coursework_code_1/dataset_visualization.ipynb) | Exploratory data analysis |
| [`models.ipynb`](AIFoundations/coursework_code_1/models.ipynb) | Classical ML model comparison |
| [`neural_networks.ipynb`](AIFoundations/coursework_code_1/neural_networks.ipynb) | Neural network training and evaluation |
| [`tsne_visualisation.ipynb`](AIFoundations/coursework_code_1/tsne_visualisation.ipynb) | Dimensionality reduction and cluster visualisation |
| [`run_models.py`](AIFoundations/coursework_code_1/run_models.py) | Script to run all models end-to-end |

Generated artefacts (trained models, plots, tuner results): [`AIFoundations/output/`](AIFoundations/output/)

**📄 Report**

[Stroke Prediction.pdf](Stroke%20Prediction.pdf)

---

### 4 · ☁️ Virtualisation and Cloud Design
**Module:** Cloud Computing

A design and analysis report on virtualisation strategies and cloud architecture. This was a report-only submission — everything detailed was implemented on the University Computer.

**📄 Report**

[Virtualisation and Cloud Design.pdf](Virtualisation%20and%20Cloud%20Design.pdf)

---

## 🗂️ Repository Structure

```
.
├── code_stroke_assignment.ipynb        # Programming for AI — Heart Attack Prediction notebook
├── traffic_classification.ipynb        # AI Foundations 2 — Traffic Light Detection notebook
│
├── ProgrammingForAI/
│   ├── output/                         # Model outputs and tuner results
│   └── *.csv                           # Processed stroke datasets
│
├── AIFoundations/
│   ├── coursework_code_1/              # AI Foundations 1 — multi-file submission
│   ├── output/                         # Trained models, plots, tuner results
│   └── traffic/                        # CNN training image dataset
│
├── Heart Attack Prediction.pdf
├── Stroke Prediction.pdf
├── Traffic Light Image Detection.pdf
└── Virtualisation and Cloud Design.pdf
```

---

*Erasmus exchange semester at the University of Portsmouth, UK.*
