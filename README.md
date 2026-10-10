# AI Content Analysis System

An end-to-end intelligent content analysis platform featuring **AI-Generated Text Detection** and **Neural Grammar Error Correction**, backed by a production-ready **FastAPI** service and an interactive **React (Vite + Tailwind CSS)** web dashboard.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Model Performance & Evaluation](#model-performance--evaluation)
  - [Module A: AI Text Detection](#module-a-ai-text-detection)
  - [Module B: Grammar Error Correction](#module-b-grammar-error-correction)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
  - [1. Backend Setup](#1-backend-setup)
  - [2. Frontend Setup](#2-frontend-setup)
- [Running the Application](#running-the-application)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Automated Testing](#automated-testing)
- [API Documentation](#api-documentation)
- [Future Roadmap](#future-roadmap)
- [License](#license)

---

## Overview

The **AI Content Analysis System** is engineered to assist educators, researchers, content creators, and editors in evaluating written text. The platform addresses two core challenges:

1. **AI vs. Human Authorship:** Identifying machine-generated content from large language models (ChatGPT, GPT-3, etc.) versus human-written text using stylometric and n-gram statistical representations.
2. **Grammar & Fluency Correction:** Detecting grammatical mistakes, verb tense inconsistencies, agreement issues, and typographical slips using a sequence-to-sequence transformer, coupled with word-level visual diffing.

---

## Key Features

- **Module A: AI Text Detection**
  - **Feature Extraction:** Hybrid `FeatureUnion` combining word n-grams (1–2) and character n-grams (3–5) with sublinear term-frequency scaling (50,000 max features).
  - **Classifier:** Calibrated Logistic Regression tuned for class balance and generalization.
  - **Metrics:** **96.34% accuracy** and **0.995 ROC-AUC** on held-out test data.
  - **Length-Bias Safeguards:** Built-in heuristic detecting short passages (< 50 words) with confidence attenuation to reduce false positives.

- **Module B: Grammar Error Correction**
  - **Transformer Architecture:** Sequence-to-Sequence model (`vennify/t5-base-grammar-correction`) evaluated across CoEdit and JFLEG benchmarks.
  - **Word-Level Diff Analysis:** Sentence-level token diffing identifying exact insertions, deletions, and substitutions.
  - **Categorization:** Automatically flags subject-verb agreement, verb tenses, prepositions, and punctuation.

- **Unified Web Interface**
  - Built with **React 19**, **Vite**, and **Tailwind CSS**.
  - Three interactive tool views:
    - **AI Detector:** Visual confidence meter, class probabilities, and stylometric statistics.
    - **Grammar Checker:** Side-by-side original vs. corrected text with clickable error badges.
    - **Combined Analysis:** Simultaneous detection and grammar correction in a single pass.
  - Live backend health status badge with auto-polling.

- **Production-Ready REST API**
  - Powered by **FastAPI** with Pydantic v2 schemas and CORS middleware.
  - Lifespan model singleton caching (`ModelManager`) to prevent redundant cold-start loads.
  - Interactive OpenAPI / Swagger UI at `/docs`.

---

## System Architecture

```text
+-------------------------------------------------------------------------+
|                              USER BROWSER                               |
|                  React 19 + Vite + Tailwind CSS Dashboard               |
|         [ AI Detector Tab ]  [ Grammar Tab ]  [ Combined View ]         |
+------------------------------------+------------------------------------+
                                     |  HTTP / REST (JSON)
                                     v
+-------------------------------------------------------------------------+
|                           FASTAPI REST SERVER                           |
|       /api/health | /api/analyze/ai-detection | /api/analyze/grammar    |
|                      /api/analyze/combined                              |
+------------------------------------+------------------------------------+
                                     |  Singleton Model Manager
                  +------------------+------------------+
                  |                                     |
                  v                                     v
+-----------------------------------+ +-----------------------------------+
|      MODULE A: AI DETECTION       | |     MODULE B: GRAMMAR ENGINE      |
|  * Word TF-IDF (1-2 n-grams)      | |  * T5 Seq2Seq Transformer         |
|  * Char TF-IDF (3-5 n-grams)      | |  * Sentence Tokenization          |
|  * Logistic Regression (C=2.0)    | |  * Word-Level Diff Highlighter    |
|  * Short-Text Heuristic Guard     | |  * Error Category Extraction      |
+-----------------------------------+ +-----------------------------------+
                  ^                                     ^
                  |                                     |
+-----------------+-----------------+ +-----------------+-----------------+
|      OFFLINE TRAINING PIPELINE    | |      DATA & BENCHMARK SUITE       |
|  * HC3 (Human ChatGPT Corpus)     | |  * CoEdit Grammar Benchmark       |
|  * GPT-Wiki-Intro (20k cleaned)   | |  * JFLEG Multi-Reference Test     |
+-----------------------------------+ +-----------------------------------+
```

---

## Model Performance & Evaluation

### Module A: AI Text Detection

Evaluated on 1,992 held-out test samples across human and AI sources:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **96.34%** |
| **Precision** | **96.12%** |
| **Recall** | **96.60%** |
| **F1-Score** | **96.36%** |
| **ROC-AUC** | **0.9949** |

#### Subgroup Performance by Source
- **HC3 Corpus:** 98.47% Accuracy | 0.9994 ROC-AUC
- **GPT-Wiki-Intro:** 94.27% Accuracy | 0.9869 ROC-AUC

#### Performance by Text Length
- **> 100 words:** 96.81% Accuracy (F1: 96.84%)
- **50 – 100 words:** 96.89% Accuracy (F1: 97.35%)
- **30 – 50 words:** 88.33% Accuracy (F1: 82.93%)
- **< 30 words:** 86.36% Accuracy *(Flagged as "Uncertain - Short Text" in UI)*

### Module B: Grammar Error Correction

Evaluated on standard grammatical error correction benchmarks:

| Benchmark / Dataset | Metric | Score |
| :--- | :--- | :--- |
| **CoEdit Test Set** | SacreBLEU | **89.39** |
| **CoEdit Test Set** | chrF+ | **94.04** |
| **JFLEG Benchmark** | Multi-Reference BLEU | **93.21** |
| **JFLEG Benchmark** | Multi-Reference chrF | **95.79** |

---

## Project Structure

```text
AI content analysis system/
|-- backend/
|   |-- main.py                       # FastAPI application & API endpoints
|   |-- schemas.py                    # Pydantic request/response data contracts
|-- configs/
|   |-- ai_detection_config.yaml      # AI detection hyperparameters & dataset paths
|   |-- grammar_config.yaml           # Grammar correction model configuration
|-- data/
|   |-- raw/                          # Raw extracted datasets
|   |-- processed/                    # Cleaned and normalized text data
|   |-- splits/                       # Leakage-free train / val / test splits
|-- frontend/
|   |-- src/
|   |   |-- components/
|   |   |   |-- AIDetectorTool.jsx    # AI detection UI component
|   |   |   |-- GrammarCheckerTool.jsx# Grammar correction & diff viewer
|   |   |   |-- CombinedTool.jsx      # Simultaneous dual-analysis view
|   |   |   |-- Navbar.jsx            # Top navigation bar with health indicator
|   |   |-- api.js                    # Frontend API client
|   |   |-- App.jsx                   # Main layout container
|   |-- package.json                  # Frontend dependencies and scripts
|   |-- vite.config.js                # Vite build configuration
|-- models/
|   |-- ai_detection/                 # Serialized model (.joblib)
|   |-- grammar_correction/           # Model metadata & cached weights
|-- reports/
|   |-- dataset_quality/              # Quality audits & class balance summaries
|   |-- evaluation/                   # Evaluation metrics & length breakdown JSONs
|   |-- plots/                        # Confusion matrices and ROC curves
|-- src/
|   |-- data_acquisition/             # Scripts to download and extract datasets
|   |-- preprocessing/                # Text sanitization, normalization, cleaning
|   |-- splitting/                    # Stratified, leakage-free dataset splitters
|   |-- training/                     # Model training & parameter tuning scripts
|   |-- evaluation/                   # Test set evaluation & metric calculators
|   |-- inference/                    # Production inference pipelines & service wrapper
|-- tests/
|   |-- test_api.py                   # FastAPI endpoint integration tests
|   |-- test_inference.py             # Inference pipeline verification
|   |-- test_preprocessing.py         # Text cleaning unit tests
|   |-- test_splitting.py             # Split integrity and leakage tests
|-- evaluate_length_and_leakage.py    # Script to audit data leakage and length bias
|-- run_pipeline.py                   # Master ML pipeline runner (Stages 1-5)
|-- verify_modules.py                 # Manual verification script with sample test cases
|-- requirements.txt                  # Python dependencies
|-- README.md                         # Project documentation
```

---

## Prerequisites

- **Python:** Version 3.10 or 3.11 recommended
- **Node.js:** Version 18.x or higher
- **npm:** Version 9.x or higher

---

## Installation & Setup

### 1. Backend Setup

1. Open a terminal and navigate to the project directory:
   ```bash
   cd "AI content analysis system"
   ```

2. Create and activate a Python virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install required Python packages:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### 2. Frontend Setup

1. In a separate terminal, navigate to the frontend folder:
   ```bash
   cd "AI content analysis system/frontend"
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

---

## Running the Application

### Start the Backend Server

From the project root with the virtual environment activated:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- The API will be accessible at: `http://localhost:8000`
- Interactive Swagger documentation: `http://localhost:8000/docs`
- Health check status: `http://localhost:8000/api/health`

### Start the Frontend Client

From the `frontend/` directory:

```bash
npm run dev
```

- The web interface will open at: `http://localhost:5173`

---

## Machine Learning Pipeline

The project includes an end-to-end reproducible pipeline managed by `run_pipeline.py`.

To run all pipeline stages sequentially:
```bash
python run_pipeline.py --stage all
```

To run individual stages:
```bash
# Stage 1: Download raw datasets
python run_pipeline.py --stage acquire

# Stage 2: Clean and normalize data
python run_pipeline.py --stage preprocess

# Stage 3: Generate train/val/test splits
python run_pipeline.py --stage split

# Stage 4: Train models
python run_pipeline.py --stage train

# Stage 5: Evaluate on held-out test sets
python run_pipeline.py --stage evaluate
```

### Additional Diagnostic Scripts

- **Audit Data Leakage and Length Bias:**
  ```bash
  python evaluate_length_and_leakage.py
  ```
- **Smoke-Test Inference Modules:**
  ```bash
  python verify_modules.py
  ```

---

## Automated Testing

Run the test suite using `pytest` or `unittest`:

```bash
# Using pytest
pytest tests/ -v

# Using unittest
python -m unittest discover -s tests
```

### Test Coverage Overview
- `test_preprocessing.py`: Verifies whitespace normalization, null inputs, and prefix removal.
- `test_splitting.py`: Verifies stratification, zero set overlap, and class proportions.
- `test_inference.py`: Verifies AI detection score ranges, short text warnings, and grammar diff structures.
- `test_api.py`: Tests `/api/health`, `/api/analyze/ai-detection`, `/api/analyze/grammar`, and input validation errors.

---

## API Documentation

### 1. Health Check
- **Endpoint:** `GET /api/health`
- **Response:**
  ```json
  {
    "status": "healthy",
    "version": "1.0.0",
    "ai_detector_loaded": true,
    "grammar_model_loaded": true,
    "device": "cpu",
    "server_time": "2026-10-10T19:00:00Z"
  }
  ```

### 2. AI Detection
- **Endpoint:** `POST /api/analyze/ai-detection`
- **Request Body:**
  ```json
  {
    "text": "Deep learning architectures have transformed natural language processing..."
  }
  ```
- **Response:**
  ```json
  {
    "is_ai": true,
    "label": "AI-Generated",
    "ai_probability": 0.942,
    "human_probability": 0.058,
    "confidence_score": 0.942,
    "confidence_level": "High Confidence",
    "assessment": "The text shows strong structural and lexical patterns characteristic of AI language models.",
    "is_short_text": false,
    "word_count": 85,
    "execution_time_seconds": 0.012
  }
  ```

### 3. Grammar Correction
- **Endpoint:** `POST /api/analyze/grammar`
- **Request Body:**
  ```json
  {
    "text": "She go to school yesterday and have many book."
  }
  ```
- **Response:**
  ```json
  {
    "original_text": "She go to school yesterday and have many book.",
    "corrected_text": "She went to school yesterday and has many books.",
    "has_errors": true,
    "total_errors": 3,
    "sentence_corrections": [
      {
        "original": "She go to school yesterday and have many book.",
        "corrected": "She went to school yesterday and has many books.",
        "has_errors": true,
        "edits": [
          { "type": "replace", "original": "go", "corrected": "went" },
          { "type": "replace", "original": "have", "corrected": "has" },
          { "type": "replace", "original": "book", "corrected": "books" }
        ]
      }
    ],
    "execution_time_seconds": 0.45
  }
  ```

### 4. Combined Analysis
- **Endpoint:** `POST /api/analyze/combined`
- **Request Body:**
  ```json
  {
    "text": "Your sample text here...",
    "enable_ai_detection": true,
    "enable_grammar": true
  }
  ```

---

## Future Roadmap

- [ ] Add batch document processing for `.docx`, `.pdf`, and `.txt` files.
- [ ] Export analysis summaries as formatted PDF and CSV reports.
- [ ] Fine-tune a lightweight transformer (`DistilBERT`) to benchmark against the statistical baseline for short-text inputs.
- [ ] Containerize backend and frontend with Docker and Docker Compose.
- [ ] Support multilingual grammar checking and language identification.

---

## License

This project is developed for educational and research purposes under the **MIT License**.
