# Amazon Review Sentiment Analysis with Local LLMs

A sentiment analysis project that evaluates the ability of small, locally deployed Large Language Models (LLMs) to classify Amazon product reviews as **positive** or **negative**.

The project uses **Ollama** to run two instruction-tuned language models locally and compares their classification performance on a held-out test set.

## Models

The following models are evaluated:

- **Qwen2.5-0.5B-Instruct**
- **Phi-3.5-mini-instruct**

The models are used for **inference only**. No fine-tuning or additional model training is performed.

---

## Project Overview

The project follows an end-to-end sentiment analysis pipeline:

```text
Raw Amazon Reviews
        │
        ▼
Data Preprocessing
        │
        ▼
Balanced Dataset
        │
        ▼
Train/Test Split
        │
        ├───────────────┐
        ▼               ▼
     Qwen            Phi-3.5
        │               │
        └───────┬───────┘
                ▼
        Sentiment Predictions
                │
                ▼
        Evaluation & Comparison
                │
                ▼
        Metrics + Visualizations
```

The goal is to investigate how effectively relatively small local LLMs can perform a simple binary sentiment classification task without task-specific fine-tuning.

---

## Dataset

The project uses an Amazon product review dataset containing review text and rating information.

### Preprocessing

The preprocessing pipeline performs the following steps:

1. Loads the raw Amazon review dataset.
2. Selects the review text and rating columns.
3. Removes missing values.
4. Converts ratings into numeric values.
5. Converts ratings into binary sentiment labels:
   - **4–5 stars → Positive (`1`)**
   - **1–2 stars → Negative (`0`)**
   - **3 stars → Removed**

6. Balances the dataset by sampling the same number of examples from each class.
7. Shuffles the resulting dataset.
8. Limits review length to 1000 characters.
9. Renames columns into a simpler structure.
10. Saves the processed dataset as:

```text
data/dataprocessed_reviews.csv
```

The resulting dataset contains:

```text
text
rating
label
```

---

## Dataset Split

The processed dataset is divided into development and test subsets using `train_test_split`.

```text
80% → Development Set
20% → Test Set
```

The split uses:

- `random_state = 42`
- Stratification based on the sentiment label

This keeps the class distribution consistent between the two subsets.

---

## Inference Pipeline

The models are executed locally through **Ollama**.

Two locally available Ollama model aliases are used by the scripts:

```text
myqwen
myphi
```

For each review, the selected model receives the review text and generates a sentiment prediction.

The inference pipeline includes:

- Model selection
- Review-by-review inference
- Prediction extraction
- Progress tracking
- Intermediate backup files
- Final prediction CSV generation

Development-set predictions are stored in:

```text
results/
```

For example:

```text
results/final_qwen.csv
results/final_phi.csv
```

Backup files are also maintained during inference so that a long-running process can be resumed without losing previously generated results.

---

## Test-Set Evaluation

After the development-set inference stage, the models are evaluated on the held-out test set.

Test predictions are stored separately from development results:

```text
testset_results/
```

Example files:

```text
testset_results/final_qwen.csv
testset_results/final_phi.csv
```

Keeping test-set predictions separate helps prevent accidental mixing of development and final evaluation results.

---

## Evaluation Metrics

The evaluation pipeline calculates several standard classification metrics for each model:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

A classification report is also generated to provide a more detailed view of model performance.

### Confusion Matrix

Confusion matrices are generated to visualize:

- True Positives
- True Negatives
- False Positives
- False Negatives

These provide additional insight into the types of classification errors made by each model.

---

## Visualizations

The project generates several visualizations for model evaluation and comparison.

Generated charts are stored in:

```text
charts/
```

The evaluation pipeline supports visualizations such as:

- Model metric comparison
- Individual metric charts
- Confusion matrices
- Model performance comparisons

These visualizations make it easier to analyze differences between the two models beyond a single accuracy value.

---

## Project Structure

```text
amazon_sentiment_llm/
│
├── backups/
│   ├── backup_qwen.csv
│   └── backup_phi.csv
│
├── charts/
│   └── generated evaluation charts
│
├── data/
│   ├── Amazon_Reviews.csv
│   └── dataprocessed_reviews.csv
│
├── results/
│   ├── final_qwen.csv
│   └── final_phi.csv
│
├── testset_results/
│   ├── backup_qwen.csv
│   ├── backup_phi.csv
│   ├── final_qwen.csv
│   └── final_phi.csv
│
├── scripts/
│   ├── preprocessing.ipynb
│   ├── split.py
│   ├── inference.py
│   ├── test.py
│   ├── eval.py
│   ├── model_test.py
│   └── main.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Script Responsibilities

### `preprocessing.ipynb`

Handles dataset preprocessing and preparation.

Main tasks:

- Data loading
- Cleaning
- Rating conversion
- Sentiment labeling
- Class balancing
- Review length limitation
- Exporting the processed dataset

### `split.py`

Loads the processed dataset and creates the development/test split.

### `inference.py`

Runs model inference on the development set using Ollama.

It supports:

- Qwen
- Phi
- Prediction storage
- Progress tracking
- Backup files

### `test.py`

Runs the selected model on the held-out test set and stores the final test predictions.

### `eval.py`

Evaluates the saved test predictions using classification metrics and generates visualizations.

### `model_test.py`

Provides a simple sanity check to verify that the local Ollama models are accessible and responding correctly.

### `main.py`

Provides a simple command-line interface for running the evaluation workflow and generating different analysis outputs.

---

## Technologies & Libraries

### Programming

- Python

### Machine Learning & Data Processing

- Pandas
- NumPy
- Scikit-learn

### Visualization

- Matplotlib

### Local LLM Inference

- Ollama

### Development Environment

- Jupyter Notebook
- IPykernel

---

## Installation

Clone the repository:

```bash
git clone https://github.com/starcolder/amazon-review-sentiment-llm.git
cd amazon-review-sentiment-llm
```

Create and activate a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Ollama Setup

This project requires Ollama to run the language models locally.

After installing Ollama, make sure the required models are available locally.

The scripts expect the following Ollama model aliases:

```text
myqwen
myphi
```

You can verify that Ollama is working by running:

```bash
python scripts/model_test.py
```

Make sure the model names configured in the script match the models available in your local Ollama environment.

---

## Running the Project

### 1. Preprocess the Dataset

Open:

```text
scripts/preprocessing.ipynb
```

Run the notebook to generate:

```text
data/dataprocessed_reviews.csv
```

### 2. Verify the Local Models

Run:

```bash
python scripts/model_test.py
```

### 3. Create the Dataset Split

The development/test split is created through:

```python
from split import dev_df, test_df
```

The split uses a fixed random state to keep the results reproducible.

### 4. Run Development Inference

Run:

```bash
python scripts/inference.py
```

Select the model when prompted.

The resulting predictions are saved under:

```text
results/
```

### 5. Run Test-Set Inference

Run:

```bash
python scripts/test.py
```

The test predictions are saved under:

```text
testset_results/
```

### 6. Evaluate the Models

Run:

```bash
python scripts/main.py
```

The evaluation interface can be used to generate:

- Model comparisons
- Metric visualizations
- Confusion matrices
- Individual evaluation charts

Generated charts are stored in:

```text
charts/
```

---

## Reproducibility

Several steps in the pipeline use fixed random seeds:

```python
random_state = 42
```

This is used during:

- Dataset balancing
- Dataset shuffling
- Development/test splitting

This helps produce consistent dataset partitions and makes experiments easier to reproduce.

However, LLM inference itself can still depend on the local model configuration and inference environment.

---

## Important Notes

- The models are **not fine-tuned** on the Amazon review dataset.
- The project evaluates **local LLM inference**, rather than training a dedicated sentiment classifier.
- The `results/` directory contains development-set inference results.
- The `testset_results/` directory contains predictions generated on the held-out test set and used for final evaluation.
- Ollama must be installed and configured locally before running inference.
- Running inference over a large dataset can take significant time because reviews are processed through the local LLMs.

---

## Learning Outcomes

Through this project, I worked with:

- End-to-end machine learning workflows
- Text preprocessing and dataset cleaning
- Binary sentiment classification
- Dataset balancing
- Stratified train/test splitting
- Local LLM deployment with Ollama
- Prompt-based inference
- Automated prediction pipelines
- Backup and recovery mechanisms for long-running inference
- Classification metrics
- Confusion matrix analysis
- Data visualization with Matplotlib
- Comparative model evaluation
- Reproducible experimentation
- Python project organization

---

## Project Purpose

This project was developed as an exploration of how small instruction-tuned language models can be used for practical NLP classification tasks in a fully local environment.

Rather than focusing on model training, the project explores the complete workflow surrounding **data preparation, local LLM inference, evaluation, and model comparison**.
