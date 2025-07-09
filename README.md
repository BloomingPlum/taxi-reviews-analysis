# Taxi Reviews NLP Project

This project analyzes and classifies taxi service reviews using Natural Language Processing (NLP) techniques and BERT-based models. It includes data cleaning, exploratory data analysis (EDA), sentiment analysis, role (Rider/Driver) classification, and topic modeling.

## Project Structure

- `data/` — Contains raw and cleaned taxi review datasets.
- `results/` — Stores model checkpoints and training artifacts.
- `results_role/` — Stores checkpoints for the role classification model.
- `utils.py` — Utility functions for metrics and model explainability.
- `1.EDA.ipynb` — Exploratory Data Analysis of the dataset.
- `2.Bert_sentiment_analysis.ipynb` — Sentiment classification using BERT.
- `3.Bert_classification.ipynb` — Rider/Driver role classification using BERT.
- `4.Bert_topic_modelling.ipynb` — Topic modeling on reviews using BERT.

## Main Features

- **Data Cleaning:** Handles missing values, outliers, and inconsistent labels.
- **EDA:** Visualizes sentiment, role, and review length distributions.
- **Sentiment Analysis:** Classifies reviews as Positive, Negative, or Mixed using BERT.
- **Role Classification:** Predicts whether a review is from a Rider or Driver.
- **Topic Modeling:** Extracts main topics from reviews.
- **Model Explainability:** Uses transformers-interpret for model explanations.

## Requirements

- Python 3.8+
- pandas, numpy, seaborn, matplotlib
- scikit-learn
- transformers
- torch
- wordcloud
- transformers-interpret

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preparation & EDA:**
   - Run `1.EDA.ipynb` to clean and explore the data. Outputs `data/taxi_data_clean.csv`.
2. **Sentiment Analysis:**
   - Run `2.Bert_sentiment_analysis.ipynb` to train and evaluate sentiment models.
3. **Role Classification:**
   - Run `3.Bert_classification.ipynb` to classify reviews as Rider or Driver.
4. **Topic Modeling:**
   - Run `4.Bert_topic_modelling.ipynb` for topic extraction.


